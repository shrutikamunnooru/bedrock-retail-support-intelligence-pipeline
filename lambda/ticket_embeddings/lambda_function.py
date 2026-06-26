"""
AWS Lambda: ticket_embeddings

Purpose:
    This Lambda converts enriched support ticket text into numerical vector
    representations (embeddings) using Amazon Bedrock's embedding model
    (Titan Text Embeddings V2). These vectors capture the semantic meaning
    of each ticket so that tickets with similar issues can be identified
    mathematically in the next pipeline stage.

Why embeddings are needed:
    After the ticket_enrichment Lambda produces structured fields like
    issue_category and customer_summary, the pipeline needs a way to find
    tickets that are semantically similar -- not just tickets with the same
    category label, but tickets where the actual meaning of the customer's
    complaint is close. For example, "my package never arrived" and "the
    courier marked it delivered but I did not receive it" are semantically
    similar even though the exact words differ. Embeddings make this possible.

How the embedding model is used:
    For each enriched ticket, a composite text string is built by combining:
      - issue_category   (e.g. "DELIVERY_DELAY")
      - customer_summary (e.g. "Customer reports package delayed by two days")
      - pii_redacted_message (the cleaned original message)

    This combined text is sent to the Bedrock InvokeModel API, which returns
    a high-dimensional float vector (the embedding). Each ticket gets one
    vector. These vectors are written to S3 as JSONL records.

    API used : Bedrock InvokeModel API (not Converse, because embedding
               models use the InvokeModel interface)
    Model    : amazon.titan-embed-text-v2:0 (configurable via env variable)

Who consumes this output:
    - ticket_similarity Lambda reads these embedding vectors, computes cosine
      similarity between every pair of tickets, and identifies the top N most
      similar historical tickets for each one.

Input:
    Reads enriched ticket records (JSONL) from:
      s3://<PIPELINE_BUCKET>/<BEDROCK_PREFIX>/ticket_enrichment/

Output:
    Writes embedding records (JSONL with ticket_id + vector) to:
      s3://<PIPELINE_BUCKET>/<BEDROCK_PREFIX>/ticket_embeddings/ticket_embeddings.jsonl

Environment variables (set in the AWS Lambda console):
    PIPELINE_BUCKET            - S3 bucket name for all pipeline data
    BEDROCK_PREFIX             - prefix for Bedrock-generated outputs (default: bedrock)
    TICKET_EMBEDDING_MODEL_ID  - Bedrock embedding model ID (default: amazon.titan-embed-text-v2:0)
"""

from __future__ import annotations

import json
import os
from typing import Any

import boto3


# --------------------------------------------------------------------------
# AWS SDK clients
# s3_client      : reads enriched tickets, writes embedding output back to S3
# bedrock_client : invokes the Bedrock embedding model via InvokeModel API
# --------------------------------------------------------------------------
s3_client = boto3.client("s3")
bedrock_client = boto3.client("bedrock-runtime")


def list_object_keys(bucket: str, prefix: str) -> list[str]:
    """List all actual files under a prefix."""
    paginator = s3_client.get_paginator("list_objects_v2")
    keys: list[str] = []

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/") or key.endswith("_SUCCESS"):
                continue
            keys.append(key)

    return sorted(keys)


def read_json_lines_from_s3(bucket: str, prefix: str) -> list[dict[str, Any]]:
    """Read all JSONL files under the prefix."""
    records: list[dict[str, Any]] = []

    for key in list_object_keys(bucket, prefix):
        body = s3_client.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
        for line in body.splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return records


def invoke_embedding_model(text: str) -> list[float]:
    """Invoke the Titan Text Embeddings V2 model via the Bedrock InvokeModel API.

    How this works:
    1. The input text (a composite string of issue_category + summary + message)
       is sent to the embedding model as {"inputText": text}.
    2. The model returns a JSON response containing an "embedding" key whose
       value is a list of floats -- a high-dimensional vector that represents
       the semantic meaning of the input text.
    3. This vector is returned to the caller for storage and later similarity
       computation.

    Note: The embedding model uses InvokeModel (not Converse). The Converse API
    is designed for text generation models that produce conversational responses.
    Embedding models return numerical vectors, so they use the simpler InvokeModel
    interface with a direct JSON body.
    """
    model_id = os.environ["TICKET_EMBEDDING_MODEL_ID"]
    response = bedrock_client.invoke_model(
        modelId=model_id,
        body=json.dumps({"inputText": text}),
    )
    payload = json.loads(response["body"].read())
    return payload["embedding"]


def write_json_lines_to_s3(bucket: str, key: str, records: list[dict[str, Any]]) -> None:
    """Write newline-delimited JSON for downstream Lambda and Athena use."""
    payload = "\n".join(json.dumps(record) for record in records)
    s3_client.put_object(Bucket=bucket, Key=key, Body=payload.encode("utf-8"))


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """Main Lambda entry point.

    Orchestration flow:
    1. Read enriched ticket records produced by the ticket_enrichment Lambda.
    2. For each record, build a composite text string that combines the
       issue_category, customer_summary, and pii_redacted_message. This gives
       the embedding model enough context to capture the full semantic meaning.
    3. Send the composite text to the Bedrock embedding model. The model returns
       a numerical vector (list of floats) for each ticket.
    4. Write ticket_id, order_id, issue_category, and the embedding vector to
       S3 as JSONL. The ticket_similarity Lambda reads this file next.

    Optional event overrides (useful for Step Functions or manual testing):
      - pipeline_bucket
      - bedrock_prefix
    """
    bucket = event.get("pipeline_bucket", os.environ["PIPELINE_BUCKET"])
    bedrock_prefix = event.get("bedrock_prefix", os.environ["BEDROCK_PREFIX"]).strip("/")

    # Step 1: Read enriched tickets (output of ticket_enrichment Lambda)
    enrichment_prefix = f"{bedrock_prefix}/ticket_enrichment/"
    enriched_records = read_json_lines_from_s3(bucket, enrichment_prefix)

    output_records: list[dict[str, Any]] = []
    for record in enriched_records:
        # Step 2: Build a composite text for embedding.
        # Combining category + summary + message gives the embedding model a
        # richer input, producing vectors that reflect both the issue type and
        # the specific details of the customer's complaint.
        text_for_embedding = (
            f"Issue category: {record['issue_category']}. "
            f"Summary: {record['customer_summary']}. "
            f"Customer message: {record['pii_redacted_message']}"
        )

        # Step 3: Get the embedding vector from the Bedrock embedding model
        embedding = invoke_embedding_model(text_for_embedding)

        output_records.append(
            {
                "ticket_id": record["ticket_id"],
                "order_id": record["order_id"],
                "issue_category": record["issue_category"],
                "embedding": embedding,
            }
        )

    # Step 4: Write embedding records to S3 for the ticket_similarity Lambda
    output_key = f"{bedrock_prefix}/ticket_embeddings/ticket_embeddings.jsonl"
    write_json_lines_to_s3(bucket, output_key, output_records)

    return {
        "status": "SUCCESS",
        "embedding_count": len(output_records),
        "output_s3_uri": f"s3://{bucket}/{output_key}",
    }
