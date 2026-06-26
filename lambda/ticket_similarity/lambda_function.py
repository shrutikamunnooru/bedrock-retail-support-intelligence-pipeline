"""
AWS Lambda: ticket_similarity

Purpose:
    This Lambda identifies semantically similar support tickets by comparing the
    embedding vectors produced by the ticket_embeddings Lambda. It uses cosine
    similarity (a standard measure of vector closeness) to rank every ticket
    against all other tickets and retains the top N closest matches per ticket.

Why this matters:
    When a new support ticket arrives, operations teams benefit from knowing
    which past tickets had a similar complaint and what action resolved them.
    Instead of relying on exact keyword matches or category labels alone, this
    function uses the embedding vectors (which capture semantic meaning) to find
    truly similar cases. For example, a ticket about "package never arrived" will
    match highly with "courier marked delivered but I did not receive it" because
    their embeddings are close in vector space.

How it works:
    1. Reads embedding vectors from ticket_embeddings output (each record has
       a ticket_id and a float vector).
    2. Reads enrichment records from ticket_enrichment output (to look up
       customer_summary and next_best_action for matched tickets).
    3. For each ticket, computes cosine similarity against every other ticket's
       embedding vector.
    4. Ranks matches by similarity score and keeps the top N (default 3).
    5. Writes a JSONL output where each record is a (ticket, matched_ticket)
       pair with the similarity score, both summaries, and the matched ticket's
       recommended action.

    No Bedrock model is called in this Lambda. The embedding model was already
    used in the previous stage. This function only performs mathematical
    comparison of the vectors that were already generated.

Who consumes this output:
    - build_gold_support_mart Glue job (joins the top-1 similar ticket into
      the order_support_360 gold table)
    - Athena analytical queries (04_similarity_patterns.sql aggregates
      cross-category similarity patterns)
    - daily_ops_brief Lambda indirectly benefits because the gold mart it
      queries includes similarity data

Input:
    - s3://<PIPELINE_BUCKET>/<BEDROCK_PREFIX>/ticket_embeddings/  (vectors)
    - s3://<PIPELINE_BUCKET>/<BEDROCK_PREFIX>/ticket_enrichment/  (summaries, actions)

Output:
    - s3://<PIPELINE_BUCKET>/<BEDROCK_PREFIX>/ticket_similarity/ticket_similarity.jsonl

Environment variables (set in the AWS Lambda console):
    PIPELINE_BUCKET        - S3 bucket name for all pipeline data
    BEDROCK_PREFIX         - prefix for Bedrock-generated outputs (default: bedrock)
    TOP_MATCHES_PER_TICKET - number of similar tickets to retain per ticket (default: 3)
"""

from __future__ import annotations

import json
import math
import os
from typing import Any

import boto3


# --------------------------------------------------------------------------
# AWS SDK client
# s3_client : reads embedding and enrichment data, writes similarity output
# No Bedrock client is needed here -- all model work was done earlier.
# --------------------------------------------------------------------------
s3_client = boto3.client("s3")


def list_object_keys(bucket: str, prefix: str) -> list[str]:
    """List all real S3 objects under the prefix."""
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
    """Read JSONL records from one or more objects."""
    records: list[dict[str, Any]] = []

    for key in list_object_keys(bucket, prefix):
        body = s3_client.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
        for line in body.splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return records


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Compute cosine similarity between two embedding vectors.

    Cosine similarity measures how close two vectors are in direction (not
    magnitude). A score of 1.0 means the vectors point in the same direction
    (identical meaning), 0.0 means they are orthogonal (unrelated), and -1.0
    means they point in opposite directions.

    In this pipeline, scores typically range from 0.7 to 1.0 because all
    tickets are about retail support issues and share some baseline similarity.

    This implementation uses only Python standard library (no numpy) so
    the Lambda does not require any external package layers.
    """
    dot_product = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))

    if left_norm == 0 or right_norm == 0:
        return 0.0

    return dot_product / (left_norm * right_norm)


def write_json_lines_to_s3(bucket: str, key: str, records: list[dict[str, Any]]) -> None:
    """Persist the similarity output as JSONL."""
    payload = "\n".join(json.dumps(record) for record in records)
    s3_client.put_object(Bucket=bucket, Key=key, Body=payload.encode("utf-8"))


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """Main Lambda entry point.

    Orchestration flow:
    1. Load embedding vectors (from ticket_embeddings Lambda output).
    2. Load enrichment records (from ticket_enrichment Lambda output) and
       build a lookup dictionary so summaries and actions can be attached
       to each similarity pair.
    3. For each ticket, compute cosine similarity against every other ticket.
    4. Keep the top N matches (ranked by score, highest first).
    5. Write the similarity pairs to S3 as JSONL.

    Each output record contains:
      - ticket_id and matched_ticket_id
      - both issue categories (useful for cross-category pattern analysis)
      - similarity_rank (1 = most similar)
      - similarity_score (0 to 1 float)
      - both customer summaries (human-readable context)
      - recommended_action_pattern from the matched ticket (so operations
        teams can see what action was suggested for a similar past case)

    Optional event overrides:
      - pipeline_bucket
      - bedrock_prefix
      - top_matches_per_ticket
    """
    bucket = event.get("pipeline_bucket", os.environ["PIPELINE_BUCKET"])
    bedrock_prefix = event.get("bedrock_prefix", os.environ["BEDROCK_PREFIX"]).strip("/")
    top_matches = int(event.get("top_matches_per_ticket", os.environ.get("TOP_MATCHES_PER_TICKET", "3")))

    # Step 1: Load embedding vectors produced by the ticket_embeddings Lambda
    embeddings = read_json_lines_from_s3(bucket, f"{bedrock_prefix}/ticket_embeddings/")

    # Step 2: Load enrichment records for summary and action lookups
    enrichments = read_json_lines_from_s3(bucket, f"{bedrock_prefix}/ticket_enrichment/")
    enrichment_lookup = {record["ticket_id"]: record for record in enrichments}

    # Step 3 & 4: Compare every ticket pair and keep top N matches
    similarity_records: list[dict[str, Any]] = []
    for left in embeddings:
        scored_matches: list[tuple[float, dict[str, Any]]] = []

        for right in embeddings:
            # Skip self-comparison
            if left["ticket_id"] == right["ticket_id"]:
                continue

            score = cosine_similarity(left["embedding"], right["embedding"])
            scored_matches.append((score, right))

        # Sort by similarity score descending and keep the top N
        top_scored_matches = sorted(scored_matches, key=lambda item: item[0], reverse=True)[:top_matches]
        for rank, (score, match) in enumerate(top_scored_matches, start=1):
            similarity_records.append(
                {
                    "ticket_id": left["ticket_id"],
                    "matched_ticket_id": match["ticket_id"],
                    "ticket_issue_category": left["issue_category"],
                    "matched_issue_category": match["issue_category"],
                    "similarity_rank": rank,
                    "similarity_score": round(score, 6),
                    # Include human-readable summaries for both tickets
                    "ticket_summary": enrichment_lookup[left["ticket_id"]]["customer_summary"],
                    "matched_ticket_summary": enrichment_lookup[match["ticket_id"]]["customer_summary"],
                    # The action that was recommended for the similar past ticket --
                    # useful for operations teams to see what worked before
                    "recommended_action_pattern": enrichment_lookup[match["ticket_id"]]["next_best_action"],
                }
            )

    # Step 5: Write similarity output to S3
    output_key = f"{bedrock_prefix}/ticket_similarity/ticket_similarity.jsonl"
    write_json_lines_to_s3(bucket, output_key, similarity_records)

    return {
        "status": "SUCCESS",
        "pair_count": len(similarity_records),
        "output_s3_uri": f"s3://{bucket}/{output_key}",
    }
