"""
AWS Lambda: ticket_enrichment

Purpose:
    This Lambda converts unstructured customer support ticket text into structured,
    analytics-ready fields using Amazon Bedrock's text generation model (Nova Lite).

What problem does this solve:
    Raw support tickets contain free-text customer messages like "my order is delayed"
    or "the wrong item was delivered". These messages cannot be directly used in SQL
    queries, dashboards, or KPI calculations. This function sends each ticket's text
    to a Bedrock text model, which returns structured JSON containing:
      - issue_category   (e.g. DELIVERY_DELAY, DAMAGED_ITEM, WRONG_ITEM_SENT)
      - sentiment        (POSITIVE / NEUTRAL / NEGATIVE)
      - urgency          (LOW / MEDIUM / HIGH)
      - root_cause_hint  (a short phrase identifying the likely cause)
      - pii_redacted_message (the original message with phone/email masked)
      - customer_summary (a single-sentence plain-English summary)
      - next_best_action (a concrete recommendation for the operations team)

How the text model is used:
    For each support ticket, a prompt is constructed containing the ticket_id,
    order_id, channel, customer_message, and agent_notes. This prompt is sent to
    the Bedrock Converse API along with a system prompt that instructs the model
    to return strict JSON with the required keys. The model acts as a classification
    and extraction engine -- it reads the free-text message and produces structured
    fields that downstream Glue jobs and Athena queries can consume.

    API used : Bedrock Converse API
    Model    : amazon.nova-lite-v1:0 (configurable via environment variable)

Who consumes this output:
    - ticket_embeddings Lambda (reads the enriched output to build embedding vectors)
    - ticket_similarity Lambda (reads enrichment for summary and action fields)
    - build_gold_support_mart Glue job (joins enrichment with orders, shipments, returns)
    - Athena analytical queries (query issue_category, urgency, sentiment, etc.)

Input:
    Reads silver-layer support tickets (JSONL) from:
      s3://<PIPELINE_BUCKET>/<SILVER_PREFIX>/support_tickets/

Output:
    Writes enriched ticket records (JSONL) to:
      s3://<PIPELINE_BUCKET>/<BEDROCK_PREFIX>/ticket_enrichment/ticket_enrichment.jsonl

Environment variables (set in the AWS Lambda console):
    PIPELINE_BUCKET              - S3 bucket name for all pipeline data
    SILVER_PREFIX                - prefix for cleaned silver-layer data (default: silver)
    BEDROCK_PREFIX               - prefix for Bedrock-generated outputs (default: bedrock)
    TICKET_ENRICHMENT_MODEL_ID   - Bedrock text model ID (default: amazon.nova-lite-v1:0)
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
from typing import Any

import boto3


# --------------------------------------------------------------------------
# AWS SDK clients
# s3_client      : reads silver tickets, writes enriched output back to S3
# bedrock_client : invokes the Bedrock text model via the Converse API
# --------------------------------------------------------------------------
s3_client = boto3.client("s3")
bedrock_client = boto3.client("bedrock-runtime")


# --------------------------------------------------------------------------
# System prompt sent to the Bedrock text model with every request.
# This prompt instructs the model to behave as a structured data extraction
# engine rather than a conversational assistant. It defines:
#   - the exact JSON keys the model must return
#   - allowed values for categorical fields (issue_category, sentiment, urgency)
#   - rules for PII redaction, summary length, and action specificity
# The model receives this system prompt once per ticket alongside the
# user prompt containing the actual ticket data.
# --------------------------------------------------------------------------
SYSTEM_PROMPT = """
You are enriching retail support tickets for a downstream analytics pipeline.
Return JSON only.

Required keys:
- issue_category
- sentiment
- urgency
- requires_human_followup
- root_cause_hint
- pii_redacted_message
- customer_summary
- next_best_action

Allowed issue_category values:
- DELIVERY_DELAY
- DAMAGED_ITEM
- WRONG_ITEM_SENT
- DELIVERY_EXCEPTION
- REFUND_DELAY
- PAYMENT_ISSUE
- RETURN_PICKUP_DELAY
- REPLACEMENT_REQUEST
- GENERAL_QUERY

Allowed sentiment values:
- POSITIVE
- NEUTRAL
- NEGATIVE

Allowed urgency values:
- LOW
- MEDIUM
- HIGH

Rules:
- pii_redacted_message must mask phone numbers and email addresses
- customer_summary must be one sentence
- next_best_action must be practical for an operations team
""".strip()


def list_object_keys(bucket: str, prefix: str) -> list[str]:
    """List all S3 keys under a prefix, excluding folder placeholders."""
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
    """Read newline-delimited JSON records from all files under a prefix."""
    records: list[dict[str, Any]] = []

    for key in list_object_keys(bucket, prefix):
        body = s3_client.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
        for line in body.splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return records


def redact_pii(text: str) -> str:
    """Apply a defensive local redaction step before writing downstream output."""
    text = re.sub(r"\b\d{10}\b", "[REDACTED_PHONE]", text)
    text = re.sub(r"[\w\.-]+@[\w\.-]+\.\w+", "[REDACTED_EMAIL]", text)
    return text


def extract_json_object(raw_text: str) -> dict[str, Any]:
    """Parse JSON even if the model wraps it in markdown fences."""
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```json", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"^```", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    return json.loads(cleaned)


def invoke_bedrock_enrichment(ticket: dict[str, Any]) -> dict[str, Any]:
    """Call the Bedrock Converse API for one ticket record.

    What happens here step by step:
    1. A user prompt is built containing the ticket's raw fields: ticket_id,
       order_id, channel, customer_message, and agent_notes.
    2. This user prompt is sent to the Bedrock Converse API together with the
       SYSTEM_PROMPT (which defines the expected JSON schema and rules).
    3. The text model reads the free-text customer message and agent notes,
       then generates a JSON response with structured fields like issue_category,
       sentiment, urgency, root_cause_hint, customer_summary, etc.
    4. The JSON response is parsed and a safety-net PII redaction pass is applied
       to ensure no phone numbers or emails leak into downstream datasets.

    The model temperature is set to 0.1 (near-deterministic) because this is a
    classification task, not creative generation.
    """
    model_id = os.environ["TICKET_ENRICHMENT_MODEL_ID"]

    # -----------------------------------------------------------------------
    # Build the user prompt with the ticket's raw data.
    # The model uses these fields to classify the issue, assess sentiment and
    # urgency, and generate a summary and recommended action.
    # -----------------------------------------------------------------------
    user_prompt = f"""
Enrich this support ticket for downstream analytics.

Ticket ID: {ticket['ticket_id']}
Order ID: {ticket['order_id']}
Channel: {ticket['channel']}
Customer message: {ticket['customer_message']}
Agent notes: {ticket['agent_notes']}
""".strip()

    # -----------------------------------------------------------------------
    # Invoke the Bedrock Converse API.
    # - system  : the SYSTEM_PROMPT defining output schema and rules
    # - messages: the user prompt with the ticket data
    # - temperature 0.1: near-deterministic for consistent classification
    # - maxTokens 500 : sufficient for the structured JSON output
    # -----------------------------------------------------------------------
    response = bedrock_client.converse(
        modelId=model_id,
        system=[{"text": SYSTEM_PROMPT}],
        messages=[
            {
                "role": "user",
                "content": [{"text": user_prompt}],
            }
        ],
        inferenceConfig={
            "temperature": 0.1,
            "maxTokens": 500,
        },
    )

    # -----------------------------------------------------------------------
    # Parse the model's JSON response.
    # The model may wrap the JSON in markdown code fences (```json ... ```),
    # so extract_json_object strips those before parsing.
    # -----------------------------------------------------------------------
    raw_text = response["output"]["message"]["content"][0]["text"]
    parsed = extract_json_object(raw_text)

    # Safety-net PII redaction: even though the model is instructed to redact
    # phone numbers and emails, a local regex pass ensures nothing leaks
    # into downstream analytics if the model misses any.
    parsed["pii_redacted_message"] = redact_pii(parsed["pii_redacted_message"])
    return parsed


def write_json_lines_to_s3(bucket: str, key: str, records: list[dict[str, Any]]) -> None:
    """Write newline-delimited JSON so Athena can query the output easily."""
    payload = "\n".join(json.dumps(record) for record in records)
    s3_client.put_object(Bucket=bucket, Key=key, Body=payload.encode("utf-8"))


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """Main Lambda entry point.

    Orchestration flow:
    1. Read all silver-layer support ticket records from S3 (JSONL format).
    2. For each ticket, call the Bedrock text model to generate structured fields.
    3. Combine original ticket metadata with the model-generated enrichment fields.
    4. Write the combined enriched records back to S3 as a single JSONL file.

    The output file is consumed by:
      - ticket_embeddings Lambda (to generate vector embeddings)
      - ticket_similarity Lambda (to look up summaries and actions)
      - build_gold_support_mart Glue job (to join with orders, shipments, returns)

    Optional event overrides (useful for Step Functions or manual testing):
      - pipeline_bucket
      - silver_prefix
      - bedrock_prefix
    """
    bucket = event.get("pipeline_bucket", os.environ["PIPELINE_BUCKET"])
    silver_prefix = event.get("silver_prefix", os.environ["SILVER_PREFIX"]).strip("/")
    bedrock_prefix = event.get("bedrock_prefix", os.environ["BEDROCK_PREFIX"]).strip("/")

    # Step 1: Read silver-layer support tickets
    ticket_prefix = f"{silver_prefix}/support_tickets/"
    tickets = read_json_lines_from_s3(bucket, ticket_prefix)

    # Step 2 & 3: Enrich each ticket via the Bedrock text model and merge fields
    enriched_records: list[dict[str, Any]] = []
    for ticket in tickets:
        enrichment = invoke_bedrock_enrichment(ticket)
        enriched_records.append(
            {
                # Original ticket metadata (carried forward for downstream joins)
                "ticket_id": ticket["ticket_id"],
                "order_id": ticket["order_id"],
                "customer_id": ticket["customer_id"],
                "created_at": ticket["created_at"],
                "channel": ticket["channel"],
                # Bedrock text model generated fields
                "issue_category": enrichment["issue_category"],
                "sentiment": enrichment["sentiment"],
                "urgency": enrichment["urgency"],
                "requires_human_followup": enrichment["requires_human_followup"],
                "root_cause_hint": enrichment["root_cause_hint"],
                "pii_redacted_message": enrichment["pii_redacted_message"],
                "customer_summary": enrichment["customer_summary"],
                "next_best_action": enrichment["next_best_action"],
                # Original agent notes (preserved for reference)
                "agent_notes": ticket["agent_notes"],
            }
        )

    # Step 4: Write enriched output to S3 for downstream consumers
    output_key = f"{bedrock_prefix}/ticket_enrichment/ticket_enrichment.jsonl"
    write_json_lines_to_s3(bucket, output_key, enriched_records)

    return {
        "status": "SUCCESS",
        "ticket_count": len(enriched_records),
        "output_s3_uri": f"s3://{bucket}/{output_key}",
    }
