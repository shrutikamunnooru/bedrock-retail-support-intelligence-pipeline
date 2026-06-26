"""
AWS Lambda: daily_ops_brief

Purpose:
    This Lambda generates a concise, business-facing daily operations brief by
    querying gold-layer metrics from Athena and then sending those metrics to
    Amazon Bedrock's text generation model (Nova Lite) to produce a narrative
    summary.

Why a text model is used here:
    The gold-layer tables (issue_kpis, city_issue_hotspots) contain numerical
    aggregates -- ticket counts, average delay hours, refund amounts, etc. While
    these numbers are useful in dashboards, operations leaders often need a
    written summary that highlights what matters most today: which issue category
    is highest, which cities are hotspots, and what actions to prioritize. The
    text model reads the structured metrics and generates a brief that a human
    would otherwise write manually each morning.

How the text model is used:
    1. Two Athena queries run against the gold-layer tables:
       - issue_kpis: top issue categories ranked by ticket volume and delay
       - city_issue_hotspots: top city + issue combinations
    2. The query results are formatted into a compact metrics block (plain text).
    3. This metrics block is sent to the Bedrock Converse API with a system
       prompt instructing the model to write a business-tone daily brief with:
       - a one-line headline
       - a paragraph on major issue patterns
       - a paragraph on actions to prioritize
       - explicit mention of the highest-risk issue category
    4. The generated brief is written to S3 as a markdown file.

    API used : Bedrock Converse API
    Model    : amazon.nova-lite-v1:0 (configurable via environment variable)

What the text model receives as input:
    A formatted text block like:
      Top issue KPI rows:
      - DELIVERY_DELAY: tickets=25, impacted_orders=22, avg_delay_hours=38.5, avg_refund_amount=0.0
      - DAMAGED_ITEM: tickets=12, impacted_orders=12, avg_delay_hours=0.0, avg_refund_amount=89.50
      Top city hotspot rows:
      - Mumbai / DELIVERY_DELAY: tickets=8, avg_delay_hours=42.1
      - Bengaluru / DAMAGED_ITEM: tickets=5, avg_delay_hours=0.0

What the text model generates:
    A markdown narrative such as:
      "Delivery delays remain the dominant issue category with 25 tickets..."

Auto-table creation:
    This function also creates the required Athena external tables (ticket_similarity,
    order_support_360, issue_kpis, city_issue_hotspots) automatically if they do not
    already exist. The Athena database must still be created manually beforehand.

Input:
    Reads gold-layer metrics via Athena queries against:
      - issue_kpis table
      - city_issue_hotspots table

Output:
    Writes the generated brief to:
      s3://<PIPELINE_BUCKET>/<GOLD_PREFIX>/daily_ops_brief/daily_operations_brief.md

Environment variables (set in the AWS Lambda console):
    PIPELINE_BUCKET        - S3 bucket name for all pipeline data
    BEDROCK_PREFIX         - prefix for Bedrock-generated outputs (default: bedrock)
    GOLD_PREFIX            - prefix for gold-layer analytical tables (default: gold)
    ATHENA_DATABASE        - Athena database name (default: retail_support_intelligence)
    ATHENA_WORKGROUP       - Athena workgroup (default: primary)
    ATHENA_RESULTS_PREFIX  - S3 prefix for Athena query results (default: athena-results)
    OPS_BRIEF_MODEL_ID     - Bedrock text model ID (default: amazon.nova-lite-v1:0)
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import boto3


# --------------------------------------------------------------------------
# AWS SDK clients
# athena_client  : runs SQL queries against gold-layer tables in Athena
# bedrock_client : invokes the Bedrock text model to generate the narrative
# s3_client      : writes the final operations brief back to S3
# --------------------------------------------------------------------------
athena_client = boto3.client("athena")
bedrock_client = boto3.client("bedrock-runtime")
s3_client = boto3.client("s3")


# --------------------------------------------------------------------------
# System prompt for the Bedrock text model.
# This instructs the model to produce a professional daily brief suitable for
# operations and support leadership -- not a chatbot response, not marketing
# copy, but a concise operational narrative with clear action items.
# --------------------------------------------------------------------------
SYSTEM_PROMPT = """
You are generating a concise daily operations brief for retail operations and
customer support leaders. Use a business tone, focus on actionability, and do
not use marketing language.
""".strip()


# --------------------------------------------------------------------------
# Athena query: fetch top issue categories ranked by ticket volume and delay.
# This provides the "what are the biggest problems" dimension of the brief.
# --------------------------------------------------------------------------
ISSUE_KPI_QUERY = """
SELECT issue_category,
       ticket_count,
       impacted_orders,
       average_delay_hours,
       average_order_amount,
       average_refund_amount
FROM issue_kpis
ORDER BY ticket_count DESC, average_delay_hours DESC
LIMIT 10
""".strip()


# --------------------------------------------------------------------------
# Athena query: fetch top city + issue category hotspots.
# This provides the "where are the problems concentrated" dimension.
# --------------------------------------------------------------------------
HOTSPOT_QUERY = """
SELECT city,
       issue_category,
       ticket_count,
       average_delay_hours
FROM city_issue_hotspots
ORDER BY ticket_count DESC, average_delay_hours DESC
LIMIT 10
""".strip()


def _athena_context() -> tuple[str, str, str, str]:
    """Return the common Athena execution context from environment variables."""
    database = os.environ["ATHENA_DATABASE"]
    workgroup = os.environ["ATHENA_WORKGROUP"]
    pipeline_bucket = os.environ["PIPELINE_BUCKET"]
    athena_results_prefix = os.environ["ATHENA_RESULTS_PREFIX"].strip("/")
    return database, workgroup, pipeline_bucket, athena_results_prefix


def _start_athena_execution(query: str) -> str:
    """Start an Athena query execution and return the execution ID."""
    database, workgroup, pipeline_bucket, athena_results_prefix = _athena_context()

    response = athena_client.start_query_execution(
        QueryString=query,
        QueryExecutionContext={"Database": database},
        WorkGroup=workgroup,
        ResultConfiguration={
            "OutputLocation": f"s3://{pipeline_bucket}/{athena_results_prefix}/",
        },
    )
    return response["QueryExecutionId"]


def _wait_for_athena_success(query_execution_id: str) -> None:
    """Poll Athena until the submitted statement completes."""
    while True:
        status_response = athena_client.get_query_execution(QueryExecutionId=query_execution_id)
        state = status_response["QueryExecution"]["Status"]["State"]

        if state == "SUCCEEDED":
            return
        if state in {"FAILED", "CANCELLED"}:
            reason = status_response["QueryExecution"]["Status"].get("StateChangeReason", "Unknown Athena failure")
            raise RuntimeError(f"Athena query failed with state {state}: {reason}")

        time.sleep(2)


def execute_athena_statement(query: str) -> None:
    """Execute Athena SQL where the result rows are not needed."""
    query_execution_id = _start_athena_execution(query)
    _wait_for_athena_success(query_execution_id)


def run_athena_query(query: str) -> list[dict[str, str]]:
    """Execute Athena SQL and return rows as dictionaries."""
    query_execution_id = _start_athena_execution(query)
    _wait_for_athena_success(query_execution_id)

    result_response = athena_client.get_query_results(QueryExecutionId=query_execution_id)
    rows = result_response["ResultSet"]["Rows"]
    headers = [cell.get("VarCharValue", "") for cell in rows[0]["Data"]]

    parsed_rows: list[dict[str, str]] = []
    for row in rows[1:]:
        values = [cell.get("VarCharValue", "") for cell in row["Data"]]
        parsed_rows.append(dict(zip(headers, values)))

    return parsed_rows


def build_required_table_ddls() -> list[str]:
    """Create the minimal Athena tables required by this Lambda and query pack.

    These are created lazily so Step Functions orchestration can succeed even if
    the user created only the Athena database manually beforehand.
    """
    database = os.environ["ATHENA_DATABASE"]
    bucket = os.environ["PIPELINE_BUCKET"]
    bedrock_prefix = os.environ["BEDROCK_PREFIX"].strip("/")
    gold_prefix = os.environ["GOLD_PREFIX"].strip("/")

    return [
        f"""
        CREATE EXTERNAL TABLE IF NOT EXISTS {database}.ticket_similarity (
          ticket_id string,
          matched_ticket_id string,
          ticket_issue_category string,
          matched_issue_category string,
          similarity_rank int,
          similarity_score double,
          ticket_summary string,
          matched_ticket_summary string,
          recommended_action_pattern string
        )
        ROW FORMAT SERDE 'org.openx.data.jsonserde.JsonSerDe'
        LOCATION 's3://{bucket}/{bedrock_prefix}/ticket_similarity/'
        """.strip(),
        f"""
        CREATE EXTERNAL TABLE IF NOT EXISTS {database}.order_support_360 (
          order_id string,
          customer_id string,
          customer_name string,
          product_id string,
          product_name string,
          product_category string,
          order_date string,
          city string,
          quantity int,
          unit_price double,
          order_amount double,
          payment_status string,
          shipment_id string,
          carrier string,
          dispatch_time string,
          expected_delivery_time string,
          delivery_time string,
          delay_hours double,
          shipment_status string,
          return_id string,
          reason_code string,
          refund_amount double,
          return_status string,
          ticket_id string,
          issue_category string,
          sentiment string,
          urgency string,
          requires_human_followup boolean,
          root_cause_hint string,
          customer_summary string,
          next_best_action string,
          most_similar_ticket_id string,
          most_similar_ticket_score double,
          has_ticket boolean,
          has_return boolean,
          risk_flag boolean
        )
        ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
        WITH SERDEPROPERTIES (
          'separatorChar' = ',',
          'quoteChar' = '"',
          'escapeChar' = '\\\\'
        )
        LOCATION 's3://{bucket}/{gold_prefix}/order_support_360/'
        TBLPROPERTIES (
          'skip.header.line.count'='1',
          'use.null.for.invalid.data'='true'
        )
        """.strip(),
        f"""
        CREATE EXTERNAL TABLE IF NOT EXISTS {database}.issue_kpis (
          issue_category string,
          ticket_count int,
          impacted_orders int,
          average_delay_hours double,
          average_order_amount double,
          average_refund_amount double
        )
        ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
        WITH SERDEPROPERTIES (
          'separatorChar' = ',',
          'quoteChar' = '"',
          'escapeChar' = '\\\\'
        )
        LOCATION 's3://{bucket}/{gold_prefix}/issue_kpis/'
        TBLPROPERTIES (
          'skip.header.line.count'='1',
          'use.null.for.invalid.data'='true'
        )
        """.strip(),
        f"""
        CREATE EXTERNAL TABLE IF NOT EXISTS {database}.city_issue_hotspots (
          city string,
          issue_category string,
          ticket_count int,
          average_delay_hours double
        )
        ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
        WITH SERDEPROPERTIES (
          'separatorChar' = ',',
          'quoteChar' = '"',
          'escapeChar' = '\\\\'
        )
        LOCATION 's3://{bucket}/{gold_prefix}/city_issue_hotspots/'
        TBLPROPERTIES (
          'skip.header.line.count'='1',
          'use.null.for.invalid.data'='true'
        )
        """.strip(),
    ]


def ensure_required_athena_tables() -> None:
    """Create required external tables if they do not already exist."""
    for ddl in build_required_table_ddls():
        execute_athena_statement(ddl)


def build_metrics_block(issue_rows: list[dict[str, str]], hotspot_rows: list[dict[str, str]]) -> str:
    """Format Athena output into a compact prompt payload for Bedrock."""
    issue_lines = [
        f"- {row['issue_category']}: tickets={row['ticket_count']}, impacted_orders={row['impacted_orders']}, "
        f"avg_delay_hours={row['average_delay_hours']}, avg_refund_amount={row['average_refund_amount']}"
        for row in issue_rows
    ]
    hotspot_lines = [
        f"- {row['city']} / {row['issue_category']}: tickets={row['ticket_count']}, avg_delay_hours={row['average_delay_hours']}"
        for row in hotspot_rows
    ]

    return "\n".join(
        [
            "Top issue KPI rows:",
            *issue_lines,
            "",
            "Top city hotspot rows:",
            *hotspot_lines,
        ]
    )


def generate_brief(metrics_block: str) -> str:
    """Send the formatted metrics block to the Bedrock text model and return
    the generated daily operations brief.

    The text model receives:
      - A system prompt defining the tone and format (business, actionable)
      - A user prompt containing the actual metrics data plus instructions
        to write a headline, issue pattern paragraph, and action paragraph

    The model temperature is set to 0.2 -- slightly creative to produce
    readable prose, but still grounded closely in the input data.
    """
    model_id = os.environ["OPS_BRIEF_MODEL_ID"]
    user_prompt = f"""
Write a daily operations brief using the curated metrics below.

Requirements:
- Start with a one-line headline
- Then write one paragraph on major issue patterns
- Then write one paragraph on actions to prioritize today
- Mention the highest-risk issue category explicitly

Metrics:
{metrics_block}
""".strip()

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
            "temperature": 0.2,
            "maxTokens": 650,
        },
    )
    return response["output"]["message"]["content"][0]["text"]


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """Main Lambda entry point.

    Orchestration flow:
    1. Create required Athena external tables if they do not already exist
       (ticket_similarity, order_support_360, issue_kpis, city_issue_hotspots).
    2. Run two Athena queries to pull aggregated metrics from the gold layer.
    3. Format the query results into a compact text block.
    4. Send the metrics block to the Bedrock text model to generate a narrative.
    5. Write the generated brief to S3 as a markdown file.
    """
    # Step 1: Ensure Athena tables exist (idempotent, uses IF NOT EXISTS)
    ensure_required_athena_tables()

    # Step 2: Query gold-layer metrics via Athena
    issue_rows = run_athena_query(ISSUE_KPI_QUERY)
    hotspot_rows = run_athena_query(HOTSPOT_QUERY)

    # Step 3: Format metrics into a compact text block for the text model
    metrics_block = build_metrics_block(issue_rows, hotspot_rows)

    # Step 4: Generate the narrative brief using the Bedrock text model
    brief = generate_brief(metrics_block)

    # Step 5: Write the brief to S3
    bucket = os.environ["PIPELINE_BUCKET"]
    gold_prefix = os.environ["GOLD_PREFIX"].strip("/")
    output_key = f"{gold_prefix}/daily_ops_brief/daily_operations_brief.md"

    s3_client.put_object(Bucket=bucket, Key=output_key, Body=brief.encode("utf-8"))

    return {
        "status": "SUCCESS",
        "output_s3_uri": f"s3://{bucket}/{output_key}",
        "issue_rows_used": len(issue_rows),
        "hotspot_rows_used": len(hotspot_rows),
    }
