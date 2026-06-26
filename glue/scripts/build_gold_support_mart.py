"""
AWS Glue Job: build_gold_support_mart

Purpose:
    This Glue job creates the gold-layer analytical tables by joining silver-layer
    operational data (orders, shipments, returns) with Bedrock-generated outputs
    (ticket enrichment and ticket similarity). The result is a set of analytics-
    ready datasets that can be queried directly in Athena and consumed by the
    daily_ops_brief Lambda.

What this job produces (three gold tables):

    1. order_support_360
       A wide, denormalized table with one row per order. Each order is enriched
       with its shipment details, return details (if any), Bedrock-generated
       ticket enrichment fields (issue_category, sentiment, urgency, summary,
       next_best_action), and the most similar historical ticket (from the
       similarity output). Also includes derived flags:
         - has_ticket  : whether the order has an associated support ticket
         - has_return  : whether the order has an associated return
         - risk_flag   : true if the order has a ticket, a return, or a delay > 24h

    2. issue_kpis
       Aggregated metrics per issue_category: ticket count, impacted order count,
       average delay hours, average order amount, and average refund amount.
       This table is queried by the daily_ops_brief Lambda to build the
       operations narrative.

    3. city_issue_hotspots
       Aggregated metrics per (city, issue_category): ticket count and average
       delay hours. This table is also queried by the daily_ops_brief Lambda
       to identify geographic concentrations of specific issues.

Input (read from S3):
    Silver layer:
      s3://<PIPELINE_BUCKET>/<SILVER_PREFIX>/orders/
      s3://<PIPELINE_BUCKET>/<SILVER_PREFIX>/shipments/
      s3://<PIPELINE_BUCKET>/<SILVER_PREFIX>/returns/
    Bedrock layer:
      s3://<PIPELINE_BUCKET>/<BEDROCK_PREFIX>/ticket_enrichment/
      s3://<PIPELINE_BUCKET>/<BEDROCK_PREFIX>/ticket_similarity/

Output (written to S3 gold zone):
    s3://<PIPELINE_BUCKET>/<GOLD_PREFIX>/order_support_360/   (CSV)
    s3://<PIPELINE_BUCKET>/<GOLD_PREFIX>/issue_kpis/          (CSV)
    s3://<PIPELINE_BUCKET>/<GOLD_PREFIX>/city_issue_hotspots/ (CSV)

Glue job arguments (set in the AWS Glue console):
    --PIPELINE_BUCKET  - S3 bucket name for all pipeline data
    --SILVER_PREFIX    - prefix for cleaned silver-layer data (default: silver)
    --BEDROCK_PREFIX   - prefix for Bedrock-generated outputs (default: bedrock)
    --GOLD_PREFIX      - prefix for gold-layer analytical tables (default: gold)
"""

from __future__ import annotations

import sys

from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from pyspark.sql import Window
from pyspark.sql import functions as F


# --------------------------------------------------------------------------
# Resolve Glue job arguments passed via the console or Step Functions
# --------------------------------------------------------------------------
args = getResolvedOptions(
    sys.argv,
    [
        "JOB_NAME",
        "PIPELINE_BUCKET",
        "SILVER_PREFIX",
        "BEDROCK_PREFIX",
        "GOLD_PREFIX",
    ],
)

# --------------------------------------------------------------------------
# Initialize the Spark + Glue runtime
# --------------------------------------------------------------------------
sc = SparkContext()
glue_context = GlueContext(sc)
spark = glue_context.spark_session
job = Job(glue_context)
job.init(args["JOB_NAME"], args)

# --------------------------------------------------------------------------
# Derive S3 paths from job arguments
# --------------------------------------------------------------------------
PIPELINE_BUCKET = args["PIPELINE_BUCKET"]
SILVER_PREFIX = args["SILVER_PREFIX"].strip("/")
BEDROCK_PREFIX = args["BEDROCK_PREFIX"].strip("/")
GOLD_PREFIX = args["GOLD_PREFIX"].strip("/")
S3_ROOT = f"s3://{PIPELINE_BUCKET}"


def write_single_csv(df, target_prefix: str) -> None:
    """Write small classroom-sized gold outputs as single CSV part files."""
    (
        df.coalesce(1)
        .write.mode("overwrite")
        .option("header", "true")
        .csv(f"{S3_ROOT}/{target_prefix}")
    )


# ==========================================================================
# Read source datasets from the silver layer and Bedrock output layer
# ==========================================================================
orders = spark.read.option("header", "true").csv(f"{S3_ROOT}/{SILVER_PREFIX}/orders/")
shipments = spark.read.option("header", "true").csv(f"{S3_ROOT}/{SILVER_PREFIX}/shipments/")
returns = spark.read.option("header", "true").csv(f"{S3_ROOT}/{SILVER_PREFIX}/returns/")

# ticket_enrichment: contains the Bedrock text model outputs (issue_category,
# sentiment, urgency, customer_summary, next_best_action, etc.) for each ticket
ticket_enrichment = spark.read.json(f"{S3_ROOT}/{BEDROCK_PREFIX}/ticket_enrichment/")

# ticket_similarity: contains (ticket_id, matched_ticket_id, similarity_score)
# pairs generated by the ticket_similarity Lambda using embedding-based cosine
# similarity. Each ticket may have multiple matches ranked by similarity_rank.
ticket_similarity = spark.read.json(f"{S3_ROOT}/{BEDROCK_PREFIX}/ticket_similarity/")

# Cast numeric columns that Spark reads as strings from CSV
shipments = shipments.withColumn("delay_hours", F.col("delay_hours").cast("double"))
orders = orders.withColumn("order_amount", F.col("order_amount").cast("double"))
returns = returns.withColumn("refund_amount", F.col("refund_amount").cast("double"))

# ==========================================================================
# Extract only the top-1 most similar ticket per ticket_id.
# The similarity output may contain up to N matches per ticket (default 3).
# For the gold table, only the single best match is joined to keep the
# order_support_360 table at one row per order.
# ==========================================================================
top_similarity_window = Window.partitionBy("ticket_id").orderBy(F.col("similarity_rank").asc())
top_ticket_similarity = (
    ticket_similarity.withColumn("row_number", F.row_number().over(top_similarity_window))
    .filter(F.col("row_number") == 1)
    .drop("row_number")
)

# ==========================================================================
# BUILD order_support_360: the primary gold table
#
# This join chain creates a single wide row per order by left-joining:
#   orders -> shipments   (delivery details, delay hours)
#   orders -> returns     (return status, refund amount)
#   orders -> ticket_enrichment (Bedrock text model outputs: issue category,
#                          sentiment, urgency, summary, next best action)
#   ticket_enrichment -> top_ticket_similarity (the single most similar
#                          historical ticket and its similarity score)
#
# Left joins ensure that orders without tickets, returns, or shipments
# still appear in the output (with null values in those columns).
# ==========================================================================
order_support_360 = (
    orders.alias("o")
    .join(shipments.alias("s"), F.col("o.order_id") == F.col("s.order_id"), "left")
    .join(returns.alias("r"), F.col("o.order_id") == F.col("r.order_id"), "left")
    .join(
        ticket_enrichment.alias("te"),
        (F.col("o.order_id") == F.col("te.order_id")) & (F.col("o.customer_id") == F.col("te.customer_id")),
        "left",
    )
    .join(top_ticket_similarity.alias("ts"), F.col("te.ticket_id") == F.col("ts.ticket_id"), "left")
    .select(
        F.col("o.order_id"),
        F.col("o.customer_id"),
        F.col("o.customer_name"),
        F.col("o.product_id"),
        F.col("o.product_name"),
        F.col("o.product_category"),
        F.col("o.order_date"),
        F.col("o.city"),
        F.col("o.quantity"),
        F.col("o.unit_price"),
        F.col("o.order_amount"),
        F.col("o.payment_status"),
        F.col("s.shipment_id"),
        F.col("s.carrier"),
        F.col("s.dispatch_time"),
        F.col("s.expected_delivery_time"),
        F.col("s.delivery_time"),
        F.col("s.delay_hours"),
        F.col("s.shipment_status"),
        F.col("r.return_id"),
        F.col("r.reason_code"),
        F.col("r.refund_amount"),
        F.col("r.return_status"),
        F.col("te.ticket_id"),
        F.col("te.issue_category"),
        F.col("te.sentiment"),
        F.col("te.urgency"),
        F.col("te.requires_human_followup"),
        F.col("te.root_cause_hint"),
        F.col("te.customer_summary"),
        F.col("te.next_best_action"),
        F.col("ts.matched_ticket_id").alias("most_similar_ticket_id"),
        F.col("ts.similarity_score").alias("most_similar_ticket_score"),
    )
    .withColumn("has_ticket", F.col("ticket_id").isNotNull())
    .withColumn("has_return", F.col("return_id").isNotNull())
    # Athena external CSV tables are sensitive to empty strings in numeric
    # columns. We coalesce nullable numeric outputs to 0.0 so downstream
    # analytical queries do not fail with BAD_DATA parse errors.
    .withColumn("delay_hours", F.coalesce(F.col("delay_hours"), F.lit(0.0)))
    .withColumn("refund_amount", F.coalesce(F.col("refund_amount"), F.lit(0.0)))
    .withColumn("most_similar_ticket_score", F.coalesce(F.col("most_similar_ticket_score"), F.lit(0.0)))
    .withColumn(
        "risk_flag",
        F.col("ticket_id").isNotNull()
        | F.col("return_id").isNotNull()
        | (F.col("delay_hours") > F.lit(24)),
    )
)

# ==========================================================================
# BUILD issue_kpis: aggregated metrics per Bedrock-classified issue category
#
# Filters to only orders that have an associated support ticket, then groups
# by the issue_category assigned by the Bedrock text model. This table is
# queried by the daily_ops_brief Lambda to generate the operations narrative.
# ==========================================================================
issue_kpis = (
    order_support_360.filter(F.col("has_ticket"))
    .groupBy("issue_category")
    .agg(
        F.count("ticket_id").alias("ticket_count"),
        F.countDistinct("order_id").alias("impacted_orders"),
        F.round(F.avg("delay_hours"), 2).alias("average_delay_hours"),
        F.round(F.avg("order_amount"), 2).alias("average_order_amount"),
        F.round(F.avg("refund_amount"), 2).alias("average_refund_amount"),
    )
    .withColumn("average_delay_hours", F.coalesce(F.col("average_delay_hours"), F.lit(0.0)))
    .withColumn("average_order_amount", F.coalesce(F.col("average_order_amount"), F.lit(0.0)))
    .withColumn("average_refund_amount", F.coalesce(F.col("average_refund_amount"), F.lit(0.0)))
    .orderBy(F.desc("ticket_count"), F.desc("average_delay_hours"))
)

# ==========================================================================
# BUILD city_issue_hotspots: geographic concentration of issue categories
#
# Groups by (city, issue_category) to identify which cities have the most
# support tickets for each issue type. This table is also queried by the
# daily_ops_brief Lambda to highlight geographic hotspots in the narrative.
# ==========================================================================
city_issue_hotspots = (
    order_support_360.filter(F.col("has_ticket"))
    .groupBy("city", "issue_category")
    .agg(
        F.count("ticket_id").alias("ticket_count"),
        F.round(F.avg("delay_hours"), 2).alias("average_delay_hours"),
    )
    .withColumn("average_delay_hours", F.coalesce(F.col("average_delay_hours"), F.lit(0.0)))
    .orderBy(F.desc("ticket_count"), F.desc("average_delay_hours"))
)

# ==========================================================================
# Write all gold tables to S3 as single CSV files with headers.
# These are queryable via Athena external tables and consumed by the
# daily_ops_brief Lambda for generating the operations narrative.
# ==========================================================================
write_single_csv(order_support_360, f"{GOLD_PREFIX}/order_support_360/")
write_single_csv(issue_kpis, f"{GOLD_PREFIX}/issue_kpis/")
write_single_csv(city_issue_hotspots, f"{GOLD_PREFIX}/city_issue_hotspots/")

job.commit()
