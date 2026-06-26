"""
AWS Glue Job: bronze_to_silver_standardize

Purpose:
    This Glue job reads raw (bronze-layer) operational datasets from S3, applies
    type casting, timestamp parsing, and value normalization, then writes cleaned
    (silver-layer) datasets back to S3. The silver layer is the foundation for
    all downstream processing -- Lambda functions and the gold-layer Glue job
    read exclusively from silver, never from raw.

What this job standardizes:
    - orders     : casts quantity/price/amount to numeric types, normalizes
                   payment_status and product_category to uppercase, parses
                   order_date as a timestamp.
    - shipments  : parses all timestamp columns, casts delay_hours to double,
                   normalizes shipment_status to uppercase, adds boolean flags
                   for delivery delays (>24h) and delivery exceptions.
    - returns    : parses requested_at timestamp, casts refund_amount to double,
                   normalizes reason_code and return_status to uppercase.
    - products   : casts catalog_price to double, normalizes category to uppercase.
    - support_tickets : parses created_at as a timestamp, normalizes channel to
                   uppercase. These are the free-text records that the Bedrock
                   ticket_enrichment Lambda processes in the next pipeline stage.

Input (read from S3 raw zone):
    s3://<PIPELINE_BUCKET>/<RAW_PREFIX>/orders/
    s3://<PIPELINE_BUCKET>/<RAW_PREFIX>/shipments/
    s3://<PIPELINE_BUCKET>/<RAW_PREFIX>/returns/
    s3://<PIPELINE_BUCKET>/<RAW_PREFIX>/products/
    s3://<PIPELINE_BUCKET>/<RAW_PREFIX>/support_tickets/

Output (written to S3 silver zone):
    s3://<PIPELINE_BUCKET>/<SILVER_PREFIX>/orders/          (CSV)
    s3://<PIPELINE_BUCKET>/<SILVER_PREFIX>/shipments/       (CSV)
    s3://<PIPELINE_BUCKET>/<SILVER_PREFIX>/returns/         (CSV)
    s3://<PIPELINE_BUCKET>/<SILVER_PREFIX>/products/        (CSV)
    s3://<PIPELINE_BUCKET>/<SILVER_PREFIX>/support_tickets/ (JSONL)

Glue job arguments (set in the AWS Glue console):
    --PIPELINE_BUCKET  - S3 bucket name for all pipeline data
    --RAW_PREFIX       - prefix for raw source data (default: raw)
    --SILVER_PREFIX    - prefix for cleaned silver-layer data (default: silver)
"""

from __future__ import annotations

import sys

from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from pyspark.sql import functions as F


# --------------------------------------------------------------------------
# Resolve Glue job arguments passed via the console or Step Functions
# --------------------------------------------------------------------------
args = getResolvedOptions(
    sys.argv,
    [
        "JOB_NAME",
        "PIPELINE_BUCKET",
        "RAW_PREFIX",
        "SILVER_PREFIX",
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
RAW_PREFIX = args["RAW_PREFIX"].strip("/")
SILVER_PREFIX = args["SILVER_PREFIX"].strip("/")
S3_ROOT = f"s3://{PIPELINE_BUCKET}"


def write_single_csv(df, target_prefix: str) -> None:
    """Write a small classroom-sized dataset as a single CSV part file."""
    (
        df.coalesce(1)
        .write.mode("overwrite")
        .option("header", "true")
        .csv(f"{S3_ROOT}/{target_prefix}")
    )


def write_single_json(df, target_prefix: str) -> None:
    """Write JSONL-like output to a single object folder for easy Athena reads."""
    df.coalesce(1).write.mode("overwrite").json(f"{S3_ROOT}/{target_prefix}")


# ==========================================================================
# ORDERS: cast numeric columns, parse order_date, normalize categorical values
# The silver orders table is used by the build_gold_support_mart Glue job
# to form the backbone of the order_support_360 gold table.
# ==========================================================================
orders = (
    spark.read.option("header", "true").csv(f"{S3_ROOT}/{RAW_PREFIX}/orders/")
    .withColumn("order_date", F.to_timestamp("order_date"))
    .withColumn("quantity", F.col("quantity").cast("int"))
    .withColumn("unit_price", F.col("unit_price").cast("double"))
    .withColumn("order_amount", F.col("order_amount").cast("double"))
    .withColumn("payment_status", F.upper(F.col("payment_status")))
    .withColumn("product_category", F.upper(F.col("product_category")))
)

# ==========================================================================
# SHIPMENTS: parse all timestamps, cast delay_hours, normalize status,
# and derive two boolean flags:
#   - delivery_delayed_flag : true if the shipment was delayed by more than 24 hours
#   - delivery_exception_flag : true if the shipment status is DELAYED or DELIVERED_DAMAGED
# These flags are useful for identifying problematic shipments in the gold layer.
# ==========================================================================
shipments = (
    spark.read.option("header", "true").csv(f"{S3_ROOT}/{RAW_PREFIX}/shipments/")
    .withColumn("dispatch_time", F.to_timestamp("dispatch_time"))
    .withColumn("expected_delivery_time", F.to_timestamp("expected_delivery_time"))
    .withColumn("delivery_time", F.to_timestamp("delivery_time"))
    .withColumn("delay_hours", F.col("delay_hours").cast("double"))
    .withColumn("shipment_status", F.upper(F.col("shipment_status")))
    .withColumn("delivery_delayed_flag", F.col("delay_hours") > F.lit(24))
    .withColumn(
        "delivery_exception_flag",
        F.col("shipment_status").isin("DELAYED", "DELIVERED_DAMAGED"),
    )
)

# ==========================================================================
# RETURNS: parse requested_at, cast refund_amount, normalize categorical values
# ==========================================================================
returns = (
    spark.read.option("header", "true").csv(f"{S3_ROOT}/{RAW_PREFIX}/returns/")
    .withColumn("requested_at", F.to_timestamp("requested_at"))
    .withColumn("refund_amount", F.col("refund_amount").cast("double"))
    .withColumn("reason_code", F.upper(F.col("reason_code")))
    .withColumn("return_status", F.upper(F.col("return_status")))
)

# ==========================================================================
# PRODUCTS: cast catalog_price, normalize category
# ==========================================================================
products = (
    spark.read.option("header", "true").csv(f"{S3_ROOT}/{RAW_PREFIX}/products/")
    .withColumn("catalog_price", F.col("catalog_price").cast("double"))
    .withColumn("category", F.upper(F.col("category")))
)

# ==========================================================================
# SUPPORT TICKETS: parse created_at timestamp, normalize channel to uppercase.
# These records contain the free-text customer_message and agent_notes that
# the ticket_enrichment Lambda will send to the Bedrock text model for
# structured classification and extraction in the next pipeline stage.
# Written as JSONL (not CSV) because the customer_message field contains
# free-text with commas, quotes, and newlines that would break CSV formatting.
# ==========================================================================
support_tickets = (
    spark.read.json(f"{S3_ROOT}/{RAW_PREFIX}/support_tickets/")
    .withColumn("created_at", F.to_timestamp("created_at"))
    .withColumn("channel", F.upper(F.col("channel")))
)

# ==========================================================================
# Write all silver datasets to S3
# CSV datasets are written with headers; support tickets are written as JSONL.
# Each dataset is coalesced to a single partition file for simplicity.
# ==========================================================================
write_single_csv(orders, f"{SILVER_PREFIX}/orders/")
write_single_csv(shipments, f"{SILVER_PREFIX}/shipments/")
write_single_csv(returns, f"{SILVER_PREFIX}/returns/")
write_single_csv(products, f"{SILVER_PREFIX}/products/")
write_single_json(support_tickets, f"{SILVER_PREFIX}/support_tickets/")

job.commit()
