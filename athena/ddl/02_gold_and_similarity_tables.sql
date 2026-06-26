USE retail_support_intelligence;

CREATE EXTERNAL TABLE IF NOT EXISTS ticket_similarity (
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
LOCATION 's3://<PIPELINE_BUCKET>/<BEDROCK_PREFIX>/ticket_similarity/';

CREATE EXTERNAL TABLE IF NOT EXISTS order_support_360 (
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
  'escapeChar' = '\\'
)
LOCATION 's3://<PIPELINE_BUCKET>/<GOLD_PREFIX>/order_support_360/'
TBLPROPERTIES ('skip.header.line.count'='1');

CREATE EXTERNAL TABLE IF NOT EXISTS issue_kpis (
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
  'escapeChar' = '\\'
)
LOCATION 's3://<PIPELINE_BUCKET>/<GOLD_PREFIX>/issue_kpis/'
TBLPROPERTIES ('skip.header.line.count'='1');

CREATE EXTERNAL TABLE IF NOT EXISTS city_issue_hotspots (
  city string,
  issue_category string,
  ticket_count int,
  average_delay_hours double
)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
WITH SERDEPROPERTIES (
  'separatorChar' = ',',
  'quoteChar' = '"',
  'escapeChar' = '\\'
)
LOCATION 's3://<PIPELINE_BUCKET>/<GOLD_PREFIX>/city_issue_hotspots/'
TBLPROPERTIES ('skip.header.line.count'='1');
