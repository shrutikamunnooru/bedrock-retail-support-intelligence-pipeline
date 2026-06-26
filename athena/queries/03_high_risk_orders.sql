SELECT order_id,
       customer_id,
       city,
       product_name,
       carrier,
       shipment_status,
       delay_hours,
       issue_category,
       urgency,
       root_cause_hint,
       next_best_action
FROM order_support_360
WHERE risk_flag = true
ORDER BY
  CASE urgency
    WHEN 'HIGH' THEN 1
    WHEN 'MEDIUM' THEN 2
    ELSE 3
  END,
  delay_hours DESC;
