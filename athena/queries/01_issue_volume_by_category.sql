SELECT issue_category,
       COUNT(*) AS ticket_count,
       ROUND(AVG(delay_hours), 2) AS average_delay_hours,
       ROUND(AVG(order_amount), 2) AS average_order_amount
FROM order_support_360
WHERE has_ticket = true
GROUP BY issue_category
ORDER BY ticket_count DESC, average_delay_hours DESC;
