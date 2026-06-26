SELECT city,
       issue_category,
       COUNT(*) AS ticket_count,
       ROUND(AVG(delay_hours), 2) AS average_delay_hours
FROM order_support_360
WHERE has_ticket = true
GROUP BY city, issue_category
ORDER BY ticket_count DESC, average_delay_hours DESC
LIMIT 25;
