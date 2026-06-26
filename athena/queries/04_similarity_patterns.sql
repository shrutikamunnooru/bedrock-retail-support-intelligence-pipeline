SELECT ticket_issue_category,
       matched_issue_category,
       COUNT(*) AS similar_pair_count,
       ROUND(AVG(similarity_score), 4) AS average_similarity_score
FROM ticket_similarity
WHERE similarity_rank = 1
GROUP BY ticket_issue_category, matched_issue_category
ORDER BY similar_pair_count DESC, average_similarity_score DESC;
