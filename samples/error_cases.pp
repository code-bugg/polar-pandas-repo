# Error / caveat samples — each block should trigger a single diagnostic.
# Run individually with: python src/main.py examples/error_cases.pp --check

# 1. LEX — unterminated string
READ 'orders.csv

# 2. PARSE — missing END for FOR
READ 'data.csv'
FOR $c IN a, b, c :
    CAST $c TO NUMBER

# 3. SEMANTIC — pipeline before any READ
orders -> FILTER WHERE total > 0

# 4. SEMANTIC — unknown CAST type
READ 'a.csv'
CAST age TO INTEGER

# 5. SEMANTIC — unknown aggregator
READ 'a.csv'
GROUP BY region TOTAL amount

# 6. SEMANTIC — MERGE source not READ
READ 'a.csv'
MERGE other ON id LEFT

# 7. SEMANTIC — unsupported PLOT kind
READ 'a.csv'
PLOT TYPE radar X region Y total

# 8. SEMANTIC — bar plot missing Y
READ 'a.csv'
PLOT TYPE bar X region

# 9. SEMANTIC — loop variable used outside FOR
READ 'a.csv'
SET total = $col + 1

# 10. SEMANTIC — BETWEEN with low > high
READ 'a.csv'
FILTER WHERE age BETWEEN 90 AND 18
