# Multi-table analysis (spec section 7.2)
READ 'orders.csv'
READ 'customers.csv'

customers -> DROP EMPTY email
          -> RENAME cust_id TO id

orders -> FILTER WHERE status == 'complete'
       -> MERGE customers ON id LEFT
       -> GROUP BY region SUM total
       -> SORT BY total DESC
       -> PLOT TYPE bar X region Y total TITLE 'Revenue by Region' SAVE 'revenue.png'
       -> SAVE 'regional_revenue.csv'
