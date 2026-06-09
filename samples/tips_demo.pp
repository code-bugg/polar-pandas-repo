# Tips dataset demo
READ 'samples/tips.csv'
INFO
PREVIEW 3

SET tip_pct = tip / total_bill
GROUP BY day AVERAGE tip_pct
SORT BY tip_pct DESC
SAVE 'samples/tips_summary.csv'