# Conditional cleaning (spec section 7.3)
READ 'survey.csv'
INFO

IF ROWCOUNT survey > 1000 :
    DROP EMPTY
    DROP DUPLICATES
ELSE
    FILL EMPTY score WITH median
END

FOR $col IN q1, q2, q3 :
    CAST $col TO NUMBER (ON ERROR SKIP)
    FILTER WHERE $col BETWEEN 1 AND 10
END

GROUP BY region AVERAGE score
SORT BY score DESC
PREVIEW
