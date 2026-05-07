# EDA + cleaning pipeline (spec section 7.1)
READ 'health.csv'
INFO
PREVIEW 5

DROP EMPTY
DROP DUPLICATES

FOR $col IN age, weight, bmi :
    CAST $col TO NUMBER (ON ERROR SKIP)
    FILL EMPTY $col WITH mean
END

FILTER WHERE age BETWEEN 18 AND 90
SET bmi = weight / height
SORT BY bmi DESC
SAVE 'clean_health.csv'
