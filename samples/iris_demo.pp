# Iris dataset demo
READ 'samples/iris.csv'
RENAME sepal_length TO sepal_len
RENAME sepal_width TO sepal_wid
SET sepal_area = sepal_len * sepal_wid
GROUP BY species AVERAGE sepal_area
SORT BY sepal_area DESC
PREVIEW 3
SAVE 'samples/iris_summary.csv'