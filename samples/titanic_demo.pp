# Titanic dataset demo
READ 'samples/titanic.csv'
DROP EMPTY age
FILL EMPTY age WITH median
DROP deck, embark_town, alive, who, adult_male
GROUP BY sex AVERAGE survived
SORT BY survived DESC
PREVIEW 3
SAVE 'samples/titanic_survival_by_sex.csv'