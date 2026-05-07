# =====================================================================
# stroke_analysis.pp — PolarPandas v5
# Exploratory analysis & cleaning of the Kaggle Healthcare Stroke dataset.
# =====================================================================

# ---------- 1. Load ----------------------------------------------------------
READ 'healthcare-dataset-stroke-data.csv'
INFO
PREVIEW 5

# ---------- 2. Cleaning ------------------------------------------------------
# Drop the useless identifier column and fully-empty / duplicate rows.
DROP id
DROP EMPTY
DROP DUPLICATES

# The 'bmi' column stores "N/A" as a literal string. Replace it, then cast.
FILL EMPTY bmi WITH 'N/A'
CAST bmi TO NUMBER (ON ERROR SKIP)
FILL EMPTY bmi WITH median

# Coerce numeric features defensively.
FOR $col IN age, avg_glucose_level, hypertension, heart_disease, stroke :
    CAST $col TO NUMBER (ON ERROR SKIP)
END

# ---------- 3. Feature engineering ------------------------------------------
SET glucose_to_bmi = avg_glucose_level / bmi
SET risk_score     = age + avg_glucose_level / 10 + bmi / 5

# ---------- 4. Cohort analytics ---------------------------------------------
# Mean glucose by smoking status, sorted, previewed and persisted.
GROUP BY smoking_status AVERAGE avg_glucose_level
    -> SORT BY avg_glucose_level DESC
    -> PREVIEW 10
    -> SAVE 'glucose_by_smoking.csv'

# ---------- 5. Stroke counts per work type ----------------------------------
READ 'healthcare-dataset-stroke-data.csv'
DROP EMPTY
GROUP BY work_type SUM stroke
    -> SORT BY stroke DESC
    -> PREVIEW
    -> SAVE 'strokes_by_work_type.csv'

# ---------- 6. High-risk subset ---------------------------------------------
READ 'healthcare-dataset-stroke-data.csv'
FILL EMPTY bmi WITH 'N/A'
CAST bmi TO NUMBER (ON ERROR SKIP)
FILL EMPTY bmi WITH median

FILTER WHERE age >= 60 AND hypertension == 1 AND avg_glucose_level > 140
    -> SORT BY avg_glucose_level DESC
    -> PREVIEW 10
    -> SAVE 'high_risk_patients.parquet'

# ---------- 7. Conditional cleaning ----------------------------------------
READ 'healthcare-dataset-stroke-data.csv'
IF ROWCOUNT healthcare_dataset_stroke_data > 1000 :
    DROP EMPTY
    DROP DUPLICATES
ELSE
    FILL EMPTY bmi WITH 0
END
PREVIEW 5

# ---------- 8. Visualisation ------------------------------------------------
READ 'healthcare-dataset-stroke-data.csv'
FILL EMPTY bmi WITH 'N/A'
CAST bmi TO NUMBER (ON ERROR SKIP)
FILL EMPTY bmi WITH median

GROUP BY smoking_status SUM stroke
    -> PLOT TYPE bar X smoking_status Y stroke TITLE 'Strokes by smoking status' SAVE 'strokes_by_smoking.png'

# Re-load full table for the scatter (the GROUP BY above narrowed the active df).
READ 'healthcare-dataset-stroke-data.csv'
FILL EMPTY bmi WITH 'N/A'
CAST bmi TO NUMBER (ON ERROR SKIP)
FILL EMPTY bmi WITH median
PLOT TYPE scatter X bmi Y avg_glucose_level TITLE 'BMI vs glucose' SAVE 'bmi_vs_glucose.png'
