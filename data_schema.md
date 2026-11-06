# Data schema

The scripts expect the derived cohort files described below under `$NFA_DATA_DIR`. These
files are **not** distributed with this repository; see the README, section
"What is and is not in this repository".

Cohort A covers measurement years 2022–2026 with 20 input variables. Cohort B is exactly
the Cohort A records measured from 2024 onward (295,912 of 526,050) and adds waist
circumference, giving 21 input variables. The 85/15 development–test split was drawn
independently within each cohort. Stratification is on the age band alone; sex is not a
stratification variable. The split is in fact a nested 70/15/15 (train / validation / test), and
the 85% development set is the train and validation blocks concatenated in that order.

## Files per cohort

| File | Rows (Cohort A / B) | Written by | Contents |
|---|---|---|---|
| `full_dataset.csv` | 526,050 / 295,912 | `00a` | All eligible records with raw NFA measurement columns |
| `development_set.csv` | 447,142 / 251,525 | `00` | Development split: the 70% training and 15% validation blocks concatenated in that order. Cross-validation for algorithm selection runs inside the training block alone |
| `validation_set.csv` | 78,907 / 44,387 | `00` | The 15% validation block on its own |
| `result_set.csv` | 78,908 / 44,387 | `00` | Held-out test split with per-record predictions from all five algorithms |
| `final_test_metrics.csv` | 5 / 5 | `00` | Held-out MAE, RMSE, R² and Pearson r per algorithm |
| `cv_summary.csv` | 5 / 5 | `00` | 5-fold cross-validation means and SDs per algorithm |
| `deltaage_correction.csv` | 2 / 2 | `00` | Kronmal coefficients and the raw/corrected aging-gap correlations, one row for the development split and one for the test split |

## `result_set.csv`

The input to `02a`, `02b` and `03`–`06`. One row per held-out test record.

| Column | Meaning |
|---|---|
| `MODEL` | `Model A` or `Model B` |
| `RESULT_SPLIT` | Always `test` |
| `MESURE_AGE_CO` | Chronological age in years (the regression target) |
| `AGE_BAND` | Age decade: `19-29`, `30-39`, `40-49`, `50-59`, `60-64` |
| `YEAR` | Measurement year |
| `SEXDSTN_FLAG_CD` | Sex, `M` or `F` |
| `MESURE_PLACE_FLAG_NM` | Measurement setting |
| `MESURE_IEM_003_VALUE` | Body fat percentage (withheld from model inputs) |
| `MESURE_IEM_005_VALUE` | Diastolic blood pressure (withheld) |
| `MESURE_IEM_006_VALUE` | Systolic blood pressure (withheld) |
| `MESURE_IEM_040_VALUE` | 3 × 3 button-press reaction time (withheld) |
| `PRED_EN`, `PRED_DT`, `PRED_RF`, `PRED_XGB`, `PRED_LGBM` | Predicted BA from elastic net, decision tree, random forest, XGBoost, LightGBM |
| `BA_PRED` | Primary predicted BA (identical to `PRED_LGBM`) |
| `DELTA_RAW` | Raw aging gap, `BA_PRED − MESURE_AGE_CO` |
| `DELTA_CORR` | Kronmal-corrected aging gap |
| `BA_CORRECTED` | Corrected BA, `MESURE_AGE_CO + DELTA_CORR` |

The Kronmal correction is estimated once on the development set and applied without
re-estimation:

```
DELTA_CORR = DELTA_RAW − (slope × MESURE_AGE_CO + intercept)
```

with `slope = −0.2012514386615332`, `intercept = 7.0725629681266025` for Cohort A and
`slope = −0.1990409313624863`, `intercept = 6.964529433240011` for Cohort B. These are
stored in `deltaage_correction.csv` alongside each `result_set.csv`. `06_bootstrap_ci.py` checks
that the stored `DELTA_CORR` follows from them, using the coefficients as module constants rather
than reading that file.

## Model input variables

`development_set.csv` and `full_dataset.csv` carry the raw NFA measurement codes. Feature
Set A is items 1–20; Feature Set B adds waist circumference (`MESURE_IEM_004_VALUE`).

| Code | Variable | Units |
|---|---|---|
| `MESURE_IEM_001_VALUE` | Height | cm |
| `MESURE_IEM_002_VALUE` | Body weight | kg |
| `MESURE_IEM_007_VALUE` | Grip strength, left | kg |
| `MESURE_IEM_008_VALUE` | Grip strength, right | kg |
| `MESURE_IEM_012_VALUE` | Sit-and-reach flexibility | cm |
| `MESURE_IEM_018_VALUE` | Body mass index | kg/m² |
| `MESURE_IEM_019_VALUE` | Cross sit-up | repetitions |
| `MESURE_IEM_020_VALUE` | 20-m shuttle run | count |
| `MESURE_IEM_021_VALUE` | 10-m × 4 shuttle run | s |
| `MESURE_IEM_022_VALUE` | Standing long jump | cm |
| `MESURE_IEM_028_VALUE` | Relative grip strength | % |
| `MESURE_IEM_030_VALUE` | Shuttle-run-derived VO₂max | mL/kg/min |
| `MESURE_IEM_031_VALUE` | Treadmill resting heart rate | bpm |
| `MESURE_IEM_032_VALUE` | Treadmill heart rate at 3 min | bpm |
| `MESURE_IEM_033_VALUE` | Treadmill heart rate at 6 min | bpm |
| `MESURE_IEM_034_VALUE` | Treadmill heart rate at 9 min | bpm |
| `MESURE_IEM_035_VALUE` | Treadmill-derived VO₂max | mL/kg/min |
| `MESURE_IEM_036_VALUE` | Step-test recovery heart rate | bpm |
| `MESURE_IEM_037_VALUE` | Step-test-derived VO₂max | mL/kg/min |
| `MESURE_IEM_041_VALUE` | Adult flight time | s |
| `MESURE_IEM_004_VALUE` | Waist circumference (Feature Set B only) | cm |

Missing values are median-imputed inside the modelling pipeline for all five algorithms,
including LightGBM and XGBoost; no native NaN routing and no stratified imputation is used. See
`00_train_model.py`, `build_models()`.
