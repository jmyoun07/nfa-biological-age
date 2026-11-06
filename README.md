# Biological age estimation from national physical fitness data — analysis code

Analysis code for:

> Lee Y-S, Song D, Youn B-Y. *Development and internal validation of an interpretable
> machine-learning model for biological age estimation from national physical fitness
> data in Korean adults.* (submitted)

The study estimates biological age (BA) from multidomain physical fitness measurements in
the Korea National Fitness Award (NFA) programme using LightGBM, applies a Kronmal-type
linear bias correction to the aging gap (ΔAge = BA − chronological age), and reports
discrimination, calibration, SHAP interpretability, and convergent validity against four
withheld biomarkers.

---

## What is and is not in this repository

**Included** — the analysis scripts that produce the tables, figures and statistics
reported in the manuscript, given the derived cohort files as input.

**Model development is reproducible.** `00_train_model.py` regenerates the five candidate
models, the cross-validation table, the final held-out metrics and the Kronmal-type aging-gap
correction from the derived cohort files. Run it with `--verify` to diff its output against the
stored artefacts. On the reference machine it reproduces, in both cohorts:

- the five candidate models' predictions, with LightGBM (the primary model), random forest,
  decision tree and elastic net matching to ~1e-14, i.e. bit-exact within floating-point noise;
- the 5-fold cross-validation table, the selected best model, and the Kronmal intercept and
  slope to the same precision;
- the corrected aging gap to ~1e-14;
- the split membership of the development and validation sets, and both rows of the
  aging-gap correction table including its four correlation fields.

**Cohort derivation is reproducible too.** `00a_build_cohorts.py` turns the 51 monthly NFA
extracts into the analysis datasets. Run with `--verify`: it rebuilds both published
`full_dataset.csv` files with the **row order identical and zero value mismatches** across
every measurement column (Cohort A 526,050 rows, Cohort B 295,912). Row order is load-bearing,
because the development/test partition is derived from this frame.

The published cohort is defined by exactly two conditions — adult record and age 19–64 — with
no further exclusions, and by a plausibility screen that is applied to 18 of the 25 measurement
columns and, faithfully to the published data, not to the other seven.

**Not included, and required to run anything:**

- **The exact data snapshot.** The published cohorts were built from 51 monthly files spanning
  2022-01 to 2026-03 inclusive. `00a_build_cohorts.py` does not enforce that window: it takes
  whatever monthly files it finds under `NFA_RAW_DIR`, so a later or partial download will
  produce different cohorts and therefore different numbers. Check the file count and range
  before trusting a rebuild.
- **The NFA data.** Both the monthly extracts and the derived cohort files are licensed
  public-use data obtained after registration with the Big Data Culture Platform
  (https://www.bigdata-culture.kr) and are not redistributable. No participant-level data of
  any kind is contained in this repository, so a reader cannot execute these scripts without
  first obtaining the data themselves. The reproduction results reported above were verified on
  the authors' machine.

Two details in `00_train_model.py` are easy to get wrong and are load-bearing: waist
circumference must be **last** in the Cohort B feature list (`colsample_bytree=0.9` makes
column position matter; putting it first moves the Cohort B MAE from 3.8346 to 3.8187), and
missing values are **median-imputed inside the pipeline for all five algorithms** — native NaN
handling does not reproduce the published models.

Contact the corresponding author for anything not covered here.

---

## Scripts

All scripts sit at the repository root. Each is independent; `05` requires the SHAP outputs
from `02`.

| Script | Produces | Manuscript output |
|---|---|---|
| `00a_build_cohorts.py` | The Cohort A and Cohort B analysis datasets from the monthly NFA extracts | Fig. 1 counts, Table 1 denominators |
| `00_train_model.py` | The five candidate models, 5-fold cross-validation, final held-out metrics, Kronmal correction coefficients | Tables 2–3, and the predictions every later script consumes |
| `01_baseline_table.py` | Cohort descriptive statistics | Table 1 |
| `02a_shap_cohort_a.py` | Per-instance SHAP values, dependence and interaction data; iterates over both cohorts and writes the `.npz` archives `05_figures.py` consumes | Figures 5–6, feature-importance text |
| `02b_shap_cohort_b.py` | A Cohort B variant that computes the full SHAP interaction tensor in a single pass; does not write the `.npz` archive | — |
| `03_calibration.py` | Calibration-in-the-large, calibration slope and intercept, decile and age-stratum tables, and a two-panel calibration figure (written as `Figure6_Calibration_AB.png`; it is Figure 2 in the manuscript) | Table 4, Figure 2 |
| `04_aging_gap_distribution.py` | Corrected ΔAge distribution, ANOVA by BMI category, sex and age decade, measurement year | Figures 3–4, Results text |
| `05_figures.py` | Per-cohort calibration, distribution, age × sex, SHAP beeswarm and dependence figures | Figures 2–6 |
| `06_bootstrap_ci.py` | Nonparametric bootstrap 95% confidence intervals for every reported discrimination and calibration metric | Tables 2–4, Additional file 1 |

`06_bootstrap_ci.py` case-resamples the held-out test records 50,000 times with the fitted
models held fixed, reuses one set of resample indices across all five algorithms so
between-algorithm differences are paired, and runs a separate paired bootstrap on the records
held out from training in both cohorts. Before bootstrapping each cohort it checks that
`BA_PRED` equals `PRED_LGBM` and that the stored raw and corrected aging gaps follow from the
Kronmal coefficients, which it carries as module constants rather than reading
`deltaage_correction.csv`. It then recomputes the original-sample metrics, stored as the `point`
values in its JSON output. It does not compare those point values with a published reference.
`00_train_model.py --verify` separately checks the model-development metrics, predictions,
correction table and split files; it does not verify the full set of bootstrap point values.

`02a`/`02b` deliberately refit LightGBM with a different configuration from `00_train_model.py`
before computing SHAP values, which is what the manuscript reports and what produced the
published Figures 5–6: "the primary LightGBM model was retrained on the development set with a
closely matched configuration", with retrained per-row predictions concordant with the reported
predictions at Pearson *r* = 0.976 in both cohorts. Their output therefore will not match
`00_train_model.py` exactly; that is expected, not a defect.

`02a` processes both cohorts and writes the `shap_per_instance_*.npz` archives that
`05_figures.py` reads, so the SHAP panels of Figures 5–6 come from `02a` for both cohorts.
`02b` is a Cohort B variant that computes the full interaction tensor in a single pass; it does
not write an `.npz` archive.

Figure 1 (participant flow) is a hand-authored diagram and is not generated by any script.

---

## Data layout

Point `NFA_RAW_DIR` at the monthly extracts and `NFA_DATA_DIR` at the directory for the derived
cohort files (`00a_build_cohorts.py` writes the latter from the former):

```
$NFA_DATA_DIR/
    Model A/                 # Cohort A, 2022-2026, 20 input variables
        full_dataset.csv         # written by 00a
        development_set.csv      # written by 00, = the train and validation blocks
        validation_set.csv       # written by 00
        result_set.csv           # written by 00, held-out test rows + predictions
        deltaage_correction.csv  # written by 00
        final_test_metrics.csv   # written by 00
        cv_summary.csv           # written by 00
    Model B/                 # Cohort B, 2024-2026, adds waist circumference
        (the same seven files)
```

`00a_build_cohorts.py` writes `full_dataset.csv`; `00_train_model.py` writes the other six.
Both write into `NFA_DATA_DIR`, mirroring the layout of the stored artefacts, so `--verify`
works against a freshly generated tree as well as against the originals. `NFA_OUT_DIR` receives
the analysis output of `03`-`06`: the calibration tables, the distribution JSON and the figures.
Running `00a` then `00` leaves the tree in the state `01`, `03`, `04` and `06` expect. `05`
additionally needs the SHAP archives that `02a` writes.

Column definitions are in [`data_schema.md`](data_schema.md). The cohort derivation
is implemented in `00a_build_cohorts.py` and the partition in `00_train_model.py`; both are
documented in their module docstrings, which record where the implementation and the manuscript
Methods describe the procedure differently.

---

## Running

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export NFA_RAW_DIR=/path/to/monthly/nfa/csvs
export NFA_DATA_DIR=/path/to/derived/cohorts
export NFA_OUT_DIR=./outputs          # optional, defaults to ./outputs

python 00a_build_cohorts.py           # or --verify against stored datasets
python 00_train_model.py              # or --verify against stored metrics
python 06_bootstrap_ci.py
```

`00a` and `00` write their artefacts into `NFA_DATA_DIR`. Everything `01`-`06` produce goes to
`NFA_OUT_DIR`, except `05_figures.py`, which writes to `NFA_FIG_DIR` — that defaults to
`NFA_OUT_DIR/figures` and can be overridden on its own.

## Environment

Verified on Python 3.9.6 with the versions pinned in `requirements.txt`. Install those pins:
gradient-boosting predictions are build-sensitive, so the pins are what make a run
deterministic. Random seeds are fixed inside each script.

## Licence

MIT — see [LICENSE](LICENSE). The licence covers this code only, not the NFA data.

## Citation


