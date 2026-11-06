"""Resolve input and output locations for every analysis script.

The Korea National Fitness Award (NFA) extracts are licensed data and are not
redistributed with this repository. Obtain them from the Big Data Culture
Platform (https://www.bigdata-culture.kr), reproduce the cohort derivation
described in the manuscript Methods, and point NFA_DATA_DIR at the result.

Expected layout under NFA_DATA_DIR (see data_schema.md for columns):

    NFA_DATA_DIR/
        Model A/
            full_dataset.csv         written by 00a_build_cohorts.py
            development_set.csv      written by 00_train_model.py
            validation_set.csv       written by 00_train_model.py
            result_set.csv           written by 00_train_model.py
            final_test_metrics.csv   written by 00_train_model.py
            cv_summary.csv           written by 00_train_model.py
            deltaage_correction.csv  written by 00_train_model.py
        Model B/
            (the same seven files)

NFA_OUT_DIR receives the analysis output of 01-06; NFA_FIG_DIR the figures of 05.
"""

import os

_ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.environ.get("NFA_DATA_DIR", os.path.join(_ROOT, "data"))
OUT_DIR = os.environ.get("NFA_OUT_DIR", os.path.join(_ROOT, "outputs"))
FIG_DIR = os.environ.get("NFA_FIG_DIR", os.path.join(OUT_DIR, "figures"))

for _d in (OUT_DIR, FIG_DIR):
    os.makedirs(_d, exist_ok=True)


def cohort_dir(cohort):
    """cohort is "A" or "B"."""
    return os.path.join(DATA_DIR, f"Model {cohort}")


def require(path):
    if not os.path.exists(path):
        raise SystemExit(
            f"Missing input: {path}\n"
            "The NFA extracts are not distributed with this repository. "
            "See README.md, section 'Data'."
        )
    return path
