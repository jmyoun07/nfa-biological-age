#!/usr/bin/env python3
"""Cohort derivation: the 51 monthly Korea National Fitness Award (NFA) extracts
into the analysis datasets for Cohort A and Cohort B.

VERIFICATION
------------
Run with --verify: it reproduces the published `full_dataset.csv` for BOTH
cohorts with the row order identical and ZERO value mismatches across every
measurement column
(Cohort A 526,050 x 20 inputs + 4 withheld; Cohort B 295,912 x 21 + 4).

Row order matters. `00_train_model.py` derives the development/test partition by
calling `train_test_split` on this frame, so a different row order yields a
different partition and different published numbers.

COHORT DEFINITION, as implemented
---------------------------------
* Monthly files are read in ascending filename order (which is chronological) and
  concatenated with `ignore_index=True`.
* YEAR is taken from the filename, not from any column.
* Eligibility is exactly two conditions: `AGE_FLAG_NM == "성인"` (adult) and
  `19 <= MESURE_AGE_CO <= 64`. Nothing else is excluded. This yields 526,050
  records for 2022-2026 and 295,912 for 2024-2026, matching the published totals.
* Missing measurement setting is filled with "미상" (unknown).
* Values outside PLAUSIBLE_RANGES are set to NaN. Note that six of the twenty
  Cohort A inputs (021, 031, 032, 033, 034, 041) and waist circumference (004)
  are NOT in PLAUSIBLE_RANGES and therefore receive no range screening. That
  asymmetry is faithful to the published data, and was confirmed by the
  zero-mismatch reproduction; it is not an oversight in this file.
* AGE_BAND is the stratification key for every split: 19-29, 30-39, 40-49,
  50-59, 60-64 (the top band is truncated at 64, not a full decade).

USAGE
-----
    export NFA_RAW_DIR=/directory/holding/the/monthly/csv/files
    export NFA_DATA_DIR=/path/to/write/derived/cohorts
    python 00a_build_cohorts.py
    python 00a_build_cohorts.py --verify   # diff against stored full_dataset.csv
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

from paths import DATA_DIR

RAW_DIR = os.environ.get("NFA_RAW_DIR", os.path.join(DATA_DIR, "raw"))
ADULT = "성인"
UNKNOWN_PLACE = "미상"

# Ranges applied in the published derivation. Codes absent here are not screened.
PLAUSIBLE_RANGES = {
    "MESURE_IEM_001_VALUE": (120, 220),   # height, cm
    "MESURE_IEM_002_VALUE": (25, 200),    # body weight, kg
    "MESURE_IEM_003_VALUE": (3, 60),      # body fat %          (withheld)
    "MESURE_IEM_005_VALUE": (30, 200),    # diastolic BP        (withheld)
    "MESURE_IEM_006_VALUE": (50, 300),    # systolic BP         (withheld)
    "MESURE_IEM_007_VALUE": (1, 100),     # grip strength, left
    "MESURE_IEM_008_VALUE": (1, 100),     # grip strength, right
    "MESURE_IEM_012_VALUE": (-50, 60),    # sit-and-reach
    "MESURE_IEM_018_VALUE": (10, 60),     # BMI
    "MESURE_IEM_019_VALUE": (0, 200),     # cross sit-up
    "MESURE_IEM_020_VALUE": (0, 300),     # 20-m shuttle run
    "MESURE_IEM_022_VALUE": (0, 400),     # standing long jump
    "MESURE_IEM_028_VALUE": (1, 200),     # relative grip strength
    "MESURE_IEM_030_VALUE": (5, 90),      # shuttle-run VO2max
    "MESURE_IEM_035_VALUE": (5, 90),      # treadmill VO2max
    "MESURE_IEM_036_VALUE": (30, 250),    # step-test recovery HR
    "MESURE_IEM_037_VALUE": (5, 90),      # step-test VO2max
    "MESURE_IEM_040_VALUE": (0, 10),      # reaction time       (withheld)
}

# 20 model inputs (Cohort A), ascending by code, plus the 4 withheld biomarkers.
INPUTS_A = [
    "MESURE_IEM_001_VALUE", "MESURE_IEM_002_VALUE", "MESURE_IEM_007_VALUE",
    "MESURE_IEM_008_VALUE", "MESURE_IEM_012_VALUE", "MESURE_IEM_018_VALUE",
    "MESURE_IEM_019_VALUE", "MESURE_IEM_020_VALUE", "MESURE_IEM_021_VALUE",
    "MESURE_IEM_022_VALUE", "MESURE_IEM_028_VALUE", "MESURE_IEM_030_VALUE",
    "MESURE_IEM_031_VALUE", "MESURE_IEM_032_VALUE", "MESURE_IEM_033_VALUE",
    "MESURE_IEM_034_VALUE", "MESURE_IEM_035_VALUE", "MESURE_IEM_036_VALUE",
    "MESURE_IEM_037_VALUE", "MESURE_IEM_041_VALUE",
]
WAIST = "MESURE_IEM_004_VALUE"
WITHHELD = ["MESURE_IEM_003_VALUE", "MESURE_IEM_005_VALUE",
            "MESURE_IEM_006_VALUE", "MESURE_IEM_040_VALUE"]
BANDS = [(19, 29, "19-29"), (30, 39, "30-39"), (40, 49, "40-49"),
         (50, 59, "50-59"), (60, 64, "60-64")]


def age_band(age):
    out = pd.Series(pd.NA, index=age.index, dtype="object")
    for lo, hi, lab in BANDS:
        out[age.between(lo, hi)] = lab
    return out


def build(cohort):
    """cohort is "A" (2022 onward) or "B" (2024 onward, adds waist)."""
    inputs = INPUTS_A + ([WAIST] if cohort == "B" else [])
    keep = (["AGE_FLAG_NM", "MESURE_AGE_CO", "SEXDSTN_FLAG_CD",
             "MESURE_PLACE_FLAG_NM"] + inputs + WITHHELD)
    first_year = 2024 if cohort == "B" else 0

    files = sorted(glob.glob(os.path.join(
        RAW_DIR, "KS_NFA_FTNESS_MESURE_ITEM_MESURE_INFO_*.csv")))
    if not files:
        raise SystemExit(
            f"No monthly NFA files under {RAW_DIR}.\n"
            "These are licensed data and are not distributed with this "
            "repository; see README.md, section 'Data'.")

    frames = []
    for path in files:
        m = re.search(r"(\d{4})(\d{2})", os.path.basename(path))
        if m is None:
            continue
        year = int(m.group(1))
        if year < first_year:
            continue
        frame = pd.read_csv(path, usecols=lambda c: c in keep,
                            low_memory=False, encoding="utf-8-sig")
        frame["YEAR"] = year
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)

    df["MESURE_AGE_CO"] = pd.to_numeric(df["MESURE_AGE_CO"], errors="coerce")
    for col in inputs + WITHHELD:
        df[col] = pd.to_numeric(df.get(col), errors="coerce")

    df = df[(df["AGE_FLAG_NM"] == ADULT)
            & df["MESURE_AGE_CO"].between(19, 64)].copy()
    df["AGE_BAND"] = age_band(df["MESURE_AGE_CO"])
    df["MESURE_PLACE_FLAG_NM"] = df["MESURE_PLACE_FLAG_NM"].fillna(UNKNOWN_PLACE)

    for col, (low, high) in PLAUSIBLE_RANGES.items():
        if col in df.columns:
            df.loc[df[col].notna() & ~df[col].between(low, high), col] = np.nan

    df["MODEL"] = f"Model {cohort}"
    df["SPLIT_SOURCE"] = "full"
    ordered = (["MODEL", "SPLIT_SOURCE", "MESURE_AGE_CO", "AGE_BAND", "YEAR",
                "SEXDSTN_FLAG_CD", "MESURE_PLACE_FLAG_NM"]
               + ([WAIST] if cohort == "B" else [])
               + [c for c in INPUTS_A] + WITHHELD)
    seen, cols = set(), []
    for c in ordered:
        if c in df.columns and c not in seen:
            cols.append(c); seen.add(c)
    return df[cols].reset_index(drop=True)


def verify(cohort, df):
    ref_path = os.path.join(DATA_DIR, f"Model {cohort}", "full_dataset.csv")
    if not os.path.exists(ref_path):
        print(f"   Cohort {cohort}: FAILED - nothing to compare against at {ref_path}")
        return False
    ref = pd.read_csv(ref_path, encoding="utf-8-sig", low_memory=False)
    print(f"   Cohort {cohort}: rebuilt {len(df):,} rows, stored {len(ref):,}")
    key = ["MESURE_AGE_CO", "YEAR", "SEXDSTN_FLAG_CD", "MESURE_PLACE_FLAG_NM"] + WITHHELD
    same_order = (df[key].astype(str).agg("|".join, axis=1)
                  .equals(ref[key].astype(str).agg("|".join, axis=1)))

    # The column sets and their order must match in both directions: checking
    # only "reference column missing from rebuild" would pass a reference that
    # had lost a column, or one whose columns were reordered.
    if list(df.columns) != list(ref.columns):
        only_ref = [c for c in ref.columns if c not in df.columns]
        only_new = [c for c in df.columns if c not in ref.columns]
        if only_ref or only_new:
            print(f"      FAILED - column sets differ. Missing from rebuild: "
                  f"{only_ref or 'none'}; unexpected in rebuild: {only_new or 'none'}")
        else:
            print(f"      FAILED - same columns in a different order")
        return False

    if len(df) != len(ref):
        print(f"      FAILED - {len(df):,} rows rebuilt vs {len(ref):,} stored")
        return False

    # Compare every column rather than a hand-written list of measurement and
    # label names: a list would silently stop covering a column the day one is
    # added. Numeric columns use exact NaN-aware equality (np.isclose would let a
    # drift of ~1e-8 relative pass as "zero mismatches"); everything else is
    # compared as text. Numeric columns are never cast through an integer dtype,
    # which would truncate a corrupted 23.7 back to 23.
    mismatch = 0
    for c in ref.columns:
        if (np.issubdtype(ref[c].dtype, np.number)
                and np.issubdtype(df[c].dtype, np.number)):
            a, b = df[c].to_numpy(float), ref[c].to_numpy(float)
            differs = (a != b) & ~(np.isnan(a) & np.isnan(b))
        else:
            differs = (df[c].astype(str).to_numpy()
                       != ref[c].astype(str).to_numpy())
        n = int(differs.sum())
        if n:
            print(f"      FAILED - {n:,} differing values in {c}")
        mismatch += n

    print(f"      row order identical: {same_order}")
    print(f"      value mismatches across all {len(ref.columns)} columns: {mismatch}")
    return same_order and mismatch == 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="diff against the stored full_dataset.csv")
    args = ap.parse_args()

    ok = True
    for cohort in ["A", "B"]:
        df = build(cohort)
        if args.verify:
            ok &= verify(cohort, df)
        else:
            out = os.path.join(DATA_DIR, f"Model {cohort}")
            os.makedirs(out, exist_ok=True)
            df.to_csv(os.path.join(out, "full_dataset.csv"),
                      index=False, encoding="utf-8-sig")
            print(f"   Cohort {cohort}: {len(df):,} rows -> {out}/full_dataset.csv")
    if args.verify:
        if ok:
            print("\nCohort derivation reproduces the published datasets exactly.")
        else:
            print("\nMismatch: see the diffs above.")
            sys.exit(1)
