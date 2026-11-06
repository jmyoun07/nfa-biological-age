#!/usr/bin/env python3
"""Model development: the five candidate algorithms, cross-validation, the final
refit, and the Kronmal-type aging-gap correction.

VERIFICATION
------------
Run with --verify to diff this script's output against the stored artefacts. On
the reference machine it reproduces

  * LightGBM (the primary model), random forest, decision tree and elastic net
    predictions to ~1e-14, i.e. bit-exact within floating-point noise, in both
    cohorts;
  * the 5-fold cross-validation table, the selected best model, and the Kronmal
    intercept and slope to the same precision;
  * the corrected aging gap (DELTA_CORR) to ~1e-14.

--verify rebuilds each artefact this script writes and diffs the whole frame -
column set, column order, row count and every value - rather than only the
prediction columns, so a corrupted AGE_BAND, withheld biomarker, DELTA_RAW or
split tag fails rather than passing on matching predictions. A NaN anywhere, a
missing file, a missing field and duplicate or absent correction rows all fail.

Install the pinned versions in requirements.txt; gradient-boosting predictions
are build-sensitive, so the pins are what make the run deterministic.

NOTES ON DETAILS THAT ARE LOAD-BEARING
--------------------------------------
* Feature order matters. colsample_bytree=0.9 makes LightGBM and XGBoost
  sensitive to column position; waist circumference must come LAST in Cohort B.
  Placing it first changes the Cohort B MAE from 3.8346 to 3.8187.
* The split is a nested 70/15/15 (train / validation / test) stratified on
  AGE_BAND alone, with random_state=42 at both calls. The 85% development set is
  train + validation concatenated in that order. Row order of the input
  full_dataset.csv therefore determines the partition.
* Missing values are median-imputed inside the pipeline for all five algorithms,
  including LightGBM and XGBoost. Native NaN handling gives materially different
  predictions (Cohort A MAE 3.7818) and does not reproduce the published models.
* Cross-validation runs inside the 70% training block, not the 85% development
  set. The final models are refit on the full development set.

USAGE
-----
    export NFA_DATA_DIR=/path/to/derived/cohorts
    python 00_train_model.py            # write the artefacts into NFA_DATA_DIR
    python 00_train_model.py --verify   # diff against the artefacts already there
"""

import argparse
import math
import os
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from lightgbm import LGBMRegressor
from scipy.stats import pearsonr, spearmanr
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from paths import DATA_DIR, OUT_DIR

warnings.filterwarnings("ignore")

# 20 raw NFA codes in ascending order; Cohort B appends waist LAST.
# Order is load-bearing: colsample_bytree=0.9 makes LightGBM/XGBoost sensitive
# to column position; putting waist first shifts the Cohort B MAE by ~0.016 years.
FEATS_A = [
    "MESURE_IEM_001_VALUE", "MESURE_IEM_002_VALUE", "MESURE_IEM_007_VALUE",
    "MESURE_IEM_008_VALUE", "MESURE_IEM_012_VALUE", "MESURE_IEM_018_VALUE",
    "MESURE_IEM_019_VALUE", "MESURE_IEM_020_VALUE", "MESURE_IEM_021_VALUE",
    "MESURE_IEM_022_VALUE", "MESURE_IEM_028_VALUE", "MESURE_IEM_030_VALUE",
    "MESURE_IEM_031_VALUE", "MESURE_IEM_032_VALUE", "MESURE_IEM_033_VALUE",
    "MESURE_IEM_034_VALUE", "MESURE_IEM_035_VALUE", "MESURE_IEM_036_VALUE",
    "MESURE_IEM_037_VALUE", "MESURE_IEM_041_VALUE",
]
WAIST = "MESURE_IEM_004_VALUE"
FEATS = {"A": FEATS_A, "B": FEATS_A + [WAIST]}


def metric_row(y_true, pred):
    y = np.asarray(y_true)
    return {
        "MAE": float(mean_absolute_error(y, pred)),
        "RMSE": float(math.sqrt(mean_squared_error(y, pred))),
        "R2": float(r2_score(y, pred)),
        "PEARSON_R": float(pearsonr(y, pred).statistic),
    }


def _diff(a, b):
    """Absolute difference that treats a stray NaN as a failure, not as zero.

    max(0.0, float("nan")) evaluates to 0.0, so a NaN in either operand would
    otherwise pass the tolerance check.
    """
    a, b = float(a), float(b)
    if math.isnan(a) or math.isnan(b):
        return 0.0 if (math.isnan(a) and math.isnan(b)) else math.inf
    return abs(a - b)


def corrected_delta(age, pred, intercept, slope):
    return (pred - age) - (intercept + slope * age)


def split_data(df):
    train_df, holdout_df = train_test_split(
        df, test_size=0.30, random_state=42, stratify=df["AGE_BAND"])
    valid_df, test_df = train_test_split(
        holdout_df, test_size=0.50, random_state=42, stratify=holdout_df["AGE_BAND"])
    return train_df.copy(), valid_df.copy(), test_df.copy()


def build_models(numeric_cols):
    scaled = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler", StandardScaler())]), numeric_cols)])
    tree_pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric_cols)])
    return {
        "EN": Pipeline([("preprocess", scaled),
                        ("model", ElasticNet(alpha=0.1, l1_ratio=0.5,
                                             random_state=42, max_iter=10000))]),
        "DT": Pipeline([("preprocess", tree_pre),
                        ("model", DecisionTreeRegressor(random_state=42,
                                                        min_samples_leaf=20,
                                                        max_depth=12))]),
        "RF": Pipeline([("preprocess", tree_pre),
                        ("model", RandomForestRegressor(random_state=42,
                                                        n_estimators=300, n_jobs=-1,
                                                        min_samples_leaf=5,
                                                        max_features="sqrt"))]),
        "XGB": Pipeline([("preprocess", tree_pre),
                         ("model", XGBRegressor(objective="reg:squarederror",
                                                n_estimators=300, max_depth=6,
                                                learning_rate=0.07, subsample=0.9,
                                                colsample_bytree=0.9,
                                                random_state=42, n_jobs=4,
                                                verbosity=0))]),
        "LGBM": Pipeline([("preprocess", tree_pre),
                          ("model", LGBMRegressor(random_state=42, n_estimators=300,
                                                  num_leaves=31, learning_rate=0.07,
                                                  subsample=0.9, colsample_bytree=0.9,
                                                  verbosity=-1))]),
    }


def run_cv(x_train, y_train, age_band, models):
    rows = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for name, model in models.items():
        for k, (tr, va) in enumerate(skf.split(x_train, age_band.astype(str)), 1):
            m = clone(model)
            m.fit(x_train.iloc[tr], y_train.iloc[tr])
            rows.append({"MODEL": name, "FOLD": k,
                         **metric_row(y_train.iloc[va], m.predict(x_train.iloc[va]))})
    fold = pd.DataFrame(rows)
    summary = (fold.groupby("MODEL", as_index=False)
               .agg(CV_MAE_MEAN=("MAE", "mean"), CV_MAE_SD=("MAE", "std"),
                    CV_RMSE_MEAN=("RMSE", "mean"), CV_RMSE_SD=("RMSE", "std"),
                    CV_R2_MEAN=("R2", "mean"), CV_R2_SD=("R2", "std"),
                    CV_PEARSON_R_MEAN=("PEARSON_R", "mean"),
                    CV_PEARSON_R_SD=("PEARSON_R", "std"))
               .sort_values(["CV_MAE_MEAN", "CV_R2_MEAN"], ascending=[True, False])
               .reset_index(drop=True))
    return fold, summary


def run_cohort(coh):
    feats = FEATS[coh]
    full = pd.read_csv(f"{DATA_DIR}/Model {coh}/full_dataset.csv",
                       encoding="utf-8-sig", low_memory=False)
    train_df, valid_df, test_df = split_data(full)
    tv = pd.concat([train_df, valid_df], ignore_index=True)

    models = build_models(feats)
    _, cv_summary = run_cv(train_df[feats], train_df["MESURE_AGE_CO"],
                           train_df["AGE_BAND"], models)
    best = str(cv_summary.iloc[0]["MODEL"])

    # Pass the Series through to metric_row, as the source pipeline did. Its
    # np.asarray leaves the integer ages integral, and each model's predictions
    # keep their own dtype (XGBoost predicts float32). Forcing a common float64
    # shifts the XGBoost MAE from the published 4.020024299621582 to
    # 4.020024062344755; forcing float32 shifts the Pearson correlation instead.
    y_te_s = test_df["MESURE_AGE_CO"]
    y_te = y_te_s.to_numpy(float)
    y_tv = tv["MESURE_AGE_CO"].to_numpy(float)
    rows, pred_te, pred_tv = [], {}, {}
    for name, model in models.items():
        m = clone(model)
        m.fit(tv[feats], tv["MESURE_AGE_CO"])
        pred_te[name] = m.predict(test_df[feats])
        pred_tv[name] = m.predict(tv[feats])
        rows.append({"MODEL": name, **metric_row(y_te_s, pred_te[name])})
    final = (pd.DataFrame(rows).sort_values(["MAE", "R2"], ascending=[True, False])
             .reset_index(drop=True))

    bias = smf.ols("DELTA_RAW ~ CA", data=pd.DataFrame(
        {"CA": y_tv, "DELTA_RAW": pred_tv[best] - y_tv})).fit()
    b0 = float(bias.params["Intercept"])
    b1 = float(bias.params["CA"])
    d_corr_te = corrected_delta(y_te, pred_te[best], b0, b1)
    d_corr_tv = corrected_delta(y_tv, pred_tv[best], b0, b1)
    return dict(best=best, cv=cv_summary, final=final, b0=b0, b1=b1,
                pred_te=pred_te, pred_tv=pred_tv, y_te=y_te, y_tv=y_tv,
                d_corr=d_corr_te, d_corr_tv=d_corr_tv,
                train_df=train_df, valid_df=valid_df, test_df=test_df, tv=tv,
                n=(len(train_df), len(valid_df), len(test_df)))


def report(coh, R):
    print("=" * 74)
    print(f"Cohort {coh}   train/valid/test = {R['n'][0]:,} / {R['n'][1]:,} / {R['n'][2]:,}")
    print(f"CV 최적 모델: {R['best']}")
    pub_final = pd.read_csv(f"{DATA_DIR}/Model {coh}/final_test_metrics.csv", encoding="utf-8-sig")
    pub_cv = pd.read_csv(f"{DATA_DIR}/Model {coh}/cv_summary.csv", encoding="utf-8-sig")
    pub_bias = pd.read_csv(f"{DATA_DIR}/Model {coh}/deltaage_correction.csv", encoding="utf-8-sig")
    pub_res = pd.read_csv(f"{DATA_DIR}/Model {coh}/result_set.csv", encoding="utf-8-sig", low_memory=False)

    print("\n-- final_test_metrics (재현 vs 논문) --")
    worst = 0.0
    for _, r in R["final"].iterrows():
        p = pub_final[pub_final["MODEL"] == r["MODEL"]].iloc[0]
        d = max(_diff(r[c], p[c]) for c in ["MAE", "RMSE", "R2", "PEARSON_R"])
        worst = max(worst, d)
        print(f"   {r['MODEL']:5s} MAE {r['MAE']:.12f} vs {p['MAE']:.12f}   최대차 {d:.2e}")
    print(f"   -> final_test_metrics 최대 절대차 {worst:.3e}")

    dcv = 0.0
    for _, r in R["cv"].iterrows():
        p = pub_cv[pub_cv["MODEL"] == r["MODEL"]].iloc[0]
        dcv = max(dcv, max(_diff(r[c], p[c]) for c in r.index if c != "MODEL"))
    print(f"-- cv_summary 최대 절대차 {dcv:.3e}")

    dbias = 0.0
    for label, y, pred, dc in (
            ("train_valid", R["y_tv"], R["pred_tv"][R["best"]], R["d_corr_tv"]),
            ("test", R["y_te"], R["pred_te"][R["best"]], R["d_corr"])):
        pb = pub_bias[pub_bias["SPLIT"] == label]
        if len(pb) != 1:
            print(f"-- Kronmal FAILED: expected exactly one '{label}' row, "
                  f"found {len(pb)}")
            dbias = math.inf
            continue
        pb = pb.iloc[0]
        raw = pred - y
        mine = {
            "BIAS_INTERCEPT": R["b0"], "BIAS_SLOPE": R["b1"],
            "PEARSON_RAW_WITH_CA": float(pearsonr(raw, y).statistic),
            "SPEARMAN_RAW_WITH_CA": float(spearmanr(raw, y).statistic),
            "PEARSON_CORR_WITH_CA": float(pearsonr(dc, y).statistic),
            "SPEARMAN_CORR_WITH_CA": float(spearmanr(dc, y).statistic),
        }
        absent = [k for k in mine if k not in pb.index]
        if absent:
            print(f"-- Kronmal/{label} FAILED: fields absent from the reference: "
                  f"{absent}")
            dbias = math.inf
            continue
        worst_f = max(_diff(v, pb[k]) for k, v in mine.items())
        dbias = max(dbias, worst_f)
        print(f"-- Kronmal/{label}: 6개 필드 최대 절대차 {worst_f:.3e}")

    # Rather than checking only the prediction columns, rebuild each artefact
    # this script writes and diff it wholesale. Otherwise a corrupted MODEL,
    # AGE_BAND, withheld biomarker, DELTA_RAW or BA_CORRECTED - all of which
    # 02-06 consume - would pass while the predictions matched.
    def diff_frame(mine, ref, label):
        if list(mine.columns) != list(ref.columns):
            only_ref = [c for c in ref.columns if c not in mine.columns]
            only_new = [c for c in mine.columns if c not in ref.columns]
            if only_ref or only_new:
                print(f"-- {label}: FAILED - column sets differ. missing "
                      f"{only_ref or 'none'}, unexpected {only_new or 'none'}")
            else:
                print(f"-- {label}: FAILED - same columns in a different order")
            return math.inf
        if len(mine) != len(ref):
            print(f"-- {label}: FAILED - {len(mine):,} rows vs {len(ref):,}")
            return math.inf
        worst_c = 0.0
        for c in ref.columns:
            a, b = mine[c].to_numpy(), ref[c].to_numpy()
            if np.issubdtype(ref[c].dtype, np.number) and np.issubdtype(mine[c].dtype, np.number):
                af = a.astype(float)
                mdt = np.dtype(mine[c].dtype)
                if np.issubdtype(mdt, np.floating) and mdt.itemsize < 8:
                    # Round the reference to the precision this column was
                    # written at, so a float32 prediction column re-read as
                    # float64 does not look like a ~1e-6 mismatch.
                    bf = b.astype(mdt).astype(float)
                else:
                    # Never cast through an integer dtype: it would truncate a
                    # corrupted 23.7 back to 23 and turn an injected NaN into
                    # garbage, hiding the very corruption being checked for.
                    bf = b.astype(float)
                both_nan = np.isnan(af) & np.isnan(bf)
                d = np.abs(af - bf)
                d[both_nan] = 0.0
                if np.isnan(d).any():
                    print(f"-- {label}: FAILED - NaN mismatch in {c}")
                    return math.inf
                worst_c = max(worst_c, float(d.max()) if d.size else 0.0)
            else:
                bad = int((pd.Series(a).astype(str).to_numpy()
                           != pd.Series(b).astype(str).to_numpy()).sum())
                if bad:
                    print(f"-- {label}: FAILED - {bad:,} differing values in {c}")
                    return math.inf
        print(f"-- {label}: {len(ref):,} rows x {len(ref.columns)} columns, "
              f"최대 절대차 {worst_c:.3e}")
        return worst_c

    dpred = diff_frame(
        _prediction_frame(coh, "test", R["test_df"], R["pred_te"], R["y_te"],
                          R["best"], R["b0"], R["b1"]),
        pub_res, "result_set.csv")

    dd = np.abs(R["d_corr"] - pub_res["DELTA_CORR"].to_numpy(float)).max()
    print(f"-- DELTA_CORR 최대차 {dd:.3e}")

    # the metric tables must carry exactly the five models, once each
    for label, mine, ref in (("final_test_metrics.csv", R["final"], pub_final),
                             ("cv_summary.csv", R["cv"], pub_cv)):
        if (len(ref) != len(mine)
                or sorted(ref["MODEL"]) != sorted(mine["MODEL"])):
            print(f"-- {label}: FAILED - expected exactly {len(mine)} rows for "
                  f"{sorted(mine['MODEL'])}, found {len(ref)} for "
                  f"{sorted(ref['MODEL'])}")
            worst = math.inf

    dsplit = 0.0
    for fname in ("development_set.csv", "validation_set.csv"):
        path = os.path.join(DATA_DIR, f"Model {coh}", fname)
        if not os.path.exists(path):
            print(f"-- {fname}: FAILED - 참조 파일 없음 ({path})")
            dsplit = math.inf
            continue
        ref = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        # Rebuild exactly what write_outputs would emit and diff the whole frame,
        # so a reduced file, a wrong MODEL tag or swapped train/valid labels all
        # fail. Comparing only intersecting columns would let those through.
        dsplit = max(dsplit, diff_frame(_split_frame(coh, fname, R), ref, fname))

    ok = (worst < 1e-9 and dcv < 1e-9 and dpred < 1e-9 and dd < 1e-9
          and dbias < 1e-9 and dsplit < 1e-9)
    print(f"\n>>> Cohort {coh}: {'완전 재현' if ok else '불일치 있음'}")
    return ok



ID_COLS = ["MESURE_AGE_CO", "AGE_BAND", "YEAR", "SEXDSTN_FLAG_CD",
           "MESURE_PLACE_FLAG_NM"]
WITHHELD = ["MESURE_IEM_003_VALUE", "MESURE_IEM_005_VALUE",
            "MESURE_IEM_006_VALUE", "MESURE_IEM_040_VALUE"]


def _prediction_frame(coh, split_label, source_df, preds, y, best, b0, b1):
    """Assemble a result_set.csv-shaped frame for one split."""
    out = source_df[ID_COLS + WITHHELD].reset_index(drop=True).copy()
    out.insert(0, "MODEL", f"Model {coh}")
    out.insert(1, "RESULT_SPLIT", split_label)
    for name in ["EN", "DT", "RF", "XGB", "LGBM"]:
        out[f"PRED_{name}"] = preds[name]
    out["BA_PRED"] = preds[best]
    out["DELTA_RAW"] = out["BA_PRED"].to_numpy() - y
    out["DELTA_CORR"] = corrected_delta(y, preds[best], b0, b1)
    out["BA_CORRECTED"] = y + out["DELTA_CORR"].to_numpy()
    return out


def _split_frame(coh, fname, R):
    """Rebuild development_set.csv / validation_set.csv exactly as written."""
    front = ["MODEL", "SPLIT_SOURCE"]
    if fname == "development_set.csv":
        f = pd.concat([R["train_df"].assign(SPLIT_SOURCE="train"),
                       R["valid_df"].assign(SPLIT_SOURCE="valid")],
                      ignore_index=True)
    else:
        f = R["valid_df"].assign(SPLIT_SOURCE="valid").reset_index(drop=True)
    f = f.copy()
    f["MODEL"] = f"Model {coh}"
    return f[front + [c for c in f.columns if c not in front]]


def write_outputs(coh, R):
    """Write every artefact the downstream scripts (02-06) consume.

    Everything goes to NFA_DATA_DIR, mirroring the layout of the stored
    artefacts: that is where 01-06 read the datasets from, where
    00a_build_cohorts.py writes full_dataset.csv, and where --verify looks for
    the reference tables. NFA_OUT_DIR is left for the analysis output that
    01-06 produce.
    """
    d = os.path.join(DATA_DIR, f"Model {coh}")
    os.makedirs(d, exist_ok=True)

    R["final"].to_csv(os.path.join(d, "final_test_metrics.csv"), index=False)
    R["cv"].to_csv(os.path.join(d, "cv_summary.csv"), index=False)

    # result_set.csv: held-out test rows with all five models' predictions,
    # the corrected aging gap, and the withheld biomarkers. 03-06 read this.
    _prediction_frame(coh, "test", R["test_df"], R["pred_te"], R["y_te"],
                      R["best"], R["b0"], R["b1"]).to_csv(
        os.path.join(d, "result_set.csv"), index=False, encoding="utf-8-sig")

    # development_set.csv / validation_set.csv: the split membership, tagged the
    # way the stored artefacts tag it. 01 and 02a/02b read these.
    for fname in ("development_set.csv", "validation_set.csv"):
        _split_frame(coh, fname, R).to_csv(os.path.join(d, fname), index=False,
                                          encoding="utf-8-sig")

    # deltaage_correction.csv, in the stored seven-column two-row shape.
    rows = []
    for label, y, pred, dc in (
            ("train_valid", R["y_tv"], R["pred_tv"][R["best"]], R["d_corr_tv"]),
            ("test", R["y_te"], R["pred_te"][R["best"]], R["d_corr"])):
        raw = pred - y
        rows.append({
            "SPLIT": label,
            "PEARSON_RAW_WITH_CA": float(pearsonr(raw, y).statistic),
            "SPEARMAN_RAW_WITH_CA": float(spearmanr(raw, y).statistic),
            "PEARSON_CORR_WITH_CA": float(pearsonr(dc, y).statistic),
            "SPEARMAN_CORR_WITH_CA": float(spearmanr(dc, y).statistic),
            "BIAS_INTERCEPT": R["b0"],
            "BIAS_SLOPE": R["b1"],
        })
    pd.DataFrame(rows).to_csv(os.path.join(d, "deltaage_correction.csv"),
                              index=False)
    print(f"   -> {d}")
    print(f"      full_dataset.csv (from 00a), development_set.csv, "
          f"validation_set.csv, result_set.csv,")
    print(f"      final_test_metrics.csv, cv_summary.csv, deltaage_correction.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="diff the outputs against the stored artefacts")
    args = ap.parse_args()
    ok = True
    for coh in ["A", "B"]:
        R = run_cohort(coh)
        if args.verify:
            ok &= report(coh, R)
        else:
            print(f"Cohort {coh}: best model {R['best']}, "
                  f"test MAE {R['final'].iloc[0]['MAE']:.6f}")
            write_outputs(coh, R)
    if args.verify:
        print("\n" + "=" * 74)
        if ok:
            print("Reproduces the published outputs.")
        else:
            print("Differences remain; see the diffs above.")
            sys.exit(1)
