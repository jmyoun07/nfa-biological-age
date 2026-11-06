#!/usr/bin/env python3
"""
Publication-grade nonparametric bootstrap 95% CIs for held-out test-set
discrimination and calibration metrics — Korea NFA biological-age study.

Design
------
* Case (nonparametric) bootstrap: held-out TEST records resampled WITH
  replacement to the original test-set size. Fitted models and the
  development-set-derived Kronmal correction parameters are held FIXED, so the
  intervals quantify sampling uncertainty of the performance estimate
  conditional on the fitted model.
* B = 50,000 replicates, seed fixed. Percentile method primary; bias-corrected
  (BC) percentile computed as a sensitivity check.
* Within a cohort, the SAME resample index set is reused across all five
  algorithms, so between-algorithm differences are paired.
* Feature Set A vs Feature Set B are additionally compared with a paired
  bootstrap restricted to records held out in BOTH cohorts.

All arithmetic uses explicit reductions (no BLAS matmul) to avoid the spurious
FP-status warnings raised by NumPy 2.0 on the macOS Accelerate backend.
"""


# NOTE: machine-specific paths in the original working copy have been replaced
# by the shared resolver in paths.py so the script runs from a checkout.
from paths import DATA_DIR, OUT_DIR, FIG_DIR  # noqa: E402
import json
import numpy as np
import pandas as pd
from scipy.stats import norm

BASE = DATA_DIR
OUT = OUT_DIR

B = 50000
SEED = 20250726
ALGOS = [("LightGBM", "PRED_LGBM"), ("XGBoost", "PRED_XGB"),
         ("Random forest", "PRED_RF"), ("Decision tree", "PRED_DT"),
         ("Elastic net", "PRED_EN")]
DECADES = ["19-29", "30-39", "40-49", "50-59", "60-64"]
BIAS = {"A": dict(intercept=7.0725629681266025, slope=-0.2012514386615332),
        "B": dict(intercept=6.964529433240011, slope=-0.1990409313624863)}


def ssum(a):
    return float(np.sum(a, dtype=np.float64))


def ols(x, y):
    """OLS of y on x -> (slope, intercept)."""
    xm = float(np.mean(x)); ym = float(np.mean(y))
    dx = x - xm
    return ssum(dx * (y - ym)) / ssum(dx * dx), None if False else ym - (
        ssum(dx * (y - ym)) / ssum(dx * dx)) * xm


def pearson(a, b):
    am = a - np.mean(a); bm = b - np.mean(b)
    return ssum(am * bm) / np.sqrt(ssum(am * am) * ssum(bm * bm))


def block(y, preds, dec, d_raw, d_cor):
    out = {}
    ym = float(np.mean(y))
    dy = y - ym
    sstot = ssum(dy * dy)
    for name, p in preds.items():
        e = p - y
        out[f"MAE|{name}"] = float(np.mean(np.abs(e)))
        sse = ssum(e * e)
        out[f"RMSE|{name}"] = np.sqrt(sse / len(y))
        out[f"R2|{name}"] = 1.0 - sse / sstot
        out[f"r|{name}"] = pearson(p, y)
    p = preds["LightGBM"]
    slope, intercept = ols(p, y)
    out["CITL"] = float(np.mean(p)) - ym
    out["CalSlope"] = slope
    out["CalIntercept"] = intercept
    out["dAge_mean"] = float(np.mean(d_cor))
    out["dAge_sd"] = float(np.std(d_cor, ddof=1))
    out["dAge_p2.5"] = float(np.percentile(d_cor, 2.5))
    out["dAge_p97.5"] = float(np.percentile(d_cor, 97.5))
    out["dAge_pos_pct"] = 100.0 * float(np.mean(d_cor > 0))
    out["corr_dAgeCorr_CA"] = pearson(d_cor, y)
    out["corr_dAgeRaw_CA"] = pearson(d_raw, y)
    for k, d in enumerate(DECADES):
        m = dec == k
        out[f"MSE_raw|{d}"] = float(np.mean(d_raw[m])) if m.any() else np.nan
        out[f"MSE_corr|{d}"] = float(np.mean(d_cor[m])) if m.any() else np.nan
    for nm, _ in ALGOS[1:]:
        out[f"dMAE|LightGBM-{nm}"] = out["MAE|LightGBM"] - out[f"MAE|{nm}"]
    out["MAEred_pct|vs Elastic net"] = 100.0 * (
        out["MAE|Elastic net"] - out["MAE|LightGBM"]) / out["MAE|Elastic net"]
    return out


def summarize(point, draws):
    lo, hi = np.percentile(draws, [2.5, 97.5])
    # bias-corrected (BC) percentile sensitivity check
    prop = float(np.mean(draws < point))
    prop = min(max(prop, 1.0 / (2 * len(draws))), 1 - 1.0 / (2 * len(draws)))
    z0 = norm.ppf(prop)
    bclo, bchi = np.percentile(
        draws, [100 * norm.cdf(2 * z0 - 1.959964),
                100 * norm.cdf(2 * z0 + 1.959964)])
    return dict(point=float(point), lo=float(lo), hi=float(hi),
                se=float(np.std(draws, ddof=1)),
                bc_lo=float(bclo), bc_hi=float(bchi),
                boot_mean=float(np.mean(draws)))


def run_cohort(tag, model, seed=SEED, nboot=B):
    df = pd.read_csv(f"{BASE}/{model}/result_set.csv",
                     encoding="utf-8-sig", low_memory=False)
    y = df["MESURE_AGE_CO"].astype(float).to_numpy()
    preds = {n: df[c].astype(float).to_numpy() for n, c in ALGOS}
    assert np.allclose(df["BA_PRED"].astype(float).to_numpy(), preds["LightGBM"])
    d_raw = df["DELTA_RAW"].astype(float).to_numpy()
    d_cor = df["DELTA_CORR"].astype(float).to_numpy()
    recon = d_raw - (BIAS[tag]["slope"] * y + BIAS[tag]["intercept"])
    assert np.abs(recon - d_cor).max() < 1e-8
    assert np.abs((preds["LightGBM"] - y) - d_raw).max() < 1e-8
    dec = np.array([DECADES.index(b) for b in df["AGE_BAND"].astype(str)])

    n = len(df)
    point = block(y, preds, dec, d_raw, d_cor)
    keys = list(point.keys())
    draws = {k: np.empty(nboot) for k in keys}
    rng = np.random.default_rng(seed)
    for b in range(nboot):
        i = rng.integers(0, n, n)
        mb = block(y[i], {k: v[i] for k, v in preds.items()},
                   dec[i], d_raw[i], d_cor[i])
        for k in keys:
            draws[k][b] = mb[k]
    return dict(n=n, results={k: summarize(point[k], draws[k]) for k in keys})


def paired_AB(seed=SEED + 1, nboot=B):
    dfA = pd.read_csv(f"{BASE}/Model A/result_set.csv",
                      encoding="utf-8-sig", low_memory=False)
    dfB = pd.read_csv(f"{BASE}/Model B/result_set.csv",
                      encoding="utf-8-sig", low_memory=False)
    kc = ["MESURE_AGE_CO", "YEAR", "SEXDSTN_FLAG_CD", "MESURE_PLACE_FLAG_NM",
          "MESURE_IEM_003_VALUE", "MESURE_IEM_005_VALUE",
          "MESURE_IEM_006_VALUE", "MESURE_IEM_040_VALUE"]
    ka = dfA[kc].astype(str).agg("|".join, axis=1)
    kb = dfB[kc].astype(str).agg("|".join, axis=1)
    ua, ub = ka.value_counts(), kb.value_counts()
    ok = set(ua[ua == 1].index) & set(ub[ub == 1].index)
    a = dfA[ka.isin(ok)].assign(_k=ka[ka.isin(ok)]).set_index("_k").sort_index()
    b = dfB[kb.isin(ok)].assign(_k=kb[kb.isin(ok)]).set_index("_k").sort_index()
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]
    y = a["MESURE_AGE_CO"].astype(float).to_numpy()
    assert np.array_equal(y, b["MESURE_AGE_CO"].astype(float).to_numpy())
    pa = a["PRED_LGBM"].astype(float).to_numpy()
    pb = b["PRED_LGBM"].astype(float).to_numpy()
    n = len(common)

    def stat(yy, aa, bb):
        ym = float(np.mean(yy)); dy = yy - ym; sstot = ssum(dy * dy)
        ea, eb = aa - yy, bb - yy
        sa, sb = ssum(ea * ea), ssum(eb * eb)
        mA, mB = float(np.mean(np.abs(ea))), float(np.mean(np.abs(eb)))
        rA, rB = np.sqrt(sa / len(yy)), np.sqrt(sb / len(yy))
        r2A, r2B = 1 - sa / sstot, 1 - sb / sstot
        return dict(MAE_A=mA, MAE_B=mB, dMAE=mB - mA,
                    RMSE_A=rA, RMSE_B=rB, dRMSE=rB - rA,
                    R2_A=r2A, R2_B=r2B, dR2=r2B - r2A)

    point = stat(y, pa, pb)
    keys = list(point.keys())
    draws = {k: np.empty(nboot) for k in keys}
    rng = np.random.default_rng(seed)
    for i in range(nboot):
        j = rng.integers(0, n, n)
        s = stat(y[j], pa[j], pb[j])
        for k in keys:
            draws[k][i] = s[k]
    return dict(n=n, n_A_test=len(dfA), n_B_test=len(dfB),
                results={k: summarize(point[k], draws[k]) for k in keys})


if __name__ == "__main__":
    res = dict(B=B, seed=SEED, method="percentile (BC as sensitivity check)")
    res["cohortA"] = run_cohort("A", "Model A"); print("A done", flush=True)
    res["cohortB"] = run_cohort("B", "Model B"); print("B done", flush=True)
    res["paired_AB"] = paired_AB(); print("paired done", flush=True)
    # Full seed-stability replication: every analysis is repeated under an
    # independent seed so the supplement can speak about the whole procedure.
    SEED2 = 99991
    res["stability_A_seed2"] = run_cohort("A", "Model A", seed=SEED2, nboot=B)
    print("stability A done", flush=True)
    res["stability_B_seed2"] = run_cohort("B", "Model B", seed=SEED2 + 10, nboot=B)
    print("stability B done", flush=True)
    res["stability_paired_seed2"] = paired_AB(seed=SEED2 + 20, nboot=B)
    print("stability paired done", flush=True)
    res["seed2"] = SEED2
    res["paired_seed"] = SEED + 1
    with open(f"{OUT}/bootstrap_v2.json", "w") as f:
        json.dump(res, f, indent=2)
    print("written")
