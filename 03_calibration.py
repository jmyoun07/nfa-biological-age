"""
Calibration analysis for the fitness-based biological-age (BA) LightGBM model.
Computes, on the held-out test set for Model A and Model B:
  - Calibration-in-the-large (CITL = mean predicted - mean observed)
  - Calibration slope & intercept (observed CA regressed on predicted BA; ideal slope=1, intercept=0)
  - Decile-of-predicted observed-vs-predicted table
  - Calibration across chronological age-decade strata
  - A two-panel calibration plot, written as Figure6_Calibration_AB.png (Figure 2 in the manuscript)
Uses only the existing held-out test predictions in $NFA_DATA_DIR/Model {A,B}/result_set.csv.
No model re-training or other-PC access required.
"""

# NOTE: machine-specific paths in the original working copy have been replaced
# by the shared resolver in paths.py so the script runs from a checkout.
from paths import DATA_DIR, OUT_DIR, FIG_DIR  # noqa: E402
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = DATA_DIR
FINAL = BASE
OUTDIR = os.path.join(OUT_DIR, "calibration")
os.makedirs(OUTDIR, exist_ok=True)

AGE_BANDS = [(19, 29), (30, 39), (40, 49), (50, 59), (60, 64)]


def ols(x, y):
    """Return slope, intercept of y ~ x via least squares."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    sx, sy = x.sum(), y.sum()
    sxx = (x * x).sum()
    sxy = (x * y).sum()
    slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    intercept = (sy - slope * sx) / n
    return slope, intercept


def analyze(model_name, folder):
    df = pd.read_csv(os.path.join(FINAL, folder, "result_set.csv"))
    df = df[df["RESULT_SPLIT"] == "test"].copy()
    ca = df["MESURE_AGE_CO"].astype(float).values        # observed chronological age
    pred = df["BA_PRED"].astype(float).values            # predicted biological age (raw LGBM)
    corr = df["BA_CORRECTED"].astype(float).values       # bias-corrected BA

    n = len(df)
    mae = float(np.mean(np.abs(pred - ca)))
    rmse = float(np.sqrt(np.mean((pred - ca) ** 2)))
    r = float(np.corrcoef(pred, ca)[0, 1])
    r2 = 1 - np.sum((ca - pred) ** 2) / np.sum((ca - ca.mean()) ** 2)

    # --- Calibration of the raw predicted age vs chronological age ---
    citl = float(pred.mean() - ca.mean())                # calibration-in-the-large
    slope, intercept = ols(pred, ca)                     # observed ~ predicted (ideal 1, 0)

    # --- Decile-of-predicted observed-vs-predicted table ---
    qs = pd.qcut(pred, 10, labels=False, duplicates="drop")
    dec_rows = []
    for q in sorted(pd.unique(qs)):
        m = qs == q
        dec_rows.append({
            "decile": int(q) + 1,
            "n": int(m.sum()),
            "mean_predicted_BA": float(pred[m].mean()),
            "mean_observed_CA": float(ca[m].mean()),
            "diff_pred_minus_obs": float(pred[m].mean() - ca[m].mean()),
        })
    dec = pd.DataFrame(dec_rows)

    # --- Calibration across chronological age-decade strata ---
    band_rows = []
    for lo, hi in AGE_BANDS:
        m = (ca >= lo) & (ca <= hi)
        if m.sum() == 0:
            continue
        band_rows.append({
            "age_band": f"{lo}-{hi}",
            "n": int(m.sum()),
            "mean_observed_CA": float(ca[m].mean()),
            "mean_predicted_BA": float(pred[m].mean()),
            "mean_bias_pred_minus_obs": float((pred[m] - ca[m]).mean()),
            "mean_corrected_BA": float(corr[m].mean()),
            "mean_bias_corr_minus_obs": float((corr[m] - ca[m]).mean()),
        })
    band = pd.DataFrame(band_rows)

    summary = {
        "model": model_name,
        "n_test": n,
        "MAE": mae, "RMSE": rmse, "R2": float(r2), "pearson_r": r,
        "calibration_in_the_large": citl,
        "calibration_slope": float(slope),
        "calibration_intercept": float(intercept),
        "mean_predicted_BA": float(pred.mean()),
        "mean_observed_CA": float(ca.mean()),
    }
    return summary, dec, band, (ca, pred)


results = {}
plot_data = {}
for model_name, folder in [("Model A", "Model A"), ("Model B", "Model B")]:
    summary, dec, band, raw = analyze(model_name, folder)
    results[model_name] = summary
    dec.to_csv(os.path.join(OUTDIR, f"calibration_deciles_{folder.replace(' ', '_')}.csv"), index=False)
    band.to_csv(os.path.join(OUTDIR, f"calibration_ageband_{folder.replace(' ', '_')}.csv"), index=False)
    plot_data[model_name] = (raw, dec, band)
    print(f"================ {model_name} ================")
    print(json.dumps(summary, indent=2))
    print("\nDecile observed-vs-predicted:")
    print(dec.to_string(index=False))
    print("\nAge-band calibration (raw and corrected):")
    print(band.to_string(index=False))
    print()

with open(os.path.join(OUTDIR, "calibration_summary.json"), "w") as f:
    json.dump(results, f, indent=2)

# --------- Figure 6: two-panel calibration plot ---------
fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), dpi=300)
for ax, model_name in zip(axes, ["Model A", "Model B"]):
    (ca, pred), dec, band = plot_data[model_name]
    s = results[model_name]
    lims = [17, 66]
    # reference (ideal) diagonal
    ax.plot(lims, lims, color="0.4", lw=1.2, ls="--", zorder=1, label="Ideal (y = x)")
    # density-style background scatter (subsample for legibility)
    idx = np.linspace(0, len(ca) - 1, min(4000, len(ca))).astype(int)
    ax.scatter(pred[idx], ca[idx], s=4, alpha=0.08, color="#4C72B0", zorder=2, rasterized=True)
    # decile observed-vs-predicted points + connecting line
    ax.plot(dec["mean_predicted_BA"], dec["mean_observed_CA"],
            "-o", color="#C44E52", lw=1.6, ms=6, zorder=4,
            label="Observed mean (deciles of predicted)")
    # fitted calibration line
    xs = np.array(lims)
    ax.plot(xs, s["calibration_slope"] * xs + s["calibration_intercept"],
            color="#55A868", lw=1.6, zorder=3, label="Fitted calibration")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("Predicted biological age (years)")
    ax.set_ylabel("Observed chronological age (years)")
    txt = (f"{model_name}\n"
           f"Calibration slope = {s['calibration_slope']:.3f}\n"
           f"Calibration intercept = {s['calibration_intercept']:.2f} yr\n"
           f"CITL = {s['calibration_in_the_large']:+.3f} yr")
    ax.text(0.04, 0.96, txt, transform=ax.transAxes, va="top", ha="left",
            fontsize=9, bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    ax.legend(loc="lower right", fontsize=7.5, framealpha=0.9)
plt.tight_layout()
fig_path = os.path.join(OUT_DIR, "Figure6_Calibration_AB.png")
os.makedirs(os.path.dirname(fig_path), exist_ok=True)
fig.savefig(fig_path, dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(OUTDIR, "Figure6_Calibration_AB.png"), dpi=300, bbox_inches="tight")
print("Saved figure:", fig_path)
