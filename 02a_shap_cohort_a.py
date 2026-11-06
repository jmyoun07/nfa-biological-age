"""Refit LightGBM with the SHAP configuration and compute TreeExplainer values
per instance for the held-out test set.

Covers both cohorts. The configuration differs from 00_train_model.py, as the
manuscript reports, so the predictions here are close to but not identical with
the reported ones.

Steps:
1. Load development_set (the 70% training plus 15% validation blocks) and result_set (the held-out test predictions used as a reference)
2. Train LightGBM on development_set with reasonable hyperparams
3. Predict on test rows and report the correlation with PRED_LGBM. This is a closeness check, not a match: the configuration here differs from 00_train_model.py, as the manuscript reports
4. Compute SHAP TreeExplainer values per-instance on test rows
5. Save per-instance SHAP, dependence data, and interaction values
6. Generate SHAP dependence plots for top features

Outputs (written to NFA_OUT_DIR):
- shap_per_instance_<Model>.npz       (SHAP values + feature names + X_test)
- shap_dependence_data_<Model>.json   (binned dependence data for top features)
- shap_interaction_<Model>.json       (selected pairwise interaction values)
- shap_dep_<feature>_<Model>.png       (matplotlib figures)
- retrain_validation_<Model>.json     (sanity check vs PRED_LGBM)
"""

# NOTE: machine-specific paths in the original working copy have been replaced
# by the shared resolver in paths.py so the script runs from a checkout.
from paths import DATA_DIR, OUT_DIR, FIG_DIR  # noqa: E402
import os, json
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

BASE = DATA_DIR
OUT = OUT_DIR
os.makedirs(OUT, exist_ok=True)

# Feature mapping (raw NFA codes -> human-readable column names)
FEATURE_MAP = {
    'MESURE_IEM_001_VALUE': 'HEIGHT',
    'MESURE_IEM_002_VALUE': 'WEIGHT',
    'MESURE_IEM_007_VALUE': 'GRIP_LEFT',
    'MESURE_IEM_008_VALUE': 'GRIP_RIGHT',
    'MESURE_IEM_012_VALUE': 'FLEX',
    'MESURE_IEM_018_VALUE': 'BMI',
    'MESURE_IEM_019_VALUE': 'SITUP',
    'MESURE_IEM_020_VALUE': 'SHUTTLE',
    'MESURE_IEM_021_VALUE': 'RUN10',
    'MESURE_IEM_022_VALUE': 'JUMP',
    'MESURE_IEM_028_VALUE': 'REL_GRIP',
    'MESURE_IEM_030_VALUE': 'VO2_SHUTTLE',
    'MESURE_IEM_031_VALUE': 'TREADMILL_REST_HR',
    'MESURE_IEM_032_VALUE': 'TREADMILL_3MIN_HR',
    'MESURE_IEM_033_VALUE': 'TREADMILL_6MIN_HR',
    'MESURE_IEM_034_VALUE': 'TREADMILL_9MIN_HR',
    'MESURE_IEM_035_VALUE': 'VO2_TREADMILL',
    'MESURE_IEM_036_VALUE': 'RECOVERY_HR',
    'MESURE_IEM_037_VALUE': 'VO2_STEP',
    'MESURE_IEM_041_VALUE': 'ADULT_FLIGHT',
    'MESURE_IEM_004_VALUE': 'WAIST',
}


def get_feature_cols(df):
    """Return ordered list of feature column codes present in df."""
    order = ['MESURE_IEM_001_VALUE', 'MESURE_IEM_002_VALUE',
             'MESURE_IEM_007_VALUE', 'MESURE_IEM_008_VALUE',
             'MESURE_IEM_012_VALUE', 'MESURE_IEM_018_VALUE',
             'MESURE_IEM_019_VALUE', 'MESURE_IEM_020_VALUE',
             'MESURE_IEM_021_VALUE', 'MESURE_IEM_022_VALUE',
             'MESURE_IEM_028_VALUE', 'MESURE_IEM_030_VALUE',
             'MESURE_IEM_031_VALUE', 'MESURE_IEM_032_VALUE',
             'MESURE_IEM_033_VALUE', 'MESURE_IEM_034_VALUE',
             'MESURE_IEM_035_VALUE', 'MESURE_IEM_036_VALUE',
             'MESURE_IEM_037_VALUE', 'MESURE_IEM_041_VALUE']
    # Add waist if Model B
    if 'MESURE_IEM_004_VALUE' in df.columns:
        order.append('MESURE_IEM_004_VALUE')
    return [c for c in order if c in df.columns]


def fit_predict(model_name):
    print(f"\n========== {model_name} retrain ==========")
    dev = pd.read_csv(os.path.join(BASE, model_name, "development_set.csv"))
    rs  = pd.read_csv(os.path.join(BASE, model_name, "result_set.csv"))
    full = pd.read_csv(os.path.join(BASE, model_name, "full_dataset.csv"))
    print(f"  development rows: {len(dev):,}, result_set rows: {len(rs):,}, "
          f"full rows: {len(full):,}")

    feat_cols = get_feature_cols(dev)
    feat_names = [FEATURE_MAP[c] for c in feat_cols]
    print(f"  features ({len(feat_cols)}): {feat_names}")

    # Merge result_set with full_dataset to retrieve input features
    join_keys = ['MESURE_AGE_CO', 'YEAR', 'SEXDSTN_FLAG_CD',
                 'MESURE_PLACE_FLAG_NM',
                 'MESURE_IEM_003_VALUE', 'MESURE_IEM_005_VALUE',
                 'MESURE_IEM_006_VALUE', 'MESURE_IEM_040_VALUE']
    full_dedup = full.drop_duplicates(subset=join_keys, keep='first')
    rs_full = rs.merge(full_dedup[join_keys + feat_cols], on=join_keys,
                        how='left')
    print(f"  rs_full rows: {len(rs_full):,}")
    # Drop rows where merge failed (no input features) — should be < 0.1%
    miss = rs_full[feat_cols].isna().all(axis=1).sum()
    print(f"  Rows without any input features: {miss:,}")
    rs_full = rs_full.dropna(subset=feat_cols, how='all')

    X_train = dev[feat_cols].copy()
    y_train = dev['MESURE_AGE_CO'].astype(float)

    X_test  = rs_full[feat_cols].copy()
    y_test  = rs_full['MESURE_AGE_CO'].astype(float)

    # Rename columns for SHAP readability
    X_train.columns = feat_names
    X_test.columns  = feat_names

    # Train LightGBM
    print("  Training LightGBM ...")
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': -1,
        'min_child_samples': 20,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.9,
        'bagging_freq': 5,
        'random_state': 42,
        'verbose': -1,
    }
    model = lgb.LGBMRegressor(n_estimators=1000, **params)
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              callbacks=[lgb.early_stopping(30, verbose=False),
                         lgb.log_evaluation(0)])
    print(f"  Best iteration: {model.best_iteration_}")

    # Predict
    y_pred = model.predict(X_test)
    mae    = float(np.mean(np.abs(y_pred - y_test)))
    rmse   = float(np.sqrt(np.mean((y_pred - y_test)**2)))
    r2     = float(1 - np.sum((y_pred - y_test)**2)
                    / np.sum((y_test - y_test.mean())**2))
    print(f"  Retrained LGBM test MAE = {mae:.4f}, RMSE = {rmse:.4f}, R² = {r2:.4f}")

    # Compare with original PRED_LGBM
    orig_pred = rs_full['PRED_LGBM'].astype(float).values
    diff = y_pred - orig_pred
    mae_vs_orig = float(np.mean(np.abs(diff)))
    rmse_vs_orig = float(np.sqrt(np.mean(diff**2)))
    pearson = float(np.corrcoef(y_pred, orig_pred)[0, 1])
    print(f"  Agreement with original PRED_LGBM: "
          f"MAE_diff = {mae_vs_orig:.4f}, RMSE_diff = {rmse_vs_orig:.4f}, "
          f"r = {pearson:.5f}")

    sanity = {
        "retrain_test_MAE": mae,
        "retrain_test_RMSE": rmse,
        "retrain_test_R2": r2,
        "vs_orig_MAE_diff": mae_vs_orig,
        "vs_orig_RMSE_diff": rmse_vs_orig,
        "pearson_with_orig": pearson,
        "best_iteration": int(model.best_iteration_),
        "n_features": len(feat_cols),
    }
    with open(os.path.join(OUT, f"retrain_validation_{model_name.replace(' ','_')}.json"),
              'w', encoding='utf-8') as f:
        json.dump(sanity, f, indent=2, ensure_ascii=False)
    return model, X_test, y_test, feat_names, rs_full


def compute_shap(model, X_test, model_name, feat_names):
    print(f"\n  Computing SHAP TreeExplainer for {model_name} ...")
    explainer = shap.TreeExplainer(model)
    # SHAP values: shape (n, n_features)
    print(f"  Running shap_values on {len(X_test):,} rows ...")
    shap_values = explainer.shap_values(X_test)
    expected_value = float(explainer.expected_value)
    print(f"  SHAP done. Mean |SHAP| top 5:")
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(-mean_abs)
    for i in order[:5]:
        print(f"    {feat_names[i]:20s}  {mean_abs[i]:.3f}")

    # Save per-instance SHAP
    out_path = os.path.join(OUT,
                f"shap_per_instance_{model_name.replace(' ','_')}.npz")
    np.savez_compressed(out_path,
                        shap_values=shap_values.astype(np.float32),
                        X_test=X_test.values.astype(np.float32),
                        feat_names=np.array(feat_names),
                        expected_value=expected_value)
    print(f"  Saved: {out_path}")

    return shap_values, mean_abs, order


def dependence_data(shap_values, X_test, feat_names, feature,
                    color_feature=None, n_bins=25):
    """Build binned dependence data for a feature."""
    i = feat_names.index(feature)
    x = X_test[feature].values
    s = shap_values[:, i]
    mask = ~(np.isnan(x) | np.isnan(s))
    x, s = x[mask], s[mask]

    # Bin by feature percentile
    bins = np.linspace(np.percentile(x, 1), np.percentile(x, 99), n_bins+1)
    bin_idx = np.digitize(x, bins) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins-1)

    out = {"feature": feature, "n": int(len(x)),
           "x_p1": float(np.percentile(x,1)),
           "x_p99": float(np.percentile(x,99)),
           "bins": []}
    for b in range(n_bins):
        m = bin_idx == b
        if m.sum() < 10:
            continue
        out['bins'].append({
            "x_center": float((bins[b] + bins[b+1]) / 2),
            "x_lo": float(bins[b]),
            "x_hi": float(bins[b+1]),
            "n": int(m.sum()),
            "shap_mean": float(s[m].mean()),
            "shap_sd": float(s[m].std()),
            "shap_p25": float(np.percentile(s[m], 25)),
            "shap_p75": float(np.percentile(s[m], 75)),
        })
    if color_feature is not None and color_feature in feat_names:
        cj = feat_names.index(color_feature)
        c = X_test[color_feature].values[mask]
        out['color_feature'] = color_feature
    return out


def interaction_value(model, X_test, feat_names, f1, f2, sample_n=2000):
    """Compute mean absolute SHAP interaction value between two features on a sample."""
    if f1 not in feat_names or f2 not in feat_names:
        return None
    explainer = shap.TreeExplainer(model)
    # Sample for speed (shap interaction is O(n_features^2))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_test), size=min(sample_n, len(X_test)), replace=False)
    Xs = X_test.iloc[idx]
    iv = explainer.shap_interaction_values(Xs)
    i1 = feat_names.index(f1)
    i2 = feat_names.index(f2)
    # Off-diagonal pair (i1, i2)
    pair = iv[:, i1, i2]
    return {
        "f1": f1, "f2": f2,
        "sample_n": int(len(idx)),
        "mean_abs_interaction": float(np.mean(np.abs(pair))),
        "mean_interaction": float(np.mean(pair)),
        "sd_interaction": float(np.std(pair)),
    }


def plot_dependence(shap_values, X_test, feat_names, feature, color_feature,
                    out_path, title):
    i = feat_names.index(feature)
    x = X_test[feature].values
    s = shap_values[:, i]
    if color_feature in feat_names:
        cj = feat_names.index(color_feature)
        c = X_test[color_feature].values
    else:
        c = None

    fig, ax = plt.subplots(figsize=(7, 5))
    mask = ~(np.isnan(x) | np.isnan(s))
    if c is not None:
        mask = mask & ~np.isnan(c)
        sc = ax.scatter(x[mask], s[mask], c=c[mask], cmap='RdBu_r',
                         alpha=0.25, s=4, edgecolors='none')
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label(color_feature, fontsize=10)
    else:
        ax.scatter(x[mask], s[mask], alpha=0.25, s=4, edgecolors='none',
                   color='steelblue')
    ax.axhline(0, color='gray', linestyle='--', lw=0.7)
    ax.set_xlabel(feature)
    ax.set_ylabel(f"SHAP value for {feature} (years)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def process_model(model_name):
    model, X_test, y_test, feat_names, rs = fit_predict(model_name)
    shap_values, mean_abs, order = compute_shap(model, X_test, model_name,
                                                  feat_names)

    # Top features for dependence
    top5 = [feat_names[i] for i in order[:5]]
    print(f"\n  Top 5 by mean |SHAP|: {top5}")

    # Build dependence data for top 5 features
    dep_data = {}
    for f in top5:
        dep_data[f] = dependence_data(shap_values, X_test, feat_names, f)
    with open(os.path.join(OUT,
              f"shap_dependence_data_{model_name.replace(' ','_')}.json"),
              'w', encoding='utf-8') as fp:
        json.dump(dep_data, fp, indent=2, ensure_ascii=False)

    # Interaction values: top pairs
    pairs = [(top5[0], top5[1]), (top5[0], top5[3]),
             (top5[3], top5[4] if len(top5) > 4 else top5[2])]
    if 'BMI' in feat_names and 'WAIST' in feat_names:
        pairs.append(('BMI', 'WAIST'))
    if 'GRIP_LEFT' in feat_names and 'WEIGHT' in feat_names:
        pairs.append(('GRIP_LEFT', 'WEIGHT'))
    interactions = []
    for f1, f2 in pairs:
        iv = interaction_value(model, X_test, feat_names, f1, f2)
        if iv:
            interactions.append(iv)
            print(f"    {f1}×{f2}: mean|SHAP_int| = {iv['mean_abs_interaction']:.4f}")
    with open(os.path.join(OUT,
              f"shap_interaction_{model_name.replace(' ','_')}.json"),
              'w', encoding='utf-8') as fp:
        json.dump(interactions, fp, indent=2, ensure_ascii=False)

    # Plot dependence for top 3 features (color by chronological age)
    X_aug = X_test.copy()
    X_aug['CA'] = y_test.values
    feat_aug = feat_names + ['CA']
    shap_aug = np.column_stack([shap_values, np.zeros(len(shap_values))])
    suffix = model_name.replace(' ', '_')
    for i, f in enumerate(top5[:3]):
        out_png = os.path.join(OUT, f"shap_dep_{f}_{suffix}.png")
        # Color by CA (age)
        plot_dependence(shap_aug, X_aug, feat_aug, f, 'CA',
                        out_png, f"{model_name}: SHAP dependence — {f}")
        print(f"    Saved: {out_png}")

    # Extra: BMI × WAIST plot for Model B
    if 'WAIST' in feat_names:
        out_png = os.path.join(OUT, f"shap_dep_BMI_color_WAIST_{suffix}.png")
        plot_dependence(shap_values, X_test, feat_names, 'BMI', 'WAIST',
                        out_png, f"{model_name}: SHAP dependence — BMI (colored by waist)")
        print(f"    Saved: {out_png}")


if __name__ == "__main__":
    for m in ["Model A", "Model B"]:
        process_model(m)
    print("\nDone.")
