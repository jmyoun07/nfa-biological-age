"""Efficient Model B SHAP analysis — computes the full SHAP interaction
tensor ONCE for a 2000-row sample, then extracts all desired pairs.

This replaces the inefficient retrain_and_shap.py path for Model B (which
called shap_interaction_values 5 times). Targets a total runtime of ~25 min
(retrain ~3 min, per-instance SHAP ~3 min, full interaction tensor ~15-20 min).

Outputs (written to NFA_OUT_DIR):
- shap_interaction_Model_B.json
- retrain_validation_Model_B.json
- shap_dep_<feature>_Model_B.png  (3 top features + BMI colored by WAIST)

This script does not write shap_per_instance_Model_B.npz. 05_figures.py reads that
archive, and 02a produces it for both cohorts.
"""


# NOTE: machine-specific paths in the original working copy have been replaced
# by the shared resolver in paths.py so the script runs from a checkout.
from paths import DATA_DIR, OUT_DIR, FIG_DIR  # noqa: E402
import os, json, sys
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Force unbuffered output so progress is visible
sys.stdout.reconfigure(line_buffering=True)

BASE = DATA_DIR
OUT = OUT_DIR

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
    if 'MESURE_IEM_004_VALUE' in df.columns:
        order.append('MESURE_IEM_004_VALUE')
    return [c for c in order if c in df.columns]


def retrain_model_b():
    print("[Step 1/5] Loading Model B data...", flush=True)
    dev = pd.read_csv(os.path.join(BASE, "Model B", "development_set.csv"))
    rs  = pd.read_csv(os.path.join(BASE, "Model B", "result_set.csv"))
    full = pd.read_csv(os.path.join(BASE, "Model B", "full_dataset.csv"))

    feat_cols = get_feature_cols(dev)
    feat_names = [FEATURE_MAP[c] for c in feat_cols]
    print(f"  Features ({len(feat_cols)}): {feat_names}", flush=True)

    # Merge result_set with full_dataset for input features
    join_keys = ['MESURE_AGE_CO', 'YEAR', 'SEXDSTN_FLAG_CD',
                 'MESURE_PLACE_FLAG_NM',
                 'MESURE_IEM_003_VALUE', 'MESURE_IEM_005_VALUE',
                 'MESURE_IEM_006_VALUE', 'MESURE_IEM_040_VALUE']
    full_dedup = full.drop_duplicates(subset=join_keys, keep='first')
    rs_full = rs.merge(full_dedup[join_keys + feat_cols], on=join_keys,
                        how='left').dropna(subset=feat_cols, how='all')

    X_train = dev[feat_cols].copy()
    y_train = dev['MESURE_AGE_CO'].astype(float)
    X_test  = rs_full[feat_cols].copy()
    y_test  = rs_full['MESURE_AGE_CO'].astype(float)
    X_train.columns = feat_names
    X_test.columns  = feat_names

    print(f"  Train: {len(X_train):,}, Test: {len(X_test):,}", flush=True)

    # Train LightGBM — avoid test-set leakage by NOT using test as eval_set
    # Use a small internal validation slice instead
    print("[Step 2/5] Training LightGBM (no leakage)...", flush=True)
    rng = np.random.default_rng(42)
    val_idx = rng.choice(len(X_train), size=int(0.1 * len(X_train)),
                          replace=False)
    train_mask = np.ones(len(X_train), dtype=bool)
    train_mask[val_idx] = False
    X_tr = X_train.iloc[train_mask]
    y_tr = y_train.iloc[train_mask]
    X_va = X_train.iloc[val_idx]
    y_va = y_train.iloc[val_idx]

    params = {
        'objective': 'regression', 'metric': 'mae',
        'learning_rate': 0.05, 'num_leaves': 63, 'max_depth': -1,
        'min_child_samples': 20, 'feature_fraction': 0.9,
        'bagging_fraction': 0.9, 'bagging_freq': 5,
        'random_state': 42, 'verbose': -1,
    }
    model = lgb.LGBMRegressor(n_estimators=1000, **params)
    model.fit(X_tr, y_tr,
              eval_set=[(X_va, y_va)],
              callbacks=[lgb.early_stopping(30, verbose=False),
                         lgb.log_evaluation(0)])
    print(f"  Best iteration: {model.best_iteration_}", flush=True)

    y_pred = model.predict(X_test)
    mae    = float(np.mean(np.abs(y_pred - y_test)))
    rmse   = float(np.sqrt(np.mean((y_pred - y_test)**2)))
    r2     = float(1 - np.sum((y_pred - y_test)**2)
                    / np.sum((y_test - y_test.mean())**2))
    orig_pred = rs_full['PRED_LGBM'].astype(float).values
    pearson = float(np.corrcoef(y_pred, orig_pred)[0, 1])
    print(f"  Test MAE: {mae:.4f}, R²: {r2:.4f}, Pearson with orig: {pearson:.5f}",
          flush=True)

    return model, X_test, y_test, feat_names, mae, rmse, r2, pearson


def compute_shap_full(model, X_test, feat_names):
    print("[Step 3/5] Computing per-instance SHAP values...", flush=True)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(-mean_abs)
    print("  Top 5 by mean |SHAP|:", flush=True)
    for i in order[:5]:
        print(f"    {feat_names[i]:20s}  {mean_abs[i]:.3f}", flush=True)
    return shap_values, mean_abs, order, explainer


def compute_interactions_once(explainer, X_test, feat_names,
                                sample_n=2000, pair_list=None):
    """Compute the full SHAP interaction tensor ONCE on a random sample,
    then extract all requested pairs from it."""
    print(f"[Step 4/5] Computing FULL SHAP interaction tensor "
          f"({sample_n} samples × {len(feat_names)}^2 pairs)...", flush=True)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_test), size=min(sample_n, len(X_test)),
                      replace=False)
    Xs = X_test.iloc[idx]
    iv = explainer.shap_interaction_values(Xs)  # (n_samples, n_features, n_features)
    print(f"  Tensor shape: {iv.shape}", flush=True)

    interactions = []
    for f1, f2 in pair_list:
        if f1 not in feat_names or f2 not in feat_names:
            continue
        i1 = feat_names.index(f1)
        i2 = feat_names.index(f2)
        # SHAP interaction is symmetric; off-diagonal cell is the pair value
        pair = iv[:, i1, i2]
        rec = {
            "f1": f1, "f2": f2,
            "sample_n": int(len(idx)),
            "mean_abs_interaction": float(np.mean(np.abs(pair))),
            "mean_interaction": float(np.mean(pair)),
            "sd_interaction": float(np.std(pair)),
        }
        interactions.append(rec)
        print(f"    {f1}×{f2}: mean|SHAP_int| = {rec['mean_abs_interaction']:.4f}",
              flush=True)
    return interactions, iv, idx


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


def main():
    model, X_test, y_test, feat_names, mae, rmse, r2, pearson = retrain_model_b()
    shap_values, mean_abs, order, explainer = compute_shap_full(model, X_test, feat_names)

    # Top features for dependence + interactions
    top5 = [feat_names[i] for i in order[:5]]
    print(f"\n  Top 5: {top5}", flush=True)

    pair_list = [
        (top5[0], top5[1]),
        (top5[0], top5[3]),
        (top5[3], top5[4] if len(top5) > 4 else top5[2]),
    ]
    if 'BMI' in feat_names and 'WAIST' in feat_names:
        pair_list.append(('BMI', 'WAIST'))
    if 'GRIP_LEFT' in feat_names and 'WEIGHT' in feat_names:
        pair_list.append(('GRIP_LEFT', 'WEIGHT'))

    interactions, iv, idx = compute_interactions_once(
        explainer, X_test, feat_names, sample_n=2000, pair_list=pair_list)

    # Save interactions
    with open(os.path.join(OUT, "shap_interaction_Model_B.json"), 'w',
              encoding='utf-8') as f:
        json.dump(interactions, f, indent=2, ensure_ascii=False)

    # Update retrain validation with clean (non-leaky) metrics
    sanity = {
        "retrain_test_MAE": mae,
        "retrain_test_RMSE": rmse,
        "retrain_test_R2": r2,
        "pearson_with_orig": pearson,
        "best_iteration": int(model.best_iteration_),
        "n_features": len(feat_names),
        "note": "Retrained with internal 10% validation split (no test-set leakage)"
    }
    with open(os.path.join(OUT, "retrain_validation_Model_B.json"), 'w',
              encoding='utf-8') as f:
        json.dump(sanity, f, indent=2, ensure_ascii=False)

    print("[Step 5/5] Generating dependence plots...", flush=True)
    # Augment data with CA column for color
    X_aug = X_test.copy()
    X_aug['CA'] = y_test.values
    feat_aug = feat_names + ['CA']
    shap_aug = np.column_stack([shap_values, np.zeros(len(shap_values))])

    for i, f in enumerate(top5[:3]):
        out_png = os.path.join(OUT, f"shap_dep_{f}_Model_B.png")
        plot_dependence(shap_aug, X_aug, feat_aug, f, 'CA', out_png,
                        f"Model B: SHAP dependence — {f}")
        print(f"  Saved: {out_png}", flush=True)

    if 'WAIST' in feat_names and 'BMI' in feat_names:
        out_png = os.path.join(OUT, "shap_dep_BMI_color_WAIST_Model_B.png")
        plot_dependence(shap_values, X_test, feat_names, 'BMI', 'WAIST',
                        out_png, "Model B: SHAP dependence — BMI (colored by waist)")
        print(f"  Saved: {out_png}", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
