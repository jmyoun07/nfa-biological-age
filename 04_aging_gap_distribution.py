"""Compute Distribution and Demographic Patterns of the Aging Gap.

Outputs (written to NFA_OUT_DIR):
- distribution_stats.json — all numerical results
"""

# NOTE: machine-specific paths in the original working copy have been replaced
# by the shared resolver in paths.py so the script runs from a checkout.
from paths import DATA_DIR, OUT_DIR, FIG_DIR  # noqa: E402
import os, json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

BASE = DATA_DIR
OUT = OUT_DIR
os.makedirs(OUT, exist_ok=True)


def load_test_set(model_name):
    """Load result_set (held-out test predictions) and merge with full_dataset
    on multi-key composite to obtain BMI (and waist for Model B)."""
    rs = pd.read_csv(os.path.join(BASE, model_name, "result_set.csv"))
    full = pd.read_csv(os.path.join(BASE, model_name, "full_dataset.csv"))
    print(f"  result_set rows: {len(rs):,}, full_dataset rows: {len(full):,}")

    keys = ['MESURE_AGE_CO', 'YEAR', 'SEXDSTN_FLAG_CD',
            'MESURE_PLACE_FLAG_NM',
            'MESURE_IEM_003_VALUE', 'MESURE_IEM_005_VALUE',
            'MESURE_IEM_006_VALUE', 'MESURE_IEM_040_VALUE']
    extra_cols = ['MESURE_IEM_018_VALUE']  # BMI
    if 'MESURE_IEM_004_VALUE' in full.columns:
        extra_cols.append('MESURE_IEM_004_VALUE')  # waist

    # Drop duplicates in full_dataset on the composite key
    full_dedup = full.drop_duplicates(subset=keys, keep='first')
    print(f"  full_dataset dedup rows: {len(full_dedup):,}")

    # Inner join
    merged = rs.merge(full_dedup[keys + extra_cols], on=keys,
                       how='left')
    # Rename for clarity
    merged = merged.rename(columns={
        'MESURE_IEM_018_VALUE': 'BMI',
        'MESURE_IEM_004_VALUE': 'WAIST',
    })
    n_with_bmi = merged['BMI'].notna().sum()
    print(f"  Test rows with BMI matched: {n_with_bmi:,} / {len(rs):,} "
          f"({100*n_with_bmi/len(rs):.2f}%)")
    return merged


def korean_bmi_category(bmi):
    if pd.isna(bmi):
        return np.nan
    if bmi < 18.5:    return 'Underweight'
    if bmi < 23.0:    return 'Normal'
    if bmi < 25.0:    return 'Overweight'
    return 'Obese'


def shapiro_subsample(x, n=5000, seed=42):
    """Shapiro-Wilk is unreliable for huge samples; subsample to n."""
    rng = np.random.default_rng(seed)
    x = x.dropna().values
    if len(x) > n:
        x = rng.choice(x, n, replace=False)
    W, p = stats.shapiro(x)
    return W, p, len(x)


def analyze(model_name):
    print(f"\n========== {model_name} ==========")
    df = load_test_set(model_name)
    print(f"  Loaded {len(df):,} test rows")

    out = {"model": model_name, "n_test": int(len(df))}

    # =========================================================
    # A. ΔAge distribution
    # =========================================================
    delta = df['DELTA_CORR'].dropna()
    out['delta_corr'] = {
        "n": int(len(delta)),
        "mean": float(delta.mean()),
        "sd": float(delta.std()),
        "median": float(delta.median()),
        "min": float(delta.min()),
        "max": float(delta.max()),
        "p2_5": float(delta.quantile(0.025)),
        "p97_5": float(delta.quantile(0.975)),
        "positive_n": int((delta > 0).sum()),
        "positive_pct": float((delta > 0).mean() * 100),
    }
    W, p, n_sub = shapiro_subsample(delta)
    out['delta_corr']['shapiro_W'] = float(W)
    out['delta_corr']['shapiro_p'] = float(p)
    out['delta_corr']['shapiro_n'] = int(n_sub)

    # =========================================================
    # B. BMI group ANOVA + Tukey HSD
    # =========================================================
    df['BMI_GROUP'] = df['BMI'].apply(korean_bmi_category)
    bmi_df = df[['DELTA_CORR', 'BMI_GROUP']].dropna()
    bmi_groups = ['Underweight', 'Normal', 'Overweight', 'Obese']
    bmi_arrays = [bmi_df.loc[bmi_df['BMI_GROUP']==g, 'DELTA_CORR'].values
                  for g in bmi_groups]
    F_bmi, p_bmi = stats.f_oneway(*bmi_arrays)
    # eta-squared
    ss_between = sum(len(a) * (a.mean() - bmi_df['DELTA_CORR'].mean())**2
                     for a in bmi_arrays)
    ss_total = ((bmi_df['DELTA_CORR'] - bmi_df['DELTA_CORR'].mean())**2).sum()
    eta_sq_bmi = ss_between / ss_total

    bmi_means = {g: {"n": int(len(a)),
                     "mean_delta": float(a.mean()),
                     "sd_delta": float(a.std())}
                 for g, a in zip(bmi_groups, bmi_arrays)}
    tukey_bmi = pairwise_tukeyhsd(bmi_df['DELTA_CORR'], bmi_df['BMI_GROUP'])
    tukey_bmi_rows = []
    for row in tukey_bmi._results_table.data[1:]:
        tukey_bmi_rows.append({
            "g1": str(row[0]), "g2": str(row[1]),
            "meandiff": float(row[2]), "p_adj": float(row[3]),
            "lower": float(row[4]), "upper": float(row[5]),
            "reject": bool(row[6])
        })

    out['bmi_anova'] = {
        "df_between": len(bmi_groups) - 1,
        "df_within": int(len(bmi_df) - len(bmi_groups)),
        "F": float(F_bmi),
        "p": float(p_bmi),
        "eta_squared": float(eta_sq_bmi),
        "groups": bmi_means,
        "tukey": tukey_bmi_rows
    }

    # =========================================================
    # C. Sex × Age decade two-way ANOVA
    # =========================================================
    # Build age decade
    bins = [18, 29, 39, 49, 59, 64]
    labels = ['19-29', '30-39', '40-49', '50-59', '60-64']
    df['DECADE'] = pd.cut(df['MESURE_AGE_CO'], bins=bins, labels=labels,
                          include_lowest=True)
    twoway_df = df[['DELTA_CORR', 'SEXDSTN_FLAG_CD', 'DECADE']].dropna()
    model = ols('DELTA_CORR ~ C(SEXDSTN_FLAG_CD) + C(DECADE) + '
                'C(SEXDSTN_FLAG_CD):C(DECADE)', data=twoway_df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    # Compute partial eta-squared
    ss_total_2way = ((twoway_df['DELTA_CORR'] - twoway_df['DELTA_CORR'].mean())**2).sum()
    twoway_results = {}
    for term in ['C(SEXDSTN_FLAG_CD)', 'C(DECADE)',
                 'C(SEXDSTN_FLAG_CD):C(DECADE)']:
        twoway_results[term] = {
            "sum_sq": float(anova_table.loc[term, 'sum_sq']),
            "df": float(anova_table.loc[term, 'df']),
            "F": float(anova_table.loc[term, 'F']),
            "p": float(anova_table.loc[term, 'PR(>F)']),
            "eta_squared": float(anova_table.loc[term, 'sum_sq'] / ss_total_2way)
        }

    # Cell means
    cells = twoway_df.groupby(['DECADE', 'SEXDSTN_FLAG_CD'],
                              observed=True)['DELTA_CORR'].agg(
                                  ['mean', 'std', 'count']).reset_index()
    cell_means = []
    for _, row in cells.iterrows():
        cell_means.append({
            "decade": str(row['DECADE']),
            "sex": str(row['SEXDSTN_FLAG_CD']),
            "mean_delta": float(row['mean']),
            "sd_delta": float(row['std']),
            "n": int(row['count'])
        })

    out['sex_age_twoway'] = {
        "n": int(len(twoway_df)),
        "effects": twoway_results,
        "cell_means": cell_means
    }

    # =========================================================
    # D. Year one-way ANOVA + Tukey
    # =========================================================
    year_df = df[['DELTA_CORR', 'YEAR']].dropna()
    years = sorted(year_df['YEAR'].unique())
    year_arrays = [year_df.loc[year_df['YEAR']==y, 'DELTA_CORR'].values
                   for y in years]
    F_y, p_y = stats.f_oneway(*year_arrays)
    ss_between_y = sum(len(a) * (a.mean() - year_df['DELTA_CORR'].mean())**2
                       for a in year_arrays)
    ss_total_y = ((year_df['DELTA_CORR'] - year_df['DELTA_CORR'].mean())**2).sum()
    eta_y = ss_between_y / ss_total_y

    year_means = {int(y): {"n": int(len(a)),
                            "mean_delta": float(a.mean()),
                            "sd_delta": float(a.std())}
                  for y, a in zip(years, year_arrays)}

    tukey_year = pairwise_tukeyhsd(year_df['DELTA_CORR'], year_df['YEAR'])
    tukey_year_rows = []
    for row in tukey_year._results_table.data[1:]:
        tukey_year_rows.append({
            "g1": str(row[0]), "g2": str(row[1]),
            "meandiff": float(row[2]), "p_adj": float(row[3]),
            "lower": float(row[4]), "upper": float(row[5]),
            "reject": bool(row[6])
        })

    out['year_anova'] = {
        "df_between": len(years) - 1,
        "df_within": int(len(year_df) - len(years)),
        "F": float(F_y),
        "p": float(p_y),
        "eta_squared": float(eta_y),
        "years": year_means,
        "tukey": tukey_year_rows
    }

    return out


if __name__ == "__main__":
    results = {}
    for model in ["Model A", "Model B"]:
        results[model] = analyze(model)

    with open(os.path.join(OUT, "distribution_stats.json"), "w",
              encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n\n========== SUMMARY (Model A) ==========")
    a = results['Model A']
    print(f"Test n: {a['n_test']:,}")
    d = a['delta_corr']
    print(f"\nΔAge_corr: mean={d['mean']:.3f}, SD={d['sd']:.3f}, "
          f"range={d['min']:.2f} to {d['max']:.2f}")
    print(f"  Shapiro-Wilk (n={d['shapiro_n']}): W={d['shapiro_W']:.4f}, "
          f"p={d['shapiro_p']:.2e}")
    print(f"  Positive ΔAge: {d['positive_n']:,} ({d['positive_pct']:.2f}%)")

    b = a['bmi_anova']
    print(f"\nBMI ANOVA: F({b['df_between']},{b['df_within']:,}) = "
          f"{b['F']:.3f}, p={b['p']:.2e}, η²={b['eta_squared']:.4f}")
    for g, v in b['groups'].items():
        print(f"  {g:12s}: n={v['n']:>7,}, mean ΔAge={v['mean_delta']:+.3f} "
              f"(SD={v['sd_delta']:.3f})")

    t = a['sex_age_twoway']['effects']
    print(f"\nSex × Age two-way ANOVA:")
    for k, v in t.items():
        print(f"  {k:38s}: F({v['df']:.0f}) = {v['F']:.3f}, "
              f"p={v['p']:.2e}, η²={v['eta_squared']:.4f}")

    y = a['year_anova']
    print(f"\nYear ANOVA: F({y['df_between']},{y['df_within']:,}) = "
          f"{y['F']:.3f}, p={y['p']:.2e}, η²={y['eta_squared']:.4f}")
    for yr, v in y['years'].items():
        print(f"  {yr}: n={v['n']:>7,}, mean ΔAge={v['mean_delta']:+.3f} "
              f"(SD={v['sd_delta']:.3f})")

    print("\n\n========== SUMMARY (Model B) ==========")
    b = results['Model B']
    print(f"Test n: {b['n_test']:,}")
    d = b['delta_corr']
    print(f"\nΔAge_corr: mean={d['mean']:.3f}, SD={d['sd']:.3f}")
    bb = b['bmi_anova']
    print(f"BMI ANOVA: F = {bb['F']:.3f}, p={bb['p']:.2e}, η²={bb['eta_squared']:.4f}")
    for g, v in bb['groups'].items():
        print(f"  {g:12s}: n={v['n']:>7,}, mean ΔAge={v['mean_delta']:+.3f}")

    print("\nSaved: distribution_stats.json")
