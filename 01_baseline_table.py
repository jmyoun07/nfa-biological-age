"""Compute baseline characteristics from Model A and Model B full datasets."""

# NOTE: machine-specific paths in the original working copy have been replaced
# by the shared resolver in paths.py so the script runs from a checkout.
from paths import DATA_DIR, OUT_DIR, FIG_DIR  # noqa: E402
import pandas as pd
import numpy as np
import os, json

BASE = DATA_DIR

def desc(model_name):
    p = os.path.join(BASE, model_name, "full_dataset.csv")
    df = pd.read_csv(p)
    n = len(df)
    out = {"model": model_name, "n": n}

    # Year breakdown
    year_counts = df['YEAR'].value_counts().sort_index().to_dict()
    out['year_counts'] = {int(k): int(v) for k, v in year_counts.items()}

    # Sex (assume M=1, F=2 or 'M'/'F'; check)
    sex_counts = df['SEXDSTN_FLAG_CD'].value_counts().to_dict()
    out['sex_counts'] = {str(k): int(v) for k, v in sex_counts.items()}

    # Age stats
    out['age_mean'] = float(df['MESURE_AGE_CO'].mean())
    out['age_sd']   = float(df['MESURE_AGE_CO'].std())
    out['age_min']  = float(df['MESURE_AGE_CO'].min())
    out['age_max']  = float(df['MESURE_AGE_CO'].max())

    # Age band counts (using AGE_BAND if available)
    if 'AGE_BAND' in df.columns:
        band_counts = df['AGE_BAND'].value_counts().sort_index().to_dict()
        out['age_band_counts'] = {str(k): int(v) for k, v in band_counts.items()}

    # Decade buckets (19-29, 30-39, 40-49, 50-59, 60-64)
    bins = [18, 29, 39, 49, 59, 64]
    labels = ['19-29','30-39','40-49','50-59','60-64']
    df['decade'] = pd.cut(df['MESURE_AGE_CO'], bins=bins, labels=labels, include_lowest=True)
    decade_counts = df['decade'].value_counts().sort_index().to_dict()
    out['decade_counts'] = {str(k): int(v) for k, v in decade_counts.items()}

    # Decade x sex
    cross = df.groupby(['decade', 'SEXDSTN_FLAG_CD'], observed=True).size().to_dict()
    out['decade_sex_counts'] = {f"{k[0]}|{k[1]}": int(v) for k, v in cross.items()}

    # Key fitness items: mean (SD)
    items = {
        'MESURE_IEM_001_VALUE': 'Height (cm)',
        'MESURE_IEM_002_VALUE': 'Weight (kg)',
        'MESURE_IEM_018_VALUE': 'BMI (kg/m^2)',
        'MESURE_IEM_007_VALUE': 'Grip strength left (kg)',
        'MESURE_IEM_008_VALUE': 'Grip strength right (kg)',
        'MESURE_IEM_012_VALUE': 'Sit-and-reach (cm)',
        'MESURE_IEM_019_VALUE': 'Cross sit-up (reps)',
        'MESURE_IEM_020_VALUE': '20-m shuttle run (count)',
        'MESURE_IEM_022_VALUE': 'Standing long jump (cm)',
        'MESURE_IEM_028_VALUE': 'Relative grip strength (%)',
        'MESURE_IEM_030_VALUE': 'Shuttle-run VO2max',
        'MESURE_IEM_035_VALUE': 'Treadmill VO2max',
        'MESURE_IEM_037_VALUE': 'Step-test VO2max',
        'MESURE_IEM_036_VALUE': 'Step recovery HR',
        'MESURE_IEM_003_VALUE': 'Body fat % (ext val)',
        'MESURE_IEM_006_VALUE': 'SBP (ext val)',
        'MESURE_IEM_005_VALUE': 'DBP (ext val)',
        'MESURE_IEM_040_VALUE': 'Reaction time (ext val)',
    }
    if 'MESURE_IEM_004_VALUE' in df.columns:
        items['MESURE_IEM_004_VALUE'] = 'Waist circumference (cm)'

    item_stats = {}
    for col, label in items.items():
        if col in df.columns:
            s = df[col].dropna()
            item_stats[label] = {
                'n': int(s.shape[0]),
                'mean': float(s.mean()),
                'sd': float(s.std()),
                'missing_pct': float(100 * (1 - s.shape[0]/n))
            }
    out['item_stats'] = item_stats

    # BMI categories (Korean criteria: <18.5, 18.5-22.9, 23.0-24.9, 25.0+)
    bmi = df['MESURE_IEM_018_VALUE'].dropna()
    underweight = int((bmi < 18.5).sum())
    normal = int(((bmi >= 18.5) & (bmi < 23)).sum())
    overweight = int(((bmi >= 23) & (bmi < 25)).sum())
    obese = int((bmi >= 25).sum())
    out['bmi_categories'] = {
        'underweight_<18.5': underweight,
        'normal_18.5-22.9': normal,
        'overweight_23-24.9': overweight,
        'obese_>=25': obese,
        'total_bmi_n': int(bmi.shape[0])
    }
    # Mean BMI by sex
    bmi_by_sex = df.groupby('SEXDSTN_FLAG_CD')['MESURE_IEM_018_VALUE'].agg(['mean','std','count']).to_dict('index')
    out['bmi_by_sex'] = {str(k): {kk: float(vv) if not pd.isna(vv) else None for kk,vv in v.items()} for k,v in bmi_by_sex.items()}

    return out

a = desc("Model A")
b = desc("Model B")

with open(os.path.join(OUT_DIR, "baseline_characteristics.json"), "w", encoding="utf-8") as f:
    json.dump({"Model_A": a, "Model_B": b}, f, indent=2, ensure_ascii=False)

# Print summary
print("=== Model A (2022-2026, 19-64) ===")
print(f"N = {a['n']:,}")
print(f"Sex: {a['sex_counts']}")
print(f"Age: mean={a['age_mean']:.2f} (SD={a['age_sd']:.2f}); range={a['age_min']:.0f}-{a['age_max']:.0f}")
print(f"Year: {a['year_counts']}")
print(f"Decade counts: {a['decade_counts']}")
print(f"BMI categories: {a['bmi_categories']}")
print(f"BMI by sex: {a['bmi_by_sex']}")
print()
print("=== Model B (2024-2026, 19-64) ===")
print(f"N = {b['n']:,}")
print(f"Sex: {b['sex_counts']}")
print(f"Age: mean={b['age_mean']:.2f} (SD={b['age_sd']:.2f}); range={b['age_min']:.0f}-{b['age_max']:.0f}")
print(f"Year: {b['year_counts']}")
print(f"Decade counts: {b['decade_counts']}")
print()
print("=== Saved to baseline_characteristics.json ===")
