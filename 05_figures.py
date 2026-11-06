# NOTE: machine-specific paths in the original working copy have been replaced
# by the shared resolver in paths.py so the script runs from a checkout.
from paths import DATA_DIR, OUT_DIR, FIG_DIR  # noqa: E402

# -*- coding: utf-8 -*-
"""Generate per-cohort (Cohort A / Cohort B) figures into NFA_FIG_DIR.
Splittable figures rendered separately for A and B; the participant-flow diagram
is a hand-authored diagram and is not produced here.
"""
import os, json, shutil
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

OUT = FIG_DIR
os.makedirs(OUT,exist_ok=True)
FIN = DATA_DIR
MO = OUT_DIR
plt.rcParams.update({"font.size":11,"axes.titlesize":12,"figure.dpi":200})

LABELS={"HEIGHT":"Height","WEIGHT":"Body weight","GRIP_LEFT":"Grip strength (L)",
"GRIP_RIGHT":"Grip strength (R)","FLEX":"Sit-and-reach","BMI":"BMI","SITUP":"Cross sit-up",
"SHUTTLE":"20-m shuttle run","RUN10":"10-m x4 shuttle","JUMP":"Standing long jump",
"REL_GRIP":"Relative grip","VO2_SHUTTLE":"VO2max (shuttle)","TREADMILL_REST_HR":"Treadmill rest HR",
"TREADMILL_3MIN_HR":"Treadmill 3-min HR","TREADMILL_6MIN_HR":"Treadmill 6-min HR",
"TREADMILL_9MIN_HR":"Treadmill 9-min HR","VO2_TREADMILL":"VO2max (treadmill)",
"RECOVERY_HR":"Step recovery HR","VO2_STEP":"VO2max (step)","ADULT_FLIGHT":"Flight time",
"WAIST":"Waist circumference"}
COH={"A":"Cohort A","B":"Cohort B"}

def load(coh):
    df=pd.read_csv(os.path.join(FIN,f"Model {coh}","result_set.csv"))
    df=df[df["RESULT_SPLIT"]=="test"].copy()
    z=np.load(os.path.join(MO,f"shap_per_instance_Model_{coh}.npz"),allow_pickle=True)
    return df, z

def ols(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float); n=len(x)
    sx,sy,sxx,sxy=x.sum(),y.sum(),(x*x).sum(),(x*y).sum()
    b=(n*sxy-sx*sy)/(n*sxx-sx*sx); a=(sy-b*sx)/n; return b,a

# ---------- Fig 2: Calibration (per cohort) ----------
def fig_calibration(coh,df):
    ca=df["MESURE_AGE_CO"].astype(float).values; pred=df["BA_PRED"].astype(float).values
    citl=pred.mean()-ca.mean(); slope,intc=ols(pred,ca)
    q=pd.qcut(pred,10,labels=False,duplicates="drop")
    mx=[pred[q==k].mean() for k in range(10)]; my=[ca[q==k].mean() for k in range(10)]
    fig,ax=plt.subplots(figsize=(5,5))
    lims=[17,66]; ax.plot(lims,lims,"--",color="0.4",lw=1.2,label="Ideal (y = x)")
    idx=np.linspace(0,len(ca)-1,min(4000,len(ca))).astype(int)
    ax.scatter(pred[idx],ca[idx],s=4,alpha=0.07,color="#4C72B0",rasterized=True)
    ax.plot(mx,my,"-o",color="#C44E52",lw=1.6,ms=6,label="Observed mean (deciles)")
    xs=np.array(lims); ax.plot(xs,slope*xs+intc,color="#55A868",lw=1.6,label="Fitted")
    ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect("equal","box")
    ax.set_xlabel("Predicted biological age (years)"); ax.set_ylabel("Observed chronological age (years)")
    ax.set_title(f"Calibration — {COH[coh]}")
    ax.text(0.04,0.96,f"Slope = {slope:.3f}\nCITL = {citl:+.3f} yr",transform=ax.transAxes,
            va="top",fontsize=9,bbox=dict(boxstyle="round",fc="white",ec="0.7"))
    ax.legend(loc="lower right",fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,f"Fig2_Calibration_Cohort{coh}.png"),bbox_inches="tight"); plt.close(fig)

# ---------- Fig 3: Aging-gap distribution (per cohort) ----------
def fig_distribution(coh,df):
    d=df["DELTA_CORR"].astype(float).values
    fig,ax=plt.subplots(figsize=(5.2,4))
    ax.hist(d,bins=80,range=(-20,20),color="#4C72B0",alpha=0.85,edgecolor="white",linewidth=0.2)
    ax.axvline(0,color="0.3",lw=1,ls="--"); ax.axvline(d.mean(),color="#C44E52",lw=1.4,label=f"Mean = {d.mean():+.3f} yr")
    ax.set_xlabel("Corrected aging gap, ΔAge (years)"); ax.set_ylabel("Participants")
    ax.set_title(f"Aging-gap distribution — {COH[coh]} (n = {len(d):,})"); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,f"Fig3_AgingGapDistribution_Cohort{coh}.png"),bbox_inches="tight"); plt.close(fig)

# ---------- Fig 4: Aging gap by age decade x sex (per cohort) ----------
def fig_agesex(coh,df):
    bands=["19-29","30-39","40-49","50-59","60-64"]
    g=df.groupby(["AGE_BAND","SEXDSTN_FLAG_CD"])["DELTA_CORR"].mean().unstack()
    g=g.reindex(bands)
    fig,ax=plt.subplots(figsize=(6.2,4))
    x=np.arange(len(bands)); w=0.38
    ax.bar(x-w/2,g.get("M"),w,label="Male",color="#4C72B0")
    ax.bar(x+w/2,g.get("F"),w,label="Female",color="#DD8452")
    ax.axhline(0,color="0.3",lw=0.8); ax.set_xticks(x); ax.set_xticklabels(bands)
    ax.set_xlabel("Age decade (years)"); ax.set_ylabel("Mean corrected ΔAge (years)")
    ax.set_title(f"Aging gap by age and sex — {COH[coh]}"); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,f"Fig4_AgingGap_byAgeSex_Cohort{coh}.png"),bbox_inches="tight"); plt.close(fig)

# ---------- Fig 5: SHAP beeswarm (per cohort, manual) ----------
def fig_beeswarm(coh,z,topn=15):
    sv=z["shap_values"]; X=z["X_test"]; feats=[str(f) for f in z["feat_names"]]
    order=np.argsort(np.abs(sv).mean(0))[::-1][:topn]
    rng=np.random.RandomState(0); m=min(3000,sv.shape[0]); sub=rng.choice(sv.shape[0],m,replace=False)
    fig,ax=plt.subplots(figsize=(7,0.42*topn+1))
    for row,fi in enumerate(order):
        yv=topn-1-row
        s=sv[sub,fi]; xv=X[sub,fi].astype(float)
        lo,hi=np.nanpercentile(xv,5),np.nanpercentile(xv,95); c=np.clip((xv-lo)/(hi-lo+1e-9),0,1)
        jit=(rng.rand(m)-0.5)*0.6
        ax.scatter(s,yv+jit,c=c,cmap="coolwarm",s=5,alpha=0.5,linewidths=0,rasterized=True)
    ax.axvline(0,color="0.5",lw=0.8)
    ax.set_yticks(range(topn)); ax.set_yticklabels([LABELS.get(feats[i],feats[i]) for i in order[::-1]])
    ax.set_xlabel("SHAP value (impact on predicted BA, years)")
    ax.set_title(f"SHAP beeswarm (top {topn}) — {COH[coh]}")
    sm=cm.ScalarMappable(cmap="coolwarm"); sm.set_array([]); cb=fig.colorbar(sm,ax=ax,pad=0.01,fraction=0.025)
    cb.set_ticks([0,1]); cb.set_ticklabels(["Low","High"]); cb.set_label("Feature value")
    fig.tight_layout(); fig.savefig(os.path.join(OUT,f"Fig5_SHAP_Beeswarm_Cohort{coh}.png"),bbox_inches="tight"); plt.close(fig)

# ---------- Fig 6: SHAP dependence for step-test VO2max (per cohort) ----------
def fig_dependence(coh,z):
    feats=[str(f) for f in z["feat_names"]]; fi=feats.index("VO2_STEP")
    x=z["X_test"][:,fi].astype(float); y=z["shap_values"][:,fi].astype(float)
    lo,hi=np.nanpercentile(x,1),np.nanpercentile(x,99); mk=(x>=lo)&(x<=hi)
    x,y=x[mk],y[mk]
    fig,ax=plt.subplots(figsize=(5.6,4.2))
    rng=np.random.RandomState(0); sub=rng.choice(len(x),min(5000,len(x)),replace=False)
    ax.scatter(x[sub],y[sub],s=6,alpha=0.15,color="#4C72B0",rasterized=True)
    bins=np.linspace(lo,hi,25); bc=0.5*(bins[:-1]+bins[1:]); bm=[y[(x>=bins[k])&(x<bins[k+1])].mean() for k in range(len(bins)-1)]
    ax.plot(bc,bm,"-",color="#C44E52",lw=2,label="Binned mean")
    ax.axhline(0,color="0.5",lw=0.8)
    ax.set_xlabel("Step-test VO2max"); ax.set_ylabel("SHAP value (years)")
    ax.set_title(f"SHAP dependence: step-test VO2max — {COH[coh]}"); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,f"Fig6_SHAP_Dependence_VO2step_Cohort{coh}.png"),bbox_inches="tight"); plt.close(fig)

# Figure 1 (participant flow) is a hand-authored diagram and is not
# produced by this script; see the manuscript for the published version.

for coh in ["A","B"]:
    df,z=load(coh)
    fig_calibration(coh,df); fig_distribution(coh,df); fig_agesex(coh,df)
    fig_beeswarm(coh,z); fig_dependence(coh,z)
    print(f"Cohort {coh} done")
print("ALL FIGURES ->", OUT)
print(sorted(os.listdir(OUT)))
