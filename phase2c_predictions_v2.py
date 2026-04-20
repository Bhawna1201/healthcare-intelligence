"""
phase2c_predictions_v2.py — Enhanced Shortage & Rating Prediction
Stevens Institute of Technology · BIA-660 Web Mining

NEW FEATURES FOR SHORTAGE PREDICTION:
  - Dosage risk score (injection=3, solution=2, tablet=0)
  - Brand-only flag (single manufacturer by definition)
  - Therapeutic substitutability (can it be replaced if shortage?)
  - Number of manufacturers (from NADAC)
  - Price instability (price_cv)

Run: venv/bin/python phase2c_predictions_v2.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

DATA = Path("data/raw")
PROC = Path("data/processed")
OUT  = Path("data/master")
OUT.mkdir(exist_ok=True)

TARGET_DRUGS = [
    "metformin","lisinopril","atorvastatin","levothyroxine","amlodipine",
    "metoprolol","omeprazole","simvastatin","losartan","albuterol",
    "gabapentin","hydrochlorothiazide","sertraline","montelukast",
    "furosemide","pantoprazole","escitalopram","rosuvastatin",
    "bupropion","fluoxetine","clopidogrel","tramadol","cyclobenzaprine",
    "amoxicillin","azithromycin","doxycycline","prednisone",
    "methylprednisolone","clonazepam","alprazolam","zolpidem",
    "oxycodone","hydrocodone","acetaminophen","ibuprofen","naproxen",
    "insulin glargine","insulin lispro","empagliflozin","semaglutide",
    "dulaglutide","apixaban","rivaroxaban","warfarin","digoxin",
    "diltiazem","verapamil","carvedilol","spironolactone","tamsulosin",
]

# ── New engineered features ────────────────────────────────────────────────────

# Dosage form risk score — injectables shortage most
DOSAGE_RISK = {
    "insulin glargine":    3,   # injection
    "insulin lispro":      3,   # injection
    "methylprednisolone":  3,   # injection available
    "furosemide":          3,   # IV injection critical
    "albuterol":           2,   # inhalation solution
    "digoxin":             2,   # injection + oral
    "diltiazem":           2,   # injection + oral
    "semaglutide":         2,   # subcutaneous pen
    "dulaglutide":         2,   # subcutaneous pen
    "empagliflozin":       1,   # tablet (brand only)
    "apixaban":            1,   # tablet (brand only)
    "rivaroxaban":         1,   # tablet (brand only)
}

# Brand-only drugs — single manufacturer, highest shortage risk
BRAND_ONLY = [
    "empagliflozin","semaglutide","dulaglutide","apixaban","rivaroxaban",
]

# Therapeutic substitutability — can it be replaced if shortage?
# 1 = easily substituted, 0 = no good substitute
SUBSTITUTABILITY = {
    "warfarin":            1,   # → apixaban/rivaroxaban
    "simvastatin":         1,   # → atorvastatin/rosuvastatin
    "atorvastatin":        1,   # → rosuvastatin/simvastatin
    "rosuvastatin":        1,   # → atorvastatin
    "omeprazole":          1,   # → pantoprazole
    "pantoprazole":        1,   # → omeprazole
    "sertraline":          1,   # → escitalopram/fluoxetine
    "escitalopram":        1,   # → sertraline
    "fluoxetine":          1,   # → sertraline
    "alprazolam":          1,   # → clonazepam
    "clonazepam":          1,   # → alprazolam
    "metoprolol":          1,   # → carvedilol
    "carvedilol":          1,   # → metoprolol
    "amlodipine":          1,   # → diltiazem/verapamil
    "lisinopril":          1,   # → losartan
    "losartan":            1,   # → lisinopril
    "furosemide":          0,   # no IV equivalent in acute setting
    "albuterol":           0,   # no substitute for acute bronchospasm
    "insulin glargine":    0,   # specific basal insulin
    "insulin lispro":      0,   # specific rapid insulin
    "digoxin":             0,   # unique mechanism, no substitute
    "methylprednisolone":  0,   # IV steroid, limited alternatives
    "oxycodone":           0,   # controlled substance, restricted
    "hydrocodone":         0,   # controlled substance, restricted
}

MANUAL_PRICES = {
    "empagliflozin": 369.20,
    "semaglutide":   1109.00,
    "dulaglutide":   1087.00,
    "apixaban":      260.00,
}

def load(proc, raw):
    p = proc if (proc and Path(proc).exists()) else raw
    if p and Path(p).exists():
        df = pd.read_csv(p, on_bad_lines="skip", engine="python",
                         encoding_errors="replace")
        return df.replace({"None":np.nan,"none":np.nan,"nan":np.nan})
    return pd.DataFrame()

def dcol(df):
    return "drug_name_clean" if "drug_name_clean" in df.columns else "drug_name"

def fuzzy_to_target(name):
    if not isinstance(name,str): return None
    nl = name.lower().strip()
    for drug in TARGET_DRUGS:
        if drug in nl: return drug
    return None

print("="*65)
print("PHASE 2C v2 — Enhanced Shortage Prediction")
print("Stevens Institute of Technology · BIA-660 Web Mining")
print("="*65)

prices    = load(PROC/"prices_clean.csv",   DATA/"prices.csv")
shortages = load(None,                      DATA/"shortages.csv")
adverse   = load(None,                      DATA/"adverse_events.csv")
trials    = load(PROC/"trials_clean.csv",   DATA/"trials.csv")

# ── Build shortage labels ─────────────────────────────────────────────────────
shortages_c = shortages.copy()
shortages_c.columns = shortages_c.columns.str.strip().str.lower().str.replace(" ","_")
sdc = "drug_name" if "drug_name" in shortages_c.columns else shortages_c.columns[0]

def map_s(name):
    if not isinstance(name,str): return None
    nl=name.lower()
    for d in TARGET_DRUGS:
        if d in nl: return d
    return None

shortages_c["drug_canonical"] = shortages_c[sdc].apply(map_s)
shortages_c = shortages_c.dropna(subset=["drug_canonical"])
if "status" in shortages_c.columns:
    shortages_c["is_active"] = shortages_c["status"].str.lower().str.contains(
        "current|active|unavailable|limited",na=False).astype(int)
else:
    shortages_c["is_active"] = 1

shortage_labels = (shortages_c.groupby("drug_canonical")
                   .agg(shortage_total=("drug_canonical","count"),
                        shortage_active=("is_active","sum"))
                   .reset_index().rename(columns={"drug_canonical":"drug_name"}))
shortage_labels["has_shortage"] = (shortage_labels["shortage_total"]>0).astype(int)

# ── Build feature matrix ──────────────────────────────────────────────────────
drug_df = pd.DataFrame({"drug_name": TARGET_DRUGS})

# Price features
pdc = dcol(prices)
prices_c = prices.copy()
prices_c["drug_name"] = prices_c[pdc].apply(fuzzy_to_target)
prices_c = prices_c.dropna(subset=["drug_name"])
prices_c["price"] = pd.to_numeric(prices_c["price"],errors="coerce")
prices_c = prices_c[prices_c["price"].between(0.0001,9999)]
if "pharmacy" in prices_c.columns:
    prices_c["is_brand"] = prices_c["pharmacy"].str.contains("Brand",case=False,na=False)
else:
    prices_c["is_brand"] = False

price_feats = prices_c.groupby("drug_name").apply(lambda g: pd.Series({
    "price_30day":      g["price"].median()*30,
    "price_cv":         g["price"].std()/g["price"].mean() if g["price"].mean()>0 else 0,
    "n_manufacturers":  len(g),
    "generic_available":int((~g["is_brand"]).any()),
})).reset_index()

for drug,price in MANUAL_PRICES.items():
    idx = price_feats[price_feats["drug_name"]==drug].index
    if len(idx):
        price_feats.loc[idx,"price_30day"] = price

drug_df = drug_df.merge(price_feats, on="drug_name", how="left")

# Adverse features
adc = dcol(adverse)
adv = adverse.copy()
adv["drug_name"] = adv[adc].apply(fuzzy_to_target)
adv = adv.dropna(subset=["drug_name"])
adv_feats = (adv.groupby("drug_name")
             .agg(adverse_total=("drug_name","count"),
                  adverse_unique=("event_type","nunique")
                  if "event_type" in adv.columns else ("drug_name","count"))
             .reset_index())
adv_feats["adverse_log"] = np.log1p(adv_feats["adverse_total"])
drug_df = drug_df.merge(adv_feats, on="drug_name", how="left")

# Trial features
tdc = dcol(trials)
tri = trials.copy()
tri["drug_name"] = tri[tdc].apply(fuzzy_to_target)
tri = tri.dropna(subset=["drug_name"])
if "status" in tri.columns:
    tri["is_completed"] = tri["status"].str.contains("Completed",na=False).astype(int)
else:
    tri["is_completed"] = 0
tri_feats = tri.groupby("drug_name").agg(
    trial_total=("drug_name","count"),
    trial_completed=("is_completed","sum")).reset_index()
tri_feats["trial_completion_rate"] = tri_feats["trial_completed"]/tri_feats["trial_total"]
drug_df = drug_df.merge(tri_feats, on="drug_name", how="left")

# ── NEW ENGINEERED FEATURES ───────────────────────────────────────────────────
drug_df["dosage_risk_score"]   = drug_df["drug_name"].map(DOSAGE_RISK).fillna(0)
drug_df["is_brand_only"]       = drug_df["drug_name"].isin(BRAND_ONLY).astype(int)
drug_df["has_substitute"]      = drug_df["drug_name"].map(SUBSTITUTABILITY).fillna(1)
drug_df["price_log"]           = np.log1p(drug_df["price_30day"].fillna(0))
drug_df["low_manufacturer_risk"] = (drug_df["n_manufacturers"].fillna(0) < 50).astype(int)

# Merge shortage labels
data = drug_df.merge(
    shortage_labels[["drug_name","has_shortage","shortage_total"]],
    on="drug_name", how="left").fillna(0)
data["has_shortage"] = data["has_shortage"].astype(int)

print(f"\nShortage dataset: {data.shape}")
print(f"  With shortage (Y=1): {data['has_shortage'].sum()}")
print(f"  Without shortage (Y=0): {(data['has_shortage']==0).sum()}")

# ── BASELINE vs ENHANCED features ─────────────────────────────────────────────
baseline_cols_s = [c for c in [
    "price_30day","price_cv","generic_available","n_manufacturers",
    "adverse_log","adverse_unique","trial_total","is_injectable"
    if "is_injectable" in data.columns else "dosage_risk_score",
] if c in data.columns]
baseline_cols_s = [c for c in [
    "price_30day","price_cv","generic_available",
    "adverse_log","adverse_unique","trial_total",
] if c in data.columns]

enhanced_cols_s = [c for c in [
    "price_30day","price_log","price_cv","generic_available","n_manufacturers",
    "adverse_log","adverse_unique","trial_total","trial_completion_rate",
    "dosage_risk_score","is_brand_only","has_substitute","low_manufacturer_risk",
] if c in data.columns]

# Remove zero variance
for col_list in [baseline_cols_s, enhanced_cols_s]:
    col_list[:] = [c for c in col_list if data[c].var() > 0.001]

print(f"\nBaseline features: {len(baseline_cols_s)}")
print(f"Enhanced features: {len(enhanced_cols_s)}")

X_base = data[baseline_cols_s].values
X_enh  = data[enhanced_cols_s].values
y      = data["has_shortage"].values

scaler_b = StandardScaler()
scaler_e = StandardScaler()
X_base_sc = scaler_b.fit_transform(X_base)
X_enh_sc  = scaler_e.fit_transform(X_enh)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "LR (baseline)":    (LogisticRegression(max_iter=1000,random_state=42), X_base_sc),
    "LR (enhanced)":    (LogisticRegression(max_iter=1000,random_state=42), X_enh_sc),
    "RF (baseline)":    (RandomForestClassifier(n_estimators=200,max_depth=4,random_state=42), X_base_sc),
    "RF (enhanced)":    (RandomForestClassifier(n_estimators=200,max_depth=4,random_state=42), X_enh_sc),
    "GB (enhanced)":    (GradientBoostingClassifier(n_estimators=100,max_depth=2,random_state=42), X_enh_sc),
}

print(f"\n{'Model':<25} {'Accuracy':>10} {'AUC':>8} {'F1':>8}")
print("-"*55)
results = {}
best_auc = -999
best_name = ""
for name,(model,X) in models.items():
    preds = cross_val_predict(model, X, y, cv=cv)
    proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:,1]
    acc = accuracy_score(y,preds)
    auc = roc_auc_score(y,proba)
    f1  = f1_score(y,preds,zero_division=0)
    results[name] = {"acc":acc,"auc":auc,"f1":f1}
    note = " ← best" if auc==max(r["auc"] for r in results.values()) else ""
    print(f"  {name:<23} {acc:>10.4f} {auc:>8.4f} {f1:>8.4f}{note}")
    if auc > best_auc:
        best_auc  = auc
        best_name = name

print(f"\nImprovement from feature engineering:")
print(f"  RF baseline AUC : {results['RF (baseline)']['auc']:.4f}")
print(f"  RF enhanced AUC : {results['RF (enhanced)']['auc']:.4f}")
print(f"  Improvement     : {results['RF (enhanced)']['auc']-results['RF (baseline)']['auc']:+.4f}")

# Feature importance for best enhanced model
best_model = RandomForestClassifier(n_estimators=200,max_depth=4,random_state=42)
best_model.fit(X_enh_sc, y)
perm = permutation_importance(best_model, X_enh_sc, y,
                               n_repeats=50, random_state=42, scoring="roc_auc")
imp_df = pd.DataFrame({
    "feature":    enhanced_cols_s,
    "importance": perm.importances_mean,
    "std":        perm.importances_std,
}).sort_values("importance", ascending=False)

print(f"\nTop Features for Shortage Prediction:")
for _,row in imp_df.iterrows():
    print(f"  {row['feature']:<35} {row['importance']:.4f} ± {row['std']:.4f}")

# Correlation analysis
print(f"\nPoint-biserial correlations with has_shortage:")
for feat in enhanced_cols_s:
    r,p = stats.pointbiserialr(y, data[feat].values)
    sig = "**p<0.05" if p<0.05 else "*p<0.10" if p<0.10 else "n.s."
    print(f"  {feat:<35} r={r:+.3f}  p={p:.3f}  {sig}")

# Save
shortage_preds = data[["drug_name","has_shortage","shortage_total"]].copy()
shortage_preds["predicted_shortage"] = cross_val_predict(
    RandomForestClassifier(n_estimators=200,max_depth=4,random_state=42),
    X_enh_sc, y, cv=cv)
shortage_preds["shortage_probability"] = cross_val_predict(
    RandomForestClassifier(n_estimators=200,max_depth=4,random_state=42),
    X_enh_sc, y, cv=cv, method="predict_proba")[:,1]
shortage_preds.to_csv(OUT/"shortage_predictions_v2.csv", index=False)
imp_df.to_csv(OUT/"shortage_feature_importance_v2.csv", index=False)

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 7))

# Model comparison
names_plot = list(results.keys())
aucs_plot  = [results[m]["auc"] for m in names_plot]
colors     = ["#E0E0E0","#9D1535","#E0E0E0","#2563EB","#059669"]
axes[0].bar(range(len(names_plot)), aucs_plot, color=colors, width=0.6)
axes[0].axhline(0.5, color="gray", ls="--", lw=1, label="Random baseline")
for i,v in enumerate(aucs_plot):
    axes[0].text(i, v+0.01, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
axes[0].set_xticks(range(len(names_plot)))
axes[0].set_xticklabels([n.replace(" (","\n(") for n in names_plot], fontsize=8)
axes[0].set_ylim(0,1.1)
axes[0].set_title("Baseline vs Enhanced\nShortage Prediction (AUC)", fontweight="bold")
axes[0].set_ylabel("ROC-AUC")
axes[0].legend(fontsize=8)

# Feature importance
imp_top = imp_df.head(10)
axes[1].barh(imp_top["feature"][::-1], imp_top["importance"][::-1],
              color="#9D1535", xerr=imp_top["std"][::-1], capsize=3)
axes[1].set_xlabel("Permutation Importance")
axes[1].set_title("Top 10 Features\nShortage Prediction", fontweight="bold")
axes[1].tick_params(axis="y", labelsize=8)

# Shortage probability per drug
sp = shortage_preds.sort_values("shortage_probability")
bar_c = ["#9D1535" if p>0.5 else "#E8F0FE" for p in sp["shortage_probability"]]
axes[2].barh(sp["drug_name"], sp["shortage_probability"], color=bar_c)
axes[2].axvline(0.5, color="#D97706", ls="--", lw=1.5)
axes[2].set_xlabel("Shortage Probability")
axes[2].set_title("Predicted Shortage\nProbability per Drug", fontweight="bold")
axes[2].tick_params(axis="y", labelsize=7)
axes[2].set_xlim(0, 1.15)

plt.suptitle("Phase 2C v2 — Enhanced Shortage Prediction\nStevens Institute of Technology · BIA-660",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUT/"phase2c_v2_results.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"\n{'='*65}")
print("PHASE 2C v2 COMPLETE")
print(f"{'='*65}")
print(f"  Best model     : {best_name}")
print(f"  Best AUC       : {best_auc:.4f}")
print(f"  Baseline AUC   : {results['RF (baseline)']['auc']:.4f}")
print(f"  Improvement    : {best_auc - results['RF (baseline)']['auc']:+.4f}")
print(f"  Saved to       : data/master/")
