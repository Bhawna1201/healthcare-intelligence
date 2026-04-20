"""
phase2b_run_v3.py — Patient Satisfaction Predictor (Enhanced Feature Engineering)
Stevens Institute of Technology · BIA-660 Web Mining

NEW FEATURES ADDED:
  - Drug class (one-hot encoded therapeutic categories)
  - Drug approval year / drug age
  - Review recency ratio
  - Sentiment variance (polarization)
  - Condition diversity
  - Negative review ratio
  - Manual prices for brand-only drugs missing from NADAC

Run: venv/bin/python phase2b_run_v3.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
from rapidfuzz import process, fuzz
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings("ignore")
pd.set_option("display.float_format", "{:.4f}".format)

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

# ── Drug metadata ─────────────────────────────────────────────────────────────
DRUG_CLASS = {
    "metformin":           "diabetes_oral",
    "empagliflozin":       "sglt2",
    "semaglutide":         "glp1",
    "dulaglutide":         "glp1",
    "insulin glargine":    "insulin",
    "insulin lispro":      "insulin",
    "lisinopril":          "ace_inhibitor",
    "losartan":            "arb",
    "amlodipine":          "calcium_channel_blocker",
    "diltiazem":           "calcium_channel_blocker",
    "verapamil":           "calcium_channel_blocker",
    "metoprolol":          "beta_blocker",
    "carvedilol":          "beta_blocker",
    "atorvastatin":        "statin",
    "simvastatin":         "statin",
    "rosuvastatin":        "statin",
    "furosemide":          "diuretic",
    "hydrochlorothiazide": "diuretic",
    "spironolactone":      "diuretic",
    "warfarin":            "anticoagulant",
    "apixaban":            "anticoagulant",
    "rivaroxaban":         "anticoagulant",
    "clopidogrel":         "antiplatelet",
    "digoxin":             "cardiac_glycoside",
    "sertraline":          "ssri",
    "escitalopram":        "ssri",
    "fluoxetine":          "ssri",
    "bupropion":           "antidepressant_other",
    "clonazepam":          "benzodiazepine",
    "alprazolam":          "benzodiazepine",
    "zolpidem":            "sleep",
    "gabapentin":          "anticonvulsant",
    "montelukast":         "respiratory",
    "albuterol":           "respiratory",
    "omeprazole":          "ppi",
    "pantoprazole":        "ppi",
    "oxycodone":           "opioid",
    "hydrocodone":         "opioid",
    "tramadol":            "opioid",
    "cyclobenzaprine":     "muscle_relaxant",
    "amoxicillin":         "antibiotic",
    "azithromycin":        "antibiotic",
    "doxycycline":         "antibiotic",
    "prednisone":          "corticosteroid",
    "methylprednisolone":  "corticosteroid",
    "levothyroxine":       "thyroid",
    "tamsulosin":          "alpha_blocker",
    "acetaminophen":       "analgesic_nsaid",
    "ibuprofen":           "analgesic_nsaid",
    "naproxen":            "analgesic_nsaid",
}

DRUG_APPROVAL_YEAR = {
    "metformin":           1994,
    "lisinopril":          1987,
    "atorvastatin":        1996,
    "levothyroxine":       1927,
    "amlodipine":          1992,
    "metoprolol":          1978,
    "omeprazole":          1989,
    "simvastatin":         1991,
    "losartan":            1995,
    "albuterol":           1981,
    "gabapentin":          1993,
    "hydrochlorothiazide": 1959,
    "sertraline":          1991,
    "montelukast":         1998,
    "furosemide":          1966,
    "pantoprazole":        2000,
    "escitalopram":        2002,
    "rosuvastatin":        2003,
    "bupropion":           1985,
    "fluoxetine":          1987,
    "clopidogrel":         1997,
    "tramadol":            1995,
    "cyclobenzaprine":     1977,
    "amoxicillin":         1974,
    "azithromycin":        1991,
    "doxycycline":         1967,
    "prednisone":          1955,
    "methylprednisolone":  1957,
    "clonazepam":          1975,
    "alprazolam":          1981,
    "zolpidem":            1992,
    "oxycodone":           1996,
    "hydrocodone":         1943,
    "acetaminophen":       1955,
    "ibuprofen":           1974,
    "naproxen":            1976,
    "insulin glargine":    2000,
    "insulin lispro":      1996,
    "empagliflozin":       2014,
    "semaglutide":         2017,
    "dulaglutide":         2014,
    "apixaban":            2012,
    "rivaroxaban":         2011,
    "warfarin":            1954,
    "digoxin":             1954,
    "diltiazem":           1982,
    "verapamil":           1981,
    "carvedilol":          1995,
    "spironolactone":      1960,
    "tamsulosin":          1997,
}

# Manual 30-day prices for brand-only drugs not in NADAC
MANUAL_PRICES = {
    "empagliflozin": 369.20,
    "semaglutide":   1109.00,
    "dulaglutide":   1087.00,
    "apixaban":      260.00,
}

def load(proc, raw):
    path = proc if (proc and Path(proc).exists()) else raw
    if path and Path(path).exists():
        df = pd.read_csv(path, on_bad_lines="skip", engine="python",
                         encoding_errors="replace")
        return df.replace({"None": np.nan, "none": np.nan, "nan": np.nan})
    return pd.DataFrame()

def dcol(df):
    return "drug_name_clean" if "drug_name_clean" in df.columns else "drug_name"

def fuzzy_to_target(name):
    if not isinstance(name, str): return None
    nl = name.lower().strip()
    for drug in TARGET_DRUGS:
        if drug in nl: return drug
    return None

print("=" * 65)
print("PHASE 2B v3 — Patient Satisfaction Predictor")
print("Enhanced Feature Engineering")
print("Stevens Institute of Technology · BIA-660 Web Mining")
print("=" * 65)

# ── Load ──────────────────────────────────────────────────────────────────────
reviews   = load(PROC/"reviews_clean.csv",  DATA/"reviews.csv")
prices    = load(PROC/"prices_clean.csv",   DATA/"prices.csv")
shortages = load(None,                      DATA/"shortages.csv")
adverse   = load(None,                      DATA/"adverse_events.csv")
trials    = load(PROC/"trials_clean.csv",   DATA/"trials.csv")
pubmed    = load(PROC/"pubmed_clean.csv",   DATA/"pubmed_abstracts.csv")

for name, df in [("reviews",reviews),("prices",prices),("shortages",shortages),
                  ("adverse",adverse),("trials",trials),("pubmed",pubmed)]:
    print(f"  {name:<12} {len(df):>7,} rows")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: TARGET VARIABLE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("STEP 2 — Target Variable + Review-based Features")
print("="*65)

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

dc = dcol(reviews)
reviews["drug_canonical"] = reviews[dc].apply(fuzzy_to_target)
reviews_m = reviews.dropna(subset=["drug_canonical"]).copy()

if "vader_compound" not in reviews_m.columns:
    sia = SentimentIntensityAnalyzer()
    reviews_m["vader_compound"] = reviews_m["review_text"].apply(
        lambda t: sia.polarity_scores(str(t))["compound"] if pd.notna(t) else 0)

reviews_m["sentiment_label"] = reviews_m["vader_compound"].apply(
    lambda v: "positive" if v>=0.05 else "negative" if v<=-0.05 else "neutral")

# Parse review date year
if "date" in reviews_m.columns:
    reviews_m["review_year"] = pd.to_datetime(
        reviews_m["date"], errors="coerce").dt.year

# Aggregate target + review-based features
target = reviews_m.groupby("drug_canonical").apply(lambda g: pd.Series({
    "avg_sentiment":      g["vader_compound"].mean(),
    "review_count":       len(g),
    "pct_negative":       (g["sentiment_label"]=="negative").mean(),
    "pct_positive":       (g["sentiment_label"]=="positive").mean(),
    "sentiment_std":      g["vader_compound"].std(),          # NEW: polarization
    "condition_diversity":g["condition"].nunique()             # NEW: condition diversity
                          if "condition" in g.columns else 0,
    "review_recency":     (g["review_year"] >= 2020).mean()   # NEW: recency
                          if "review_year" in g.columns else 0,
})).query("review_count >= 3").reset_index().rename(
    columns={"drug_canonical":"drug_name"})

print(f"Drugs with reviews: {len(target)}")
print(f"Sentiment range: {target['avg_sentiment'].min():.4f} to {target['avg_sentiment'].max():.4f}")
print(f"\nNew review-based features:")
print(f"  sentiment_std range:      {target['sentiment_std'].min():.4f} to {target['sentiment_std'].max():.4f}")
print(f"  condition_diversity range:{int(target['condition_diversity'].min())} to {int(target['condition_diversity'].max())}")
print(f"  review_recency range:     {target['review_recency'].min():.3f} to {target['review_recency'].max():.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: DRUG METADATA FEATURES
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("STEP 3 — Drug Metadata Features (class, age)")
print("="*65)

meta_df = pd.DataFrame({"drug_name": TARGET_DRUGS})
meta_df["drug_class"]        = meta_df["drug_name"].map(DRUG_CLASS).fillna("other")
meta_df["approval_year"]     = meta_df["drug_name"].map(DRUG_APPROVAL_YEAR)
meta_df["drug_age"]          = 2024 - meta_df["approval_year"]
meta_df["is_new_drug"]       = (meta_df["approval_year"] >= 2010).astype(int)

# One-hot encode drug class
class_dummies = pd.get_dummies(meta_df["drug_class"], prefix="class")
meta_df = pd.concat([meta_df, class_dummies], axis=1)

print(f"Drug classes found: {meta_df['drug_class'].nunique()}")
print(f"Drug age range: {meta_df['drug_age'].min():.0f} to {meta_df['drug_age'].max():.0f} years")
print(f"New drugs (post-2010): {meta_df['is_new_drug'].sum()}")
print(f"One-hot class features: {len(class_dummies.columns)}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: PRICE FEATURES
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("STEP 4 — Price Features (NADAC + manual overrides)")
print("="*65)

pdc = dcol(prices)
prices_c = prices.copy()
prices_c["drug_name"] = prices_c[pdc].apply(fuzzy_to_target)
prices_c = prices_c.dropna(subset=["drug_name"])
prices_c["price"] = pd.to_numeric(prices_c["price"], errors="coerce")
prices_c = prices_c[prices_c["price"].between(0.0001, 9999)]

if "pharmacy" in prices_c.columns:
    prices_c["is_brand"] = prices_c["pharmacy"].str.contains("Brand", case=False, na=False)
else:
    prices_c["is_brand"] = False

price_feats = prices_c.groupby("drug_name").apply(lambda g: pd.Series({
    "price_median_unit":  g["price"].median(),
    "price_cv":           g["price"].std()/g["price"].mean() if g["price"].mean()>0 else 0,
    "generic_available":  int((~g["is_brand"]).any()),
    "n_manufacturers":    len(g),
    "price_30day":        g["price"].median()*30,
})).reset_index()

# Apply manual price overrides for brand-only drugs
for drug, price in MANUAL_PRICES.items():
    idx = price_feats[price_feats["drug_name"]==drug].index
    if len(idx) > 0:
        price_feats.loc[idx, "price_30day"] = price
        print(f"  Override: {drug} → ${price}")
    else:
        new_row = {"drug_name":drug,"price_median_unit":price/30,
                   "price_cv":0,"generic_available":0,
                   "n_manufacturers":1,"price_30day":price}
        price_feats = pd.concat([price_feats,pd.DataFrame([new_row])],
                                  ignore_index=True)
        print(f"  Added: {drug} → ${price}")

# Log price (handles extreme values like semaglutide $1109 vs metformin $0.70)
price_feats["price_log"] = np.log1p(price_feats["price_30day"])
print(f"Price features: {len(price_feats)} drugs")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: SHORTAGE FEATURES
# ══════════════════════════════════════════════════════════════════════════════
shortages_c = shortages.copy()
if not shortages_c.empty:
    shortages_c.columns = shortages_c.columns.str.strip().str.lower().str.replace(" ","_")
    sdc = "drug_name" if "drug_name" in shortages_c.columns else shortages_c.columns[0]
    def map_s(name):
        if not isinstance(name,str): return None
        nl=name.lower()
        for d in TARGET_DRUGS:
            if d in nl: return d
        return None
    shortages_c["drug_name"] = shortages_c[sdc].apply(map_s)
    shortages_c = shortages_c.dropna(subset=["drug_name"])
    if "status" in shortages_c.columns:
        shortages_c["is_active"] = shortages_c["status"].str.lower().str.contains(
            "current|active|unavailable|limited",na=False).astype(int)
    else:
        shortages_c["is_active"] = 0
    shortage_feats = (shortages_c.groupby("drug_name")
                      .agg(shortage_total=("drug_name","count"),
                           shortage_active=("is_active","sum"))
                      .reset_index())
    shortage_feats["has_active_shortage"] = (shortage_feats["shortage_active"]>0).astype(int)
else:
    shortage_feats = pd.DataFrame({"drug_name":TARGET_DRUGS,"shortage_total":0,
                                    "shortage_active":0,"has_active_shortage":0})

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: ADVERSE EVENT FEATURES
# ══════════════════════════════════════════════════════════════════════════════
adc = dcol(adverse)
adverse_c = adverse.copy()
adverse_c["drug_name"] = adverse_c[adc].apply(fuzzy_to_target)
adverse_c = adverse_c.dropna(subset=["drug_name"])
adverse_feats = (adverse_c.groupby("drug_name")
                 .agg(adverse_total=("drug_name","count"),
                      adverse_unique=("event_type","nunique")
                      if "event_type" in adverse_c.columns
                      else ("drug_name","count"))
                 .reset_index())
adverse_feats["adverse_log"] = np.log1p(adverse_feats["adverse_total"])

# ══════════════════════════════════════════════════════════════════════════════
# STEP 7: TRIAL FEATURES
# ══════════════════════════════════════════════════════════════════════════════
tdc = dcol(trials)
trials_c = trials.copy()
trials_c["drug_name"] = trials_c[tdc].apply(fuzzy_to_target)
trials_c = trials_c.dropna(subset=["drug_name"])
if "status" in trials_c.columns:
    trials_c["is_completed"] = trials_c["status"].str.contains("Completed",na=False).astype(int)
else:
    trials_c["is_completed"] = 0
trial_feats = (trials_c.groupby("drug_name")
               .agg(trial_total=("drug_name","count"),
                    trial_completed=("is_completed","sum"))
               .reset_index())
trial_feats["trial_completion_rate"] = trial_feats["trial_completed"]/trial_feats["trial_total"]

# ══════════════════════════════════════════════════════════════════════════════
# STEP 8: PUBMED FEATURES
# ══════════════════════════════════════════════════════════════════════════════
pdc2 = dcol(pubmed)
pubmed_c = pubmed.copy()
pubmed_c["drug_name"] = pubmed_c[pdc2].apply(fuzzy_to_target)
pubmed_c = pubmed_c.dropna(subset=["drug_name"])
if "pub_year" in pubmed_c.columns:
    pubmed_c["pub_year"] = pd.to_numeric(pubmed_c["pub_year"],errors="coerce")
    pubmed_c["is_recent"] = (pubmed_c["pub_year"]>=2020).astype(int)
else:
    pubmed_c["is_recent"] = 0
pubmed_feats = (pubmed_c.groupby("drug_name")
                .agg(pubmed_total=("drug_name","count"),
                     pubmed_recent=("is_recent","sum"))
                .reset_index())
pubmed_feats["pubmed_recency_ratio"] = pubmed_feats["pubmed_recent"]/pubmed_feats["pubmed_total"]

# ══════════════════════════════════════════════════════════════════════════════
# STEP 9: BUILD MASTER FEATURE MATRIX
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("STEP 9 — Build Master Feature Matrix")
print("="*65)

master = target[["drug_name","avg_sentiment","review_count",
                  "pct_negative","pct_positive","sentiment_std",
                  "condition_diversity","review_recency"]].copy()

for feats, name in [
    (meta_df[["drug_name","drug_age","is_new_drug"] +
              [c for c in meta_df.columns if c.startswith("class_")]],
     "Metadata"),
    (price_feats,    "Pricing"),
    (shortage_feats, "Shortages"),
    (adverse_feats,  "Adverse"),
    (trial_feats,    "Trials"),
    (pubmed_feats,   "PubMed"),
]:
    master = master.merge(feats, on="drug_name", how="left")
    print(f"  {name}: {master.shape}")

master = master.fillna(0)
master.to_csv(OUT/"drug_feature_matrix_v3.csv", index=False)
print(f"\nFinal matrix: {master.shape[0]} drugs × {master.shape[1]} columns")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 10: FEATURE SELECTION & CORRELATIONS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("STEP 10 — Feature Correlations with Satisfaction")
print("="*65)

# Core structural features
core_features = [
    "price_30day","price_log","price_cv","generic_available","n_manufacturers",
    "shortage_total","has_active_shortage",
    "adverse_log","adverse_unique",
    "trial_completion_rate","trial_total",
    "pubmed_recency_ratio",
    "drug_age","is_new_drug",
    "sentiment_std","condition_diversity","review_recency",
]

# Add class dummies
class_cols = [c for c in master.columns if c.startswith("class_")]
all_features = [c for c in core_features + class_cols if c in master.columns]

# Drop zero-variance
variances = master[all_features].var()
all_features = [c for c in all_features if variances.get(c,0) > 0.0001]

print(f"Total features: {len(all_features)}")

corr_df = pd.DataFrame([{
    "feature": c,
    "pearson_r": stats.pearsonr(master[c], master["avg_sentiment"])[0],
    "p_value":   stats.pearsonr(master[c], master["avg_sentiment"])[1],
} for c in all_features]).sort_values("pearson_r")

print(f"\nTop 10 negative correlates with satisfaction:")
print(corr_df.head(10)[["feature","pearson_r","p_value"]].to_string())
print(f"\nTop 10 positive correlates with satisfaction:")
print(corr_df.tail(10)[["feature","pearson_r","p_value"]].to_string())

sig_features = corr_df[corr_df["p_value"]<0.10]["feature"].tolist()
print(f"\nStatistically significant features (p<0.10): {sig_features}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 11: MODEL TRAINING — TWO VERSIONS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("STEP 11 — Model Training (LOO-CV, n=50)")
print("="*65)

# Version A: Original features (baseline)
baseline_cols = [c for c in [
    "price_30day","price_cv","generic_available",
    "shortage_total","has_active_shortage",
    "adverse_log","adverse_unique",
    "trial_completion_rate","pubmed_recency_ratio",
] if c in master.columns]

# Version B: All new features
enhanced_cols = all_features

X_base = master[baseline_cols].values
X_enh  = master[enhanced_cols].values
y      = master["avg_sentiment"].values

scaler_b = StandardScaler()
scaler_e = StandardScaler()
X_base_sc = scaler_b.fit_transform(X_base)
X_enh_sc  = scaler_e.fit_transform(X_enh)

loo = LeaveOneOut()
models = {
    "Ridge α=10 (baseline)":  (Ridge(alpha=10.0),  X_base_sc),
    "Ridge α=10 (enhanced)":  (Ridge(alpha=10.0),  X_enh_sc),
    "RF (baseline)":          (RandomForestRegressor(n_estimators=200,max_depth=3,random_state=42), X_base_sc),
    "RF (enhanced)":          (RandomForestRegressor(n_estimators=200,max_depth=3,random_state=42), X_enh_sc),
}

print(f"\n{'Model':<30} {'LOO R²':>8} {'LOO MAE':>9}")
print("-"*50)
results = {}
for name, (model, X) in models.items():
    preds = cross_val_predict(model, X, y, cv=loo)
    r2  = r2_score(y, preds)
    mae = mean_absolute_error(y, preds)
    results[name] = {"r2":r2,"mae":mae}
    note = " ← best" if r2==max(v["r2"] for v in results.values()) else ""
    print(f"  {name:<30} {r2:>8.4f} {mae:>9.4f}{note}")

print(f"\nImprovement from feature engineering:")
base_r2 = results["RF (baseline)"]["r2"]
enh_r2  = results["RF (enhanced)"]["r2"]
print(f"  Baseline RF R²  : {base_r2:.4f}")
print(f"  Enhanced RF R²  : {enh_r2:.4f}")
print(f"  Improvement     : {enh_r2-base_r2:+.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 12: HYPOTHESIS TESTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("STEP 12 — Hypothesis Test Results")
print("="*65)

hypotheses = [
    ("H1","adverse_unique",     "Adverse event diversity","negative"),
    ("H2","has_active_shortage","Active shortage flag",   "negative"),
    ("H3","price_30day",        "Drug price (30-day)",    "negative"),
    ("H4","trial_completion_rate","Trial completion rate","positive"),
    ("H5","drug_age",           "Drug age (years)",       "positive"),   # NEW
    ("H6","is_new_drug",        "New drug (post-2010)",   "negative"),   # NEW
    ("H7","condition_diversity","Condition diversity",    "negative"),   # NEW
    ("H8","sentiment_std",      "Sentiment polarization", "negative"),   # NEW
]

print(f"\n{'#':<4} {'Feature':<30} {'Expected':<10} {'r':>8} {'p-val':>8}  Result")
print("-"*72)
for hid,feat,label,expected in hypotheses:
    if feat not in master.columns: continue
    x = master[feat]; ys = master["avg_sentiment"]
    valid = ~(x.isna()|ys.isna()) & (x.var()>0)
    if valid.sum()<5: continue
    r,p = stats.pearsonr(x[valid],ys[valid])
    sig  = "p<0.05**" if p<0.05 else "p<0.10*" if p<0.10 else "n.s."
    match= (expected=="negative" and r<0) or (expected=="positive" and r>0)
    res  = "✓ SUPPORTED" if match and p<0.05 else "~ TREND" if match and p<0.10 else "~ trend" if match else "✗ NOT SUPPORTED"
    print(f"  {hid:<4} {label:<30} {expected:<10} {r:>8.4f} {p:>8.4f}  {sig}  {res}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 13: FEATURE IMPORTANCE (Ridge coefficients)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("STEP 13 — Feature Importance (Enhanced Model)")
print("="*65)

best_model_name = max(results, key=lambda k: results[k]["r2"])
ridge_enh = Ridge(alpha=10.0)
ridge_enh.fit(X_enh_sc, y)

name_map = {
    "price_30day":           "Drug Price (30-day)",
    "price_log":             "Drug Price (log)",
    "price_cv":              "Price Variability",
    "generic_available":     "Generic Available",
    "n_manufacturers":       "Number of Manufacturers",
    "shortage_total":        "Total Shortage History",
    "has_active_shortage":   "Active Shortage",
    "adverse_log":           "Adverse Event Volume",
    "adverse_unique":        "Adverse Event Diversity",
    "trial_completion_rate": "Trial Completion Rate",
    "trial_total":           "Trial Volume",
    "pubmed_recency_ratio":  "Recent Research Ratio",
    "drug_age":              "Drug Age (years)",
    "is_new_drug":           "New Drug (post-2010)",
    "sentiment_std":         "Sentiment Polarization",
    "condition_diversity":   "Condition Diversity",
    "review_recency":        "Review Recency",
}

coef_df = pd.DataFrame({
    "feature":  enhanced_cols,
    "label":    [name_map.get(c, c.replace("class_","Drug Class: ")) for c in enhanced_cols],
    "coef":     ridge_enh.coef_,
    "abs_coef": np.abs(ridge_enh.coef_),
}).sort_values("coef")

print("\nTop 10 features → LOWER satisfaction:")
for _,row in coef_df.head(10).iterrows():
    print(f"  {row['label']:<40} {row['coef']:>8.4f}")
print("\nTop 10 features → HIGHER satisfaction:")
for _,row in coef_df.tail(10).iloc[::-1].iterrows():
    print(f"  {row['label']:<40} {row['coef']:>8.4f}")

# Save
coef_df[["label","coef","abs_coef"]].to_csv(OUT/"feature_importance_v3.csv", index=False)

# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 8))

# Plot 1: Model comparison
model_names = list(results.keys())
r2_vals = [results[m]["r2"] for m in model_names]
colors  = ["#E0E0E0","#9D1535","#E0E0E0","#2563EB"]
axes[0].bar(range(len(model_names)), r2_vals, color=colors, width=0.6)
axes[0].axhline(0, color="gray", ls="--", lw=1)
for i,(n,v) in enumerate(zip(model_names,r2_vals)):
    axes[0].text(i, v+0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
axes[0].set_xticks(range(len(model_names)))
axes[0].set_xticklabels([n.replace(" (","\n(") for n in model_names], fontsize=8)
axes[0].set_title("Baseline vs Enhanced\nModel Comparison (LOO R²)", fontweight="bold")
axes[0].set_ylabel("R² Score")

# Plot 2: Top feature importance
top_n = coef_df.assign(abs_c=coef_df["coef"].abs()).nlargest(15,"abs_c")
top_n = top_n.sort_values("coef")
bar_c = ["#9D1535" if v<0 else "#059669" for v in top_n["coef"]]
axes[1].barh(top_n["label"], top_n["coef"], color=bar_c)
axes[1].axvline(0, color="gray", ls="--", lw=0.8)
axes[1].set_xlabel("Ridge Coefficient")
axes[1].set_title("Top 15 Features\n(Enhanced Model)", fontweight="bold")
axes[1].tick_params(axis="y", labelsize=8)

# Plot 3: New features vs satisfaction
new_feats = ["drug_age","sentiment_std","condition_diversity","review_recency","is_new_drug"]
new_feats = [f for f in new_feats if f in master.columns]
corr_new  = [(f, stats.pearsonr(master[f], master["avg_sentiment"])[0],
               stats.pearsonr(master[f], master["avg_sentiment"])[1])
             for f in new_feats]
corr_new.sort(key=lambda x: x[1])
labels_n = [name_map.get(f,f) for f,_,_ in corr_new]
vals_n   = [r for _,r,_ in corr_new]
pvals_n  = [p for _,_,p in corr_new]
bar_c2   = ["#9D1535" if v<0 else "#059669" for v in vals_n]
bars     = axes[2].barh(labels_n, vals_n, color=bar_c2)
for bar, p in zip(bars, pvals_n):
    if p < 0.05: axes[2].text(bar.get_width()+0.005, bar.get_y()+bar.get_height()/2,
                                "**", va="center", fontsize=10, color="#D97706")
    elif p < 0.10: axes[2].text(bar.get_width()+0.005, bar.get_y()+bar.get_height()/2,
                                 "*", va="center", fontsize=10, color="#D97706")
axes[2].axvline(0, color="gray", ls="--", lw=0.8)
axes[2].set_xlabel("Pearson r with Satisfaction")
axes[2].set_title("New Features vs\nPatient Satisfaction", fontweight="bold")
axes[2].tick_params(axis="y", labelsize=9)

plt.suptitle("Phase 2B v3 — Enhanced Feature Engineering Results\n"
             "Stevens Institute of Technology · BIA-660",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUT/"phase2b_v3_results.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: data/master/phase2b_v3_results.png")

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("PHASE 2B v3 COMPLETE")
print("="*65)
print(f"\nFeature Engineering Summary:")
print(f"  Baseline features : {len(baseline_cols)}")
print(f"  Enhanced features : {len(enhanced_cols)}")
print(f"  New features added: {len(enhanced_cols)-len(baseline_cols)}")
print(f"\nModel Performance:")
print(f"  Baseline RF R²   : {results['RF (baseline)']['r2']:.4f}")
print(f"  Enhanced RF R²   : {results['RF (enhanced)']['r2']:.4f}")
print(f"  Improvement      : {results['RF (enhanced)']['r2']-results['RF (baseline)']['r2']:+.4f}")
print(f"\nNew Hypotheses Tested: H5 (drug age), H6 (new drug), H7 (condition diversity), H8 (sentiment polarization)")
print(f"\nOutputs saved to data/master/")
