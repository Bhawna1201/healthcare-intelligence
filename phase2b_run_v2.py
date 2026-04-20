"""
Phase 2B v2 — Patient Satisfaction Predictor (fixed)
Fixes: fuzzy drug name matching, reduced feature set, LOO with small N
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
from rapidfuzz import process, fuzz
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
pd.set_option("display.float_format", "{:.4f}".format)

DATA = Path("data/raw")
PROC = Path("data/processed")
OUT  = Path("data/master")
OUT.mkdir(exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
def load(proc, raw):
    path = proc if proc and Path(proc).exists() else raw
    if Path(path).exists():
        df = pd.read_csv(path, on_bad_lines="skip", engine="python",
                         encoding_errors="replace")
        return df.replace({"None": np.nan, "none": np.nan, "nan": np.nan})
    return pd.DataFrame()

reviews   = load(PROC/"reviews_clean.csv",  DATA/"reviews.csv")
prices    = load(PROC/"prices_clean.csv",   DATA/"prices.csv")
shortages = load(None,                      DATA/"shortages.csv")
adverse   = load(None,                      DATA/"adverse_events.csv")
trials    = load(PROC/"trials_clean.csv",   DATA/"trials.csv")
pubmed    = load(PROC/"pubmed_clean.csv",   DATA/"pubmed_abstracts.csv")

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

def fuzzy_to_target(name, threshold=70):
    """Map any drug name string to nearest TARGET_DRUGS entry."""
    if not isinstance(name, str): return None
    clean = name.lower().strip()
    # Direct match first
    for d in TARGET_DRUGS:
        if d in clean or clean in d: return d
    result = process.extractOne(clean, TARGET_DRUGS, scorer=fuzz.partial_ratio)
    return result[0] if result and result[1] >= threshold else None

print("=" * 60)
print("STEP 1 — Load & map drug names to TARGET_DRUGS")
print("=" * 60)
for name, df in [("reviews",reviews),("prices",prices),("shortages",shortages),
                  ("adverse",adverse),("trials",trials),("pubmed",pubmed)]:
    print(f"  {name:<12} {len(df):>6,} rows")

# ── STEP 2: Target Variable ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 — Build Target Variable (Y = avg VADER sentiment per drug)")
print("=" * 60)

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

dc = "drug_name_clean" if "drug_name_clean" in reviews.columns else "drug_name"

# Map all review drug names to TARGET_DRUGS
reviews["drug_canonical"] = reviews[dc].apply(fuzzy_to_target)
reviews_mapped = reviews.dropna(subset=["drug_canonical"])

# Compute VADER if not present
if "vader_compound" not in reviews_mapped.columns:
    sia = SentimentIntensityAnalyzer()
    reviews_mapped = reviews_mapped.copy()
    reviews_mapped["vader_compound"] = reviews_mapped["review_text"].apply(
        lambda t: sia.polarity_scores(str(t))["compound"] if pd.notna(t) else 0)
    reviews_mapped["sentiment_label"] = reviews_mapped["vader_compound"].apply(
        lambda v: "positive" if v>=0.05 else "negative" if v<=-0.05 else "neutral")

# Aggregate — lower threshold to 3 reviews minimum
target = (reviews_mapped.groupby("drug_canonical")
          .agg(
              avg_sentiment  = ("vader_compound","mean"),
              review_count   = ("vader_compound","count"),
              pct_negative   = ("sentiment_label", lambda x: (x=="negative").mean()),
              pct_positive   = ("sentiment_label", lambda x: (x=="positive").mean()),
              sentiment_std  = ("vader_compound","std"),
          )
          .query("review_count >= 3")
          .reset_index()
          .rename(columns={"drug_canonical":"drug_name"}))

print(f"Drugs with reviews (threshold ≥3): {len(target)}")
print(f"Sentiment range: {target['avg_sentiment'].min():.4f} to {target['avg_sentiment'].max():.4f}")
print(f"Mean sentiment: {target['avg_sentiment'].mean():.4f}")
print("\nAll drugs in target:")
print(target[["drug_name","avg_sentiment","review_count"]].to_string())

# ── STEP 3: Price Features ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 — Price Features (NADAC)")
print("=" * 60)

pdc = "drug_name_clean" if "drug_name_clean" in prices.columns else "drug_name"
prices["drug_name"] = prices[pdc].apply(fuzzy_to_target)
prices = prices.dropna(subset=["drug_name"])
prices["price"] = pd.to_numeric(prices["price"], errors="coerce")
prices = prices[prices["price"].between(0.0001, 9999)]

if "pharmacy" in prices.columns:
    prices["is_brand"] = prices["pharmacy"].str.contains("Brand", case=False, na=False)
else:
    prices["is_brand"] = False

price_feats = prices.groupby("drug_name").apply(lambda g: pd.Series({
    "price_median_unit": g["price"].median(),
    "price_cv":         g["price"].std() / g["price"].mean() if g["price"].mean() > 0 else 0,
    "generic_available": int((~g["is_brand"]).any()),
    "price_30day":       g["price"].median() * 30,
})).reset_index()

print(f"Price features: {len(price_feats)} drugs")

# Manual prices for brand-only drugs absent from NADAC
manual_prices = {
    "empagliflozin": 369.20,
    "semaglutide":   1109.00,
    "dulaglutide":   1087.00,
    "apixaban":      260.00,
}
for drug, price in manual_prices.items():
    idx = price_feats[price_feats["drug_name"]==drug].index
    if len(idx) > 0:
        price_feats.loc[idx, "price_30day"] = price
    else:
        price_feats = pd.concat([price_feats,
            pd.DataFrame([{"drug_name":drug,"price_30day":price,
                           "price_cv":0,"generic_available":0}])],
            ignore_index=True)
    print(f"  Manual override: {drug} → ${price}")

# ── STEP 4: Shortage Features ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4 — Shortage Features (FDA)")
print("=" * 60)

shortages = shortages.copy()
shortages.columns = shortages.columns.str.strip().str.lower().str.replace(" ","_")
sdc = "drug_name" if "drug_name" in shortages.columns else shortages.columns[0]
# Use substring search — shortage names are full descriptions like "Clonazepam Tablet"
# fuzzy_to_target misses these; substring match is more reliable
def map_shortage_name(name):
    if not isinstance(name, str): return None
    nl = name.lower()
    for drug in TARGET_DRUGS:
        if drug in nl: return drug
    return None

shortages["drug_name"] = shortages[sdc].apply(map_shortage_name)
shortages = shortages.dropna(subset=["drug_name"])

if "status" in shortages.columns:
    shortages["is_active"] = shortages["status"].str.lower().str.contains(
        "current|active|unavailable|limited", na=False).astype(int)
else:
    shortages["is_active"] = 0

shortage_feats = (shortages.groupby("drug_name")
                  .agg(shortage_total=("drug_name","count"),
                       shortage_active=("is_active","sum"))
                  .reset_index())
shortage_feats["has_active_shortage"] = (shortage_feats["shortage_active"] > 0).astype(int)

print(f"Shortage features: {len(shortage_feats)} drugs matched to TARGET_DRUGS")

# ── STEP 5: Adverse Event Features ───────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5 — Adverse Event Features (OpenFDA)")
print("=" * 60)

adc = "drug_name_clean" if "drug_name_clean" in adverse.columns else "drug_name"
adverse["drug_name"] = adverse[adc].apply(fuzzy_to_target)
adverse = adverse.dropna(subset=["drug_name"])

adverse_feats = (adverse.groupby("drug_name")
                 .agg(adverse_total=("drug_name","count"),
                      adverse_unique_types=("event_type","nunique")
                      if "event_type" in adverse.columns
                      else ("drug_name","count"))
                 .reset_index())
# NOTE: adverse_serious_ratio dropped — all = 1.0 (no variance in FAERS data)
adverse_feats["adverse_log_total"] = np.log1p(adverse_feats["adverse_total"])

print(f"Adverse event features: {len(adverse_feats)} drugs")
print(f"adverse_unique_types range: {adverse_feats['adverse_unique_types'].min()} "
      f"to {adverse_feats['adverse_unique_types'].max()}")

# ── STEP 6: Trial Features ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6 — Clinical Trial Features")
print("=" * 60)

tdc = "drug_name_clean" if "drug_name_clean" in trials.columns else "drug_name"
trials["drug_name"] = trials[tdc].apply(fuzzy_to_target)
trials = trials.dropna(subset=["drug_name"])

if "status" in trials.columns:
    trials["is_completed"] = trials["status"].str.lower().str.contains(
        "completed", na=False).astype(int)
else:
    trials["is_completed"] = 0

trial_feats = (trials.groupby("drug_name")
               .agg(trial_total=("drug_name","count"),
                    trial_completed=("is_completed","sum"))
               .reset_index())
trial_feats["trial_completion_rate"] = (
    trial_feats["trial_completed"] / trial_feats["trial_total"])

print(f"Trial features: {len(trial_feats)} drugs")
print(f"Completion rate range: {trial_feats['trial_completion_rate'].min():.3f} "
      f"to {trial_feats['trial_completion_rate'].max():.3f}")

# ── STEP 7: PubMed Features ───────────────────────────────────────────────────
pdc2 = "drug_name_clean" if "drug_name_clean" in pubmed.columns else "drug_name"
pubmed["drug_name"] = pubmed[pdc2].apply(fuzzy_to_target)
pubmed = pubmed.dropna(subset=["drug_name"])

if "pub_year" in pubmed.columns:
    pubmed["pub_year"] = pd.to_numeric(pubmed["pub_year"], errors="coerce")
    pubmed["is_recent"] = (pubmed["pub_year"] >= 2020).astype(int)
else:
    pubmed["is_recent"] = 0

pubmed_feats = (pubmed.groupby("drug_name")
                .agg(pubmed_total=("drug_name","count"),
                     pubmed_recent=("is_recent","sum"))
                .reset_index())
pubmed_feats["pubmed_recency_ratio"] = (
    pubmed_feats["pubmed_recent"] / pubmed_feats["pubmed_total"])

# ── STEP 8: Build Master Matrix ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8 — Master Feature Matrix")
print("=" * 60)

master = target[["drug_name","avg_sentiment","review_count",
                  "pct_negative","pct_positive"]].copy()

for feats, name in [(price_feats,"Pricing"),(shortage_feats,"Shortages"),
                     (adverse_feats,"Adverse"),(trial_feats,"Trials"),
                     (pubmed_feats,"PubMed")]:
    master = master.merge(feats, on="drug_name", how="left")
    print(f"  {name}: {master.shape}")

master = master.fillna(0)
master.to_csv(OUT/"drug_feature_matrix.csv", index=False)
print(f"\nFinal matrix: {master.shape[0]} drugs × {master.shape[1]} columns")
print(master[["drug_name","avg_sentiment","price_30day",
               "shortage_total","adverse_unique_types",
               "trial_completion_rate"]].to_string())

# ── STEP 9: Feature Selection — only features with real variance ──────────────
print("\n" + "=" * 60)
print("STEP 9 — Feature Correlations with Target")
print("=" * 60)

feature_cols = [
    "price_30day","price_cv","generic_available",
    "shortage_total","has_active_shortage",
    "adverse_unique_types","adverse_log_total",
    "trial_completion_rate","trial_total",
    "pubmed_recency_ratio",
]
feature_cols = [c for c in feature_cols if c in master.columns]

# Check variance — drop zero-variance features
variances = master[feature_cols].var()
feature_cols = [c for c in feature_cols if variances[c] > 0.0001]
print(f"Features with real variance: {len(feature_cols)}")

corr_target = (master[feature_cols + ["avg_sentiment"]]
               .corr()["avg_sentiment"]
               .drop("avg_sentiment")
               .dropna()
               .sort_values())
print("\nCorrelations with avg_sentiment:")
print(corr_target.to_string())

# Plot correlation bar
fig, ax = plt.subplots(figsize=(8, 6))
colors = ["#f85149" if v < 0 else "#3fb950" for v in corr_target]
ax.barh(corr_target.index, corr_target.values, color=colors)
ax.axvline(0, color="grey", lw=0.8, ls="--")
ax.set_xlabel("Pearson r with Patient Satisfaction")
ax.set_title("Feature Correlations with Patient Satisfaction", fontweight="bold")
plt.tight_layout()
plt.savefig(OUT/"correlation_bar.png", dpi=150)
plt.close()

# ── STEP 10: Models (LOO-CV) ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 10 — Model Training (Leave-One-Out CV, n=" + str(len(master)) + ")")
print("=" * 60)

X = master[feature_cols].values
y = master["avg_sentiment"].values
scaler = StandardScaler()
X_sc = scaler.fit_transform(X)
loo  = LeaveOneOut()

models = {
    "Ridge (α=0.1)":   Ridge(alpha=0.1),
    "Ridge (α=1.0)":   Ridge(alpha=1.0),
    "Ridge (α=10.0)":  Ridge(alpha=10.0),
    "Random Forest":   RandomForestRegressor(n_estimators=200, max_depth=3,
                                              max_features=0.6, random_state=42),
}

print(f"\n{'Model':<20} {'LOO R²':>9} {'LOO MAE':>9}")
print("-" * 42)
results = {}
for name, model in models.items():
    preds  = cross_val_predict(model, X_sc, y, cv=loo)
    r2     = r2_score(y, preds)
    mae    = mean_absolute_error(y, preds)
    results[name] = {"r2":r2,"mae":mae,"preds":preds,"model":model}
    print(f"  {name:<20} {r2:>9.4f} {mae:>9.4f}")

best_name  = max(results, key=lambda k: results[k]["r2"])
best_model = results[best_name]["model"]
best_model.fit(X_sc, y)
print(f"\nBest model: {best_name}  (R²={results[best_name]['r2']:.4f})")

# ── STEP 11: Feature Importance ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 11 — Feature Importance")
print("=" * 60)

name_map = {
    "price_30day":           "Drug Price (30-day generic)",
    "price_cv":              "Price Variability (CV)",
    "generic_available":     "Generic Available",
    "shortage_total":        "Total Shortage History",
    "has_active_shortage":   "Currently in Shortage",
    "adverse_unique_types":  "Adverse Event Diversity",
    "adverse_log_total":     "Adverse Event Volume",
    "trial_completion_rate": "Trial Completion Rate",
    "trial_total":           "Clinical Trial Volume",
    "pubmed_recency_ratio":  "Recent Research Ratio",
}

# Use Ridge coefficients (most interpretable for small N)
ridge = Ridge(alpha=1.0)
ridge.fit(X_sc, y)
coef_df = pd.DataFrame({
    "feature":  feature_cols,
    "label":    [name_map.get(c,c) for c in feature_cols],
    "coef":     ridge.coef_,
    "abs_coef": np.abs(ridge.coef_),
}).sort_values("coef")

fig, ax = plt.subplots(figsize=(9, 6))
colors = ["#f85149" if v < 0 else "#3fb950" for v in coef_df["coef"]]
ax.barh(coef_df["label"], coef_df["coef"], color=colors)
ax.axvline(0, color="grey", lw=0.8, ls="--")
ax.set_xlabel("Ridge Coefficient (effect on patient satisfaction score)")
ax.set_title("What Predicts Patient Drug Satisfaction?\n"
             "Green = increases satisfaction | Red = decreases satisfaction",
             fontweight="bold")
plt.tight_layout()
plt.savefig(OUT/"feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: feature_importance.png")
print("\nRidge Coefficients (effect on satisfaction):")
for _, row in coef_df.sort_values("coef").iterrows():
    direction = "→ more satisfied" if row["coef"] > 0 else "→ less satisfied"
    print(f"  {row['label']:<35} {row['coef']:>8.4f}  {direction}")

# ── STEP 12: Hypothesis Tests ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 12 — Hypothesis Test Results")
print("=" * 60)

hypotheses = [
    ("H1","adverse_unique_types","Adverse event diversity","negative"),
    ("H2","has_active_shortage", "Active shortage flag",   "negative"),
    ("H3","price_30day",         "Drug price (30-day)",    "negative"),
    ("H4","trial_completion_rate","Trial completion rate", "positive"),
]
print(f"\n{'#':<4} {'Feature':<30} {'Expected':<10} {'r':>8} {'p-val':>8}  Result")
print("-" * 70)
for hid, feat, label, expected in hypotheses:
    if feat not in master.columns:
        print(f"  {hid}  {label:<30} MISSING"); continue
    x = master[feat]; ys = master["avg_sentiment"]
    valid = ~(x.isna() | ys.isna()) & (x.var() > 0)
    if valid.sum() < 5:
        print(f"  {hid}  {label:<30} INSUFFICIENT VARIANCE"); continue
    r, pval = stats.pearsonr(x[valid], ys[valid])
    sig  = "p<0.05" if pval<0.05 else "p<0.10" if pval<0.10 else "n.s. "
    match = (expected=="negative" and r<0) or (expected=="positive" and r>0)
    res  = "✓ SUPPORTED" if match and pval<0.10 else "~ TREND" if match else "✗ NOT SUPPORTED"
    print(f"  {hid}  {label:<30} {expected:<10} {r:>8.4f} {pval:>8.4f}  {sig}  {res}")

# Save predictions
preds_df = master[["drug_name","avg_sentiment","review_count"]].copy()
preds_df["predicted_sentiment"] = results[best_name]["preds"]
preds_df["residual"]            = (preds_df["avg_sentiment"] -
                                    preds_df["predicted_sentiment"])
preds_df.to_csv(OUT/"drug_predictions.csv", index=False)
coef_df[["label","coef","abs_coef"]].to_csv(OUT/"feature_importance.csv", index=False)

print(f"\n{'='*60}")
print("COMPLETE")
print(f"{'='*60}")
print(f"  Drugs analysed : {len(master)}")
print(f"  Features used  : {len(feature_cols)}")
print(f"  Best model     : {best_name}")
print(f"  LOO R²         : {results[best_name]['r2']:.4f}")
print(f"  Outputs saved to data/master/")