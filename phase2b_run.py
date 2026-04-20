import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path

from sklearn.linear_model import Ridge, LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
from scipy import stats

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")
pd.set_option("display.max_colwidth", 60)
pd.set_option("display.float_format", "{:.4f}".format)

DATA    = Path("data/raw")
PROC    = Path("data/processed")
OUT     = Path("data/master")
OUT.mkdir(exist_ok=True)

print("Setup complete")

def load(proc, raw):
    path = proc if proc and Path(proc).exists() else raw
    if Path(path).exists():
        df = pd.read_csv(path, on_bad_lines="skip", engine="python",
                         encoding_errors="replace")
        df = df.replace({"None": np.nan, "none": np.nan, "nan": np.nan})
        return df
    return pd.DataFrame()

reviews   = load(PROC/"reviews_clean.csv",   DATA/"reviews.csv")
prices    = load(PROC/"prices_clean.csv",    DATA/"prices.csv")
shortages = load(None,                       DATA/"shortages.csv")
adverse   = load(None,                       DATA/"adverse_events.csv")
trials    = load(PROC/"trials_clean.csv",    DATA/"trials.csv")
pubmed    = load(PROC/"pubmed_clean.csv",    DATA/"pubmed_abstracts.csv")

for name, df in [("reviews",reviews),("prices",prices),("shortages",shortages),
                  ("adverse",adverse),("trials",trials),("pubmed",pubmed)]:
    print(f"  {name:<12} {len(df):>6,} rows  | cols: {list(df.columns[:5])}")

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Use pre-computed sentiment if available, otherwise compute now
if "vader_compound" not in reviews.columns:
    sia = SentimentIntensityAnalyzer()
    reviews["vader_compound"] = reviews["review_text"].apply(
        lambda t: sia.polarity_scores(str(t))["compound"] if pd.notna(t) else 0)
    reviews["sentiment_label"] = reviews["vader_compound"].apply(
        lambda v: "positive" if v>=0.05 else "negative" if v<=-0.05 else "neutral")

# Normalize drug name column
dc = "drug_name_clean" if "drug_name_clean" in reviews.columns else "drug_name"

# Aggregate per drug — need at least 5 reviews to be reliable
target = (reviews.groupby(dc)
          .agg(
              avg_sentiment   = ("vader_compound", "mean"),
              review_count    = ("vader_compound", "count"),
              pct_negative    = ("sentiment_label", lambda x: (x=="negative").mean()),
              pct_positive    = ("sentiment_label", lambda x: (x=="positive").mean()),
              sentiment_std   = ("vader_compound", "std"),
          )
          .query("review_count >= 5")
          .reset_index()
          .rename(columns={dc: "drug_name"}))

target["drug_name"] = target["drug_name"].str.lower().str.strip()

print(f"Drugs with sufficient reviews: {len(target)}")
print(f"\nTarget variable distribution:")
print(f"  Mean sentiment  : {target['avg_sentiment'].mean():.4f}")
print(f"  Std sentiment   : {target['avg_sentiment'].std():.4f}")
print(f"  Min / Max       : {target['avg_sentiment'].min():.4f} / {target['avg_sentiment'].max():.4f}")
print(f"\nTop 5 most positive drugs:")
print(target.nlargest(5, "avg_sentiment")[["drug_name","avg_sentiment","review_count"]])
print(f"\nTop 5 most negative drugs:")
print(target.nsmallest(5, "avg_sentiment")[["drug_name","avg_sentiment","review_count"]])

prices = prices.copy()
pdc = "drug_name_clean" if "drug_name_clean" in prices.columns else "drug_name"
prices["drug_name"] = prices[pdc].str.lower().str.strip()
prices["price"] = pd.to_numeric(prices["price"], errors="coerce")
prices = prices[prices["price"].between(0.0001, 9999)]

# Detect brand vs generic
if "pharmacy" in prices.columns:
    prices["is_brand"] = prices["pharmacy"].str.contains("Brand", case=False, na=False)
elif "drug_type" in prices.columns:
    prices["is_brand"] = prices["drug_type"].str.upper().str.strip() == "B"
else:
    prices["is_brand"] = False

price_feats = prices.groupby("drug_name").apply(lambda g: pd.Series({
    "price_median_unit"   : g["price"].median(),
    "price_std"           : g["price"].std(),
    "price_min"           : g["price"].min(),
    "price_max"           : g["price"].max(),
    "price_range"         : g["price"].max() - g["price"].min(),
    "generic_available"   : int((~g["is_brand"]).any()),
    "brand_available"     : int(g["is_brand"].any()),
    "n_ndc_variants"      : len(g),
    "brand_generic_ratio" : (g[g["is_brand"]]["price"].median() /
                              g[~g["is_brand"]]["price"].median()
                              if g["is_brand"].any() and (~g["is_brand"]).any()
                              else np.nan),
})).reset_index()

# 30-day cost (more intuitive)
price_feats["price_30day"] = price_feats["price_median_unit"] * 30

print(f"Price features: {len(price_feats)} drugs")
print(price_feats[["drug_name","price_30day","generic_available",
                    "brand_generic_ratio","n_ndc_variants"]].head(8))

shortages = shortages.copy()
shortages.columns = shortages.columns.str.strip().str.lower().str.replace(" ","_")

# Normalize drug name
sdc = "drug_name" if "drug_name" in shortages.columns else shortages.columns[0]
shortages["drug_name"] = shortages[sdc].str.lower().str.strip()

# Clean status
if "status" in shortages.columns:
    shortages["is_active"] = shortages["status"].str.lower().str.contains(
        "current|active|unavailable|limited", na=False).astype(int)
else:
    shortages["is_active"] = 0

shortage_feats = shortages.groupby("drug_name").agg(
    shortage_total   = ("drug_name",  "count"),
    shortage_active  = ("is_active",  "sum"),
    shortage_pct_active = ("is_active", "mean"),
).reset_index()

shortage_feats["has_active_shortage"] = (shortage_feats["shortage_active"] > 0).astype(int)

# Match target drugs via fuzzy matching
from rapidfuzz import process, fuzz

target_names = target["drug_name"].tolist()
def fuzzy_match(name, choices, threshold=75):
    result = process.extractOne(name, choices, scorer=fuzz.partial_ratio)
    return result[0] if result and result[1] >= threshold else name

shortage_feats["drug_name_matched"] = shortage_feats["drug_name"].apply(
    lambda x: fuzzy_match(x, target_names))

# Aggregate by matched name
shortage_feats = (shortage_feats.groupby("drug_name_matched")
                  .agg(shortage_total=("shortage_total","sum"),
                       shortage_active=("shortage_active","sum"),
                       shortage_pct_active=("shortage_pct_active","mean"),
                       has_active_shortage=("has_active_shortage","max"))
                  .reset_index()
                  .rename(columns={"drug_name_matched":"drug_name"}))

print(f"Shortage features: {len(shortage_feats)} drugs matched")
print(shortage_feats[["drug_name","shortage_total","shortage_active",
                        "has_active_shortage"]].head(8))

adverse = adverse.copy()
adc = "drug_name_clean" if "drug_name_clean" in adverse.columns else "drug_name"
adverse["drug_name"] = adverse[adc].str.lower().str.strip()

if "severity" in adverse.columns:
    adverse["is_serious"] = adverse["severity"].str.lower().str.contains(
        "serious", na=False).astype(int)
else:
    adverse["is_serious"] = 0

adverse_feats = adverse.groupby("drug_name").agg(
    adverse_total        = ("drug_name",   "count"),
    adverse_serious      = ("is_serious",  "sum"),
    adverse_unique_types = ("event_type",  "nunique") if "event_type" in adverse.columns
                           else ("drug_name", "count"),
).reset_index()

adverse_feats["adverse_serious_ratio"] = (
    adverse_feats["adverse_serious"] / adverse_feats["adverse_total"])

# Normalize event count by log (heavy tail)
adverse_feats["adverse_log_total"] = np.log1p(adverse_feats["adverse_total"])

print(f"Adverse event features: {len(adverse_feats)} drugs")
print(adverse_feats[["drug_name","adverse_total","adverse_serious",
                       "adverse_serious_ratio","adverse_unique_types"]].head(8))

trials = trials.copy()
tdc = "drug_name_clean" if "drug_name_clean" in trials.columns else "drug_name"
trials["drug_name"] = trials[tdc].str.lower().str.strip()

if "status" in trials.columns:
    trials["is_completed"] = trials["status"].str.lower().str.contains(
        "completed", na=False).astype(int)
else:
    trials["is_completed"] = 0

if "phase" in trials.columns:
    trials["is_late_phase"] = trials["phase"].str.contains(
        "Phase 3|Phase 4|phase 3|phase 4|PHASE 3|PHASE 4", na=False).astype(int)
else:
    trials["is_late_phase"] = 0

trial_feats = trials.groupby("drug_name").agg(
    trial_total        = ("drug_name",     "count"),
    trial_completed    = ("is_completed",  "sum"),
    trial_late_phase   = ("is_late_phase", "sum"),
).reset_index()

trial_feats["trial_completion_rate"] = (
    trial_feats["trial_completed"] / trial_feats["trial_total"])
trial_feats["trial_late_phase_ratio"] = (
    trial_feats["trial_late_phase"] / trial_feats["trial_total"])
trial_feats["trial_log_total"] = np.log1p(trial_feats["trial_total"])

print(f"Trial features: {len(trial_feats)} drugs")
print(trial_feats[["drug_name","trial_total","trial_completion_rate",
                    "trial_late_phase_ratio"]].head(8))

pubmed = pubmed.copy()
pdc = "drug_name_clean" if "drug_name_clean" in pubmed.columns else "drug_name"
pubmed["drug_name"] = pubmed[pdc].str.lower().str.strip()

if "pub_year" in pubmed.columns:
    pubmed["pub_year"] = pd.to_numeric(pubmed["pub_year"], errors="coerce")
    pubmed["is_recent"] = (pubmed["pub_year"] >= 2020).astype(int)
else:
    pubmed["is_recent"] = 0

pubmed_feats = pubmed.groupby("drug_name").agg(
    pubmed_total  = ("drug_name",  "count"),
    pubmed_recent = ("is_recent",  "sum"),
).reset_index()

pubmed_feats["pubmed_recency_ratio"] = (
    pubmed_feats["pubmed_recent"] / pubmed_feats["pubmed_total"])
pubmed_feats["pubmed_log_total"] = np.log1p(pubmed_feats["pubmed_total"])

print(f"PubMed features: {len(pubmed_feats)} drugs")
print(pubmed_feats.head(8))

# Start with target
master = target[["drug_name","avg_sentiment","review_count",
                  "pct_negative","pct_positive"]].copy()

# Left-join each feature table
for feats, name in [
    (price_feats,    "NADAC Pricing"),
    (shortage_feats, "FDA Shortages"),
    (adverse_feats,  "OpenFDA FAERS"),
    (trial_feats,    "ClinicalTrials"),
    (pubmed_feats,   "PubMed"),
]:
    before = len(master)
    master = master.merge(feats, on="drug_name", how="left")
    print(f"  {name:<20} joined — {master.shape[1]} cols total")

print(f"\nMaster matrix: {master.shape[0]} drugs × {master.shape[1]} columns")
print(f"Missing values per column:")
missing = master.isnull().sum()
print(missing[missing > 0])

# Fill missing with 0 (drug not in that source = no activity)
master = master.fillna(0)

# Save
master.to_csv(OUT/"drug_feature_matrix.csv", index=False)
print(f"\nSaved: data/master/drug_feature_matrix.csv")
master.head(5)

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.flatten()

feature_pairs = [
    ("adverse_serious_ratio",  "avg_sentiment", "H1: Adverse Events vs Satisfaction"),
    ("has_active_shortage",    "avg_sentiment", "H2: Active Shortage vs Satisfaction"),
    ("price_30day",            "avg_sentiment", "H3: Price vs Satisfaction"),
    ("trial_completion_rate",  "avg_sentiment", "H4: Trial Completion vs Satisfaction"),
    ("adverse_log_total",      "avg_sentiment", "Adverse Volume vs Sentiment"),
    ("shortage_total",         "avg_sentiment", "Shortage History vs Sentiment"),
    ("pubmed_log_total",       "avg_sentiment", "Research Volume vs Sentiment"),
    ("n_ndc_variants",         "avg_sentiment", "NDC Variants vs Sentiment"),
]

for ax, (x_col, y_col, title) in zip(axes, feature_pairs):
    if x_col not in master.columns:
        ax.set_visible(False)
        continue
    x = master[x_col]
    y = master[y_col]
    valid = ~(x.isna() | y.isna())
    if valid.sum() < 5:
        ax.set_visible(False)
        continue
    ax.scatter(x[valid], y[valid], alpha=0.7, color="#58a6ff", edgecolor="none", s=60)
    # Trend line
    if valid.sum() > 3:
        z = np.polyfit(x[valid], y[valid], 1)
        p = np.poly1d(z)
        ax.plot(sorted(x[valid]), p(sorted(x[valid])), "r--", alpha=0.8, lw=1.5)
    # Pearson r
    r, pval = stats.pearsonr(x[valid], y[valid])
    ax.set_title(f"{title}\nr={r:.3f}, p={pval:.3f}", fontsize=9)
    ax.set_xlabel(x_col, fontsize=8)
    ax.set_ylabel("Avg Sentiment", fontsize=8)

plt.suptitle("Feature vs Target Relationships", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(OUT/"feature_correlations.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: data/master/feature_correlations.png")

# Correlation matrix
feature_cols = [
    "price_30day","price_std","generic_available","brand_generic_ratio","n_ndc_variants",
    "shortage_total","shortage_active","has_active_shortage","shortage_pct_active",
    "adverse_log_total","adverse_serious_ratio","adverse_unique_types",
    "trial_log_total","trial_completion_rate","trial_late_phase_ratio",
    "pubmed_log_total","pubmed_recency_ratio",
]
feature_cols = [c for c in feature_cols if c in master.columns]

corr_with_target = (master[feature_cols + ["avg_sentiment"]]
                    .corr()["avg_sentiment"]
                    .drop("avg_sentiment")
                    .sort_values())

fig, ax = plt.subplots(figsize=(8, 9))
colors = ["#f85149" if v < 0 else "#3fb950" for v in corr_with_target]
ax.barh(corr_with_target.index, corr_with_target.values, color=colors)
ax.axvline(0, color="white", lw=0.8, ls="--")
ax.set_xlabel("Pearson Correlation with Avg Sentiment")
ax.set_title("Feature Correlations with Patient Satisfaction",
             fontweight="bold", fontsize=12)
plt.tight_layout()
plt.savefig(OUT/"correlation_bar.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nTop positive correlates with satisfaction:")
print(corr_with_target.tail(5))
print("\nTop negative correlates with satisfaction:")
print(corr_with_target.head(5))

X = master[feature_cols].values
y = master["avg_sentiment"].values

scaler = StandardScaler()
X_sc   = scaler.fit_transform(X)

models = {
    "Ridge Regression":       Ridge(alpha=1.0),
    "Linear Regression":      LinearRegression(),
    "Random Forest":          RandomForestRegressor(n_estimators=200, max_depth=4,
                                                     random_state=42),
    "Gradient Boosting":      GradientBoostingRegressor(n_estimators=100,
                                                          max_depth=2, random_state=42),
}

loo = LeaveOneOut()

print(f"{'Model':<25} {'LOO R²':>8} {'LOO MAE':>9} {'Notes'}")
print("-" * 60)

results = {}
for name, model in models.items():
    preds = cross_val_predict(model, X_sc, y, cv=loo)
    r2  = r2_score(y, preds)
    mae = mean_absolute_error(y, preds)
    results[name] = {"r2": r2, "mae": mae, "preds": preds, "model": model}
    note = "← best" if r2 == max(r["r2"] for r in results.values()) else ""
    print(f"  {name:<23} {r2:>8.4f} {mae:>9.4f}  {note}")

best_name  = max(results, key=lambda k: results[k]["r2"])
best_model = results[best_name]["model"]
best_model.fit(X_sc, y)
print(f"\nBest model: {best_name}")

# Predicted vs Actual plot
best_preds = results[best_name]["preds"]

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y, best_preds, color="#58a6ff", s=80, alpha=0.8, edgecolor="#30363d")

# Label each drug
for i, drug in enumerate(master["drug_name"]):
    ax.annotate(drug, (y[i], best_preds[i]), fontsize=7,
                alpha=0.7, xytext=(4, 4), textcoords="offset points")

# Perfect prediction line
mn, mx = min(y.min(), best_preds.min()), max(y.max(), best_preds.max())
ax.plot([mn, mx], [mn, mx], "r--", lw=1.5, label="Perfect prediction")
ax.set_xlabel("Actual Avg Sentiment (VADER)")
ax.set_ylabel("Predicted Avg Sentiment")
ax.set_title(f"Predicted vs Actual Patient Satisfaction\n"
             f"{best_name} — LOO R²={results[best_name]['r2']:.4f}",
             fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(OUT/"predicted_vs_actual.png", dpi=150)
plt.show()

# Use permutation importance (model-agnostic, works for all model types)
perm = permutation_importance(best_model, X_sc, y,
                               n_repeats=50, random_state=42)
imp_df = pd.DataFrame({
    "feature":    feature_cols,
    "importance": perm.importances_mean,
    "std":        perm.importances_std,
}).sort_values("importance", ascending=True)

# Clean feature names for display
name_map = {
    "price_30day":            "Drug Price (30-day)",
    "price_std":              "Price Variability",
    "generic_available":      "Generic Available",
    "brand_generic_ratio":    "Brand/Generic Price Gap",
    "n_ndc_variants":         "# Manufacturer Variants",
    "shortage_total":         "Total Shortages (historical)",
    "shortage_active":        "Active Shortage Count",
    "has_active_shortage":    "Currently in Shortage",
    "shortage_pct_active":    "Shortage Active Rate",
    "adverse_log_total":      "Adverse Event Volume",
    "adverse_serious_ratio":  "Serious Event Ratio",
    "adverse_unique_types":   "Adverse Event Diversity",
    "trial_log_total":        "Clinical Trial Volume",
    "trial_completion_rate":  "Trial Completion Rate",
    "trial_late_phase_ratio": "Phase 3/4 Trial Ratio",
    "pubmed_log_total":       "Research Publications",
    "pubmed_recency_ratio":   "Recent Publications Ratio",
}
imp_df["label"] = imp_df["feature"].map(name_map).fillna(imp_df["feature"])
imp_df["color"] = imp_df["importance"].apply(
    lambda v: "#3fb950" if v > 0 else "#f85149")

fig, ax = plt.subplots(figsize=(10, 9))
bars = ax.barh(imp_df["label"], imp_df["importance"],
               xerr=imp_df["std"], color=imp_df["color"],
               capsize=3, error_kw=dict(ecolor="#8b949e", lw=1))
ax.axvline(0, color="white", lw=0.8, ls="--")
ax.set_xlabel("Permutation Importance (impact on prediction accuracy)")
ax.set_title("Feature Importance — What Predicts Patient Drug Satisfaction?",
             fontweight="bold", fontsize=12)
ax.text(0.98, 0.02, f"Model: {best_name}\nLOO R²={results[best_name]['r2']:.3f}",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=9, color="grey")
plt.tight_layout()
plt.savefig(OUT/"feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: data/master/feature_importance.png")

print("\nTOP PREDICTORS OF PATIENT SATISFACTION:")
top5 = imp_df.nlargest(5, "importance")
for _, row in top5.iterrows():
    direction = "↑ higher = more satisfied" if row["importance"] > 0 else "↑ higher = less satisfied"
    print(f"  {row['label']:<35} importance={row['importance']:.4f}  ({direction})")

hypotheses = [
    ("H1", "adverse_serious_ratio",  "Adverse event rate",      "negative"),
    ("H2", "has_active_shortage",    "Active shortage",          "negative"),
    ("H3", "price_30day",            "Drug price",               "negative"),
    ("H4", "trial_completion_rate",  "Trial completion rate",    "positive"),
]

print(f"{'#':<4} {'Feature':<30} {'Expected':<10} {'r':>8} {'p-value':>10} {'Result'}")
print("-" * 75)

for hid, feat, label, expected_dir in hypotheses:
    if feat not in master.columns:
        print(f"  {hid}  {label:<30} FEATURE NOT FOUND")
        continue
    x = master[feat]
    y_s = master["avg_sentiment"]
    valid = ~(x.isna() | y_s.isna()) & (x != 0)
    if valid.sum() < 5:
        print(f"  {hid}  {label:<30} INSUFFICIENT DATA")
        continue
    r, pval = stats.pearsonr(x[valid], y_s[valid])
    sig   = "p<0.05" if pval < 0.05 else "p<0.10" if pval < 0.10 else "n.s."
    match = ((expected_dir=="negative" and r < 0) or
             (expected_dir=="positive" and r > 0))
    result = ("✓ SUPPORTED" if match and pval < 0.10 else
              "~ TREND"     if match else
              "✗ NOT SUPPORTED")
    print(f"  {hid:<4} {label:<30} {expected_dir:<10} {r:>8.4f} {pval:>10.4f}  "
          f"{sig}  {result}")

# Save predictions alongside drug names
preds_df = master[["drug_name","avg_sentiment","review_count"]].copy()
preds_df["predicted_sentiment"] = results[best_name]["preds"]
preds_df["residual"]            = preds_df["avg_sentiment"] - preds_df["predicted_sentiment"]
preds_df["satisfaction_tier"]   = pd.cut(preds_df["avg_sentiment"],
    bins=[-1,-0.1,0.05,1], labels=["Negative","Neutral","Positive"])
preds_df.to_csv(OUT/"drug_predictions.csv", index=False)

# Save feature importance
imp_df[["label","importance","std"]].to_csv(OUT/"feature_importance.csv", index=False)

print("Saved to data/master/:")
for f in sorted(OUT.glob("*")):
    print(f"  {f.name:<45} {f.stat().st_size/1024:.1f} KB")

print(f"\n{'='*60}")
print("PHASE 2B COMPLETE — Patient Satisfaction Predictor")
print(f"{'='*60}")
print(f"  Drugs analysed     : {len(master)}")
print(f"  Features used      : {len(feature_cols)}")
print(f"  Best model         : {best_name}")
print(f"  LOO R²             : {results[best_name]['r2']:.4f}")
print(f"  LOO MAE            : {results[best_name]['mae']:.4f}")
print(f"\nKey finding: See feature_importance.png for which drug")
print(f"characteristics most strongly predict patient satisfaction.")
