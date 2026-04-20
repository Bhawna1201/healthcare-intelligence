"""
dashboard.py — Healthcare Intelligence
Stevens Institute of Technology · BIA-660 Web Mining
Run: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from scipy import stats as scipy_stats
import re

st.set_page_config(
    page_title="Healthcare Intelligence · Stevens",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Source+Sans+3:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family:'Source Sans 3',sans-serif; color:#2d2d2d; }
h1,h2,h3 { font-family:'Playfair Display',serif; }

.stApp {
    background:#F9F9F9;
    background-image:
        radial-gradient(ellipse at 96% 4%, rgba(157,21,53,0.05) 0%, transparent 45%),
        radial-gradient(ellipse at 4% 96%, rgba(157,21,53,0.03) 0%, transparent 45%);
}

section[data-testid="stSidebar"] {
    background:linear-gradient(180deg,#9D1535 0%,#7A1028 100%) !important;
    border-right:none;
    box-shadow:4px 0 20px rgba(157,21,53,0.18);
}

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div { color:white !important; }
section[data-testid="stSidebar"] .stRadio label {
    line-height:1.4 !important;
    display:flex !important;
    align-items:center !important;
    gap:6px !important;
    color:white !important;
    font-size:1.08rem !important;
    font-weight:400 !important;
    padding:5px 2px;
    white-space:nowrap !important;
}          
section[data-testid="stSidebar"] hr { border-color:rgba(255,255,255,0.2) !important; }

[data-baseweb="select"]>div:first-child { background:white !important; border-color:#E0E0E0 !important; color:#2d2d2d !important; }
[data-baseweb="select"] div { color:#2d2d2d !important; }
[data-baseweb="menu"] li { color:#2d2d2d !important; background:white !important; }
[data-baseweb="input"] input { color:#2d2d2d !important; background:white !important; }

.stTabs [data-baseweb="tab-list"] { background:#F0F0F0; border-radius:8px; padding:4px; border:1px solid #E0E0E0; }
.stTabs [data-baseweb="tab"] { background:transparent !important; color:#949594 !important; border-radius:6px; font-weight:400; }
.stTabs [aria-selected="true"] { background:white !important; color:#9D1535 !important; font-weight:600; box-shadow:0 1px 4px rgba(0,0,0,0.1); }

.stSelectbox label,.stMultiSelect label,.stTextInput label,.stToggle label { color:#949594 !important; font-weight:400; }
[data-testid="stMarkdownContainer"] p,[data-testid="stMarkdownContainer"] li { color:#2d2d2d !important; }
.stAlert p { color:#2d2d2d !important; }
.js-plotly-plot { border-radius:12px; box-shadow:0 2px 12px rgba(0,0,0,0.06); }
.stCaption { color:#949594 !important; }
</style>
""", unsafe_allow_html=True)

# ── Design tokens ──────────────────────────────────────────────────────────────
MAR   = "#9D1535"   # Stevens Maroon
MAR_D = "#7A1028"   # Dark maroon
MAR_L = "#F7EEF1"   # Light maroon tint
GRAY  = "#949594"   # Stevens Gray
WHT   = "#FFFFFE"   # Stevens White

P_BLUE   = "#E8F0FE"
P_GREEN  = "#E6F4EA"
P_AMBER  = "#FEF9E7"
P_PURPLE = "#F3E8FF"
P_TEAL   = "#E8F5F5"

C_BLUE   = "#2563EB"
C_GREEN  = "#059669"
C_AMBER  = "#D97706"
C_PURPLE = "#7C3AED"
C_TEAL   = "#0891B2"

PT = dict(
    paper_bgcolor="white", plot_bgcolor="#FAFAFA",
    font=dict(color="#2d2d2d", family="Source Sans 3"),
    title_font=dict(family="Playfair Display", color="#2d2d2d", size=16),
    colorway=[MAR, C_BLUE, C_GREEN, C_AMBER, C_PURPLE, C_TEAL, "#DB2777", "#65A30D"],
)

# ── Helper components ──────────────────────────────────────────────────────────
def kpi(value, label, color=MAR, bg=MAR_L, sub=""):
    return (
        f"<div style='background:{bg};border-top:3px solid {color};"
        f"border-radius:0 0 10px 10px;padding:18px 16px;text-align:center;"
        f"box-shadow:0 2px 8px rgba(0,0,0,0.05)'>"
        f"<div style='font-family:Playfair Display,serif;font-size:2rem;"
        f"color:{color};font-weight:700;line-height:1'>{value}</div>"
        f"<div style='font-size:0.7rem;color:{GRAY};text-transform:uppercase;"
        f"letter-spacing:0.09em;margin-top:5px;font-weight:600'>{label}</div>"
        f"{'<div style=font-size:0.8rem;color:' + color + ';margin-top:3px>' + sub + '</div>' if sub else ''}"
        f"</div>")

def info_box(title, body, color=MAR, bg=MAR_L):
    return (
        f"<div style='background:{bg};border-left:4px solid {color};"
        f"border-radius:0 10px 10px 0;padding:14px 18px;margin:12px 0;"
        f"box-shadow:0 1px 6px rgba(0,0,0,0.05)'>"
        f"<div style='font-weight:700;color:{color};font-size:0.9rem;margin-bottom:6px'>{title}</div>"
        f"<div style='color:#2d2d2d;font-size:0.88rem;line-height:1.7'>{body}</div>"
        f"</div>")

def sec(text, sub=""):
    st.markdown(
        f"<div style='margin:28px 0 14px 0'>"
        f"<div style='font-family:Playfair Display,serif;font-size:1.35rem;"
        f"color:{MAR};font-weight:600;border-bottom:2px solid {MAR_L};"
        f"padding-bottom:8px'>{text}</div>"
        f"{'<div style=font-size:0.83rem;color:' + GRAY + ';margin-top:5px>' + sub + '</div>' if sub else ''}"
        f"</div>", unsafe_allow_html=True)

def narrative(text):
    st.markdown(
        f"<div style='background:white;border:1px solid #E9ECEF;"
        f"border-left:4px solid {MAR};border-radius:0 10px 10px 0;"
        f"padding:14px 18px;margin:10px 0;font-size:0.88rem;"
        f"color:#2d2d2d;line-height:1.75;box-shadow:0 1px 4px rgba(0,0,0,0.04)'>"
        f"📖 <i>{text}</i></div>", unsafe_allow_html=True)

# ── Data loading ──────────────────────────────────────────────────────────────
BASE  = Path(__file__).parent
RAW   = BASE / "data" / "raw"
PROC  = BASE / "data" / "processed"
MAST  = BASE / "data" / "master"

TARGET = [
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

@st.cache_data
def load(proc, raw):
    p = proc if (proc and Path(proc).exists()) else raw
    if p and Path(p).exists():
        df = pd.read_csv(p, on_bad_lines="skip", engine="python", encoding_errors="replace")
        return df.replace({"None": np.nan, "none": np.nan, "nan": np.nan})
    return pd.DataFrame()

@st.cache_data
def load_all():
    return {
        "reviews":   load(PROC/"reviews_clean.csv",  RAW/"reviews.csv"),
        "prices":    load(PROC/"prices_clean.csv",   RAW/"prices.csv"),
        "trials":    load(PROC/"trials_clean.csv",   RAW/"trials.csv"),
        "shortages": load(None,                      RAW/"shortages.csv"),
        "adverse":   load(None,                      RAW/"adverse_events.csv"),
        "pubmed":    load(PROC/"pubmed_clean.csv",   RAW/"pubmed_abstracts.csv"),
    }

data = load_all()

def dc(df):
    return "drug_name_clean" if "drug_name_clean" in df.columns else "drug_name"

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:18px 0 10px'>
        <div style='font-family:Playfair Display,serif;font-size:1.5rem;
                    font-weight:700;color:white;letter-spacing:2px'>STEVENS</div>
        <div style='font-size:0.62rem;letter-spacing:3px;color:rgba(255,255,255,0.65);
                    text-transform:uppercase;margin-top:2px'>Institute of Technology</div>
        <div style='width:36px;height:2px;background:rgba(255,255,255,0.35);margin:10px auto'></div>
        <div style='font-size:0.7rem;color:rgba(255,255,255,0.65)'>BIA-660 · Web Mining</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("Navigate", [
        "📊 Overview",
        "😊 Sentiment Analysis",
        "💰 Drug Pricing",
        "⚠️ Drug Shortages",
        "🔬 Clinical Trials",
        "🧬 Adverse Events",
        "📚 PubMed Research",
        "🎯 Satisfaction Predictor",
        "🔮 Shortage & Rating Prediction",
    ], label_visibility="collapsed")
    st.markdown("---")
    sel = st.multiselect("Filter by drug", sorted(TARGET),
                         default=[], placeholder="All 50 drugs")
    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.7rem;color:rgba(255,255,255,0.55);line-height:1.9'>
        Phase 1 + Phase 2 complete<br>
        131,843 records · 6 sources<br>
        50 drugs · 6,827 reviews
    </div>""", unsafe_allow_html=True)

# def filt(df, col=None):
#     if not sel or df.empty: return df
#     c = col or dc(df)
#     if c not in df.columns: return df
#     return df[df[c].str.lower().isin([d.lower() for d in sel])]
def filt(df, col=None):
    if not sel or df.empty: return df
    c = col or dc(df)
    if c not in df.columns:
        # try fallback column
        for alt in ["drug_name", "drug_name_clean"]:
            if alt in df.columns:
                c = alt
                break
        else:
            return df
    sel_lower = [d.lower() for d in sel]
    mask = df[c].str.lower().str.strip().isin(sel_lower)
    filtered = df[mask]
    # If nothing matched, try fuzzy substring match
    if filtered.empty and sel:
        mask2 = df[c].str.lower().apply(
            lambda x: any(s in str(x) for s in sel_lower))
        filtered = df[mask2]
    return filtered
# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":

    # Hero
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,{MAR} 0%,{MAR_D} 100%);
                border-radius:16px;padding:32px 36px;margin-bottom:24px;
                position:relative;overflow:hidden;box-shadow:0 4px 20px rgba(157,21,53,0.25)'>
        <div style='position:absolute;right:-30px;top:-30px;width:220px;height:220px;
                    background:rgba(255,255,255,0.05);border-radius:50%'></div>
        <div style='position:absolute;right:60px;bottom:-50px;width:160px;height:160px;
                    background:rgba(255,255,255,0.04);border-radius:50%'></div>
        <div style='font-family:Playfair Display,serif;font-size:2.3rem;
                    color:white;font-weight:700;line-height:1.2'>
            Healthcare Intelligence</div>
        <div style='color:rgba(255,255,255,0.8);font-size:1rem;margin-top:8px'>
            Comprehensive drug intelligence platform across 6 data sources</div>
        <div style='display:flex;gap:28px;margin-top:18px;flex-wrap:wrap'>
            <div style='color:rgba(255,255,255,0.7);font-size:0.82rem'>🏫 Stevens Institute of Technology</div>
            <div style='color:rgba(255,255,255,0.7);font-size:0.82rem'>📚 BIA-660 Web Mining</div>
            <div style='color:rgba(255,255,255,0.7);font-size:0.82rem'>💊 50 Target Drugs · 131,843 Records</div>
        </div>
    </div>""", unsafe_allow_html=True)

    # 6 KPI cards
    kpi_cfg = [
        ("💬","Patient Reviews",  len(data["reviews"]),  "WebMD via Selenium scraping",      C_BLUE,  P_BLUE),
        ("💰","Drug Prices",      len(data["prices"]),   "NADAC/CMS Socrata API",            C_GREEN, P_GREEN),
        ("🧪","Clinical Trials",  len(data["trials"]),   "ClinicalTrials.gov REST API v2",   C_AMBER, P_AMBER),
        ("⚠️","Drug Shortages",  len(data["shortages"]),"OpenFDA shortage API",             MAR,     MAR_L),
        ("🚨","Adverse Events",   len(data["adverse"]),  "OpenFDA FAERS API",                C_PURPLE,P_PURPLE),
        ("📄","PubMed Abstracts", len(data["pubmed"]),   "NCBI Entrez E-utilities",          C_TEAL,  P_TEAL),
    ]
    cols = st.columns(6)
    for col, (icon, label, count, source, color, bg) in zip(cols, kpi_cfg):
        with col:
            st.markdown(
                f"<div style='background:{bg};border-top:3px solid {color};"
                f"border-radius:0 0 10px 10px;padding:16px 10px;text-align:center;"
                f"box-shadow:0 2px 8px rgba(0,0,0,0.05)'>"
                f"<div style='font-size:1.4rem'>{icon}</div>"
                f"<div style='font-family:Playfair Display,serif;font-size:1.9rem;"
                f"color:{color};font-weight:700;line-height:1'>{count:,}</div>"
                f"<div style='font-size:0.67rem;color:{GRAY};text-transform:uppercase;"
                f"letter-spacing:0.07em;margin-top:4px;font-weight:600'>{label}</div>"
                f"<div style='font-size:0.7rem;color:{color};margin-top:3px'>{source}</div>"
                f"</div>", unsafe_allow_html=True)

    # Narrative
    narrative("We scraped 131,843 records from 6 authoritative sources using Selenium, "
              "REST APIs, and NCBI E-utilities. We then applied VADER sentiment analysis, "
              "TF-IDF classification (Logistic Regression, 76% accuracy, 0.863 AUC), LDA "
              "topic modeling (8 topics), K-Means clustering (5 clusters), and Word2Vec "
              "embeddings to answer one unified research question: what predicts patient satisfaction?")

    # Architecture
    sec("Data Architecture & Pipeline")
    st.markdown(f"""
    <div style='background:white;border:1px solid #E9ECEF;border-radius:12px;
                padding:24px;box-shadow:0 2px 10px rgba(0,0,0,0.05)'>
        <div style='display:grid;grid-template-columns:repeat(5,1fr);gap:12px;align-items:center'>
            <!-- Sources -->
            <div>
                <div style='font-family:Playfair Display,serif;font-size:0.8rem;
                            color:{GRAY};text-transform:uppercase;letter-spacing:0.08em;
                            margin-bottom:8px'>6 Sources</div>
                {''.join([f"<div style='background:{bg};border-left:3px solid {c};padding:5px 8px;border-radius:0 5px 5px 0;margin:3px 0;font-size:0.76rem;color:#2d2d2d'>{icon} {label}</div>"
                          for icon,label,_,_,c,bg in kpi_cfg])}
            </div>
            <!-- Arrow -->
            <div style='text-align:center;font-size:1.5rem;color:{GRAY}'>→</div>
            <!-- Phase 1 -->
            <div style='background:{MAR_L};border:1px solid {MAR}22;border-radius:10px;padding:14px;text-align:center'>
                <div style='font-family:Playfair Display,serif;color:{MAR};font-weight:700;font-size:0.9rem'>Phase 1</div>
                <div style='font-size:0.75rem;color:{GRAY};margin-top:4px'>Web Scraping<br>& Data Collection</div>
                <div style='font-size:0.72rem;color:#2d2d2d;margin-top:6px;line-height:1.5'>6 CSV files<br>131,843 records</div>
            </div>
            <!-- Arrow -->
            <div style='text-align:center;font-size:1.5rem;color:{GRAY}'>→</div>
            <!-- Phase 2 -->
            <div style='background:{P_BLUE};border:1px solid {C_BLUE}22;border-radius:10px;padding:14px;text-align:center'>
                <div style='font-family:Playfair Display,serif;color:{C_BLUE};font-weight:700;font-size:0.9rem'>Phase 2</div>
                <div style='font-size:0.75rem;color:{GRAY};margin-top:4px'>NLP Pipeline<br>& ML Models</div>
                <div style='font-size:0.72rem;color:#2d2d2d;margin-top:6px;line-height:1.5'>Sentiment · Topics<br>Clustering · Prediction</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    # ML Models
    sec("Machine Learning Models Used")
    m1, m2, m3 = st.columns(3)
    ml_groups = [
        (m1, C_BLUE, P_BLUE, "🤖 Supervised Learning",
         "Classifies patient sentiment from review text",
         [("Logistic Regression ✅", "Best model — 76% accuracy, 0.863 AUC. Predicts positive vs negative sentiment from TF-IDF features"),
          ("Linear SVM", "75.5% accuracy. Finds optimal hyperplane separating positive/negative reviews"),
          ("Random Forest", "75.0% accuracy. Ensemble of 100 decision trees on review text"),
          ("Naive Bayes", "70.4% accuracy. Probabilistic baseline — assumes word independence")]),
        (m2, C_GREEN, P_GREEN, "🗂️ Unsupervised Learning",
         "Discovers hidden patterns without labels",
         [("LDA Topic Model", "8 hidden topics in 3,068 reviews — Side Effects, Dosage, Effectiveness, Blood Pressure, Pain, Mental Health, Weight, General"),
          ("K-Means (5 clusters)", "Groups 50 drugs by adverse event profile. Semaglutide isolated — unique GLP-1 mechanism confirmed by data"),
          ("t-SNE Visualization", "2D projection of drug clusters. Confirms clinically meaningful groupings in embedding space")]),
        (m3, C_AMBER, P_AMBER, "🧬 Advanced NLP",
         "Deep language representations of drug text",
         [("VADER Sentiment", "Rule-based scorer for review text. Returns compound score −1 to +1. Trained for social/review language"),
          ("TextBlob", "Polarity + subjectivity scoring. Confirms VADER labels"),
          ("Word2Vec (skip-gram)", "100-dim embeddings on 3,068 reviews + PubMed. Learns: stomach → omeprazole, reflux, acid"),
          ("TF-IDF Vectorizer", "5,000 features, unigrams + bigrams. Input to all classifiers")]),
    ]
    for container, color, bg, title, sub_text, models in ml_groups:
        with container:
            st.markdown(
                f"<div style='background:{bg};border-top:3px solid {color};"
                f"border-radius:0 0 10px 10px;padding:16px;height:100%'>"
                f"<div style='font-family:Playfair Display,serif;font-size:1rem;"
                f"color:{color};font-weight:700'>{title}</div>"
                f"<div style='font-size:0.76rem;color:{GRAY};margin-bottom:10px'>{sub_text}</div>"
                + "".join([
                    f"<div style='background:white;border-radius:6px;padding:8px 10px;"
                    f"margin:5px 0;border-left:3px solid {color}'>"
                    f"<div style='font-size:0.8rem;font-weight:600;color:#2d2d2d'>{name}</div>"
                    f"<div style='font-size:0.74rem;color:{GRAY};margin-top:2px;line-height:1.4'>{desc}</div>"
                    f"</div>" for name, desc in models
                ]) + "</div>", unsafe_allow_html=True)

    # Coverage chart
    sec("Record Coverage by Drug",
        "How many records each drug has across all 6 sources")
    coverage = {}
    col_order  = ["reviews","prices","trials","shortages","adverse","pubmed"]
    col_colors = {"reviews":C_BLUE,"prices":C_GREEN,"trials":C_AMBER,
                  "shortages":MAR,"adverse":C_PURPLE,"pubmed":C_TEAL}
    col_labels = {"reviews":"Reviews","prices":"Prices","trials":"Trials",
                  "shortages":"Shortages","adverse":"Adverse","pubmed":"PubMed"}
    for name, df in data.items():
        if df.empty: continue
        c = dc(df)
        if c not in df.columns: continue
        for drug in TARGET:
            coverage.setdefault(drug, {})
            if name == "shortages":
            # substring match — shortage names are full descriptions
                count = int(df[c].str.lower().str.contains(drug, na=False).sum())
            else:
                counts = df[c].str.lower().value_counts()
                count = int(counts.get(drug, 0))
            coverage[drug][name] = count
        

    if coverage:
        cov_df = pd.DataFrame(coverage).T.fillna(0)
        cov_df = cov_df[[c for c in col_order if c in cov_df.columns]]
        cov_s  = cov_df.copy()
        cov_s["total"] = cov_s.sum(axis=1)
        cov_s = cov_s.sort_values("total", ascending=True).drop(columns="total")
        
        
        # use_log = st.toggle("Log scale (balances 86K prices vs 3K reviews)", value=True)
        # fig = go.Figure()
        # for c in [c for c in col_order if c in cov_s.columns]:
        #     fig.add_trace(go.Bar(
        #         name=col_labels[c], y=cov_s.index.tolist(), x=cov_s[c].tolist(),
        #         orientation="h", marker_color=col_colors[c],
        #         hovertemplate="<b>%{y}</b><br>" + col_labels[c] + ": %{x:,}<extra></extra>"))
        # fig.update_layout(**PT, barmode="stack", height=700,
        #     title=dict(text="Records per Drug across All 6 Sources"),
        #     xaxis_title="Records (log scale)" if use_log else "Records",
        #     xaxis_type="log" if use_log else "linear", yaxis_title="",
        #     legend=dict(orientation="h", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"),
        #     margin=dict(l=160,r=20,t=70,b=40))
        # st.plotly_chart(fig, use_container_width=True)
        # st.caption("Prices dominate on linear scale (86K NADAC records). Log scale reveals review and trial coverage.")

                    # ── Top: Records per source (6 bars) ──────────────────────────────────────
        source_totals = pd.DataFrame([
            {"Source": "Reviews (WebMD)",      "Records": len(data["reviews"]),   "color": C_BLUE},
            {"Source": "Prices (NADAC)",       "Records": len(data["prices"]),    "color": C_GREEN},
            {"Source": "Trials (CT.gov)",      "Records": len(data["trials"]),    "color": C_AMBER},
            {"Source": "Shortages (FDA)",      "Records": len(data["shortages"]), "color": MAR},
            {"Source": "Adverse Events (FDA)", "Records": len(data["adverse"]),   "color": C_PURPLE},
            {"Source": "PubMed (NCBI)",        "Records": len(data["pubmed"]),    "color": C_TEAL},
        ])
        fig_src = go.Figure(go.Bar(
            x=source_totals["Source"], y=source_totals["Records"],
            marker_color=source_totals["color"].tolist(),
            text=[f"{r:,}" for r in source_totals["Records"]],
            textposition="outside", textfont=dict(size=11),
            hovertemplate="<b>%{x}</b><br>%{y:,} records<extra></extra>",
        ))
        fig_src.update_layout(**PT, height=320,
            title=dict(text="Total Records per Source"),
            xaxis_title="", yaxis_title="Records",
            margin=dict(l=40,r=40,t=50,b=40),
            showlegend=False)
        st.plotly_chart(fig_src, use_container_width=True)

        # ── Bottom: Heatmap 50 drugs × 6 sources ─────────────────────────────────
        st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)
        text_vals, z_vals = [], []
        src_order = ["reviews","prices","trials","shortages","adverse","pubmed"]
        src_labels = ["Reviews","Prices","Trials","Shortages","Adverse","PubMed"]
        for drug in sorted(TARGET):
            row_text, row_z = [], []
            for src in src_order:
                count = coverage.get(drug, {}).get(src, 0)
                row_text.append(f"{int(count):,}" if count > 0 else "—")
                #row_z.append(min(1.0, count / 500) if count > 0 else 0)
                row_z.append(0.3 if count > 0 else 0)  # binary: has data or not
            text_vals.append(row_text)
            z_vals.append(row_z)

        fig_hm = go.Figure(go.Heatmap(
            z=z_vals,
            x=src_labels,
            y=sorted(TARGET),
            text=text_vals,
            texttemplate="%{text}",
            textfont={"size": 9},
            colorscale=[[0,"#F9F9F9"],[0.01,"#F7EEF1"],[1,MAR]],
            showscale=False,
            hovertemplate="<b>%{y}</b> — %{x}<br>%{text} records<extra></extra>",
        ))
        fig_hm.update_layout(**PT, height=1100,
            title=dict(text="Record Coverage: All 50 Drugs × 6 Sources"),
            xaxis=dict(side="top", tickfont=dict(size=12, color="#2d2d2d")),
            yaxis=dict(tickfont=dict(size=11)),
            margin=dict(l=160,r=20,t=80,b=20))
        st.plotly_chart(fig_hm, use_container_width=True)
        st.caption("Darker = more records. '—' = no data for that drug in that source.")
    # Key findings
    sec("3 Key Findings")
    f1, f2, f3 = st.columns(3)
    findings = [
        (f1, MAR,     MAR_L,   "56% of Reviews Negative",
         "Patients are 1.5× more likely to write a review when experiencing "
         "problems than when satisfied — consistent with negativity bias in "
         "online health reviews across pharmacovigilance literature."),
        (f2, C_BLUE,  P_BLUE,  "Losartan: Lowest Satisfaction (−0.673)",
         "Despite being cheap ($1.42/30-day) and widely available with zero "
         "shortage records, losartan has the worst patient sentiment of all "
         "50 drugs — suggesting pharmacological side effects drive experience "
         "more than structural factors."),
        (f3, C_GREEN, P_GREEN, "Null Finding — Structurally Meaningful",
         "No structural drug characteristic (price, shortage, adverse burden, "
         "trial activity) significantly predicts satisfaction (all p > 0.10, "
         "n=50). Patient satisfaction is pharmacologically driven, "
         "not structurally predictable."),
    ]
    for container, color, bg, title, body in findings:
        with container:
            st.markdown(
                f"<div style='background:{bg};border-top:3px solid {color};"
                f"border-radius:0 0 10px 10px;padding:16px;height:100%;"
                f"box-shadow:0 2px 8px rgba(0,0,0,0.05)'>"
                f"<div style='font-family:Playfair Display,serif;color:{color};"
                f"font-weight:700;font-size:0.95rem;margin-bottom:8px'>{title}</div>"
                f"<div style='font-size:0.83rem;color:#2d2d2d;line-height:1.65'>{body}</div>"
                f"</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — SENTIMENT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "😊 Sentiment Analysis":
    st.markdown(f"<h1 style='font-family:Playfair Display,serif;color:{MAR}'>"
                "Patient Sentiment Analysis</h1>", unsafe_allow_html=True)

    narrative("We scraped 6,827 patient reviews from WebMD using Selenium + Chrome (headless) "
              "because WebMD renders reviews with JavaScript — plain HTTP requests get blocked. "
              "Each review was scored using VADER (Valence Aware Dictionary and sEntiment "
              "Reasoner) returning a compound score from −1 (most negative) to +1 (most positive). "
              "TextBlob provided polarity confirmation. Finding: 56% of reviews are negative — "
              "patients are 1.5× more likely to review when experiencing problems.")

    df = filt(data["reviews"])
    if df.empty: st.warning("No review data."); st.stop()

    if "vader_compound" not in df.columns:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        df = df.copy()
        df["vader_compound"] = df["review_text"].apply(
            lambda t: sia.polarity_scores(str(t))["compound"] if pd.notna(t) else 0)
    if "sentiment_label" not in df.columns:
        df["sentiment_label"] = df["vader_compound"].apply(
            lambda v: "positive" if v>=0.05 else "negative" if v<=-0.05 else "neutral")
    for c in ["condition","date","review_text"]:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str).replace({"nan":""})

    dfc = dc(df)
    pct_neg = (df["sentiment_label"]=="negative").mean()*100
    mean_v  = df["vader_compound"].mean()

    # KPIs
    k1,k2,k3 = st.columns(3)
    with k1: st.markdown(kpi(f"{len(df):,}","Total Reviews Scored"), unsafe_allow_html=True)
    with k2: st.markdown(kpi(f"{pct_neg:.1f}%","Negative Reviews",MAR,MAR_L), unsafe_allow_html=True)
    with k3: st.markdown(kpi(f"{mean_v:.3f}","Mean VADER Score",C_AMBER,P_AMBER), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    tab1,tab2,tab3 = st.tabs(["📊 Sentiment by Drug","📈 Distribution & Conditions","🔍 Review Explorer"])

    with tab1:
        drug_s = (df.groupby(dfc)
                  .agg(avg=("vader_compound","mean"), n=("vader_compound","count"))
                  .query("n>=5").sort_values("avg").reset_index())

        bar_c = [MAR if v<-0.10 else C_AMBER if v<0.05 else C_GREEN
                 for v in drug_s["avg"]]
        fig = go.Figure(go.Bar(
            x=drug_s["avg"], y=drug_s[dfc], orientation="h",
            marker_color=bar_c,
            text=[f"n={int(r)}" for r in drug_s["n"]],
            textposition="outside", textfont=dict(color=GRAY, size=9),
            hovertemplate="<b>%{y}</b><br>VADER: %{x:.3f}<extra></extra>"))
        fig.add_vline(x=0,    line_dash="dash",  line_color="#CCC",  line_width=1)
        fig.add_vline(x=0.05, line_dash="dot",   line_color=C_GREEN, line_width=1,
                      annotation_text="Positive threshold",
                      annotation_font_color=C_GREEN, annotation_position="top right")
        fig.add_vline(x=-0.05,line_dash="dot",   line_color=MAR,     line_width=1,
                      annotation_text="Negative threshold",
                      annotation_font_color=MAR, annotation_position="bottom right")
        fig.update_layout(**PT, height=max(600, len(drug_s)*24),
            title=dict(text="Patient Sentiment Score by Drug — VADER Compound Score"),
            xaxis=dict(title="VADER Compound Score (−1 = very negative, +1 = very positive)",
                       range=[-0.85,0.5]),
            yaxis_title="", margin=dict(l=160,r=120,t=60,b=40))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(info_box("How to read this chart",
            "VADER compound score ranges from −1 (extremely negative) to +1 (extremely positive). "
            "Scores below −0.05 are negative, above +0.05 are positive, between is neutral. "
            "n= shows how many reviews were scored for each drug. Bars colored: "
            "<span style='color:#9D1535'>■ red = negative</span> · "
            "<span style='color:#D97706'>■ amber = neutral</span> · "
            "<span style='color:#059669'>■ green = positive</span>",
            C_BLUE, P_BLUE), unsafe_allow_html=True)

        # Top 5 best/worst
        c1,c2 = st.columns(2)
        with c1:
            st.markdown(f"<div style='font-family:Playfair Display,serif;color:{C_GREEN};"
                        f"font-weight:600;margin:12px 0 8px'>🟢 Highest Satisfaction</div>",
                        unsafe_allow_html=True)
            for _, row in drug_s.nlargest(5,"avg").iterrows():
                st.markdown(info_box(
                    f"{row[dfc]} — {row['avg']:.3f}",
                    f"{int(row['n'])} reviews scored · "
                    f"{'Positive sentiment' if row['avg']>=0.05 else 'Near neutral'}",
                    C_GREEN, P_GREEN), unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div style='font-family:Playfair Display,serif;color:{MAR};"
                        f"font-weight:600;margin:12px 0 8px'>🔴 Lowest Satisfaction</div>",
                        unsafe_allow_html=True)
            for _, row in drug_s.nsmallest(5,"avg").iterrows():
                st.markdown(info_box(
                    f"{row[dfc]} — {row['avg']:.3f}",
                    f"{int(row['n'])} reviews scored · Strong negative sentiment",
                    MAR, MAR_L), unsafe_allow_html=True)

    with tab2:
        c1,c2 = st.columns(2)
        with c1:
            counts = df["sentiment_label"].value_counts()
            fig = go.Figure(go.Pie(
                labels=counts.index, values=counts.values, hole=0.52,
                marker_colors=[MAR, C_AMBER, C_GREEN]))
            fig.update_layout(**PT, height=320,
                title=dict(text="Overall Sentiment Split"),
                margin=dict(l=20,r=20,t=50,b=20),
                legend=dict(font=dict(color="#2d2d2d")))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.histogram(df, x="vader_compound", nbins=50,
                color_discrete_sequence=[MAR], title="VADER Score Distribution")
            fig.add_vline(x=0.05, line_dash="dot", line_color=C_GREEN,
                annotation_text="Positive", annotation_font_color=C_GREEN)
            fig.add_vline(x=-0.05,line_dash="dot", line_color=MAR,
                annotation_text="Negative", annotation_font_color=MAR)
            fig.update_layout(**PT, height=320,
                xaxis_title="VADER Compound Score", yaxis_title="Number of Reviews",
                margin=dict(l=20,r=20,t=50,b=20))
            st.plotly_chart(fig, use_container_width=True)

        # Condition breakdown
        if "condition" in df.columns:
            sec("Sentiment by Medical Condition",
                "Which conditions have the most negative patient sentiment?")
            cond_df = df[df["condition"].str.len()>2].copy()
            cond_s = (cond_df.groupby("condition")
                      .agg(avg=("vader_compound","mean"), n=("vader_compound","count"))
                      .query("n>=10").sort_values("avg").head(20).reset_index())
            if not cond_s.empty:
                fig = px.bar(cond_s, x="avg", y="condition", orientation="h",
                    color="avg",
                    color_continuous_scale=[[0,MAR],[0.5,C_AMBER],[1,C_GREEN]],
                    title="Avg Sentiment by Condition (min 10 reviews)",
                    labels={"avg":"Avg VADER Score","condition":"Condition"})
                fig.update_layout(**PT, height=max(400, len(cond_s)*24),
                    xaxis_title="Avg VADER Score", yaxis_title="",
                    coloraxis_showscale=False,
                    margin=dict(l=220,r=20,t=50,b=40))
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Conditions with more negative scores may reflect drugs that are less effective "
                           "or have more side effects for that indication.")

    with tab3:
        sec("Review Explorer", "Browse actual patient reviews with sentiment labels")
        drug_opts = sorted(df[dfc].dropna().unique().tolist())
        ca,cb,cc = st.columns([2,1,1])
        with ca: sel_drug = st.selectbox("Drug", drug_opts, key="rev_drug")
        with cb: sf = st.selectbox("Sentiment", ["all","positive","neutral","negative"], key="rev_sent")
        with cc: max_show = st.selectbox("Show", [10,25,50], key="rev_n")

        view = df[df[dfc]==sel_drug].copy()
        if sf != "all": view = view[view["sentiment_label"]==sf]
        st.markdown(f"<span style='color:{GRAY}'>{len(view)} reviews found</span>",
                    unsafe_allow_html=True)
        for _, row in view.head(max_show).iterrows():
            score = row.get("vader_compound",0)
            label = row.get("sentiment_label","")
            color = C_GREEN if label=="positive" else MAR if label=="negative" else C_AMBER
            bg    = P_GREEN if label=="positive" else MAR_L if label=="negative" else P_AMBER
            cond  = str(row.get("condition","")).strip()
            date  = str(row.get("date","")).strip()
            meta  = " · ".join(filter(lambda x: x and x!="nan", [cond,date]))
            text  = str(row.get("review_text",""))
            st.markdown(
                f"<div style='background:{bg};border-left:3px solid {color};"
                f"border-radius:0 8px 8px 0;padding:12px 16px;margin:6px 0;"
                f"box-shadow:0 1px 4px rgba(0,0,0,0.04)'>"
                f"<div style='display:flex;justify-content:space-between;margin-bottom:5px'>"
                f"<span style='color:{color};font-weight:700;font-size:0.82rem'>"
                f"{label.upper()}  VADER: {score:+.3f}</span>"
                f"<span style='color:{GRAY};font-size:0.78rem'>{meta}</span></div>"
                f"<div style='color:#2d2d2d;font-size:0.88rem;line-height:1.55'>"
                f"{text[:450]}{'...' if len(text)>450 else ''}</div></div>",
                unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — DRUG PRICING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💰 Drug Pricing":
    st.markdown(f"<h1 style='font-family:Playfair Display,serif;color:{MAR}'>"
                "Drug Pricing Intelligence</h1>", unsafe_allow_html=True)

    narrative("We queried the NADAC (National Average Drug Acquisition Cost) database — "
              "a US government dataset published every Wednesday by CMS. We used the Socrata "
              "Open Data API with LIKE queries on NDC descriptions, paginated at 500 records per call. "
              "NADAC captures what pharmacies actually pay wholesalers — not retail price. "
              "Finding: Generic Metformin costs $0.021/tablet while brand Glucophage costs "
              "$1.945/tablet — a 106× premium for the same molecule.")

    df = filt(data["prices"])
    if df.empty: st.warning("No pricing data."); st.stop()
    dfc = dc(df)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df[df["price"].between(0.0001, 9999)].dropna(subset=["price"])
    if "pharmacy" in df.columns:
        df["type"] = df["pharmacy"].str.contains("Brand",case=False,na=False).map(
            {True:"Brand",False:"Generic"})
    else:
        df["type"] = "Generic"

    gen = df[df["type"]=="Generic"]
    cheapest_drug = gen.groupby(dfc)["price"].median().idxmin() if not gen.empty else "N/A"
    cheapest_val  = gen.groupby(dfc)["price"].median().min()*30 if not gen.empty else 0
    expensive_drug = df.groupby(dfc)["price"].median().idxmax() if not df.empty else "N/A"
    expensive_val  = df.groupby(dfc)["price"].median().max()*30 if not df.empty else 0

    k1,k2,k3 = st.columns(3)
    with k1: st.markdown(kpi(f"{len(df):,}","Price Records","","#E6F4EA",""), unsafe_allow_html=True)
    with k1: st.markdown(kpi(f"{len(df):,}","NADAC Records",C_GREEN,P_GREEN), unsafe_allow_html=True)
    with k2: st.markdown(kpi(f"${cheapest_val:.2f}",f"Cheapest 30-day ({cheapest_drug})",C_GREEN,P_GREEN), unsafe_allow_html=True)
    with k3: st.markdown(kpi(f"${expensive_val:.0f}",f"Most Expensive 30-day",MAR,MAR_L), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    tab1,tab2,tab3 = st.tabs(["📊 30-Day Cost",
                                    "💊 Dosage Form","🔍 Drug Detail"])

    with tab1:
        dp = (gen.groupby(dfc)["price"].median().reset_index()
              .sort_values("price", ascending=True))
        dp["cost_30"] = dp["price"]*30
        use_log = st.toggle("Log scale",value=False,key="p_log")
        fig = go.Figure(go.Bar(
            y=dp[dfc], x=dp["cost_30"], orientation="h",
            marker_color=C_GREEN,
            text=["$"+f"{v:.2f}" for v in dp["cost_30"]],
            textposition="outside", textfont=dict(color=GRAY,size=9),
            hovertemplate="<b>%{y}</b><br>30-day cost: $%{x:.2f}<extra></extra>"))
        fig.update_layout(**PT, height=max(600,len(dp)*22),
            title=dict(text="Estimated 30-Day Generic Drug Cost (Pharmacy Acquisition Price)"),
            xaxis_title="30-Day Cost ($) — pharmacy acquisition, not retail",
            xaxis_type="log" if use_log else "linear",
            yaxis_title="", margin=dict(l=160,r=120,t=60,b=40))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(info_box("What is NADAC?",
            "NADAC = National Average Drug Acquisition Cost. This is what pharmacies pay "
            "wholesalers before adding markup. Retail prices (what patients pay) are typically "
            "20–40% higher before insurance. Updated every Wednesday by CMS.",
            C_BLUE, P_BLUE), unsafe_allow_html=True)

    
        

    with tab2:
        if "dosage_form" in df.columns:
            form_prices = (gen.groupby("dosage_form")["price"]
                           .agg(median="median",count="count")
                           .query("count>=5")
                           .reset_index()
                           .sort_values("median",ascending=False))
            form_prices["cost_30"] = form_prices["median"]*30
            fig = px.bar(form_prices, x="dosage_form", y="cost_30",
                color="cost_30", color_continuous_scale=[[0,P_GREEN],[1,MAR]],
                title="Average 30-Day Cost by Dosage Form",
                labels={"dosage_form":"Dosage Form","cost_30":"Avg 30-Day Cost ($)"})
            fig.update_layout(**PT, height=400, coloraxis_showscale=False,
                xaxis_tickangle=-30, margin=dict(l=40,r=20,t=50,b=100))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(info_box("Why do forms cost differently?",
                "Injectable drugs require sterile manufacturing, cold chain logistics, and "
                "specialized equipment — driving much higher costs than tablets. Solutions and "
                "capsules fall between. This explains why insulin (injection) costs 100× more "
                "per dose than metformin (tablet) even after controlling for molecule complexity.",
                C_AMBER, P_AMBER), unsafe_allow_html=True)
        else:
            st.info("Dosage form data not available in current dataset.")

    with tab3:
        drug_opts = sorted(df[dfc].dropna().unique())
        sel_d = st.selectbox("Select drug", drug_opts, key="price_drug")
        ddf = df[df[dfc]==sel_d]
        gm = ddf[ddf["type"]=="Generic"]["price"].median()
        gn = ddf[ddf["type"]=="Generic"]["price"].min()
        bm = ddf[ddf["type"]=="Brand"]["price"].median()
        k1,k2,k3 = st.columns(3)
        with k1: st.markdown(kpi(f"${gm*30:.2f}" if pd.notna(gm) else "N/A","Generic 30-day",C_GREEN,P_GREEN),unsafe_allow_html=True)
        with k2: st.markdown(kpi(f"${gn*30:.2f}" if pd.notna(gn) else "N/A","Cheapest Option",C_BLUE,P_BLUE),unsafe_allow_html=True)
        #with k3: st.markdown(kpi(f"${bm*30:.2f}" if pd.notna(bm) else "N/A","Brand 30-day",MAR,MAR_L),unsafe_allow_html=True)
        with k3: st.markdown(kpi(f"${(gm*30)*12:.0f}","Annual Cost (generic)",MAR,MAR_L),unsafe_allow_html=True)
        st.markdown("")
        gen_data = ddf[ddf["type"]=="Generic"]["price"]*30
        if not gen_data.empty:
            stats_df = pd.DataFrame({
                "Metric": ["Cheapest Manufacturer","Median Price","Most Expensive"],
                "Cost":   [gen_data.min(), gen_data.median(), gen_data.max()]
            })
            fig = go.Figure(go.Bar(
                x=stats_df["Metric"], y=stats_df["Cost"],
                marker_color=[C_GREEN, C_BLUE, MAR],
                text=["$"+f"{v:.2f}" for v in stats_df["Cost"]],
                textposition="outside", textfont=dict(size=12, color="#2d2d2d"),
            ))
            fig.update_layout(**PT, height=380,
                title=dict(text=f"{sel_d.title()} — Generic Price Range (30-Day Cost)"),
                yaxis_title="30-Day Cost ($)", xaxis_title="",
                showlegend=False, margin=dict(l=40,r=40,t=60,b=40))
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Based on {len(gen_data)} NDC variants from different manufacturers.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — DRUG SHORTAGES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚠️ Drug Shortages":
    st.markdown(f"<h1 style='font-family:Playfair Display,serif;color:{MAR}'>"
                "FDA Drug Shortage Tracker</h1>", unsafe_allow_html=True)

    narrative("We used the OpenFDA drug shortage API to fetch all 1,215 shortage records. "
              "Drug names are full pharmaceutical descriptions like 'Albuterol Sulfate Solution' "
              "— requiring substring matching to link to our 50 target drugs. "
              "Finding: 20 of our 50 drugs appear in shortage data. 30 have zero records "
              "— not a data gap, but a meaningful finding about generic drug supply stability.")

    df_raw = data["shortages"].copy()
    if df_raw.empty: st.warning("No shortage data."); st.stop()
    df_raw.columns = df_raw.columns.str.strip().str.lower().str.replace(" ","_")
    df_raw = df_raw.replace({"None":"","none":"","NaN":"","nan":""})

    status_col = "status" if "status" in df_raw.columns else None
    total   = len(df_raw)
    current = int(df_raw[status_col].str.lower().str.contains(
        "current|active|unavailable|limited",na=False).sum()) if status_col else 0

    # SECTION 1 — KEY FINDING comparison
    sec("Key Finding: Our 50 Drugs vs All FDA Shortage Data")
    c1,c2,c3 = st.columns([2,1,2])
    with c1:
        st.markdown(
            f"<div style='background:{MAR_L};border:2px solid {MAR};"
            f"border-radius:12px;padding:20px;text-align:center'>"
            f"<div style='font-family:Playfair Display,serif;color:{MAR};"
            f"font-size:1.1rem;font-weight:700'>Our 50 Target Drugs</div>"
            f"<div style='font-size:2.5rem;font-weight:700;color:{MAR};margin:10px 0'>20 / 50</div>"
            f"<div style='font-size:0.83rem;color:#2d2d2d;line-height:1.65'>"
            f"have shortage records<br><br>"
            f"<b>30 of 50 = zero shortages</b><br>"
            f"These are mature oral generics<br>(metformin, lisinopril, atorvastatin)<br>"
            f"with 10–50 manufacturers each</div></div>",
            unsafe_allow_html=True)
    with c2:
        st.markdown(
            f"<div style='text-align:center;padding-top:60px;"
            f"font-size:2rem;color:{GRAY}'>⟺</div>"
            f"<div style='text-align:center;font-size:0.78rem;color:{GRAY}'>"
            f"comparison</div>", unsafe_allow_html=True)
    with c3:
        st.markdown(
            f"<div style='background:{P_AMBER};border:2px solid {C_AMBER};"
            f"border-radius:12px;padding:20px;text-align:center'>"
            f"<div style='font-family:Playfair Display,serif;color:{C_AMBER};"
            f"font-size:1.1rem;font-weight:700'>All FDA Shortage Records</div>"
            f"<div style='font-size:2.5rem;font-weight:700;color:{C_AMBER};margin:10px 0'>{total:,}</div>"
            f"<div style='font-size:0.83rem;color:#2d2d2d;line-height:1.65'>"
            f"total shortage records<br><br>"
            f"<b>{current:,} active ({current/total*100:.0f}%)</b><br>"
            f"Dominated by injectable hospital drugs<br>"
            f"(carboplatin, bumetanide, furosemide injection)</div></div>",
            unsafe_allow_html=True)

    st.markdown(info_box(
        "Why do commodity generics rarely shortage?",
        "Mature oral generics like metformin and lisinopril have 10–50 approved manufacturers. "
        "If one manufacturer has a supply problem, pharmacies switch to another immediately. "
        "Injectable hospital drugs (chemotherapy, IV solutions) have 1–3 manufacturers — "
        "any disruption cascades into a shortage. This is why 63% of FDA shortage records "
        "are injectable dosage forms.", C_BLUE, P_BLUE), unsafe_allow_html=True)

    # SECTION 2 — The 20 matched drugs
    sec("The 20 Drugs in Our Dataset with Shortage Records",
        "Shortage reason, count, and status for each matched drug")

    def map_shortage(name):
        if not isinstance(name, str): return None
        nl = name.lower()
        for drug in TARGET:
            if drug in nl: return drug
        return None

    df_matched = df_raw.copy()
    if "drug_name" in df_matched.columns:
        df_matched["drug_canonical"] = df_matched["drug_name"].apply(map_shortage)
        matched = df_matched.dropna(subset=["drug_canonical"])

        if not matched.empty:
            if status_col:
                matched = matched.copy()
                matched["is_active"] = matched[status_col].str.lower().str.contains(
                    "current|active|unavailable|limited",na=False).astype(int)

            summary = (matched.groupby("drug_canonical")
                       .agg(count=("drug_canonical","count"),
                            active=("is_active","sum") if "is_active" in matched.columns
                            else ("drug_canonical","count"))
                       .reset_index().sort_values("count",ascending=True))

            c1,c2 = st.columns([3,2])
            with c1:
                fig = go.Figure(go.Bar(
                    x=summary["count"], y=summary["drug_canonical"],
                    orientation="h", marker_color=MAR,
                    text=summary["count"],
                    textposition="outside", textfont=dict(color=GRAY,size=9),
                    hovertemplate="<b>%{y}</b><br>%{x} shortage records<extra></extra>"))
                fig.update_layout(**PT, height=500,
                    title=dict(text="Shortage Records for Our 20 Matched Drugs"),
                    xaxis_title="Number of Shortage Records",
                    yaxis_title="", margin=dict(l=160,r=60,t=50,b=40))
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                st.markdown(f"**Shortage Details**")
                show_cols = [c for c in ["drug_canonical","shortage_reason","status",
                                          "dosage_form","start_date"]
                             if c in matched.columns]
                disp = (matched[show_cols].drop_duplicates(subset=["drug_canonical"])
                        .sort_values("drug_canonical"))
                disp["shortage_reason"] = disp["shortage_reason"].replace(
                    "","Reason not disclosed by FDA")
                st.dataframe(disp.rename(columns={"drug_canonical":"Drug"}),
                             use_container_width=True, hide_index=True)

    # SECTION 3 — All FDA shortages context
    sec("All FDA Shortage Data — Broader Context")
    tab1,tab2 = st.tabs(["📊 Top 25 Drugs","💊 By Dosage Form"])

    with tab1:
        if "drug_name" in df_raw.columns:
            top25 = df_raw["drug_name"].value_counts().head(25).reset_index()
            top25.columns = ["drug","count"]
            top25["drug_short"] = top25["drug"].str[:45]
            fig = px.bar(top25, x="count", y="drug_short", orientation="h",
                color="count",
                color_continuous_scale=[[0,MAR_L],[1,MAR]],
                title="Top 25 Drugs in Full FDA Shortage Database")
            fig.update_layout(**PT, height=550, coloraxis_showscale=False,
                xaxis_title="Shortage Records", yaxis_title="",
                margin=dict(l=280,r=20,t=50,b=20))
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if "dosage_form" in df_raw.columns:
            fc = (df_raw["dosage_form"].replace("","Unknown")
                  .value_counts().head(12).reset_index())
            fc.columns = ["form","count"]
            c1,c2 = st.columns(2)
            with c1:
                fig = go.Figure(go.Pie(
                    labels=fc["form"], values=fc["count"], hole=0.48,
                    marker_colors=[MAR,C_AMBER,C_BLUE,C_GREEN,C_PURPLE,
                                   C_TEAL,MAR_L,P_AMBER,P_BLUE,P_GREEN,P_PURPLE,P_TEAL]))
                fig.update_layout(**PT, height=380,
                    title=dict(text="Shortages by Dosage Form"),
                    legend=dict(font=dict(color="#2d2d2d"),x=1.0),
                    margin=dict(l=20,r=140,t=50,b=20))
                fig.update_traces(textposition="inside",textinfo="percent")
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                st.markdown(info_box("Why injections dominate shortages",
                    "Injectable drugs require sterile manufacturing facilities with "
                    "FDA-approved cleanrooms. Few companies have these capabilities. "
                    "When one manufacturer faces a production issue, the entire supply "
                    "can be disrupted. Tablets can be manufactured at dozens of facilities worldwide.",
                    MAR, MAR_L), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — CLINICAL TRIALS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Clinical Trials":
    st.markdown(f"<h1 style='font-family:Playfair Display,serif;color:{MAR}'>"
                "Clinical Trials Explorer</h1>", unsafe_allow_html=True)

    narrative("We queried the ClinicalTrials.gov REST API v2, paginated at 100 records per page. "
              "Each record includes NCT ID, trial phase, status, sponsor, and dates. "
              "We computed two ML features per drug: trial_total (research investment proxy) "
              "and trial_completion_rate (evidence maturity proxy). "
              "Finding: Newer drugs like semaglutide and rivaroxaban have lower completion "
              "rates — their long-term safety profile is still being established.")

    df = filt(data["trials"])
    if df.empty: st.warning("No trials data."); st.stop()
    dfc = dc(df)

    total      = len(df)
    recruiting = int(df["status"].str.contains("Recruiting",na=False).sum()) if "status" in df.columns else 0
    completed  = int(df["status"].str.contains("Completed",na=False).sum()) if "status" in df.columns else 0
    drugs_n    = df[dfc].nunique() if dfc in df.columns else 0

    k1,k2,k3,k4 = st.columns(4)
    with k1: st.markdown(kpi(f"{total:,}","Total Trials"), unsafe_allow_html=True)
    with k2: st.markdown(kpi(f"{recruiting:,}","Actively Recruiting",C_GREEN,P_GREEN), unsafe_allow_html=True)
    with k3: st.markdown(kpi(f"{completed:,}","Completed",C_BLUE,P_BLUE), unsafe_allow_html=True)
    with k4: st.markdown(kpi(f"{drugs_n}","Drugs with Trials"), unsafe_allow_html=True)

    st.markdown("<br>",unsafe_allow_html=True)
    tab1,tab2,tab3 = st.tabs(["📊 Trials per Drug","🔬 Phase & Status","🔍 Search Trials"])

    with tab1:
        dt = df[dfc].value_counts().head(25).reset_index()
        dt.columns=["drug","trials"]
        fig=px.bar(dt,x="trials",y="drug",orientation="h",
            color="trials",color_continuous_scale=[[0,P_BLUE],[1,C_BLUE]],
            title="Number of Trials per Drug (Top 25)")
        fig.update_layout(**PT,height=550,coloraxis_showscale=False,
            xaxis_title="Number of Clinical Trials",yaxis_title="",
            margin=dict(l=160,r=20,t=50,b=20))
        st.plotly_chart(fig,use_container_width=True)
        st.markdown(info_box("What does trial count tell us?",
            "More trials = more research investment. Older, established drugs like warfarin "
            "and metformin have hundreds of trials spanning decades. Newer drugs like "
            "semaglutide and empagliflozin have fewer — their research base is growing. "
            "This was used as a feature (trial_total) in our Satisfaction Predictor.",
            C_BLUE, P_BLUE), unsafe_allow_html=True)

        # Completion rate per drug
        if "status" in df.columns:
            comp_df = df.groupby(dfc).apply(lambda g: pd.Series({
                "total": len(g),
                "completed": g["status"].str.contains("COMPLETED|Completed",na=False).sum()
            })).reset_index()
            comp_df["completion_rate"] = comp_df["completed"]/comp_df["total"]
            comp_df = comp_df.sort_values("completion_rate",ascending=True)
            fig2 = px.bar(comp_df, x="completion_rate", y=dfc, orientation="h",
                color="completion_rate",
                color_continuous_scale=[[0,MAR],[0.5,C_AMBER],[1,C_GREEN]],
                title="Trial Completion Rate per Drug (completed ÷ total)",
                labels={"completion_rate":"Completion Rate","drug_name":"Drug"})
            fig2.update_layout(**PT, height=max(500,len(comp_df)*18),
                coloraxis_showscale=False,
                xaxis_title="Completion Rate (0 = none completed, 1 = all completed)",
                xaxis=dict(range=[0, 1], tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                    ticktext=["0%","25%","50%","75%","100%"]),
                yaxis_title="", margin=dict(l=160,r=20,t=50,b=40))
            st.plotly_chart(fig2, use_container_width=True)
            st.caption("Higher completion rate = more mature evidence base. "
                       "Newer drugs (semaglutide, empagliflozin) have lower rates — "
                       "many trials still ongoing.")

    with tab2:
        # Phase explanation
        st.markdown(info_box("What do Clinical Trial Phases mean?",
            "<b>Phase 1:</b> Safety testing in small groups (~20–80 people). Is it safe? What dose? "
            "<br><b>Phase 2:</b> Efficacy testing (~100–300 people). Does it work? What are side effects? "
            "<br><b>Phase 3:</b> Large-scale confirmation (~1,000–3,000 people). Compare to existing treatments. Required for FDA approval. "
            "<br><b>Phase 4:</b> Post-market surveillance. Long-term safety monitoring after approval.",
            C_PURPLE, P_PURPLE), unsafe_allow_html=True)

        c1,c2 = st.columns(2)
        with c1:
            if "phase" in df.columns:
                ph = df["phase"].fillna("Not Specified").value_counts().reset_index()
                ph.columns=["phase","count"]
                fig=px.bar(ph,x="phase",y="count",
                    color="count",color_continuous_scale=[[0,P_PURPLE],[1,C_PURPLE]],
                    title="Trials by Phase")
                fig.update_layout(**PT,height=380,coloraxis_showscale=False,
                    xaxis_title="Phase",yaxis_title="Number of Trials",
                    xaxis_tickangle=-30,margin=dict(l=40,r=20,t=50,b=80))
                st.plotly_chart(fig,use_container_width=True)
        with c2:
            if "status" in df.columns:
                sc = df["status"].fillna("Unknown").value_counts().head(8).reset_index()
                sc.columns=["status","count"]
                fig=go.Figure(go.Pie(labels=sc["status"],values=sc["count"],hole=0.48,
                    marker_colors=[C_GREEN,C_BLUE,MAR,C_AMBER,C_PURPLE,GRAY,C_TEAL,"#DB2777"]))
                fig.update_layout(**PT,height=380,
                    title=dict(text="Trial Status Distribution"),
                    legend=dict(font=dict(color="#2d2d2d")),
                    margin=dict(l=20,r=20,t=50,b=20))
                st.plotly_chart(fig,use_container_width=True)

    with tab3:
        ca,cb = st.columns([2,1])
        with ca: q=st.text_input("Search trial titles",placeholder="e.g. diabetes, phase 3",key="trial_q")
        with cb: df2=st.selectbox("Drug",["All"]+sorted(df[dfc].dropna().unique().tolist()),key="trial_d")
        view=df.copy()
        if q:
            m=(df.get("title",pd.Series()).str.contains(q,case=False,na=False)|
               df.get("description",pd.Series()).str.contains(q,case=False,na=False))
            view=view[m]
        if df2!="All" and dfc in view.columns: view=view[view[dfc]==df2]
        st.markdown(f"<span style='color:{GRAY}'>{len(view)} trials</span>",
                    unsafe_allow_html=True)
        show=[c for c in ["nct_id","drug_name","title","phase","status",
                           "start_date","sponsor"] if c in view.columns]
        st.dataframe(view[show].head(100),use_container_width=True,hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — ADVERSE EVENTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧬 Adverse Events":
    st.markdown(f"<h1 style='font-family:Playfair Display,serif;color:{MAR}'>"
                "Adverse Event Analysis</h1>", unsafe_allow_html=True)

    narrative("We queried the OpenFDA FAERS (FDA Adverse Event Reporting System) API — "
              "the same database the FDA uses for pharmacovigilance. FAERS is voluntary: "
              "doctors, patients, and manufacturers submit reports when unexpected drug "
              "problems occur. We collected up to 500 reports per drug. "
              "Important: 100% of records are classified as 'serious' — a known FAERS artifact. "
              "We therefore used adverse_unique_types (how many different event types a drug "
              "causes, range 73–370) as our ML feature rather than severity.")

    df = filt(data["adverse"])
    if df.empty: st.warning("No adverse event data."); st.stop()
    dfc = dc(df)

    total  = len(df)
    events = df["event_type"].nunique() if "event_type" in df.columns else 0
    drugs_n= df[dfc].nunique()

    k1,k2,k3 = st.columns(3)
    with k1: st.markdown(kpi(f"{total:,}","Total Reports"), unsafe_allow_html=True)
    with k2: st.markdown(kpi(f"{events:,}","Unique Event Types",C_PURPLE,P_PURPLE), unsafe_allow_html=True)
    with k3: st.markdown(kpi(f"{drugs_n}","Drugs Covered",C_GREEN,P_GREEN), unsafe_allow_html=True)

    st.markdown(info_box("⚠️ Important: FAERS Data Limitation",
        "All 24,100 reports in our dataset are classified as 'serious.' This is NOT because "
        "every event is life-threatening — it reflects reporting bias. Doctors and patients "
        "almost never submit FAERS reports for minor side effects like mild headache. "
        "Only hospitalizations, deaths, or unusual reactions get reported. "
        "This means adverse_serious_ratio = 1.0 for every drug — we excluded it as a feature "
        "and used adverse_unique_types instead, which does vary meaningfully (73–370 across drugs).",
        C_AMBER, P_AMBER), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    tab1,tab2 = st.tabs(["📊 Event Analysis","🔍 By Drug"])

    with tab1:
        # Adverse event diversity per drug
        if "event_type" in df.columns:
            div_df = (df.groupby(dfc)["event_type"].nunique()
                      .reset_index().rename(columns={"event_type":"unique_types"})
                      .sort_values("unique_types",ascending=True))
            fig = px.bar(div_df, x="unique_types", y=dfc, orientation="h",
                color="unique_types",
                color_continuous_scale=[[0,MAR_L],[1,MAR]],
                title="Adverse Event Diversity per Drug (Unique Event Types)",
                labels={"unique_types":"Unique Event Types","drug_name":"Drug"})
            fig.update_layout(**PT, height=max(600,len(div_df)*22),
                coloraxis_showscale=False,
                xaxis_title="Number of Unique Adverse Event Types",
                yaxis_title="", margin=dict(l=160,r=20,t=50,b=40))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(info_box("What does adverse event diversity mean?",
                "A higher count means a drug causes a wider variety of side effect types "
                "— not necessarily more severe. Amoxicillin (370 types) triggers many different "
                "reactions because it's a broad-spectrum antibiotic affecting multiple body systems. "
                "Atorvastatin (73 types) is more targeted. This was used as a feature in "
                "our Satisfaction Predictor (correlation with satisfaction: r=−0.05).",
                C_BLUE, P_BLUE), unsafe_allow_html=True)

            # Top 20 events overall
            top_e = df["event_type"].value_counts().head(20).reset_index()
            top_e.columns=["event","count"]
            fig2=px.bar(top_e,x="count",y="event",orientation="h",
                color="count",color_continuous_scale=[[0,MAR_L],[1,MAR]],
                title="Top 20 Most Reported Adverse Events (All 50 Drugs)")
            fig2.update_layout(**PT,height=520,coloraxis_showscale=False,
                xaxis_title="Number of Reports",yaxis_title="",
                margin=dict(l=200,r=20,t=50,b=20))
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        sel_adv = st.selectbox("Select drug", sorted(df[dfc].dropna().unique()),key="adv_drug")
        adv_df  = df[df[dfc]==sel_adv]
        st.markdown(f"<span style='color:{GRAY}'>{len(adv_df)} adverse event "
                    f"reports for <b>{sel_adv}</b></span>", unsafe_allow_html=True)

        if "event_type" in adv_df.columns:
            ta = adv_df["event_type"].value_counts().head(15).reset_index()
            ta.columns=["event","count"]
            fig=px.bar(ta,x="count",y="event",orientation="h",
                color="count",color_continuous_scale=[[0,MAR_L],[1,MAR]],
                title=f"Top 15 Adverse Events — {sel_adv.title()}")
            fig.update_layout(**PT,height=440,coloraxis_showscale=False,
                xaxis_title="Reports",yaxis_title="",
                margin=dict(l=200,r=20,t=50,b=20))
            st.plotly_chart(fig,use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — PUBMED RESEARCH
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📚 PubMed Research":
    st.markdown(f"<h1 style='font-family:Playfair Display,serif;color:{MAR}'>"
                "PubMed Research Intelligence</h1>", unsafe_allow_html=True)

    narrative("We used the NCBI Entrez E-utilities two-step pipeline: esearch to retrieve "
              "PubMed IDs for a drug query, then efetch to retrieve full abstract metadata "
              "in batches of 50. Up to 150 abstracts per drug. PubMed abstracts provided two "
              "features for our ML model: pubmed_total (how well-researched a drug is) and "
              "pubmed_recency_ratio (proportion published since 2020). Abstracts also enriched "
              "our Word2Vec training corpus with clinical terminology alongside patient language. "
              "Finding: pubmed_recency_ratio shows r=−0.195 — drugs attracting recent research "
              "(semaglutide, empagliflozin) tend toward worse patient sentiment.")

    df = filt(data["pubmed"])
    if df.empty: st.warning("No PubMed data."); st.stop()
    dfc = dc(df)

    total   = len(df)
    drugs_n = df[dfc].nunique()
    yr_min  = int(df["pub_year"].min()) if "pub_year" in df.columns and df["pub_year"].notna().any() else "?"
    yr_max  = int(df["pub_year"].max()) if "pub_year" in df.columns and df["pub_year"].notna().any() else "?"

    k1,k2,k3 = st.columns(3)
    with k1: st.markdown(kpi(f"{total:,}","Abstracts Collected"), unsafe_allow_html=True)
    with k2: st.markdown(kpi(f"{drugs_n}","Drugs Covered",C_TEAL,P_TEAL), unsafe_allow_html=True)
    with k3: st.markdown(kpi(f"{yr_min}–{yr_max}","Publication Years",C_AMBER,P_AMBER), unsafe_allow_html=True)

    st.markdown("<br>",unsafe_allow_html=True)
    tab1,tab2,tab3 = st.tabs(["📊 Research Volume","📈 Trends & Recency","🔍 Search Abstracts"])

    with tab1:
        c1,c2 = st.columns(2)
        with c1:
            pd_drug = df[dfc].value_counts().head(25).reset_index()
            pd_drug.columns=["drug","abstracts"]
            fig=px.bar(pd_drug,x="abstracts",y="drug",orientation="h",
                color="abstracts",color_continuous_scale=[[0,P_TEAL],[1,C_TEAL]],
                title="Publications per Drug (Top 25)")
            fig.update_layout(**PT,height=580,coloraxis_showscale=False,
                xaxis_title="Number of Abstracts",yaxis_title="",
                margin=dict(l=160,r=20,t=50,b=20))
            st.plotly_chart(fig,use_container_width=True)
            st.markdown(info_box("What does publication count tell us?",
                "More publications = more studied. Older drugs like warfarin and metformin "
                "have decades of literature. Newer drugs like semaglutide and empagliflozin "
                "have fewer — their evidence base is growing rapidly.",
                C_TEAL, P_TEAL), unsafe_allow_html=True)
        with c2:
            if "pub_year" in df.columns:
                yr=df.dropna(subset=["pub_year"]).copy()
                yr["pub_year"]=pd.to_numeric(yr["pub_year"],errors="coerce")
                yrc=yr[yr["pub_year"]>=2000].groupby("pub_year").size().reset_index(name="count")
                fig=px.area(yrc,x="pub_year",y="count",
                    title="Publications per Year (2000–present)",
                    color_discrete_sequence=[MAR])
                fig.update_traces(fillcolor=MAR_L)
                fig.update_layout(**PT,height=380,
                    xaxis_title="Year",yaxis_title="Abstracts Published",
                    margin=dict(l=40,r=20,t=50,b=40))
                st.plotly_chart(fig,use_container_width=True)

    with tab2:
        if "pub_year" in df.columns:
            yr=df.dropna(subset=["pub_year"]).copy()
            yr["pub_year"]=pd.to_numeric(yr["pub_year"],errors="coerce")
            yr["is_recent"]=(yr["pub_year"]>=2020).astype(int)
            rec=yr.groupby(dfc).agg(
                total=("pub_year","count"),
                recent=("is_recent","sum")).reset_index()
            rec["recency_ratio"]=rec["recent"]/rec["total"]
            rec=rec.sort_values("recency_ratio",ascending=True)

            fig=px.bar(rec,x="recency_ratio",y=dfc,orientation="h",
                color="recency_ratio",
                color_continuous_scale=[[0,P_TEAL],[1,C_TEAL]],
                title="Recent Publication Ratio per Drug (papers since 2020 ÷ total)",
                labels={"recency_ratio":"Recency Ratio","drug_name":"Drug"})
            fig.update_layout(**PT,height=max(500,len(rec)*18),
                coloraxis_showscale=False,
                xaxis_title="Recency Ratio (0 = all old literature, 1 = all recent)",
                xaxis=dict(range=[0, 1], tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                    ticktext=["0%","25%","50%","75%","100%"]),
                yaxis_title="",margin=dict(l=160,r=20,t=50,b=40))
            st.plotly_chart(fig,use_container_width=True)

            st.markdown(info_box(
                "Connection to Satisfaction Predictor",
                "pubmed_recency_ratio was used as a feature in our 50-drug feature matrix. "
                "It showed Pearson r=−0.195 with patient satisfaction — drugs attracting "
                "lots of recent research (semaglutide: r=−0.33, empagliflozin: r=−0.42) tend "
                "toward worse patient sentiment. Interpretation: newer drugs with growing "
                "research bases often have less-understood side effect profiles, "
                "reflected in more negative patient reviews.",
                MAR, MAR_L), unsafe_allow_html=True)

    with tab3:
        ca,cb = st.columns([2,1])
        with ca: q=st.text_input("Search titles & abstracts",
                                  placeholder="e.g. cardiovascular, mortality, safety",key="pub_q")
        with cb: df2=st.selectbox("Drug",["All"]+sorted(df[dfc].dropna().unique().tolist()),key="pub_d")
        view=df.copy()
        if q:
            m=(df.get("title",pd.Series()).str.contains(q,case=False,na=False)|
               df.get("abstract",pd.Series()).str.contains(q,case=False,na=False))
            view=view[m]
        if df2!="All" and dfc in view.columns: view=view[view[dfc]==df2]
        st.markdown(f"<span style='color:{GRAY}'>{len(view)} abstracts found</span>",
                    unsafe_allow_html=True)
        for _,row in view.head(10).iterrows():
            title    = str(row.get("title",""))
            journal  = str(row.get("journal",""))
            year     = str(row.get("pub_year",""))
            pmid     = str(row.get("pmid",""))
            abstract = str(row.get("abstract",""))
            meta     = " · ".join(filter(lambda x:x and x!="nan",[journal,year,f"PMID:{pmid}"]))
            st.markdown(
                f"<div style='background:white;border:1px solid #E9ECEF;"
                f"border-left:3px solid {MAR};border-radius:0 8px 8px 0;"
                f"padding:14px 16px;margin:6px 0;box-shadow:0 1px 4px rgba(0,0,0,0.04)'>"
                f"<div style='color:{MAR};font-weight:700;font-size:0.9rem;"
                f"font-family:Playfair Display,serif;margin-bottom:4px'>{title}</div>"
                f"<div style='color:{GRAY};font-size:0.76rem;margin-bottom:8px'>{meta}</div>"
                f"<div style='color:#2d2d2d;font-size:0.86rem;line-height:1.55'>"
                f"{abstract[:380]}{'...' if len(abstract)>380 else ''}</div></div>",
                unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 8 — SATISFACTION PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Satisfaction Predictor":
    st.markdown(f"<h1 style='font-family:Playfair Display,serif;color:{MAR}'>"
                "Drug Satisfaction Predictor</h1>", unsafe_allow_html=True)

    # Research question banner
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,{MAR} 0%,{MAR_D} 100%);
                border-radius:12px;padding:22px 28px;margin-bottom:20px'>
        <div style='font-size:0.7rem;color:rgba(255,255,255,0.65);text-transform:uppercase;
                    letter-spacing:0.1em;margin-bottom:6px'>Research Question</div>
        <div style='font-family:Playfair Display,serif;font-size:1.25rem;
                    color:white;font-style:italic;line-height:1.5'>
            "What structural drug characteristics — acquisition cost, shortage history,
            adverse event burden, and clinical trial activity — predict patient-reported
            satisfaction, and which factors matter most?"
        </div>
    </div>""", unsafe_allow_html=True)

    narrative("We engineered a 50-drug × 10-feature matrix by joining all 5 non-review sources "
              "at the drug level. Patient sentiment (avg VADER score) was the target variable Y. "
              "We tested four supervised models using Leave-One-Out cross-validation (appropriate "
              "for n=50): Ridge Regression (α=0.1, 1.0, 10.0) and Random Forest. "
              "Best model: Random Forest (LOO R²=−0.22). "
              "Conclusion: Structural characteristics do not predict patient satisfaction.")

    MPATH = MAST/"drug_feature_matrix.csv"
    if not MPATH.exists():
        st.warning("Run phase2b_run_v2.py first to generate data/master/drug_feature_matrix.csv")
        st.stop()

    master = pd.read_csv(MPATH)

    k1,k2,k3,k4 = st.columns(4)
    with k1: st.markdown(kpi("50","Drugs Analysed"), unsafe_allow_html=True)
    with k2: st.markdown(kpi("6,827","Reviews Scored",C_BLUE,P_BLUE), unsafe_allow_html=True)
    with k3: st.markdown(kpi("10","Features · 5 Sources",C_GREEN,P_GREEN), unsafe_allow_html=True)
    with k4: st.markdown(kpi("−0.185","Mean Satisfaction",MAR,MAR_L), unsafe_allow_html=True)

    st.markdown("<br>",unsafe_allow_html=True)

    # Feature matrix table
    sec("How All 6 Sources Feed Into One Model")
    st.markdown(f"""
    <div style='background:white;border:1px solid #E9ECEF;border-radius:12px;
                padding:20px;box-shadow:0 2px 8px rgba(0,0,0,0.05);overflow-x:auto'>
        <table style='width:100%;border-collapse:collapse;font-size:0.85rem'>
            <tr style='background:{MAR};color:white'>
                <th style='padding:10px 14px;text-align:left'>Source</th>
                <th style='padding:10px 14px;text-align:left'>Features Used</th>
                <th style='padding:10px 14px;text-align:left'>Role in Model</th>
                <th style='padding:10px 14px;text-align:left'>What it measures</th>
            </tr>
            <tr style='background:{MAR_L}'>
                <td style='padding:9px 14px;font-weight:600;color:{MAR}'>💬 WebMD Reviews</td>
                <td style='padding:9px 14px'><code>avg_sentiment</code></td>
                <td style='padding:9px 14px'><b>Target (Y)</b></td>
                <td style='padding:9px 14px'>What we are predicting — patient satisfaction</td>
            </tr>
            <tr style='background:{P_GREEN}'>
                <td style='padding:9px 14px;font-weight:600;color:{C_GREEN}'>💰 NADAC/CMS</td>
                <td style='padding:9px 14px'><code>price_30day, price_cv</code></td>
                <td style='padding:9px 14px'>Predictor (X)</td>
                <td style='padding:9px 14px'>Cost and price stability across manufacturers</td>
            </tr>
            <tr style='background:{P_AMBER}'>
                <td style='padding:9px 14px;font-weight:600;color:{C_AMBER}'>⚠️ FDA Shortages</td>
                <td style='padding:9px 14px'><code>shortage_total, has_active_shortage</code></td>
                <td style='padding:9px 14px'>Predictor (X)</td>
                <td style='padding:9px 14px'>Supply reliability and availability</td>
            </tr>
            <tr style='background:{P_PURPLE}'>
                <td style='padding:9px 14px;font-weight:600;color:{C_PURPLE}'>🚨 OpenFDA FAERS</td>
                <td style='padding:9px 14px'><code>adverse_unique_types, adverse_log_total</code></td>
                <td style='padding:9px 14px'>Predictor (X)</td>
                <td style='padding:9px 14px'>Side effect diversity and volume</td>
            </tr>
            <tr style='background:{P_BLUE}'>
                <td style='padding:9px 14px;font-weight:600;color:{C_BLUE}'>🔬 ClinicalTrials</td>
                <td style='padding:9px 14px'><code>trial_completion_rate, trial_total</code></td>
                <td style='padding:9px 14px'>Predictor (X)</td>
                <td style='padding:9px 14px'>Research maturity and evidence base</td>
            </tr>
            <tr style='background:{P_TEAL}'>
                <td style='padding:9px 14px;font-weight:600;color:{C_TEAL}'>📚 PubMed</td>
                <td style='padding:9px 14px'><code>pubmed_recency_ratio</code></td>
                <td style='padding:9px 14px'>Predictor (X)</td>
                <td style='padding:9px 14px'>Whether research interest is growing or established</td>
            </tr>
        </table>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>",unsafe_allow_html=True)
    tab1,tab2,tab3 = st.tabs(["🔬 Hypothesis Results","🏆 Drug Rankings","🔍 Drug Detail"])

    with tab1:
        sec("4 Hypothesis Tests — Results",
            "We tested whether each structural factor predicts patient satisfaction (n=50 drugs)")

        feature_map = {
            "adverse_unique_types":  ("H1","Adverse Event Diversity","negative",
                                       "Higher adverse event diversity → lower satisfaction?"),
            "has_active_shortage":   ("H2","Active Drug Shortage","negative",
                                       "Drugs in shortage → lower patient satisfaction?"),
            "price_30day":           ("H3","Drug Price (30-day)","negative",
                                       "More expensive drugs → lower patient satisfaction?"),
            "trial_completion_rate": ("H4","Trial Completion Rate","positive",
                                       "More completed trials → higher satisfaction?"),
        }

        h_cols = st.columns(2)
        h_results = []
        for col_name,(hid,label,expected,question) in feature_map.items():
            if col_name not in master.columns: continue
            x = master[col_name]; y = master["avg_sentiment"]
            valid = ~(x.isna()|y.isna())
            if valid.sum()<5: continue
            pr,_ = scipy_stats.pearsonr(x[valid],y[valid])
            sr,sp= scipy_stats.spearmanr(x[valid],y[valid])
            correct_dir = (expected=="negative" and pr<0) or (expected=="positive" and pr>0)
            if sp<0.05: result,icon,color,bg = "✅ SUPPORTED",    "✅",C_GREEN,P_GREEN
            elif sp<0.10:result,icon,color,bg = "~ WEAK TREND",    "〜",C_AMBER,P_AMBER
            elif correct_dir:result,icon,color,bg = "~ TREND (n.s.)","〜",C_AMBER,P_AMBER
            else:        result,icon,color,bg = "✗ NOT SUPPORTED","✗",MAR,    MAR_L
            h_results.append((hid,label,question,pr,sp,result,icon,color,bg,expected))

        for i,(hid,label,question,pr,sp,result,icon,color,bg,expected) in enumerate(h_results):
            with h_cols[i%2]:
                st.markdown(
                    f"<div style='background:{bg};border-top:3px solid {color};"
                    f"border-radius:0 0 12px 12px;padding:18px;margin:8px 0;"
                    f"box-shadow:0 2px 8px rgba(0,0,0,0.05)'>"
                    f"<div style='display:flex;justify-content:space-between;align-items:center'>"
                    f"<span style='font-weight:700;color:{color};font-size:0.85rem'>{hid}</span>"
                    f"<span style='font-weight:700;color:{color};font-size:0.9rem'>{result}</span>"
                    f"</div>"
                    f"<div style='font-family:Playfair Display,serif;font-size:1rem;"
                    f"color:#2d2d2d;font-weight:600;margin:8px 0'>{label}</div>"
                    f"<div style='font-size:0.82rem;color:{GRAY};margin-bottom:10px;font-style:italic'>{question}</div>"
                    f"<div style='display:flex;gap:16px'>"
                    f"<div style='background:white;border-radius:6px;padding:6px 10px;flex:1;text-align:center'>"
                    f"<div style='font-size:1.1rem;font-weight:700;color:{color}'>{pr:+.3f}</div>"
                    f"<div style='font-size:0.68rem;color:{GRAY};text-transform:uppercase'>Pearson r</div></div>"
                    f"<div style='background:white;border-radius:6px;padding:6px 10px;flex:1;text-align:center'>"
                    f"<div style='font-size:1.1rem;font-weight:700;color:{color}'>{sp:.3f}</div>"
                    f"<div style='font-size:0.68rem;color:{GRAY};text-transform:uppercase'>p-value</div></div>"
                    f"<div style='background:white;border-radius:6px;padding:6px 10px;flex:1;text-align:center'>"
                    f"<div style='font-size:1.1rem;font-weight:700;color:{GRAY}'>{'<0.05' if sp<0.05 else '<0.10' if sp<0.10 else '>0.10'}</div>"
                    f"<div style='font-size:0.68rem;color:{GRAY};text-transform:uppercase'>Significance</div></div>"
                    f"</div></div>",
                    unsafe_allow_html=True)

        st.markdown("<br>",unsafe_allow_html=True)
        st.markdown(info_box(
            "🔬 Overall Conclusion — Null Result (Structurally Meaningful)",
            "None of our 4 hypotheses reached statistical significance (all p > 0.10, n=50 drugs). "
            "Best model R²=−0.22. This is a meaningful scientific finding: "
            "<b>patient satisfaction with a drug is not predictable from its price, "
            "availability, adverse event burden, or trial maturity alone.</b> "
            "Satisfaction appears driven primarily by individual pharmacological mechanisms "
            "and patient-specific responses — consistent with pharmacovigilance literature "
            "showing that patient-reported outcomes require patient-level data to predict.",
            MAR, MAR_L), unsafe_allow_html=True)

    with tab2:
        c1,c2 = st.columns([3,2])
        with c1:
            rank = master[["drug_name","avg_sentiment","review_count"]]\
                .sort_values("avg_sentiment",ascending=True).copy()
            b_c  = [MAR if v<-0.25 else C_AMBER if v<-0.10 else C_GREEN
                    for v in rank["avg_sentiment"]]
            fig = go.Figure(go.Bar(
                x=rank["avg_sentiment"], y=rank["drug_name"],
                orientation="h", marker_color=b_c,
                text=[f"n={int(r)}" for r in rank["review_count"]],
                textposition="outside", textfont=dict(size=9,color=GRAY),
                hovertemplate="<b>%{y}</b><br>VADER: %{x:.3f}<extra></extra>"))
            fig.add_vline(x=0,line_dash="dash",line_color="#CCC",line_width=1)
            fig.update_layout(**PT, height=1050,
                title=dict(text="All 50 Drugs Ranked by Patient Satisfaction"),
                xaxis=dict(title="Avg VADER Score",range=[-0.85,0.45]),
                yaxis_title="", margin=dict(l=160,r=100,t=60,b=40))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown(f"<div style='font-family:Playfair Display,serif;color:{C_GREEN};"
                        f"font-weight:700;margin:0 0 8px'>🟢 Top 5 Satisfaction</div>",
                        unsafe_allow_html=True)
            for _,row in master.nlargest(5,"avg_sentiment").iterrows():
                st.markdown(info_box(
                    f"{row['drug_name']} — {row['avg_sentiment']:.3f}",
                    f"{int(row['review_count'])} reviews · "
                    f"${row.get('price_30day',0):.2f}/30-day",
                    C_GREEN,P_GREEN), unsafe_allow_html=True)
            st.markdown(f"<div style='font-family:Playfair Display,serif;color:{MAR};"
                        f"font-weight:700;margin:16px 0 8px'>🔴 Bottom 5 Satisfaction</div>",
                        unsafe_allow_html=True)
            for _,row in master.nsmallest(5,"avg_sentiment").iterrows():
                st.markdown(info_box(
                    f"{row['drug_name']} — {row['avg_sentiment']:.3f}",
                    f"{int(row['review_count'])} reviews · "
                    f"${row.get('price_30day',0):.2f}/30-day",
                    MAR,MAR_L), unsafe_allow_html=True)

            # Price vs satisfaction scatter
            fig2=px.scatter(master,x="price_30day",y="avg_sentiment",
                text="drug_name",color="avg_sentiment",
                color_continuous_scale=[[0,MAR],[0.5,C_AMBER],[1,C_GREEN]],
                title="Price vs Patient Satisfaction",
                labels={"price_30day":"30-Day Price ($)",
                        "avg_sentiment":"VADER Score"})
            fig2.update_traces(textposition="top center",textfont_size=7)
            fig2.update_layout(**PT,height=400,coloraxis_showscale=False,
                margin=dict(l=40,r=20,t=50,b=40))
            st.plotly_chart(fig2,use_container_width=True)

    with tab3:
        sel_d = st.selectbox("Select drug",sorted(master["drug_name"].tolist()),key="pred_d")
        row   = master[master["drug_name"]==sel_d].iloc[0]
        s     = row["avg_sentiment"]
        sc    = MAR if s<-0.25 else C_AMBER if s<-0.10 else C_GREEN
        sl    = "Low Satisfaction" if s<-0.25 else "Moderate" if s<-0.10 else "High Satisfaction"
        sbg   = MAR_L if s<-0.25 else P_AMBER if s<-0.10 else P_GREEN

        k1,k2,k3 = st.columns(3)
        with k1: st.markdown(kpi(f"{s:.3f}","VADER Score",sc,sbg), unsafe_allow_html=True)
        with k2: st.markdown(kpi(f"{int(row['review_count'])}","Reviews Scored"), unsafe_allow_html=True)
        with k3:
            p = row.get("price_30day",0)
            st.markdown(kpi(f"${p:.2f}" if p>0 else "N/A","Generic 30-Day Cost",C_GREEN,P_GREEN),
                        unsafe_allow_html=True)

        st.markdown("<br>",unsafe_allow_html=True)
        st.markdown(info_box(
            f"🎯 Assessment: {sl}",
            f"VADER Score: <b style='color:{sc}'>{s:.4f}</b> &nbsp;·&nbsp; "
            f"Shortage records: <b>{int(row.get('shortage_total',0))}</b> &nbsp;·&nbsp; "
            f"Adverse event types: <b>{int(row.get('adverse_unique_types',0))}</b><br>"
            f"Trial completion rate: <b>{row.get('trial_completion_rate',0):.1%}</b> &nbsp;·&nbsp; "
            f"Generic available: <b>{'Yes' if row.get('generic_available',0) else 'No'}</b> &nbsp;·&nbsp; "
            f"Price variability (CV): <b>{row.get('price_cv',0):.3f}</b>",
            sc, sbg), unsafe_allow_html=True)

        # Feature comparison
        feat_display = {
            "price_30day":           ("30-Day Price ($)",          C_GREEN),
            "shortage_total":        ("Total Shortage Records",     MAR),
            "adverse_unique_types":  ("Adverse Event Types",        C_PURPLE),
            "trial_completion_rate": ("Trial Completion Rate",      C_BLUE),
            "pubmed_recency_ratio":  ("Recent Research Ratio",      C_TEAL),
        }
        cols_f = st.columns(len(feat_display))
        for col,(feat,(label,color)) in zip(cols_f,feat_display.items()):
            val = row.get(feat,0)
            with col:
                if feat == "trial_completion_rate":
                    st.markdown(kpi(f"{val:.1%}",label,color,color+"22"), unsafe_allow_html=True)
                elif feat == "pubmed_recency_ratio":
                    st.markdown(kpi(f"{val:.1%}",label,color,color+"22"), unsafe_allow_html=True)
                elif feat == "price_30day":
                    st.markdown(kpi(f"${val:.2f}",label,color,color+"22"), unsafe_allow_html=True)
                else:
                    st.markdown(kpi(f"{int(val)}",label,color,color+"22"), unsafe_allow_html=True)

elif page == "🔮 Shortage & Rating Prediction":
    st.markdown(f"<h1 style='font-family:Playfair Display,serif;color:{MAR}'>"
                "Shortage & Rating Prediction</h1>", unsafe_allow_html=True)

    narrative("Phase 2C extends our analysis with two new prediction problems. "
              "Problem 1: Can structural features predict which drugs will shortage? "
              "(Random Forest, AUC=0.745, n=50 drugs). "
              "Problem 2: Can review text predict patient star ratings? "
              "(Ridge Regression, R²=0.24, n=6,688 reviews). "
              "Key connecting insight: the word 'shortage' is the single strongest "
              "predictor of low ratings (coefficient=−5.13) — supply disruption "
              "indirectly drives patient dissatisfaction.")

    tab1, tab2 = st.tabs(["⚠️ Shortage Prediction", "⭐ Rating Prediction"])

    with tab1:
        sec("Problem 1 — Drug Shortage Prediction",
            "Can structural drug characteristics predict which drugs will experience shortages?")

        k1,k2,k3,k4 = st.columns(4)
        with k1: st.markdown(kpi("50","Drugs Analysed"), unsafe_allow_html=True)
        with k2: st.markdown(kpi("0.745","Best AUC (Random Forest)",C_GREEN,P_GREEN), unsafe_allow_html=True)
        with k3: st.markdown(kpi("20/50","Drugs in Shortage",MAR,MAR_L), unsafe_allow_html=True)
        with k4: st.markdown(kpi("5-Fold","Cross Validation",C_BLUE,P_BLUE), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Model comparison
        model_results = pd.DataFrame([
            {"Model":"Logistic Regression","AUC":0.5200,"Accuracy":0.5800,"F1":0.3226},
            {"Model":"Random Forest",      "AUC":0.7450,"Accuracy":0.7400,"F1":0.6286},
            {"Model":"Gradient Boosting",  "AUC":0.7450,"Accuracy":0.7000,"F1":0.6154},
            {"Model":"SVM (RBF)",          "AUC":0.4133,"Accuracy":0.6200,"F1":0.2400},
        ])
        c1,c2 = st.columns(2)
        with c1:
            fig = go.Figure(go.Bar(
                x=model_results["Model"], y=model_results["AUC"],
                marker_color=[MAR if m=="Random Forest" else "#E0E0E0"
                              for m in model_results["Model"]],
                text=[f"{v:.3f}" for v in model_results["AUC"]],
                textposition="outside"))
            fig.add_hline(y=0.5,line_dash="dash",line_color=GRAY,
                         annotation_text="Random baseline",
                         annotation_font_color=GRAY)
            fig.update_layout(**PT, height=340,
                title=dict(text="Model Comparison — AUC Score"),
                yaxis=dict(title="ROC-AUC",range=[0,1.1]),
                xaxis_title="", showlegend=False,
                margin=dict(l=40,r=20,t=50,b=60))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            # Feature importance
            feat_imp = pd.DataFrame([
                {"Feature":"Drug Price (30-day)",       "Importance":0.2267},
                {"Feature":"Price Variability",          "Importance":0.0515},
                {"Feature":"Adverse Event Diversity",    "Importance":0.0201},
                {"Feature":"Injectable Drug",            "Importance":0.0083},
                {"Feature":"Number of Manufacturers",   "Importance":0.0024},
            ]).sort_values("Importance", ascending=True)
            fig2 = go.Figure(go.Bar(
                x=feat_imp["Importance"], y=feat_imp["Feature"],
                orientation="h", marker_color=MAR,
                text=[f"{v:.4f}" for v in feat_imp["Importance"]],
                textposition="outside"))
            fig2.update_layout(**PT, height=340,
                title=dict(text="Feature Importance (Permutation)"),
                xaxis_title="Impact on AUC", yaxis_title="",
                margin=dict(l=180,r=80,t=50,b=40))
            st.plotly_chart(fig2, use_container_width=True)

        # Shortage probability per drug from CSV
        SP = MAST/"shortage_predictions.csv"
        if SP.exists():
            sp_df = pd.read_csv(SP).sort_values("shortage_probability")
            fig3 = go.Figure(go.Bar(
                x=sp_df["shortage_probability"], y=sp_df["drug_name"],
                orientation="h",
                marker_color=[MAR if p>0.5 else P_BLUE
                              for p in sp_df["shortage_probability"]],
                text=[f"{p:.2f}" for p in sp_df["shortage_probability"]],
                textposition="outside", textfont=dict(size=9,color=GRAY)))
            fig3.add_vline(x=0.5,line_dash="dash",line_color=C_AMBER,line_width=2,
                           annotation_text="Threshold 0.5",
                           annotation_font_color=C_AMBER)
            fig3.update_layout(**PT, height=900,
                title=dict(text="Predicted Shortage Probability per Drug"),
                xaxis=dict(title="Shortage Probability",range=[0,1.2]),
                yaxis_title="", margin=dict(l=160,r=80,t=60,b=40))
            st.plotly_chart(fig3, use_container_width=True)

        st.markdown(info_box(
            "Key Finding — Shortage Prediction",
            "Drug price is the strongest predictor of shortage risk (r=+0.245, p=0.086). "
            "Expensive drugs shortage more — consistent with fewer manufacturers competing "
            "in high-cost drug markets. Adding structural features does not improve "
            "patient satisfaction prediction (Phase 2B), but DOES predict supply "
            "disruption (AUC=0.745) — two very different outcomes from the same features.",
            C_GREEN, P_GREEN), unsafe_allow_html=True)

    with tab2:
        sec("Problem 2 — Patient Rating Prediction",
            "Can TF-IDF review text predict the star rating a patient will give?")

        k1,k2,k3,k4 = st.columns(4)
        with k1: st.markdown(kpi("6,688","Reviews Used"), unsafe_allow_html=True)
        with k2: st.markdown(kpi("0.238","Best R² (Ridge)",C_BLUE,P_BLUE), unsafe_allow_html=True)
        with k3: st.markdown(kpi("3,000","TF-IDF Features",C_PURPLE,P_PURPLE), unsafe_allow_html=True)
        with k4: st.markdown(kpi("2.02","MAE (±stars)",C_AMBER,P_AMBER), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Word importance from CSV
        WI = MAST/"rating_word_importance.csv"
        if WI.exists():
            wi_df = pd.read_csv(WI)
            pos_words = wi_df.nlargest(12,"coef")
            neg_words = wi_df.nsmallest(12,"coef")

            c1,c2 = st.columns(2)
            with c1:
                fig = go.Figure(go.Bar(
                    x=pos_words["coef"], y=pos_words["word"],
                    orientation="h", marker_color=C_GREEN,
                    text=[f"+{v:.2f}" for v in pos_words["coef"]],
                    textposition="outside", textfont=dict(size=9)))
                fig.update_layout(**PT, height=400,
                    title=dict(text="🟢 Words → HIGH Star Rating"),
                    xaxis_title="Ridge Coefficient", yaxis_title="",
                    margin=dict(l=140,r=80,t=50,b=40))
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig2 = go.Figure(go.Bar(
                    x=neg_words["coef"], y=neg_words["word"],
                    orientation="h", marker_color=MAR,
                    text=[f"{v:.2f}" for v in neg_words["coef"]],
                    textposition="outside", textfont=dict(size=9)))
                fig2.update_layout(**PT, height=400,
                    title=dict(text="🔴 Words → LOW Star Rating"),
                    xaxis_title="Ridge Coefficient", yaxis_title="",
                    margin=dict(l=140,r=80,t=50,b=40))
                st.plotly_chart(fig2, use_container_width=True)

        st.markdown(info_box(
            "Key Finding — 'shortage' is #1 negative word",
            "The word 'shortage' has the strongest negative coefficient (−5.13) in rating "
            "prediction — stronger than 'worse', 'horrible', or 'useless'. Patients who "
            "experienced drug unavailability give dramatically lower ratings. This connects "
            "both prediction problems: structural shortage risk (Problem 1) indirectly "
            "drives patient dissatisfaction (Problem 2) — a pathway our Phase 2B "
            "structural analysis could not detect directly.",
            MAR, MAR_L), unsafe_allow_html=True)

        st.markdown(info_box(
            "Why adding structural features doesn't help",
            "R² with text only = 0.2376. R² with text + structural features = 0.2337 "
            "(slightly worse). This confirms our Phase 2B null finding from a completely "
            "different angle: structural drug characteristics (price, shortage count, "
            "adverse events) add no predictive power beyond what the patient's own "
            "words already capture about their experience.",
            C_BLUE, P_BLUE), unsafe_allow_html=True)