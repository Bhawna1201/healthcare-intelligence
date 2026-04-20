# 💊 Healthcare Intelligence Platform

**Stevens Institute of Technology · BIA-660 Web Mining · Final Project**

A comprehensive drug intelligence platform that aggregates 131,843 records from 6 authoritative sources across 50 commonly prescribed drugs, applying NLP and machine learning to answer: *what predicts patient satisfaction with prescription medications?*

---

## 🌐 Live Dashboard

👉 **[View Live App](https://healthcare-intelligence.streamlit.app)**

No setup required — runs in your browser.

---

## 📊 Project Overview

| Metric | Value |
|--------|-------|
| Total Records | 131,843 |
| Data Sources | 6 |
| Target Drugs | 50 |
| Patient Reviews | 6,827 |
| ML Features | 43 |
| Dashboard Pages | 9 |

---

## 🗂️ Data Sources

| Source | Records | Method | Key Feature |
|--------|---------|--------|-------------|
| WebMD Reviews | 6,827 | Selenium + Chrome | avg_sentiment (target Y) |
| NADAC/CMS Pricing | 86,742 | Socrata REST API | price_30day |
| ClinicalTrials.gov | 9,331 | REST API v2 | trial_completion_rate |
| FDA Drug Shortages | 1,215 | OpenFDA API | has_active_shortage |
| OpenFDA FAERS | 24,100 | OpenFDA REST API | adverse_unique_types |
| PubMed/NCBI | 7,181 | Entrez E-utilities | pubmed_recency_ratio |

---

## 🤖 Machine Learning Models

### Supervised Learning
- **Logistic Regression** — Best sentiment classifier: 76% accuracy, 0.863 AUC
- Linear SVM — 75.5% accuracy
- Random Forest — 75.0% accuracy
- Naive Bayes — 70.4% accuracy

### Unsupervised Learning
- **LDA Topic Modeling** — 8 hidden topics from 3,068 reviews
- **K-Means Clustering** — 5 drug clusters from adverse event profiles
- **t-SNE Visualization** — 2D cluster projection

### Advanced NLP
- **VADER Sentiment** — Primary scorer (compound −1 to +1)
- **TextBlob** — Polarity validation
- **Word2Vec** — 100-dim skip-gram embeddings on reviews + PubMed

### Prediction Problems
- **Satisfaction Predictor** — 43 features, LOO-CV, RF R²=+0.017
- **Shortage Prediction** — AUC=0.792, therapeutic substitutability feature
- **Rating Prediction** — Ridge regression R²=0.238 from text alone

---

## 📈 Key Findings

1. **Sentiment Polarization = Strongest Predictor** — r=+0.485, p=0.021. Polarizing drugs have higher average satisfaction because patients they help write intensely positive reviews.

2. **Drug Class > Structural Features** — Adding 27 therapeutic categories improved R² from −0.23 to +0.017. ARB drugs show significantly lower satisfaction (r=−0.416, p=0.003).

3. **Substitutability Predicts Shortage** — Drugs with no therapeutic alternative shortage significantly more (r=−0.423, p=0.002). Furosemide, albuterol, insulin face highest risk.

4. **"Shortage" = Most Negative Word** — Coefficient −5.13 in rating prediction, stronger than "worse", "horrible", or "useless".

5. **56% of Reviews are Negative** — Consistent with negativity bias in online health reviews.

---

## 🚀 Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/healthcare-intelligence.git
cd healthcare-intelligence
```

### 2. Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the dashboard
```bash
streamlit run dashboard.py
```

Open `http://localhost:8501` in your browser.

---

## 📁 Project Structure

```
healthcare_intelligence/
│
├── dashboard.py                  # Main Streamlit dashboard (9 pages)
├── requirements.txt              # Python dependencies
│
├── data/
│   ├── raw/                      # Raw scraped data (not in repo — re-scrape)
│   │   ├── reviews.csv
│   │   ├── prices.csv
│   │   ├── trials.csv
│   │   ├── shortages.csv
│   │   ├── adverse_events.csv
│   │   └── pubmed_abstracts.csv
│   ├── processed/                # Cleaned versions
│   └── master/                   # Feature matrices and model outputs
│       ├── drug_feature_matrix_v3.csv
│       ├── shortage_predictions_v2.csv
│       └── rating_word_importance.csv
│
├── phase2b_run_v3.py             # Satisfaction predictor (43 features)
├── phase2c_predictions_v2.py     # Shortage + rating prediction
│
└── logs/
    └── checkpoints/              # Scraper progress checkpoints
```

---

## 🔄 Re-scraping Data

If you want to collect fresh data:

```bash
# Reviews (requires Chrome + ChromeDriver)
python scrape_missing_reviews.py

# Pricing, trials, shortages, adverse events, pubmed
python phase2b_run_v3.py   # rebuilds all features from raw files
```

> **Note:** WebMD scraping requires Chrome and ChromeDriver installed. All API sources (NADAC, ClinicalTrials, OpenFDA, PubMed) are free with no authentication required.

---

## 📋 Dashboard Pages

| Page | Description |
|------|-------------|
| 📊 Overview | Architecture, ML models, key findings |
| 😊 Sentiment Analysis | VADER scores, top/bottom drugs, review explorer |
| 💰 Drug Pricing | NADAC costs, dosage form comparison, drug lookup |
| ⚠️ Drug Shortages | FDA shortage data, 20 matched drugs, shortage reasons |
| 🔬 Clinical Trials | Trial status, completion rates, trial search |
| 🧬 Adverse Events | FAERS data, event diversity, drug detail |
| 📚 PubMed Research | Publication counts, recency analysis, abstract search |
| 🎯 Satisfaction Predictor | 8 hypothesis tests, drug rankings, power analysis |
| 🔮 Shortage & Rating | AUC=0.792 shortage model, word importance |

---


**Research Question:**
> *What structural drug characteristics — acquisition cost, shortage history, adverse event burden, and clinical trial activity — predict patient-reported satisfaction with prescription medications?*

**Conclusion:** Structural features alone cannot predict patient satisfaction (all p > 0.10, n=50). Drug class (therapeutic category) and sentiment polarization are stronger predictors. The null finding is itself meaningful — pharmacological mechanism drives satisfaction more than any structural characteristic.

---

## 📦 Dependencies

```
streamlit      plotly         pandas         numpy
scipy          scikit-learn   vaderSentiment textblob
nltk           gensim         selenium       beautifulsoup4
matplotlib     Pillow         rapidfuzz      tqdm
```

---

## 📄 License

Academic project — Stevens Institute of Technology, BIA-660 Web Mining.  
Data sourced from publicly available government databases (NADAC, ClinicalTrials.gov, OpenFDA, NCBI).  
WebMD reviews collected for academic research purposes only.
