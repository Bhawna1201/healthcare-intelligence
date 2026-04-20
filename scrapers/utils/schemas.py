"""
scrapers/utils/schemas.py
─────────────────────────
Dataclass schemas for every raw output file.
These enforce consistent column names across all scrapers.
"""

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


def now_iso() -> str:
    return datetime.utcnow().isoformat()


# ── reviews.csv ───────────────────────────────────────────────────────────────
@dataclass
class ReviewRecord:
    drug_name: str
    rating: Optional[float]          # 1–10 numeric scale
    review_text: str
    condition: Optional[str]         # condition being treated
    date: Optional[str]              # ISO date string
    source: str                      # "drugs.com" | "webmd"
    review_id: Optional[str]         # source-specific ID for deduplication
    helpful_votes: Optional[int]
    scraped_at: str = field(default_factory=now_iso)


# ── trials.csv ────────────────────────────────────────────────────────────────
@dataclass
class TrialRecord:
    drug_name: str
    nct_id: str                      # NCT identifier e.g. "NCT04321234"
    title: str
    phase: Optional[str]             # "Phase 1" | "Phase 2" | "Phase 3" | "Phase 4"
    status: Optional[str]            # "Recruiting" | "Completed" | "Terminated" etc.
    start_date: Optional[str]
    completion_date: Optional[str]
    description: Optional[str]       # Brief summary — main text mining input
    conditions: Optional[str]        # comma-separated condition list
    sponsor: Optional[str]
    enrollment: Optional[int]
    scraped_at: str = field(default_factory=now_iso)


# ── shortages.csv ─────────────────────────────────────────────────────────────
@dataclass
class ShortageRecord:
    drug_name: str
    generic_name: Optional[str]
    dosage_form: Optional[str]       # "Tablet" | "Injection" | "Solution"
    shortage_reason: Optional[str]   # Free text — key for topic modeling
    start_date: Optional[str]
    end_date: Optional[str]          # None if ongoing
    status: str                      # "Current" | "Resolved"
    company: Optional[str]
    availability: Optional[str]      # "Available" | "Unavailable" | "Limited Availability"
    therapeutic_category: Optional[str]  # e.g. "Anti-Infective", "Cardiovascular"
    source_url: Optional[str]
    is_target_drug: bool = False     # True if in our 50 core monitored drugs
    scraped_at: str = field(default_factory=now_iso)


# ── prices.csv ────────────────────────────────────────────────────────────────
@dataclass
class PriceRecord:
    drug_name: str
    generic_name: Optional[str]
    brand_name: Optional[str]
    price: float                     # USD per unit (tablet/mL) for NADAC; per fill for GoodRx
    pharmacy: str                    # "NADAC/CMS (Generic)" | "CVS" | "Walgreens" etc.
    dosage: Optional[str]            # Full NDC description e.g. "METFORMIN HCL 500 MG TABLET"
    quantity: Optional[str]          # "per EA | 30-day=$0.55 | 90-day=$1.65"
    with_coupon: bool                # False for NADAC (acquisition cost); True for GoodRx
    scraped_at: str = field(default_factory=now_iso)


# ── adverse_events.csv ────────────────────────────────────────────────────────
@dataclass
class AdverseEventRecord:
    drug_name: str
    generic_name: Optional[str]
    event_type: str                  # e.g. "Nausea", "Cardiac arrest"
    severity: Optional[str]         # "serious" | "non-serious"
    outcome: Optional[str]          # "death" | "hospitalization" | "recovered"
    patient_age: Optional[str]
    patient_sex: Optional[str]
    report_date: Optional[str]
    report_id: Optional[str]        # FDA report ID
    source: str = "openfda"
    scraped_at: str = field(default_factory=now_iso)


# ── pubmed_abstracts.csv ──────────────────────────────────────────────────────
@dataclass
class PubMedRecord:
    drug_name: str
    pmid: str                        # PubMed ID
    title: str
    abstract: Optional[str]
    authors: Optional[str]           # comma-separated
    journal: Optional[str]
    pub_year: Optional[int]
    pub_date: Optional[str]
    citation_count: Optional[int]    # may not be available from free API
    keywords: Optional[str]          # MeSH keywords
    source: str = "pubmed"
    scraped_at: str = field(default_factory=now_iso)
