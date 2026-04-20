"""
config/settings.py
──────────────────
Central configuration for all Phase 1 scrapers.
Loads from .env (never hardcode secrets).
"""

import os
import random
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")


# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent.parent
RAW_DIR        = BASE_DIR / os.getenv("RAW_DATA_DIR", "data/raw")
PROCESSED_DIR  = BASE_DIR / os.getenv("PROCESSED_DATA_DIR", "data/processed")
MASTER_DIR     = BASE_DIR / os.getenv("MASTER_DATA_DIR", "data/master")
LOG_DIR        = BASE_DIR / os.getenv("LOG_DIR", "logs")

# Create directories if they don't exist
for d in [RAW_DIR, PROCESSED_DIR, MASTER_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Request settings ──────────────────────────────────────────────────────────
REQUEST_DELAY_MIN   = float(os.getenv("REQUEST_DELAY_MIN", 2.0))
REQUEST_DELAY_MAX   = float(os.getenv("REQUEST_DELAY_MAX", 4.0))
MAX_RETRIES         = int(os.getenv("MAX_RETRIES", 3))
RETRY_WAIT_SECONDS  = float(os.getenv("RETRY_WAIT_SECONDS", 5.0))

def random_delay() -> float:
    """Return a random delay (seconds) within configured range."""
    import time
    delay = random.uniform(REQUEST_DELAY_MIN, REQUEST_DELAY_MAX)
    time.sleep(delay)
    return delay

# ── API endpoints ─────────────────────────────────────────────────────────────
CLINICALTRIALS_BASE_URL  = os.getenv("CLINICALTRIALS_BASE_URL",
                                      "https://clinicaltrials.gov/api/v2/studies")
CLINICALTRIALS_PAGE_SIZE = int(os.getenv("CLINICALTRIALS_PAGE_SIZE", 100))

OPENFDA_API_KEY          = os.getenv("OPENFDA_API_KEY", "")
OPENFDA_BASE_URL         = os.getenv("OPENFDA_BASE_URL", "https://api.fda.gov")

NCBI_API_KEY             = os.getenv("NCBI_API_KEY", "")
NCBI_EMAIL               = os.getenv("NCBI_EMAIL", "research@example.com")

GOODRX_BASE_URL          = os.getenv("GOODRX_BASE_URL", "https://www.goodrx.com")
FDA_SHORTAGES_URL        = os.getenv("FDA_SHORTAGES_URL",
                                      "https://www.accessdata.fda.gov/scripts/drugshortages")
DRUGSCOM_BASE_URL        = os.getenv("DRUGSCOM_BASE_URL", "https://www.drugs.com")

# ── Drug target list ──────────────────────────────────────────────────────────
# Seed list of common drugs — expand as needed
TARGET_DRUGS = [
    "metformin", "lisinopril", "atorvastatin", "levothyroxine", "amlodipine",
    "metoprolol", "omeprazole", "simvastatin", "losartan", "albuterol",
    "gabapentin", "hydrochlorothiazide", "sertraline", "montelukast",
    "furosemide", "pantoprazole", "escitalopram", "rosuvastatin",
    "bupropion", "fluoxetine", "clopidogrel", "tramadol", "cyclobenzaprine",
    "amoxicillin", "azithromycin", "doxycycline", "prednisone",
    "methylprednisolone", "clonazepam", "alprazolam", "zolpidem",
    "oxycodone", "hydrocodone", "acetaminophen", "ibuprofen", "naproxen",
    "insulin glargine", "insulin lispro", "empagliflozin", "semaglutide",
    "dulaglutide", "apixaban", "rivaroxaban", "warfarin", "digoxin",
    "diltiazem", "verapamil", "carvedilol", "spironolactone", "tamsulosin",
]

# ── User agents (rotate to avoid blocks) ─────────────────────────────────────
USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
]

def random_headers() -> dict:
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    }
