"""
scrapers/nadac_scraper.py
─────────────────────────────────────────────────────────────────────────────
Collects drug pricing data from the CMS NADAC (National Average Drug
Acquisition Cost) database — the US government's official weekly drug
pricing dataset, published by Medicaid every Wednesday.

API:  https://data.medicaid.gov (Socrata Open Data API, no key needed)
Docs: https://data.medicaid.gov/dataset/nadac-national-average-drug-acquisition-cost

What NADAC measures:
  - The actual price pharmacies PAY to acquire drugs from wholesalers
  - Updated weekly (every Wednesday)
  - Covers ~50,000 NDC codes (National Drug Codes)
  - Separate records per manufacturer, strength, and dosage form

Why this is better than GoodRx for research:
  - Official government data (citable in academic work)
  - No bot protection / CAPTCHAs
  - Clean structured JSON
  - Includes BOTH generic and brand prices for comparison

Output CSV columns:
  drug_name        - our normalized search term (e.g. "metformin")
  ndc              - 11-digit National Drug Code (unique per product)
  ndc_description  - full raw name (e.g. "METFORMIN HCL 500 MG TABLET")
  nadac_per_unit   - price per tablet/ml in USD
  price_per_30     - calculated: nadac_per_unit × 30
  price_per_90     - calculated: nadac_per_unit × 90
  pricing_unit     - EA (each/tablet) or ML (millilitre)
  drug_type        - G (Generic) or B (Brand)
  otc              - Y (over-the-counter) or N (prescription)
  as_of_date       - date this price was effective
  pharmacy         - "NADAC/CMS" (source identifier)
  dosage           - strength extracted from ndc_description
  quantity         - pricing_unit label

Usage:
  python scrapers/nadac_scraper.py --test          # 3 drugs
  python scrapers/nadac_scraper.py                 # all 50 drugs
  python scrapers/nadac_scraper.py --drug warfarin # single drug
"""

import sys
import re
import json
import time
import random
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from scrapers.utils.base import make_session, rate_limited_get, SchemaCSVWriter, Checkpoint, get_logger
from scrapers.utils.schemas import PriceRecord
from config.settings import RAW_DIR, TARGET_DRUGS

log = get_logger("nadac_scraper")

# ── API config ────────────────────────────────────────────────────────────────
# CMS Socrata Open Data API — no authentication required
# Dataset ID: f38d0706-1239-442c-a3cc-40ef1b686ac0
NADAC_API = (
    "https://data.medicaid.gov/api/1/datastore/query"
    "/f38d0706-1239-442c-a3cc-40ef1b686ac0/0"
)
PAGE_SIZE   = 500    # max records per API call
MAX_RECORDS = 2000   # cap per drug (avoids huge files for common drugs)


# ── Strength extractor ────────────────────────────────────────────────────────

def extract_strength(description: str) -> Optional[str]:
    """
    Parse dosage strength from NADAC description string.

    Examples:
      "METFORMIN HCL 500 MG TABLET"       → "500 MG"
      "LISINOPRIL 10 MG TABLET"            → "10 MG"
      "ALBUTEROL 0.083 % SOLUTION"         → "0.083 %"
      "INSULIN GLARGINE 100 UNIT/ML INJ"   → "100 UNIT/ML"
      "AMOXICILLIN 250 MG/5 ML SUSP"       → "250 MG/5 ML"
    """
    patterns = [
        r'(\d+(?:\.\d+)?\s*(?:MG/ML|MCG/ML|UNIT/ML|MG/5\s*ML|MG/ML|MEQ/ML))',  # complex units
        r'(\d+(?:\.\d+)?\s*(?:MG|MCG|UNIT|MEQ|GM|ML|%)\b)',                       # simple units
    ]
    for pat in patterns:
        m = re.search(pat, description, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return None


def extract_form(description: str) -> Optional[str]:
    """
    Extract dosage form from NADAC description.

    Examples:
      "METFORMIN HCL 500 MG TABLET"   → "TABLET"
      "ALBUTEROL 0.083 % SOLUTION"    → "SOLUTION"
      "LISINOPRIL 10 MG TABLET ER"    → "TABLET ER"
    """
    forms = [
        "TABLET ER", "TABLET SA", "TABLET DR", "CAPSULE ER", "CAPSULE DR",
        "TABLET", "CAPSULE", "SOLUTION", "SUSPENSION", "INJECTION",
        "CREAM", "OINTMENT", "GEL", "PATCH", "INHALER", "SPRAY",
        "DROPS", "SYRUP", "ELIXIR", "POWDER", "SUPPOSITORY",
    ]
    desc_upper = description.upper()
    for form in forms:
        if form in desc_upper:
            return form
    return None


# ── API fetcher ───────────────────────────────────────────────────────────────

def fetch_nadac_records(session, drug_name: str) -> list[dict]:
    """
    Query NADAC API for all NDC records matching a drug name.

    Uses LIKE query on ndc_description — gets all strengths, forms,
    and manufacturer variants. Sorted by most recent date first.

    Returns list of raw API result dicts.
    """
    all_records = []
    offset = 0
    search_term = drug_name.upper()

    # Some drugs need special search terms for NADAC
    # e.g. "insulin glargine" → search "GLARGINE" (more records)
    search_map = {
        "insulin glargine": "GLARGINE",
        "insulin lispro":   "LISPRO",
        "semaglutide":      "SEMAGLUTIDE",
        "dulaglutide":      "DULAGLUTIDE",
        "empagliflozin":    "EMPAGLIFLOZIN",
    }
    search_term = search_map.get(drug_name.lower(), search_term)

    log.debug(f"  NADAC query: ndc_description LIKE '%{search_term}%'")

    while offset < MAX_RECORDS:
        params = {
            "limit":  PAGE_SIZE,
            "offset": offset,
            # Filter: ndc_description contains drug name (case-insensitive via LIKE)
            "conditions[0][property]": "ndc_description",
            "conditions[0][value]":    f"%{search_term}%",
            "conditions[0][operator]": "LIKE",
            # Sort: newest records first
            "sort[0][property]": "as_of_date",
            "sort[0][order]":    "desc",
        }

        resp = rate_limited_get(
            session, NADAC_API, params=params,
            headers={"Accept": "application/json"},
            logger=log,
        )

        if not resp:
            log.warning(f"  No response for {drug_name} at offset {offset}")
            break

        try:
            data = resp.json()
        except Exception as e:
            log.warning(f"  JSON parse error: {e}")
            break

        results = data.get("results", [])
        if not results:
            break

        all_records.extend(results)
        total = data.get("count", 0)
        log.debug(f"  offset={offset} | got={len(results)} | total={total}")

        # Stop if we have all records or hit our cap
        if offset + PAGE_SIZE >= min(total, MAX_RECORDS):
            break
        offset += PAGE_SIZE

        # Polite delay between pages
        time.sleep(random.uniform(0.3, 0.7))

    return all_records


# ── Record builder ────────────────────────────────────────────────────────────

def build_price_records(drug_name: str, raw_records: list[dict]) -> list[PriceRecord]:
    """
    Convert raw NADAC API dicts into PriceRecord dataclasses.

    Deduplicates by (ndc, as_of_date) to avoid re-importing the same
    weekly snapshot if the scraper is re-run.
    """
    records = []
    seen = set()

    for r in raw_records:
        ndc             = r.get("ndc", "").strip()
        ndc_description = r.get("ndc_description", "").strip()
        nadac_per_unit  = r.get("nadac_per_unit", "")
        pricing_unit    = r.get("pricing_unit", "EA").strip()
        drug_type       = r.get("drug_type", "").strip()        # G or B
        otc             = r.get("otc", "N").strip()
        as_of_date      = r.get("as_of_date", "").strip()

        # Parse price
        try:
            unit_price = float(str(nadac_per_unit).replace(",", "").strip())
        except (ValueError, TypeError):
            continue

        if unit_price <= 0:
            continue

        # Deduplication key
        dedup_key = (ndc, as_of_date)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        # Calculate standard quantities
        # NADAC prices are per tablet (EA) or per mL — compute standard fills
        price_per_30 = round(unit_price * 30, 4)
        price_per_90 = round(unit_price * 90, 4)

        # Extract dosage strength and form from description
        strength = extract_strength(ndc_description)
        form     = extract_form(ndc_description)

        # Build descriptive dosage string
        dosage_str = ndc_description  # full description as dosage field

        # Brand vs generic label
        type_label = "Brand" if drug_type == "B" else "Generic"

        # Build pharmacy label with drug type for clarity
        pharmacy_label = f"NADAC/CMS ({type_label})"

        records.append(PriceRecord(
            drug_name    = drug_name,
            generic_name = drug_name,
            brand_name   = ndc_description if drug_type == "B" else None,
            price        = unit_price,           # per-unit price (per tablet/mL)
            pharmacy     = pharmacy_label,
            dosage       = dosage_str,           # full NDC description for Phase 2 parsing
            quantity     = f"per {pricing_unit} | 30-day=${price_per_30} | 90-day=${price_per_90}",
            with_coupon  = False,                # NADAC = acquisition cost, not coupon
        ))

    return records


# ── Per-drug scraper ──────────────────────────────────────────────────────────

def scrape_drug_prices(session, drug_name: str, writer: SchemaCSVWriter) -> int:
    """Fetch NADAC prices for one drug. Returns count written."""
    log.info(f"  Querying NADAC for: {drug_name}")

    raw = fetch_nadac_records(session, drug_name)
    if not raw:
        log.warning(f"  No NADAC records found for: {drug_name}")
        return 0

    records = build_price_records(drug_name, raw)
    written = writer.write_many(records)

    # Summary stats
    if records:
        prices = [r.price for r in records]
        generics = [r for r in records if "Generic" in r.pharmacy]
        brands   = [r for r in records if "Brand"   in r.pharmacy]
        log.info(
            f"  ✓ {drug_name}: {written} records | "
            f"generic={len(generics)} brand={len(brands)} | "
            f"price range ${min(prices):.4f}–${max(prices):.4f}/unit"
        )
    return written


# ── Main runner ───────────────────────────────────────────────────────────────

def run(drugs: list = None):
    """
    Main entry point — scrape NADAC prices for all target drugs.

    Args:
        drugs: list of drug names (default: TARGET_DRUGS from settings.py)
    """
    drug_list   = drugs or TARGET_DRUGS
    output_path = RAW_DIR / "prices.csv"
    checkpoint  = Checkpoint("nadac_prices")
    session     = make_session()

    log.info("=" * 65)
    log.info("NADAC Price Scraper — CMS National Average Drug Acquisition Cost")
    log.info(f"API: {NADAC_API}")
    log.info(f"Drugs: {len(drug_list)} | Output: {output_path}")
    log.info("=" * 65)

    total_written = 0
    skipped = 0

    with SchemaCSVWriter(output_path, PriceRecord) as writer:
        for i, drug in enumerate(drug_list, 1):
            if checkpoint.is_done(drug):
                log.info(f"[{i}/{len(drug_list)}] Skipping {drug} (checkpointed)")
                skipped += 1
                continue

            log.info(f"\n[{i}/{len(drug_list)}] {drug}")
            written = scrape_drug_prices(session, drug, writer)
            total_written += written
            checkpoint.mark_done(drug)

            # Polite delay between drugs (NADAC is a govt API, be a good citizen)
            if i < len(drug_list):
                time.sleep(random.uniform(0.5, 1.5))

        stats = writer.stats()

    file_size_kb = output_path.stat().st_size / 1024 if output_path.exists() else 0

    log.info("\n" + "=" * 65)
    log.info("NADAC scrape complete")
    log.info(f"  Written : {total_written} records")
    log.info(f"  Skipped : {skipped} (checkpointed)")
    log.info(f"  File    : {output_path} ({file_size_kb:.1f} KB)")
    log.info("=" * 65)

    return {
        "written":   total_written,
        "skipped":   skipped,
        "file":      str(output_path),
        "size_kb":   round(file_size_kb, 1),
    }


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="NADAC Drug Price Scraper")
    p.add_argument("--drug",  type=str, help="Single drug name to scrape")
    p.add_argument("--test",  action="store_true", help="Test with 3 drugs")
    p.add_argument("--reset", action="store_true", help="Clear checkpoint and re-scrape")
    args = p.parse_args()

    if args.reset:
        Checkpoint("nadac_prices").reset()
        log.info("Checkpoint cleared.")

    if args.drug:
        drugs = [args.drug]
    elif args.test:
        drugs = ["metformin", "lisinopril", "atorvastatin"]
    else:
        drugs = None

    run(drugs=drugs)
