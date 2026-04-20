"""
scrapers/fda_shortages_scraper.py  (v5 — OpenFDA API primary + HTML reason enrichment)
────────────────────────────────────────────────────────────────────────────────────────
Strategy:
  Pass 1: OpenFDA API /drug/shortages.json
          - Returns 1,677 records (all current shortages), structured JSON
          - Has: generic_name, company_name, dosage_form, availability,
                 status, presentation, initial_posting_date
          - Missing: shortage_reason (not in API)
          - Filter to our 50 TARGET_DRUGS → get all their NDC rows

  Pass 2: FDA HTML detail page (only for TARGET_DRUGS)
          - Visit dsp_ActiveIngredientDetails.cfm per drug
          - Extract shortage_reason per company from the 4-column table
          - Merge reasons back into API records by (drug_name, company_name)

This gives us:
  - COMPLETE data for our 50 drugs (API fields + HTML reason)
  - shortage_reason only populated for Unavailable/Limited rows (FDA only lists
    reasons when a product is actually constrained — Available rows have none)
  - No scraping needed for the 590 non-target shortages

API endpoint: https://api.fda.gov/drug/shortages.json
  Pagination:  ?limit=100&skip=N
  Total:       1,677 records
  Filter:      search=generic_name:{drug} (or full paginated download)
"""

import sys, re, time, random, json
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from bs4 import BeautifulSoup
from scrapers.utils.base import (make_session, rate_limited_get,
                                 SchemaCSVWriter, Checkpoint, get_logger)
from scrapers.utils.schemas import ShortageRecord
from config.settings import RAW_DIR, TARGET_DRUGS, random_headers

log = get_logger("fda_shortages_v5")

FDA_API_BASE   = "https://api.fda.gov/drug/shortages.json"
FDA_DETAIL_URL = "https://www.accessdata.fda.gov/scripts/drugshortages/dsp_ActiveIngredientDetails.cfm"
CACHE_PATH     = RAW_DIR / "debug" / "fda_api_cache.json"


# ── Pass 1: OpenFDA API ───────────────────────────────────────────────────────

def fetch_all_shortages_api(session) -> list[dict]:
    """
    Paginate through ALL records in OpenFDA drug shortages API.
    Returns raw list of result dicts.
    Total ~1,677 records, 100 per page = ~17 requests.
    """
    all_records = []
    limit = 100
    skip  = 0

    while True:
        url = f"{FDA_API_BASE}?limit={limit}&skip={skip}"
        resp = rate_limited_get(session, url, headers={"User-Agent": "Mozilla/5.0"},
                                delay_min=0.3, delay_max=0.8, logger=log)
        if not resp:
            log.warning(f"  No response at skip={skip}, stopping pagination")
            break

        try:
            data = resp.json()
        except Exception:
            log.warning(f"  JSON parse error at skip={skip}")
            break

        results = data.get("results", [])
        if not results:
            break

        all_records.extend(results)
        total = data.get("meta", {}).get("results", {}).get("total", 0)
        log.info(f"  Fetched {len(all_records)}/{total} records...")

        if len(all_records) >= total or len(results) < limit:
            break
        skip += limit

    return all_records


def filter_target_drugs(all_records: list[dict], target_drugs: list) -> dict:
    """
    Filter API records to our 50 target drugs.
    Returns dict: {drug_canonical → [api_record, ...]}
    """
    matched = {}
    for rec in all_records:
        name = (rec.get("generic_name") or "").lower()
        for drug in target_drugs:
            if drug.lower() in name:
                matched.setdefault(drug, []).append(rec)
                break
    return matched


# ── Pass 2: HTML reason extraction (target drugs only) ───────────────────────

def fetch_reasons_for_drug(session, drug_name: str) -> dict:
    """
    Fetch detail page for a shortage drug name.
    Returns dict: {company_name_lower → reason_text}

    Page structure (confirmed):
      CompanyName (Reverified DATE)
        | Presentation | Availability | Related | Shortage Reason |
        | Drug ...     | Unavailable  |         | GMP issue       |
        | Drug ...     | Available    |         |                 |
    """
    # URL-encode the drug name for the detail page
    from urllib.parse import quote_plus
    encoded = quote_plus(drug_name)
    url = f"{FDA_DETAIL_URL}?AI={encoded}&st=c&tab=tabs-1"

    time.sleep(random.uniform(0.8, 1.5))
    resp = rate_limited_get(session, url, headers=random_headers(),
                            delay_min=0.3, delay_max=0.6, logger=log)
    if not resp or len(resp.text) < 200:
        return {}

    soup = BeautifulSoup(resp.text, "lxml")
    company_pattern = re.compile(
        r'^(.+?)\s*\((?:Reverified|Revised|Updated|Verified)\s+\d{2}/\d{2}/\d{4}\)',
        re.IGNORECASE
    )

    reasons_by_company = {}
    current_company = None
    current_reasons = []

    def flush():
        if current_company:
            unique = list(dict.fromkeys(r for r in current_reasons if r))
            if unique:
                reasons_by_company[current_company.lower()] = " | ".join(unique)[:1000]

    for tag in (soup.find("body") or soup).descendants:
        if not hasattr(tag, 'name') or not tag.name:
            continue

        if tag.name in ("b", "strong", "h2", "h3", "h4", "td"):
            text = tag.get_text(strip=True)
            m = company_pattern.match(text)
            if m:
                flush()
                current_company = m.group(1).strip()
                current_reasons = []
                continue

        if tag.name == "tr" and current_company:
            cells = tag.find_all("td")
            if len(cells) >= 4:
                r = cells[3].get_text(strip=True)
                if r and len(r) > 5:
                    current_reasons.append(r)

    flush()
    return reasons_by_company


# ── Build ShortageRecord from API record + optional reason ───────────────────

def api_record_to_shortage(rec: dict, reason: Optional[str],
                           is_target: bool = False) -> ShortageRecord:
    # Parse dosage form
    forms = ["Tablet", "Injection", "Solution", "Suspension", "Capsule",
             "Spray", "Ointment", "Cream", "Patch", "Powder", "Concentrate",
             "Infusion", "Implant", "Drops", "Syrup"]
    name = rec.get("generic_name", "")
    dosage_form = rec.get("dosage_form") or next(
        (f for f in forms if f.lower() in name.lower()), None
    )

    # therapeutic_category can be a list in the API
    tc = rec.get("therapeutic_category")
    if isinstance(tc, list):
        tc = ", ".join(tc)

    return ShortageRecord(
        drug_name            = name,
        generic_name         = name,
        dosage_form          = dosage_form,
        shortage_reason      = reason,
        start_date           = rec.get("initial_posting_date"),
        end_date             = None,
        status               = rec.get("status", "Current"),
        company              = rec.get("company_name"),
        availability         = rec.get("availability"),
        therapeutic_category = tc,
        is_target_drug       = is_target,
        source_url           = FDA_API_BASE,
    )


# ── Main runner ───────────────────────────────────────────────────────────────

def run(drugs: list = None):
    target_drugs = drugs or TARGET_DRUGS
    output_path  = RAW_DIR / "shortages.csv"
    checkpoint   = Checkpoint("fda_shortages_v5")
    session      = make_session()

    log.info("=" * 65)
    log.info("FDA Drug Shortages Scraper v5")
    log.info(f"  Pass 1: OpenFDA API  → all 1,677 shortage records")
    log.info(f"  Pass 2: HTML detail  → shortage_reason for target drugs only")
    log.info(f"  Output: {output_path}")
    log.info("=" * 65)

    # ── Pass 1: Full API download ──────────────────────────────────────────
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not checkpoint.is_done("api_pass"):
        log.info("\n[Pass 1] Downloading all shortages from OpenFDA API...")
        all_records = fetch_all_shortages_api(session)
        CACHE_PATH.write_text(json.dumps(all_records), encoding="utf-8")
        checkpoint.mark_done("api_pass")
        log.info(f"  {len(all_records)} records cached → {CACHE_PATH}")
    else:
        all_records = json.loads(CACHE_PATH.read_text())
        log.info(f"[Pass 1] Loaded {len(all_records)} records from cache")

    # Tag target drugs (our 50 core drugs), but keep ALL records
    matched = filter_target_drugs(all_records, target_drugs)
    log.info(f"\n  Core target drugs found in shortage data: {len(matched)}/50")
    for drug, recs in sorted(matched.items()):
        log.info(f"    ★ {drug}: {len(recs)} NDC records")
    log.info(f"  Total ALL drugs in shortage: {len(all_records)} records")

    # ── Pass 2: HTML reason enrichment for ALL drugs with Unavailable/Limited status ──
    # Build map of all unique drug names that have constrained availability
    constrained_recs = {}
    for rec in all_records:
        avail = (rec.get("availability") or "").lower()
        if "unavailable" in avail or "limited" in avail:
            drug_name = (rec.get("generic_name") or "").strip()
            if drug_name:
                constrained_recs.setdefault(drug_name, []).append(rec)

    log.info(f"\n[Pass 2] Fetching shortage reasons for {len(constrained_recs)} constrained drugs (Unavailable/Limited)...")
    log.info(f"  NOTE: shortage_reason is only listed by FDA for Unavailable/Limited products")
    log.info(f"  'Available' products will have no reason — that's expected FDA behavior\n")

    reasons_map = {}  # drug_name → {company_lower → reason_text}

    for drug_name, recs in constrained_recs.items():
        # Use the exact generic_name from API for URL
        shortage_name = recs[0].get("generic_name", drug_name)
        cp_key = f"reason_{re.sub(r'[^a-z0-9]', '_', drug_name.lower()[:40])}"

        if checkpoint.is_done(cp_key):
            debug_path = RAW_DIR / "debug" / f"reasons_{re.sub(r'[^a-z0-9]+', '_', drug_name.lower()[:60])}.json"
            if debug_path.exists():
                reasons_map[drug_name] = json.loads(debug_path.read_text())
            continue

        log.info(f"  Fetching reasons: {shortage_name}")
        company_reasons = fetch_reasons_for_drug(session, shortage_name)
        reasons_map[drug_name] = company_reasons

        debug_path = RAW_DIR / "debug" / f"reasons_{re.sub(r'[^a-z0-9]+', '_', drug_name.lower()[:60])}.json"
        debug_path.write_text(json.dumps(company_reasons))
        checkpoint.mark_done(cp_key)

        n_with_reason = len([r for r in company_reasons.values() if r])
        log.info(f"    → {len(company_reasons)} companies, {n_with_reason} with reasons")

    # ── Write output ───────────────────────────────────────────────────────
    log.info(f"\n[Writing] Merging API data + reasons → {output_path}")
    total_written = 0
    target_set = set(matched.keys())

    def get_reason(rec: dict) -> str | None:
        drug_key = (rec.get("generic_name") or "").strip()
        company  = (rec.get("company_name") or "").lower()
        drug_reasons = reasons_map.get(drug_key, {})
        for co_key, r in drug_reasons.items():
            if co_key in company or company in co_key:
                return r
        return None

    with SchemaCSVWriter(output_path, ShortageRecord) as writer:
        for rec in all_records:
            name = (rec.get("generic_name") or "").strip()
            is_target = any(t.lower() in name.lower() for t in target_drugs)
            reason = get_reason(rec)
            writer.write(api_record_to_shortage(rec, reason, is_target=is_target))
            total_written += 1

    file_kb = output_path.stat().st_size / 1024 if output_path.exists() else 0
    log.info("\n" + "=" * 65)
    log.info(f"Complete | {total_written} rows | {file_kb:.1f} KB")
    log.info(f"  Target drugs in shortage: {len(matched)}")
    log.info(f"  Rows with shortage_reason: check target drug rows")
    log.info(f"  shortage_reason blank = product is 'Available' (expected)")
    log.info("=" * 65)

    return {"written": total_written, "file": str(output_path), "size_kb": round(file_kb, 1)}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="FDA Drug Shortages Scraper v5")
    p.add_argument("--reset", action="store_true", help="Clear checkpoint and re-scrape")
    p.add_argument("--test",  action="store_true", help="API first 50 records + 3 detail pages")
    args = p.parse_args()

    if args.reset:
        Checkpoint("fda_shortages_v5").reset()
        CACHE_PATH.unlink(missing_ok=True)
        log.info("Checkpoint + cache cleared.")

    if args.test:
        session = make_session()
        # Quick API test
        url = f"{FDA_API_BASE}?limit=5&skip=0"
        resp = rate_limited_get(session, url, headers={"User-Agent": "Mozilla/5.0"},
                                delay_min=0, delay_max=0, logger=log)
        data = resp.json()
        log.info(f"API test: total={data['meta']['results']['total']}, "
                 f"first drug={data['results'][0]['generic_name']}")

        # Fetch reasons for first matched target drug
        all_recs = fetch_all_shortages_api(session)
        matched = filter_target_drugs(all_recs, TARGET_DRUGS)
        log.info(f"Target drugs in shortage: {list(matched.keys())[:10]}")

        for drug in list(matched.keys())[:3]:
            shortage_name = matched[drug][0]["generic_name"]
            log.info(f"Fetching reasons for: {shortage_name}")
            reasons = fetch_reasons_for_drug(session, shortage_name)
            log.info(f"  → {len(reasons)} companies with reasons: {list(reasons.values())[:3]}")
    else:
        run()
