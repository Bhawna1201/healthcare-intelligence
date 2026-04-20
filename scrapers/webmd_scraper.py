"""
scrapers/webmd_scraper.py
─────────────────────────────────────────────────────────────────────────────
Collects patient drug reviews from WebMD Reviews.

Source: https://reviews.webmd.com/drugs/drugreview-{slug}
Method: requests + BeautifulSoup (server-side rendered HTML, no Selenium)

Confirmed selectors (from live DevTools inspection, 2026-03-09):
  Card container  : div.review-details-holder
  Reviewer name   : div.card-header div.details span:first-child
  Date            : div.card-header div.date
  Condition       : strong.condition  (strip "Condition: " prefix)
  Overall rating  : div.overall-rating div.webmd-rate[aria-valuenow]  (0–5 scale)
  Sub-ratings     : div.categories section → strong + div.webmd-rate[aria-valuenow]
  Review text     : div.description p.description-text
  Helpful votes   : div.helpful span.likes

Rating scale:
  WebMD uses 0–5. We convert to 1–10 for ReviewRecord schema consistency.

Output: data/raw/reviews.csv  (appends, deduplicates by review_id)

Usage:
  python scrapers/webmd_scraper.py --test            # 3 drugs, 2 pages each
  python scrapers/webmd_scraper.py                   # all 50 drugs
  python scrapers/webmd_scraper.py --drug metformin  # single drug
  python scrapers/webmd_scraper.py --reset           # clear checkpoint
"""

import sys
import re
import time
import random
import hashlib
from pathlib import Path
from typing import Optional
from datetime import datetime

from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).parent.parent))

from scrapers.utils.base import make_session, SchemaCSVWriter, Checkpoint, get_logger
from scrapers.utils.schemas import ReviewRecord
from config.settings import RAW_DIR, TARGET_DRUGS, random_delay

log = get_logger("webmd_scraper")

# ── URL config ────────────────────────────────────────────────────────────────
BASE_URL    = "https://reviews.webmd.com/drugs/drugreview-{slug}"
MAX_PAGES   = 15    # up to 15 pages per drug (~150 reviews)
PAGE_SIZE   = 10    # WebMD shows 10 reviews per page

# ── Drug slug map ─────────────────────────────────────────────────────────────
# WebMD appends brand names to slug: drugreview-metformin-glucophage-glumetza-and-others
# Mapped from: reviews.webmd.com/drugs/drugreview-{drug} redirect targets
# If slug not in map, falls back to plain drug name (works for many drugs)
DRUG_SLUG_MAP = {
    "metformin":              "metformin-glucophage-glumetza-and-others",
    "lisinopril":             "lisinopril-prinivil-zestril",
    "atorvastatin":           "atorvastatin-lipitor",
    "levothyroxine":          "levothyroxine-synthroid-and-others",
    "amlodipine":             "amlodipine-norvasc",
    "metoprolol":             "metoprolol-tartrate-lopressor",
    "omeprazole":             "omeprazole-prilosec",
    "simvastatin":            "simvastatin-zocor",
    "losartan":               "losartan-cozaar",
    "albuterol":              "albuterol-proventil-hfa-ventolin-hfa",
    "gabapentin":             "gabapentin-neurontin",
    "hydrochlorothiazide":    "hydrochlorothiazide-microzide",
    "sertraline":             "sertraline-zoloft",
    "montelukast":            "montelukast-singulair",
    "furosemide":             "furosemide-lasix",
    "pantoprazole":           "pantoprazole-protonix",
    "escitalopram":           "escitalopram-lexapro",
    "rosuvastatin":           "rosuvastatin-crestor-ezallor",
    "bupropion":              "bupropion-wellbutrin-xl-zyban",
    "fluoxetine":             "fluoxetine-prozac",
    "clopidogrel":            "clopidogrel-plavix",
    "tramadol":               "tramadol-ultram",
    "cyclobenzaprine":        "cyclobenzaprine-flexeril",
    "amoxicillin":            "amoxicillin-amoxil-moxatag",
    "azithromycin":           "azithromycin-zithromax-z-pak",
    "doxycycline":            "doxycycline-vibramycin",
    "prednisone":             "prednisone-deltasone",
    "methylprednisolone":     "methylprednisolone-medrol",
    "clonazepam":             "clonazepam-klonopin",
    "alprazolam":             "alprazolam-xanax",
    "zolpidem":               "zolpidem-ambien",
    "oxycodone":              "oxycodone-roxicodone",
    "hydrocodone":            "hydrocodone-vicodin-and-others",
    "acetaminophen":          "acetaminophen-tylenol",
    "ibuprofen":              "ibuprofen-advil-motrin-ib",
    "naproxen":               "naproxen-aleve-naprosyn",
    "insulin glargine":       "insulin-glargine-lantus",
    "insulin lispro":         "insulin-lispro-humalog",
    "empagliflozin":          "empagliflozin-jardiance",
    "semaglutide":            "semaglutide-ozempic",
    "dulaglutide":            "dulaglutide-trulicity",
    "apixaban":               "apixaban-eliquis",
    "rivaroxaban":            "rivaroxaban-xarelto",
    "warfarin":               "warfarin-coumadin",
    "digoxin":                "digoxin-lanoxin",
    "diltiazem":              "diltiazem-cardizem",
    "verapamil":              "verapamil-calan-sr",
    "carvedilol":             "carvedilol-coreg",
    "spironolactone":         "spironolactone-aldactone",
    "tamsulosin":             "tamsulosin-flomax",
}


# ── HTML parsers ──────────────────────────────────────────────────────────────

def parse_rating(card, selector: str) -> Optional[float]:
    """Extract 0-5 rating from aria-valuenow, convert to 1-10 scale."""
    el = card.select_one(selector)
    if not el:
        return None
    try:
        raw = float(el.get("aria-valuenow", 0))
        # Convert 0-5 → 1-10 scale (ReviewRecord schema)
        return round(raw * 2, 1)
    except (ValueError, TypeError):
        return None


def parse_card(card, drug_name: str) -> Optional[ReviewRecord]:
    """
    Parse a single review card into a ReviewRecord.

    All selectors confirmed from live DevTools inspection.
    """
    # ── Reviewer name ──────────────────────────────────────────────────────
    name_el = card.select_one("div.card-header div.details span:first-child")
    reviewer = name_el.text.strip().rstrip("|").strip() if name_el else "Anonymous"

    # ── Date ───────────────────────────────────────────────────────────────
    date_el = card.select_one("div.card-header div.date")
    date_raw = date_el.text.strip() if date_el else None
    # Normalise M/D/YYYY → YYYY-MM-DD
    date_iso = None
    if date_raw:
        try:
            date_iso = datetime.strptime(date_raw, "%m/%d/%Y").strftime("%Y-%m-%d")
        except ValueError:
            date_iso = date_raw  # keep as-is if unexpected format

    # ── Condition ──────────────────────────────────────────────────────────
    cond_el = card.select_one("strong.condition")
    condition = None
    if cond_el:
        condition = cond_el.text.strip()
        condition = re.sub(r"^Condition\s*:\s*", "", condition, flags=re.IGNORECASE).strip()

    # ── Overall rating (aria-valuenow on 0-5 scale → convert to 1-10) ─────
    rating = parse_rating(card, "div.overall-rating div.webmd-rate")

    # ── Review text ────────────────────────────────────────────────────────
    text_el = card.select_one("div.description p.description-text")
    review_text = text_el.get_text(separator=" ").strip() if text_el else ""
    if not review_text:
        return None  # skip empty reviews

    # ── Helpful votes ──────────────────────────────────────────────────────
    likes_el = card.select_one("div.helpful span.likes")
    helpful = None
    if likes_el:
        try:
            helpful = int(likes_el.text.strip())
        except ValueError:
            helpful = 0

    # ── Review ID for deduplication ────────────────────────────────────────
    # WebMD doesn't expose a review ID in HTML, so build a stable hash
    # from drug + reviewer + date + first 80 chars of text
    id_src = f"{drug_name}|{reviewer}|{date_iso}|{review_text[:80]}"
    review_id = hashlib.md5(id_src.encode()).hexdigest()[:12]

    return ReviewRecord(
        drug_name     = drug_name,
        rating        = rating,
        review_text   = review_text,
        condition     = condition,
        date          = date_iso,
        source        = "webmd",
        review_id     = review_id,
        helpful_votes = helpful,
    )


# ── Page fetcher ──────────────────────────────────────────────────────────────

def get_slug(drug_name: str) -> str:
    """Return WebMD URL slug for a drug."""
    return DRUG_SLUG_MAP.get(drug_name.lower(), drug_name.lower().replace(" ", "-"))


def fetch_page(session, slug: str, page: int) -> Optional[BeautifulSoup]:
    """
    Fetch one page of reviews. Returns parsed BeautifulSoup or None.
    WebMD uses ?page=N for pagination (1-indexed).
    """
    url = BASE_URL.format(slug=slug)
    params = {"page": page} if page > 1 else {}

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer":         "https://www.webmd.com/",
    }

    try:
        resp = session.get(url, params=params, headers=headers, timeout=20)
        if resp.status_code == 404:
            log.warning(f"  404 for slug: {slug}")
            return None
        if resp.status_code != 200:
            log.warning(f"  HTTP {resp.status_code} for {slug} page {page}")
            return None
        return BeautifulSoup(resp.text, "html.parser")
    except Exception as e:
        log.warning(f"  Fetch error ({slug} p{page}): {e}")
        return None


# ── Per-drug scraper ──────────────────────────────────────────────────────────

def scrape_drug_reviews(
    session,
    drug_name: str,
    writer: SchemaCSVWriter,
    max_pages: int = MAX_PAGES,
) -> int:
    """Scrape all review pages for one drug. Returns count written."""
    slug = get_slug(drug_name)
    log.info(f"  WebMD reviews: {drug_name}  → slug: {slug}")

    total_written = 0
    seen_ids = set()

    for page in range(1, max_pages + 1):
        soup = fetch_page(session, slug, page)
        if not soup:
            log.info(f"  Page {page}: no response — stopping")
            break

        cards = soup.select("div.review-details-holder")
        if not cards:
            log.info(f"  Page {page}: no review cards — end of reviews")
            break

        page_records = []
        for card in cards:
            rec = parse_card(card, drug_name)
            if rec and rec.review_id not in seen_ids:
                seen_ids.add(rec.review_id)
                page_records.append(rec)

        written = writer.write_many(page_records)
        total_written += written
        log.info(f"  Page {page}: {len(cards)} cards → {written} new records")

        # Stop early if page was mostly empty (near end of reviews)
        if len(cards) < PAGE_SIZE:
            log.info(f"  Page {page}: fewer than {PAGE_SIZE} cards — last page reached")
            break

        # Polite delay between pages
        time.sleep(random.uniform(1.5, 3.0))

    log.info(f"  ✓ {drug_name}: {total_written} reviews total")
    return total_written


# ── Main runner ───────────────────────────────────────────────────────────────

def run(drugs: list = None, max_pages_per_drug: int = MAX_PAGES):
    """
    Main entry — scrape WebMD reviews for all target drugs.

    Args:
        drugs:              list of drug names (default: TARGET_DRUGS)
        max_pages_per_drug: max pages to scrape per drug (default: 15)
    """
    drug_list   = drugs or TARGET_DRUGS
    output_path = RAW_DIR / "reviews.csv"
    checkpoint  = Checkpoint("webmd_reviews")
    session     = make_session()

    log.info("=" * 65)
    log.info("WebMD Review Scraper — confirmed selectors 2026-03-09")
    log.info(f"Drugs: {len(drug_list)} | Max pages/drug: {max_pages_per_drug}")
    log.info(f"Output: {output_path}")
    log.info("=" * 65)

    total_written = 0
    skipped       = 0

    with SchemaCSVWriter(output_path, ReviewRecord) as writer:
        for i, drug in enumerate(drug_list, 1):
            if checkpoint.is_done(drug):
                log.info(f"[{i}/{len(drug_list)}] Skipping {drug} (checkpointed)")
                skipped += 1
                continue

            log.info(f"\n[{i}/{len(drug_list)}] {drug}")
            written = scrape_drug_reviews(
                session, drug, writer, max_pages=max_pages_per_drug
            )
            total_written += written
            checkpoint.mark_done(drug)

            # Polite delay between drugs
            if i < len(drug_list):
                time.sleep(random.uniform(2.0, 4.0))

    file_size_kb = output_path.stat().st_size / 1024 if output_path.exists() else 0

    log.info("\n" + "=" * 65)
    log.info("WebMD scrape complete")
    log.info(f"  Written : {total_written} reviews")
    log.info(f"  Skipped : {skipped} (checkpointed)")
    log.info(f"  File    : {output_path} ({file_size_kb:.1f} KB)")
    log.info("=" * 65)

    return {
        "written": total_written,
        "skipped": skipped,
        "file":    str(output_path),
        "size_kb": round(file_size_kb, 1),
    }


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="WebMD Drug Review Scraper")
    p.add_argument("--drug",      type=str, help="Single drug to scrape")
    p.add_argument("--test",      action="store_true", help="Test: 3 drugs, 2 pages each")
    p.add_argument("--reset",     action="store_true", help="Clear checkpoint and re-scrape")
    p.add_argument("--max-pages", type=int, default=MAX_PAGES, help=f"Max pages per drug (default: {MAX_PAGES})")
    args = p.parse_args()

    if args.reset:
        Checkpoint("webmd_reviews").reset()
        log.info("Checkpoint cleared.")

    if args.drug:
        drugs     = [args.drug]
        max_pages = args.max_pages
    elif args.test:
        drugs     = ["metformin", "lisinopril", "atorvastatin"]
        max_pages = 2
    else:
        drugs     = None
        max_pages = args.max_pages

    run(drugs=drugs, max_pages_per_drug=max_pages)
