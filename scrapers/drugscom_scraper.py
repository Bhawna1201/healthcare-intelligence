"""
scrapers/drugscom_scraper.py  — Drugs.com Review Scraper
Confirmed selectors: 2026-03-31
URL pattern: https://www.drugs.com/comments/{drug}/
"""

import sys, re, time, random, hashlib
from pathlib import Path
from typing import Optional
from datetime import datetime

from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).parent.parent))

from scrapers.utils.base import make_session, SchemaCSVWriter, Checkpoint, get_logger
from scrapers.utils.schemas import ReviewRecord
from config.settings import RAW_DIR, TARGET_DRUGS

log = get_logger("drugscom_scraper")

BASE_URL = "https://www.drugs.com/comments/{slug}/"
MAX_PAGES = 15
PAGE_SIZE = 25   # drugs.com shows 25 reviews per page

# drugs.com requires these headers — plain User-Agent gets 403
HEADERS = {
    "User-Agent":      "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/123.0.0.0 Safari/537.36",
    "Accept":          "text/html,application/xhtml+xml,application/xml;"
                       "q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection":      "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest":  "document",
    "Sec-Fetch-Mode":  "navigate",
    "Sec-Fetch-Site":  "none",
    "Cache-Control":   "max-age=0",
}

DRUG_SLUG_MAP = {
    # drugs.com uses plain hyphenated names — exceptions only
    "insulin glargine":    "insulin-glargine",
    "insulin lispro":      "insulin-lispro",
    "hydrochlorothiazide": "hydrochlorothiazide",
    "methylprednisolone":  "methylprednisolone",
    "cyclobenzaprine":     "cyclobenzaprine",
    "empagliflozin":       "empagliflozin",
}


def get_slug(drug_name: str) -> str:
    return DRUG_SLUG_MAP.get(
        drug_name.lower(),
        drug_name.lower().replace(" ", "-")
    )


def parse_card(card, drug_name: str) -> Optional[ReviewRecord]:
    """
    Parse one div.ddc-comment card.
    Confirmed selectors from live HTML 2026-03-31:
      Date:      ul.ddc-comment-header li:nth-child(3)  e.g. "September 18, 2020"
      Condition: p > b  e.g. "For High Blood Pressure"
      Text:      p (full text minus the <b> condition)
      Rating:    div.ddc-rating-summary div:first-child  e.g. "2 / 10"
    """
    # Date
    header_li = card.select("ul.ddc-comment-header li")
    date_raw  = header_li[-1].text.strip() if header_li else None
    date_iso  = None
    if date_raw:
        for fmt in ["%B %d, %Y", "%b %d, %Y", "%B %Y"]:
            try:
                date_iso = datetime.strptime(date_raw, fmt).strftime("%Y-%m-%d")
                break
            except ValueError:
                continue

    # Condition
    cond_el   = card.select_one("p b")
    condition = None
    if cond_el:
        condition = re.sub(r"^For\s+", "", cond_el.text.strip(),
                           flags=re.IGNORECASE).strip()
        cond_el.decompose()   # remove from p so it doesn't pollute review text

    # Review text
    p_el = card.select_one("p")
    review_text = p_el.get_text(separator=" ").strip().strip('"').strip() if p_el else ""
    if len(review_text) < 10:
        return None

    # Rating  "2 / 10" → 2.0
    rating = None
    rating_el = card.select_one("div.ddc-rating-summary div:first-child")
    if rating_el:
        try:
            rating = float(rating_el.text.strip().split("/")[0].strip())
        except (ValueError, IndexError):
            pass

    # Dedup ID
    review_id = hashlib.md5(
        f"{drug_name}|{date_iso}|{review_text[:80]}".encode()
    ).hexdigest()[:12]

    return ReviewRecord(
        drug_name     = drug_name,
        rating        = rating,
        review_text   = review_text,
        condition     = condition,
        date          = date_iso,
        source        = "drugs.com",
        review_id     = review_id,
        helpful_votes = None,
    )


def fetch_page(session, slug: str, page: int) -> Optional[BeautifulSoup]:
    url    = BASE_URL.format(slug=slug)
    params = {"page": page} if page > 1 else {}
    try:
        resp = session.get(url, params=params, headers=HEADERS, timeout=20)
        if resp.status_code == 404:
            log.warning(f"  404 for slug: {slug}")
            return None
        if resp.status_code == 403:
            log.warning(f"  403 blocked for {slug} — waiting 10s then retrying")
            time.sleep(10)
            resp = session.get(url, params=params, headers=HEADERS, timeout=20)
            if resp.status_code != 200:
                log.warning(f"  Still {resp.status_code} after retry — skipping")
                return None
        if resp.status_code != 200:
            log.warning(f"  HTTP {resp.status_code} for {slug} page {page}")
            return None
        return BeautifulSoup(resp.text, "html.parser")
    except Exception as e:
        log.warning(f"  Fetch error ({slug} p{page}): {e}")
        return None


def scrape_drug_reviews(session, drug_name: str, writer, max_pages: int) -> int:
    slug = get_slug(drug_name)
    log.info(f"  Drugs.com reviews: {drug_name}  → slug: {slug}")

    total_written = 0
    seen_ids      = set()

    for page in range(1, max_pages + 1):
        soup = fetch_page(session, slug, page)
        if not soup:
            break

        cards = soup.select("div.ddc-comment")
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

        if len(cards) < PAGE_SIZE:
            log.info(f"  Page {page}: last page reached")
            break

        time.sleep(random.uniform(2.0, 4.0))   # polite delay — prevents 403

    log.info(f"  ✓ {drug_name}: {total_written} reviews total")
    return total_written


def run(drugs=None, max_pages_per_drug=MAX_PAGES):
    drug_list   = drugs or TARGET_DRUGS
    output_path = RAW_DIR / "reviews.csv"
    checkpoint  = Checkpoint("webmd_reviews")   # reuse same checkpoint key
    session     = make_session()

    log.info("=" * 65)
    log.info("Drugs.com Review Scraper — confirmed selectors 2026-03-31")
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
            written = scrape_drug_reviews(session, drug, writer,
                                          max_pages=max_pages_per_drug)
            total_written += written
            checkpoint.mark_done(drug)

            if i < len(drug_list):
                time.sleep(random.uniform(3.0, 6.0))   # longer delay between drugs

    file_size_kb = output_path.stat().st_size / 1024 if output_path.exists() else 0
    log.info(f"\n{'='*65}")
    log.info(f"Complete | Written: {total_written} | Skipped: {skipped} | "
             f"File: {file_size_kb:.1f} KB")
    return {"written": total_written, "skipped": skipped,
            "file": str(output_path), "size_kb": round(file_size_kb, 1)}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--drug",      type=str)
    p.add_argument("--test",      action="store_true")
    p.add_argument("--reset",     action="store_true")
    p.add_argument("--max-pages", type=int, default=MAX_PAGES)
    args = p.parse_args()

    if args.reset:
        Checkpoint("webmd_reviews").reset()
        log.info("Checkpoint cleared.")

    if args.drug:
        drugs = [args.drug]
    elif args.test:
        drugs = ["lisinopril", "atorvastatin", "gabapentin"]
    else:
        drugs = None

    run(drugs=drugs, max_pages_per_drug=args.max_pages)
