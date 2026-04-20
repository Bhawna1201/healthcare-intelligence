"""
run_phase1.py
──────────────
Master pipeline runner for Phase 1: Web Mining Layer.

Usage:
    # Run all scrapers
    python run_phase1.py

    # Run specific scrapers only
    python run_phase1.py --scrapers trials shortages

    # Run with Selenium for GoodRx
    python run_phase1.py --selenium

    # Resume interrupted run (uses checkpoints automatically)
    python run_phase1.py

    # Reset checkpoints and re-scrape everything
    python run_phase1.py --reset

    # Test mode: scrape only 3 drugs
    python run_phase1.py --test
"""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from scrapers.utils.base import get_logger, Checkpoint
from config.settings import RAW_DIR, TARGET_DRUGS

log = get_logger("run_phase1")

TEST_DRUGS = ["metformin", "lisinopril", "atorvastatin"]


def run_scraper(name: str, func, **kwargs) -> dict:
    """Run a scraper function with timing and error handling."""
    log.info(f"\n{'='*60}")
    log.info(f"STARTING: {name.upper()}")
    log.info(f"{'='*60}")
    start = time.time()
    stats = {}
    try:
        stats = func(**kwargs) or {}
        elapsed = time.time() - start
        log.info(f"COMPLETED {name} in {elapsed:.1f}s | {stats}")
    except Exception as e:
        elapsed = time.time() - start
        log.error(f"FAILED {name} after {elapsed:.1f}s: {e}", exc_info=True)
        stats = {"error": str(e)}
    return stats


def print_summary(results: dict):
    """Print a final summary table of all scraper results."""
    print("\n" + "="*60)
    print("PHASE 1 SCRAPING SUMMARY")
    print("="*60)
    print(f"{'Scraper':<25} {'Status':<12} {'Written':<10} {'Notes'}")
    print("-"*60)
    for scraper, stats in results.items():
        if "error" in stats:
            status = "FAILED"
            written = "—"
            notes = stats["error"][:30]
        else:
            status = "OK"
            written = str(stats.get("written", "?"))
            notes = stats.get("file", "")
        print(f"{scraper:<25} {status:<12} {written:<10} {notes}")
    print("="*60)

    # Output file sizes
    print("\nOutput files:")
    for csv_file in sorted(RAW_DIR.glob("*.csv")):
        size_kb = csv_file.stat().st_size / 1024
        print(f"  {csv_file.name:<30} {size_kb:>8.1f} KB")


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Healthcare Data Scraper Pipeline")
    parser.add_argument(
        "--scrapers", nargs="+",
        choices=["trials", "shortages", "reviews", "prices", "adverse", "pubmed"],
        help="Run only specific scrapers (default: all)",
    )
    # --selenium kept for backwards compat but no longer used (GoodRx replaced by NADAC)
    parser.add_argument("--selenium", action="store_true",
                        help="(Deprecated — NADAC API used instead of GoodRx)")
    parser.add_argument("--reset", action="store_true",
                        help="Reset all checkpoints and re-scrape")
    parser.add_argument("--test", action="store_true",
                        help="Test mode: scrape 3 drugs only")
    args = parser.parse_args()

    drugs = TEST_DRUGS if args.test else TARGET_DRUGS
    active_scrapers = set(args.scrapers) if args.scrapers else {
        "trials", "shortages", "reviews", "prices", "adverse", "pubmed"
    }

    if args.reset:
        log.info("Resetting all checkpoints...")
        for name in ["clinicaltrials", "fda_shortages_v2", "drugscom_reviews",
                     "nadac_prices", "openfda_adverse", "pubmed"]:
            Checkpoint(name).reset()
        log.info("All checkpoints cleared.")

    log.info(f"\nPhase 1 Pipeline Starting | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"Mode: {'TEST (3 drugs)' if args.test else f'{len(drugs)} drugs'}")
    log.info(f"Active scrapers: {', '.join(sorted(active_scrapers))}")
    log.info(f"Output directory: {RAW_DIR}\n")

    results = {}

    # ── 1. ClinicalTrials.gov ─────────────────────────────────────────────
    if "trials" in active_scrapers:
        from scrapers.clinicaltrials_scraper import run as run_trials
        results["ClinicalTrials.gov"] = run_scraper(
            "ClinicalTrials.gov", run_trials,
            drugs=drugs, max_per_drug=100 if args.test else 200
        )

    # ── 2. FDA Drug Shortages ─────────────────────────────────────────────
    if "shortages" in active_scrapers:
        from scrapers.fda_shortages_scraper import run as run_shortages
        results["FDA Shortages"] = run_scraper(
            "FDA Shortages", run_shortages, drugs=drugs
        )

    # ── 3. Drugs.com Reviews ──────────────────────────────────────────────
    if "reviews" in active_scrapers:
        from scrapers.drugscom_scraper import run as run_reviews
        results["Drugs.com Reviews"] = run_scraper(
            "Drugs.com Reviews", run_reviews,
            drugs=drugs, max_pages_per_drug=3 if args.test else 15
        )

    # ── 4. GoodRx Prices ──────────────────────────────────────────────────
    if "prices" in active_scrapers:
        from scrapers.goodrx_scraper import run as run_prices
        results["GoodRx Prices"] = run_scraper(
            "GoodRx Prices", run_prices,
            drugs=drugs, use_selenium=args.selenium
        )

    # ── 5. OpenFDA Adverse Events ─────────────────────────────────────────
    if "adverse" in active_scrapers:
        from scrapers.openfda_scraper import run as run_adverse
        results["OpenFDA Adverse Events"] = run_scraper(
            "OpenFDA Adverse Events", run_adverse,
            drugs=drugs, max_per_drug=200 if args.test else 500
        )

    # ── 6. PubMed Abstracts ───────────────────────────────────────────────
    if "pubmed" in active_scrapers:
        from scrapers.pubmed_scraper import run as run_pubmed
        results["PubMed Abstracts"] = run_scraper(
            "PubMed Abstracts", run_pubmed,
            drugs=drugs, max_per_drug=50 if args.test else 150
        )

    print_summary(results)

    # Save run report
    report_path = RAW_DIR / f"scrape_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_path, "w") as f:
        f.write(f"Phase 1 Scraping Report\n")
        f.write(f"Run at: {datetime.now()}\n")
        f.write(f"Drugs: {len(drugs)}\n\n")
        for scraper, stats in results.items():
            f.write(f"{scraper}: {stats}\n")
    log.info(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
