"""
diagnose_goodrx.py
──────────────────
Run this BEFORE the full scrape to verify:
  1. Chrome + Selenium works
  2. GoodRx page loads correctly
  3. Price extraction succeeds on a known drug

Usage:
    python diagnose_goodrx.py                     # test metformin (headless)
    python diagnose_goodrx.py --drug atorvastatin  # test a specific drug
    python diagnose_goodrx.py --visible            # show Chrome window

Output:
    - Prints extracted prices to terminal
    - Saves page HTML to data/raw/debug/diag_{drug}.html for inspection
    - Saves screenshot to data/raw/debug/diag_{drug}.png
"""

import sys
import time
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config.settings import RAW_DIR
from scrapers.utils.base import get_logger

log = get_logger("diagnose_goodrx")

debug_dir = RAW_DIR / "debug"
debug_dir.mkdir(parents=True, exist_ok=True)


def test_selenium():
    """Verify Selenium + Chrome can launch."""
    print("\n[1/4] Testing Selenium + Chrome launch...")
    try:
        from scrapers.goodrx_scraper import make_driver
        driver = make_driver(headless=True)
        driver.get("about:blank")
        driver.quit()
        print("  ✅ Chrome launched and quit successfully")
        return True
    except Exception as e:
        print(f"  ❌ Chrome failed: {e}")
        print("\n  Fix options:")
        print("  a) brew install --cask chromedriver")
        print("  b) pip install webdriver-manager")
        print("  c) xattr -d com.apple.quarantine $(which chromedriver)")
        return False


def test_webdriver_manager():
    """Try webdriver-manager as an alternative to system chromedriver."""
    print("\n[2/4] Testing webdriver-manager fallback...")
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.service import Service
        from webdriver_manager.chrome import ChromeDriverManager

        opts = webdriver.ChromeOptions()
        opts.add_argument("--headless=new")
        opts.add_argument("--no-sandbox")
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=opts)
        driver.get("https://www.example.com")
        title = driver.title
        driver.quit()
        print(f"  ✅ webdriver-manager works! (loaded: {title})")
        return True
    except ImportError:
        print("  ⚠️  webdriver-manager not installed (pip install webdriver-manager)")
    except Exception as e:
        print(f"  ⚠️  webdriver-manager failed: {e}")
    return False


def test_goodrx_load(drug_name: str, headless: bool):
    """Load a GoodRx drug page and check what we get."""
    print(f"\n[3/4] Loading GoodRx page for '{drug_name}'...")

    try:
        from scrapers.goodrx_scraper import make_driver
        driver = make_driver(headless=headless)
    except Exception as e:
        print(f"  ❌ Could not start driver: {e}")
        return None, None

    url = f"https://www.goodrx.com/{drug_name.lower().replace(' ', '-')}"
    print(f"  URL: {url}")

    try:
        driver.get(url)
        time.sleep(3)

        # Scroll
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight * 0.5);")
        time.sleep(1.5)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1.5)

        html = driver.page_source
        page_title = driver.title

        # Save screenshot
        screenshot_path = debug_dir / f"diag_{drug_name.replace(' ','_')}.png"
        driver.save_screenshot(str(screenshot_path))
        print(f"  📸 Screenshot: {screenshot_path}")

        # Save HTML
        html_path = debug_dir / f"diag_{drug_name.replace(' ','_')}.html"
        html_path.write_text(html[:100000], encoding="utf-8")
        print(f"  📄 HTML saved: {html_path}")
        print(f"  Page title: {page_title}")
        print(f"  HTML size: {len(html):,} chars")

        # Check for bot detection
        if any(p in html.lower() for p in ["captcha", "access denied", "cloudflare"]):
            print("  ⚠️  BOT DETECTION detected in page!")
        else:
            print("  ✅ No bot detection found")

        # Check for __NEXT_DATA__
        if "__NEXT_DATA__" in html:
            print("  ✅ __NEXT_DATA__ found in page")
        else:
            print("  ⚠️  __NEXT_DATA__ NOT found — GoodRx may have changed structure")

        driver.quit()
        return html, page_title

    except Exception as e:
        print(f"  ❌ Error loading page: {e}")
        try:
            driver.quit()
        except Exception:
            pass
        return None, None


def test_extraction(html: str, drug_name: str):
    """Test all extraction strategies on the loaded HTML."""
    print(f"\n[4/4] Testing extraction strategies...")

    from scrapers.goodrx_scraper import extract_next_data, extract_apollo

    # Strategy 1: __NEXT_DATA__
    records = extract_next_data(html, drug_name)
    print(f"\n  Strategy 1 — __NEXT_DATA__: {len(records)} records")
    if records:
        for r in records[:5]:
            print(f"    ${r.price:.2f} @ {r.pharmacy} | dosage: {r.dosage} | qty: {r.quantity}")
        if len(records) > 5:
            print(f"    ... and {len(records)-5} more")

    # Strategy 2: Apollo cache
    records2 = extract_apollo(html, drug_name)
    print(f"\n  Strategy 2 — Apollo cache: {len(records2)} records")
    if records2:
        for r in records2[:5]:
            print(f"    ${r.price:.2f} @ {r.pharmacy}")

    # Summary
    total = max(len(records), len(records2))
    print(f"\n  {'✅' if total > 0 else '❌'} Total extractable prices: {total}")

    if total == 0:
        print("\n  Troubleshooting tips:")
        print("  1. Open data/raw/debug/diag_{drug}.html in a browser to inspect the page")
        print("  2. Open data/raw/debug/diag_{drug}.png to see what Chrome rendered")
        print("  3. Try --visible flag to watch Chrome in real time")
        print("  4. Check if GoodRx redirected to a different URL structure")

    return total


def main():
    parser = argparse.ArgumentParser(description="Diagnose GoodRx scraper")
    parser.add_argument("--drug", default="metformin", help="Drug to test (default: metformin)")
    parser.add_argument("--visible", action="store_true", help="Show Chrome window")
    args = parser.parse_args()

    print("=" * 55)
    print("  GoodRx Scraper Diagnostic")
    print("=" * 55)
    print(f"  Drug: {args.drug}")
    print(f"  Mode: {'Visible' if args.visible else 'Headless'}")

    # Step 1: Basic Selenium test
    selenium_ok = test_selenium()

    if not selenium_ok:
        # Try webdriver-manager as fallback
        wdm_ok = test_webdriver_manager()
        if not wdm_ok:
            print("\n❌ Cannot proceed — fix Chrome/chromedriver first.")
            print("\nQuick fix:")
            print("  pip install webdriver-manager")
            print("  Then add to goodrx_scraper.py make_driver():")
            print("  from webdriver_manager.chrome import ChromeDriverManager")
            print("  service = Service(ChromeDriverManager().install())")
            sys.exit(1)

    # Step 2-3: Load GoodRx page
    html, title = test_goodrx_load(args.drug, headless=not args.visible)

    if html is None:
        print("\n❌ Could not load GoodRx page.")
        sys.exit(1)

    # Step 4: Test extraction
    price_count = test_extraction(html, args.drug)

    print("\n" + "=" * 55)
    print("  Diagnostic Complete")
    print("=" * 55)
    if price_count > 0:
        print(f"\n✅ Ready to run full scrape:")
        print(f"   python scrapers/goodrx_scraper.py --test")
        print(f"   python run_phase1.py --scrapers prices")
    else:
        print(f"\n⚠️  Prices not extracted — check debug files and try --visible")


if __name__ == "__main__":
    main()
