"""
scrapers/goodrx_scraper.py  (v4 — confirmed data-qa selectors)
───────────────────────────────────────────────────────────────
Confirmed DOM structure from live DevTools inspection:

  <ul data-qa="pharmacy-selector-container">          ← wait for this
    <li>
      <button>
        <span data-qa="seller-name">Walgreens</span>  ← pharmacy name
        <span data-qa="seller-price">$10.83</span>    ← price
      </button>
    </li>
    <li>  ← CVS, Duane Reade, Stop n Shop, ShopRite, Costco, Walmart ...
    ...
  </ul>

data-qa attributes are stable test hooks — they survive CSS class refactors.
Strategy: load page with Selenium, wait for container, read every <li>.
"""
import sys, re, json, time, random
from pathlib import Path
from typing import Optional
sys.path.insert(0, str(Path(__file__).parent.parent))
from scrapers.utils.base import SchemaCSVWriter, Checkpoint, get_logger
from scrapers.utils.schemas import PriceRecord
from config.settings import RAW_DIR, TARGET_DRUGS

log = get_logger("goodrx_v4")
GOODRX_BASE   = "https://www.goodrx.com"
CONTAINER_SEL = "ul[data-qa='pharmacy-selector-container']"
ROW_SEL       = "ul[data-qa='pharmacy-selector-container'] li"
NAME_SEL      = "span[data-qa='seller-name']"
PRICE_SEL     = "span[data-qa='seller-price']"

def make_driver(headless=True):
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service

    opts = Options()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    opts.add_argument(f"--window-size={random.randint(1280,1920)},{random.randint(800,1080)}")
    opts.add_argument(
        "--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
    opts.add_argument("--disable-extensions")
    opts.add_argument("--blink-settings=imagesEnabled=false")
    # Point to macOS Chrome binary explicitly
    opts.binary_location = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    # webdriver-manager auto-downloads the matching chromedriver version
    # This fixes: "DevToolsActivePort file doesn't exist" (version mismatch)
    try:
        from webdriver_manager.chrome import ChromeDriverManager
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=opts)
    except ImportError:
        log.warning("webdriver-manager not installed — falling back to system chromedriver")
        log.warning("Fix: pip install webdriver-manager")
        driver = webdriver.Chrome(options=opts)
    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {"source": "Object.defineProperty(navigator,'webdriver',{get:()=>undefined})"})
    return driver

def clean_price(v):
    if v is None: return None
    cleaned = re.sub(r"[^\d.]", "", str(v))
    try:
        p = float(cleaned)
        return p if 0.01 <= p <= 9999 else None
    except ValueError:
        return None

def _save_debug(driver, slug):
    debug_dir = RAW_DIR / "debug"
    debug_dir.mkdir(exist_ok=True)
    try:
        (debug_dir / f"{slug}_FULL.html").write_text(driver.page_source, encoding="utf-8")
        driver.save_screenshot(str(debug_dir / f"{slug}_screenshot.png"))
        log.info(f"  Debug files saved to {debug_dir}")
    except Exception:
        pass

def scrape_drug(driver, drug_name):
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC

    slug = drug_name.lower().replace(" ", "-").replace("/", "-")
    url  = f"{GOODRX_BASE}/{slug}"
    log.info(f"  Loading: {url}")
    try:
        driver.get(url)
    except Exception as e:
        log.warning(f"  Load failed: {e}")
        return []

    wait = WebDriverWait(driver, 25)
    try:
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, CONTAINER_SEL)))
        log.debug("  pharmacy-selector-container ready")
    except Exception:
        log.warning(f"  Timed out waiting for pharmacy list for {drug_name}")
        _save_debug(driver, slug)
        return []

    time.sleep(random.uniform(1.5, 2.5))

    records = []
    seen = set()
    try:
        rows = driver.find_elements(By.CSS_SELECTOR, ROW_SEL)
        log.debug(f"  {len(rows)} pharmacy rows found")
        for row in rows:
            try:
                name_el  = row.find_element(By.CSS_SELECTOR, NAME_SEL)
                pharmacy = (name_el.get_attribute("innerText") or name_el.text).split("\n")[0].strip()
                price_el  = row.find_element(By.CSS_SELECTOR, PRICE_SEL)
                price_val = clean_price(price_el.text.strip())
                if not pharmacy or price_val is None:
                    continue
                key = (pharmacy.lower(), price_val)
                if key in seen: continue
                seen.add(key)
                records.append(PriceRecord(
                    drug_name=drug_name, generic_name=drug_name, brand_name=None,
                    price=price_val, pharmacy=pharmacy,
                    dosage=None, quantity=None, with_coupon=True,
                ))
            except Exception as e:
                log.debug(f"  Row error: {e}")
                continue
    except Exception as e:
        log.warning(f"  Row reading failed: {e}")
        _save_debug(driver, slug)
        return []

    log.info(f"  {drug_name}: {len(records)} prices")
    return records

def run(drugs=None, headless=True, batch_size=10, pause_between_batches=15.0):
    drug_list   = drugs or TARGET_DRUGS
    output_path = RAW_DIR / "prices.csv"
    checkpoint  = Checkpoint("goodrx_prices_v4")
    remaining   = [d for d in drug_list if not checkpoint.is_done(d)]
    done_count  = len(drug_list) - len(remaining)

    log.info("=" * 60)
    log.info("GoodRx Scraper v4 — data-qa confirmed selectors")
    log.info(f"Total: {len(drug_list)} | Done: {done_count} | Remaining: {len(remaining)}")
    log.info(f"Headless: {headless} | Batch: {batch_size} | Output: {output_path}")
    log.info("=" * 60)

    if not remaining:
        log.info("All done. Use --reset to re-run.")
        return {"written": 0, "skipped": done_count}

    total_written = 0
    total_batches = (len(remaining) + batch_size - 1) // batch_size

    for batch_num, batch_start in enumerate(range(0, len(remaining), batch_size), 1):
        batch = remaining[batch_start: batch_start + batch_size]
        log.info(f"\nBatch {batch_num}/{total_batches}: {batch}")
        driver = None
        try:
            driver = make_driver(headless=headless)
            with SchemaCSVWriter(output_path, PriceRecord) as writer:
                for i, drug in enumerate(batch):
                    gidx = done_count + batch_start + i + 1
                    log.info(f"\n[{gidx}/{len(drug_list)}] {drug}")
                    records = scrape_drug(driver, drug)
                    written = writer.write_many(records)
                    total_written += written
                    checkpoint.mark_done(drug)
                    log.info(f"  Saved {written} | Total: {total_written}")
                    if i < len(batch) - 1:
                        time.sleep(random.uniform(3.0, 6.0))
        except Exception as e:
            log.error(f"Batch {batch_num} error: {e}", exc_info=True)
        finally:
            if driver:
                try: driver.quit()
                except Exception: pass
        if batch_start + batch_size < len(remaining):
            log.info(f"Pausing {pause_between_batches}s...")
            time.sleep(pause_between_batches)

    log.info(f"\nDone. Total written: {total_written}")
    return {"written": total_written, "file": str(output_path)}

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--visible",    action="store_true")
    p.add_argument("--drugs",      nargs="+")
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--reset",      action="store_true")
    p.add_argument("--test",       action="store_true")
    args = p.parse_args()
    if args.reset:
        Checkpoint("goodrx_prices_v4").reset()
        log.info("Checkpoint cleared.")
    run(
        drugs=["metformin", "lisinopril", "atorvastatin"] if args.test else args.drugs,
        headless=not args.visible,
        batch_size=args.batch_size,
    )