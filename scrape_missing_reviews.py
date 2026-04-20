"""
scrape_missing_reviews.py  v5 — 17 remaining drugs with confirmed slugs
Run:  venv/bin/python scrape_missing_reviews.py
"""

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import time, hashlib, re, csv, json
from datetime import datetime
from pathlib import Path

OUTPUT    = "data/raw/reviews.csv"
CKPT      = "logs/checkpoints/webmd_reviews.json"
MAX_PAGES = 5
WAIT      = 5

# Confirmed URLs from browser inspection
DRUGS = [
    ("atorvastatin", [
        "lypqozet-ezetimibe-atorvastatin",   # confirmed
        "atorvastatin-lipitor",
        "atorvastatin",
    ]),
    ("bupropion", [
        "auvelity-dextromethorphan-bupropion",  # confirmed
        "bupropion-wellbutrin-xl-zyban-others",
        "bupropion-wellbutrin-xl",
        "bupropion",
    ]),
    ("tramadol", [
        "tramadol-acetaminophen-ultracet",   # confirmed
        "tramadol-ultram-others",
        "tramadol-ultram",
        "tramadol",
    ]),
    ("amoxicillin", [
        "amoxicillin-amoxil-and-others",     # confirmed
        "amoxicillin-amoxil-moxatag-others",
        "amoxicillin-amoxil",
        "amoxicillin",
    ]),
    ("azithromycin", [
        "azithromycin-zithromax-z-pak-zmax",  # confirmed
        "azithromycin-zithromax-zmax-others",
        "azithromycin-zithromax",
        "azithromycin",
    ]),
    ("oxycodone", [
        "oxycodone-oxycontin-roxicodone-xtampza-er",  # confirmed
        "oxycodone-oxycontin-roxicodone",
        "oxycodone-roxicodone",
        "oxycodone",
    ]),
    ("hydrocodone", [
        "hydrocodone-hysingla-er-zohydro-er",  # confirmed
        "hydrocodone-acetaminophen-vicodin-norco-others",
        "hydrocodone-vicodin",
        "hydrocodone",
    ]),
    ("spironolactone", [
        "spironolactone-aldactone-carospir",   # confirmed
        "spironolactone-aldactone-carospir-others",
        "spironolactone-aldactone",
        "spironolactone",
    ]),
    ("carvedilol", [
        "carvedilol-coreg-coreg-cr",           # confirmed
        "carvedilol-coreg-coreg-cr-others",
        "carvedilol-coreg",
        "carvedilol",
    ]),
    ("ibuprofen", [
        "ibuprofen-advil-caldolor-motrin",
        "ibuprofen-advil-motrin-ib",
        "ibuprofen-advil",
        "ibuprofen",
    ]),
    ("insulin lispro", [
        "insulin-lispro-admelog-humalog-lyumjev",
        "insulin-lispro-humalog",
        "insulin-lispro",
    ]),
    ("empagliflozin", [
        "jardiance-empagliflozin",
        "empagliflozin-jardiance-others",
        "empagliflozin",
    ]),
    ("semaglutide", [
        "ozempic-semaglutide",
        "semaglutide-ozempic",
        "semaglutide",
    ]),
    ("dulaglutide", [
        "trulicity-dulaglutide",
        "dulaglutide-trulicity-others",
        "dulaglutide",
    ]),
    ("apixaban", [
        "eliquis-apixaban",
        "apixaban-eliquis-others",
        "apixaban",
    ]),
    ("rivaroxaban", [
        "xarelto-rivaroxaban",
        "rivaroxaban-xarelto-others",
        "rivaroxaban",
    ]),
    ("diltiazem", [
        "diltiazem-cardizem-tiazac-others",
        "diltiazem-cardizem",
        "diltiazem",
    ]),
]


def make_driver():
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    opts.binary_location = (
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    )
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()), options=opts)
    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {"source": "Object.defineProperty(navigator,'webdriver',{get:()=>undefined})"},
    )
    return driver


def find_working_slug(driver, slugs):
    for slug in slugs:
        url = f"https://reviews.webmd.com/drugs/drugreview-{slug}?pageIndex=0"
        try:
            driver.get(url)
            time.sleep(WAIT)
            cards = driver.find_elements(By.CLASS_NAME, "review-details-holder")
            if cards:
                print(f"  ✓ {slug}  ({len(cards)} cards)")
                return slug, cards
            print(f"  ✗ no cards: {slug}")
        except Exception as e:
            print(f"  ✗ error: {slug} — {e}")
    return None, []


def parse_card(card, drug_name):
    try:
        text = card.find_element(By.CLASS_NAME, "description").text.strip()
    except Exception:
        text = ""
    if len(text) < 10:
        return None
    try:
        cond = card.find_element(By.CLASS_NAME, "condition").text
        cond = re.sub(r"Condition\s*:\s*", "", cond, flags=re.I).strip()
    except Exception:
        cond = ""
    try:
        date_raw = card.find_element(By.CLASS_NAME, "date").text.strip()
        date_iso = datetime.strptime(date_raw, "%m/%d/%Y").strftime("%Y-%m-%d")
    except Exception:
        date_iso = ""
    try:
        rel    = card.find_element(
            By.CSS_SELECTOR, "div.overall-rating div.webmd-rate")
        rating = float(rel.get_attribute("aria-valuenow")) * 2
    except Exception:
        rating = None
    rid = hashlib.md5(
        f"{drug_name}|{date_iso}|{text[:80]}".encode()).hexdigest()[:12]
    return {"drug_name": drug_name, "rating": rating, "review_text": text,
            "condition": cond, "date": date_iso, "source": "webmd",
            "review_id": rid, "helpful_votes": None,
            "scraped_at": datetime.now().isoformat()}


def scrape_drug(driver, drug_name, slugs):
    slug, first_cards = find_working_slug(driver, slugs)
    if not slug:
        return []
    reviews  = []
    base_url = f"https://reviews.webmd.com/drugs/drugreview-{slug}"
    for card in first_cards:
        rec = parse_card(card, drug_name)
        if rec: reviews.append(rec)
    print(f"  Page 1: {len(first_cards)} cards — total: {len(reviews)}")
    for page in range(1, MAX_PAGES):
        driver.get(f"{base_url}?pageIndex={page}")
        time.sleep(WAIT)
        cards = driver.find_elements(By.CLASS_NAME, "review-details-holder")
        if not cards:
            print(f"  Page {page+1}: done")
            break
        for card in cards:
            rec = parse_card(card, drug_name)
            if rec: reviews.append(rec)
        print(f"  Page {page+1}: {len(cards)} cards — total: {len(reviews)}")
        if len(cards) < 20: break
    return reviews


def save_reviews(reviews):
    if not reviews: return
    out = Path(OUTPUT)
    write_header = not out.exists() or out.stat().st_size == 0
    with open(out, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(reviews[0].keys()))
        if write_header: w.writeheader()
        w.writerows(reviews)


def is_done(drug):
    p = Path(CKPT)
    return drug in json.loads(p.read_text()).get("done", []) if p.exists() else False


def mark_done(drug):
    p    = Path(CKPT)
    data = json.loads(p.read_text()) if p.exists() else {"done": []}
    if drug not in data["done"]:
        data["done"].append(drug)
        p.write_text(json.dumps(data, indent=2))


def main():
    remaining = [(d, s) for d, s in DRUGS if not is_done(d)]
    print("=" * 65)
    print(f"WebMD Scraper v5 — {len(remaining)} drugs remaining")
    print("=" * 65)
    if not remaining:
        print("All done!")
        return
    driver = make_driver()
    total  = 0
    try:
        for i, (drug, slugs) in enumerate(remaining, 1):
            print(f"\n[{i}/{len(remaining)}] {drug}")
            try:
                reviews = scrape_drug(driver, drug, slugs)
                save_reviews(reviews)
                mark_done(drug)
                total += len(reviews)
                print(f"  ✓ {drug}: {len(reviews)} reviews saved")
            except Exception as e:
                print(f"  ✗ {drug} FAILED: {e}")
            time.sleep(3)
    finally:
        driver.quit()
    print(f"\n{'='*65}")
    print(f"Complete — {total} new reviews saved")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
