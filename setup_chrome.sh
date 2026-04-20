#!/bin/bash
# setup_chrome.sh
# ─────────────────────────────────────────────────────────────────────────────
# One-time macOS setup for Selenium + ChromeDriver
# Run this ONCE before running the GoodRx scraper.
# Usage: bash setup_chrome.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e

echo "========================================"
echo "  GoodRx Scraper — macOS Setup"
echo "========================================"

# ── 1. Check Chrome is installed ─────────────────────────────────────────────
CHROME_PATH="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
if [ ! -f "$CHROME_PATH" ]; then
    echo "❌ Chrome not found at: $CHROME_PATH"
    echo "   Please install Chrome from https://www.google.com/chrome/"
    exit 1
fi

CHROME_VERSION=$("$CHROME_PATH" --version 2>/dev/null | grep -oE "[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+" | head -1)
echo "✅ Chrome found: version $CHROME_VERSION"

# ── 2. Install selenium python package ───────────────────────────────────────
echo ""
echo "Installing Python packages..."
pip install selenium webdriver-manager --quiet
echo "✅ selenium + webdriver-manager installed"

# ── 3. Try webdriver-manager (auto-downloads matching chromedriver) ───────────
echo ""
echo "Testing webdriver-manager (auto chromedriver)..."
python3 - << 'PYEOF'
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager

    service = Service(ChromeDriverManager().install())
    opts = webdriver.ChromeOptions()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    driver = webdriver.Chrome(service=service, options=opts)
    driver.get("https://www.google.com")
    print(f"✅ webdriver-manager works! Title: {driver.title}")
    driver.quit()
    print("WEBDRIVER_MANAGER=ok")
except Exception as e:
    print(f"⚠️  webdriver-manager failed: {e}")
    print("WEBDRIVER_MANAGER=fail")
PYEOF

# ── 4. If webdriver-manager works, update goodrx_scraper to use it ───────────
echo ""
echo "========================================"
echo "  Setup complete!"
echo "========================================"
echo ""
echo "Run the scraper:"
echo ""
echo "  # Test with 3 drugs (recommended first):"
echo "  python scrapers/goodrx_scraper.py --test"
echo ""
echo "  # Test with Chrome window VISIBLE (good for debugging):"
echo "  python scrapers/goodrx_scraper.py --test --visible"
echo ""
echo "  # Full run (all 50 drugs):"
echo "  python scrapers/goodrx_scraper.py"
echo ""
echo "  # Via master pipeline:"
echo "  python run_phase1.py --scrapers prices"
echo ""
