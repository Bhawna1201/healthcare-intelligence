"""
scrapers/clinicaltrials_scraper.py
───────────────────────────────────
Fetches clinical trial data from the ClinicalTrials.gov REST API v2.

API docs: https://clinicaltrials.gov/data-api/api
- No API key required
- Returns JSON
- Paginated via nextPageToken

Output: data/raw/trials.csv
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Optional
from scrapers.utils.base import make_session, rate_limited_get, SchemaCSVWriter, Checkpoint, get_logger
from scrapers.utils.schemas import TrialRecord
from config.settings import (
    CLINICALTRIALS_BASE_URL, CLINICALTRIALS_PAGE_SIZE,
    RAW_DIR, TARGET_DRUGS, random_headers
)


log = get_logger("clinicaltrials_scraper")

# Fields we want from the API (reduces payload size)
FIELDS = [
    "NCTId", "BriefTitle", "Phase", "OverallStatus",
    "StartDate", "PrimaryCompletionDate", "BriefSummary",
    "Condition", "LeadSponsorName", "EnrollmentCount",
    "InterventionName",
]


def parse_trial(study: dict, drug_name: str) -> Optional[TrialRecord]:
    """Extract a TrialRecord from a raw API study dict."""
    try:
        ps = study.get("protocolSection", {})
        id_mod = ps.get("identificationModule", {})
        status_mod = ps.get("statusModule", {})
        desc_mod = ps.get("descriptionModule", {})
        design_mod = ps.get("designModule", {})
        sponsor_mod = ps.get("sponsorCollaboratorsModule", {})
        cond_mod = ps.get("conditionsModule", {})

        phases = design_mod.get("phases", [])
        phase_str = ", ".join(phases) if phases else None

        conditions = cond_mod.get("conditions", [])
        cond_str = "; ".join(conditions[:5]) if conditions else None  # cap at 5

        return TrialRecord(
            drug_name=drug_name,
            nct_id=id_mod.get("nctId", ""),
            title=id_mod.get("briefTitle", ""),
            phase=phase_str,
            status=status_mod.get("overallStatus"),
            start_date=status_mod.get("startDateStruct", {}).get("date"),
            completion_date=status_mod.get("primaryCompletionDateStruct", {}).get("date"),
            description=desc_mod.get("briefSummary", "")[:2000],  # cap length
            conditions=cond_str,
            sponsor=sponsor_mod.get("leadSponsor", {}).get("name"),
            enrollment=design_mod.get("enrollmentInfo", {}).get("count"),
        )
    except Exception as e:
        log.warning(f"Failed to parse trial for {drug_name}: {e}")
        return None


def scrape_trials_for_drug(
    session,
    drug_name: str,
    writer: SchemaCSVWriter,
    max_results: int = 500,
) -> int:
    """
    Fetch all trials for one drug name.
    Returns count of records written.
    """
    written = 0
    next_page_token = None

    log.info(f"Fetching trials for: {drug_name}")

    while True:
        params = {
            "query.intr": drug_name,   # intervention (drug) search
            "pageSize": min(CLINICALTRIALS_PAGE_SIZE, max_results - written),
            "format": "json",
        }
        if next_page_token:
            params["pageToken"] = next_page_token

        resp = rate_limited_get(
            session,
            CLINICALTRIALS_BASE_URL,
            params=params,
            headers=random_headers(),
            delay_min=1.5,
            delay_max=3.0,
            logger=log,
        )

        if resp is None:
            log.error(f"No response for {drug_name}, skipping remaining pages")
            break

        data = resp.json()
        studies = data.get("studies", [])

        if not studies:
            log.debug(f"No more studies for {drug_name}")
            break

        for study in studies:
            record = parse_trial(study, drug_name)
            if record and record.nct_id:
                writer.write(record)
                written += 1

        writer.flush()
        next_page_token = data.get("nextPageToken")

        if not next_page_token or written >= max_results:
            break

    log.info(f"  → {drug_name}: {written} trials written")
    return written


def run(drugs: list = None, max_per_drug: int = 200):
    """
    Main entry point.
    
    Args:
        drugs: List of drug names to scrape. Defaults to TARGET_DRUGS from settings.
        max_per_drug: Max trial records per drug (API can return thousands).
    """
    drug_list = drugs or TARGET_DRUGS
    output_path = RAW_DIR / "trials.csv"
    checkpoint = Checkpoint("clinicaltrials")
    session = make_session()

    log.info(f"Starting ClinicalTrials.gov scraper | {len(drug_list)} drugs | output: {output_path}")

    total_written = 0
    with SchemaCSVWriter(output_path, TrialRecord) as writer:
        for i, drug in enumerate(drug_list, 1):
            if checkpoint.is_done(drug):
                log.info(f"[{i}/{len(drug_list)}] Skipping {drug} (already scraped)")
                continue

            log.info(f"[{i}/{len(drug_list)}] Processing: {drug}")
            count = scrape_trials_for_drug(session, drug, writer, max_results=max_per_drug)
            total_written += count
            checkpoint.mark_done(drug)

        stats = writer.stats()

    log.info(f"ClinicalTrials scrape complete | total written: {stats['written']} | skipped: {stats['skipped']}")
    log.info(f"Output: {output_path}")
    return stats


if __name__ == "__main__":
    run()
