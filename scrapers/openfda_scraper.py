"""
scrapers/openfda_scraper.py
────────────────────────────
Fetches adverse event reports from the OpenFDA FAERS API.

API: https://api.fda.gov/drug/event.json
- Free, no auth required (optional key for higher rate limits)
- 1000 results per request max, paginated via skip

Output: data/raw/adverse_events.csv
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scrapers.utils.base import make_session, rate_limited_get, SchemaCSVWriter, Checkpoint, get_logger
from scrapers.utils.schemas import AdverseEventRecord
from config.settings import OPENFDA_BASE_URL, OPENFDA_API_KEY, RAW_DIR, TARGET_DRUGS

log = get_logger("openfda_scraper")
EVENTS_URL = f"{OPENFDA_BASE_URL}/drug/event.json"

# Severity categories we want to capture
SERIOUS_FIELDS = [
    "seriousnessdeath", "seriousnesshospitalization",
    "seriousnesslifethreatening", "seriousnessdisabling"
]


def parse_adverse_event(result: dict, drug_name: str) -> list[AdverseEventRecord]:
    """
    One FAERS report can have multiple reactions.
    Returns a list of AdverseEventRecord (one per reaction).
    """
    records = []
    try:
        patient = result.get("patient", {})
        reactions = patient.get("reaction", [])

        # Determine severity
        severity = "non-serious"
        for sf in SERIOUS_FIELDS:
            if result.get(sf) == "1":
                severity = "serious"
                break

        # Outcome
        outcomes_map = {
            "1": "recovered", "2": "recovering", "3": "not_recovered",
            "4": "fatal", "5": "unknown", "6": "not_applicable"
        }
        outcome_code = patient.get("patientdeath", {})
        outcome = outcomes_map.get(str(result.get("serious", "")), "unknown")

        report_date = result.get("receivedate", "")
        if report_date and len(report_date) == 8:
            report_date = f"{report_date[:4]}-{report_date[4:6]}-{report_date[6:8]}"

        age_str = None
        if patient.get("patientonsetage"):
            age_unit = patient.get("patientonsetageunit", "")
            age_str = f"{patient['patientonsetage']} {age_unit}"

        sex_map = {"1": "male", "2": "female", "0": "unknown"}
        sex = sex_map.get(str(patient.get("patientsex", "0")), "unknown")

        for reaction in reactions:
            event_type = reaction.get("reactionmeddrapt", "").strip()
            if not event_type:
                continue

            records.append(AdverseEventRecord(
                drug_name=drug_name,
                generic_name=drug_name,
                event_type=event_type,
                severity=severity,
                outcome=outcome,
                patient_age=age_str,
                patient_sex=sex,
                report_date=report_date,
                report_id=result.get("safetyreportid"),
            ))

    except Exception as e:
        log.debug(f"Error parsing adverse event: {e}")

    return records


def fetch_adverse_events(session, drug_name: str, max_results: int = 1000) -> list[AdverseEventRecord]:
    """Fetch adverse events for one drug from OpenFDA."""
    all_records = []
    skip = 0
    limit = 100

    while len(all_records) < max_results:
        params = {
            "search": f'patient.drug.openfda.generic_name:"{drug_name}"',
            "limit": limit,
            "skip": skip,
        }
        if OPENFDA_API_KEY:
            params["api_key"] = OPENFDA_API_KEY

        resp = rate_limited_get(
            session, EVENTS_URL,
            params=params,
            headers={"User-Agent": "HealthcareResearchBot/1.0"},
            delay_min=1.0, delay_max=2.0,
            logger=log,
        )

        if resp is None:
            break

        data = resp.json()
        if "error" in data:
            log.debug(f"OpenFDA: no adverse event results for '{drug_name}'")
            break

        results = data.get("results", [])
        if not results:
            break

        for result in results:
            all_records.extend(parse_adverse_event(result, drug_name))

        total = data.get("meta", {}).get("results", {}).get("total", 0)
        skip += limit
        if skip >= min(total, max_results):
            break

    return all_records[:max_results]


def run(drugs: list = None, max_per_drug: int = 500):
    drug_list = drugs or TARGET_DRUGS
    output_path = RAW_DIR / "adverse_events.csv"
    checkpoint = Checkpoint("openfda_adverse")
    session = make_session()

    log.info(f"Starting OpenFDA adverse events scraper | {len(drug_list)} drugs")

    with SchemaCSVWriter(output_path, AdverseEventRecord) as writer:
        for i, drug in enumerate(drug_list, 1):
            if checkpoint.is_done(drug):
                log.info(f"[{i}/{len(drug_list)}] Skipping {drug} (checkpoint)")
                continue
            log.info(f"[{i}/{len(drug_list)}] Fetching adverse events: {drug}")
            records = fetch_adverse_events(session, drug, max_results=max_per_drug)
            written = writer.write_many(records)
            log.info(f"  → {drug}: {written} adverse event records")
            checkpoint.mark_done(drug)
            writer.flush()

        stats = writer.stats()

    log.info(f"OpenFDA scrape complete | written: {stats['written']}")
    return stats


if __name__ == "__main__":
    run()
