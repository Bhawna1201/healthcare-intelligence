"""
scrapers/pubmed_scraper.py
───────────────────────────
Fetches drug research abstracts from PubMed via NCBI E-utilities API.

Free API — optional key recommended for higher rate limits.
Get key at: https://www.ncbi.nlm.nih.gov/account/

Endpoints used:
  - esearch: find PMIDs for a drug query
  - efetch: retrieve full records (including abstract) for PMIDs

Output: data/raw/pubmed_abstracts.csv
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import xml.etree.ElementTree as ET
import time
from typing import Optional

from scrapers.utils.base import make_session, rate_limited_get, SchemaCSVWriter, Checkpoint, get_logger
from scrapers.utils.schemas import PubMedRecord
from config.settings import NCBI_API_KEY, NCBI_EMAIL, RAW_DIR, TARGET_DRUGS

log = get_logger("pubmed_scraper")

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
ELINK_URL   = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"


def search_pmids(session, drug_name: str, max_results: int = 200) -> list[str]:
    """Search PubMed for articles about a drug. Returns list of PMIDs."""
    # Build a targeted query
    query = f'"{drug_name}"[Title/Abstract] AND (clinical trial[PT] OR review[PT] OR drug therapy[MeSH])'

    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "usehistory": "y",
        "email": NCBI_EMAIL,
        "sort": "relevance",
    }
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY

    delay = 0.35 if NCBI_API_KEY else 0.7  # NCBI rate: 10/s with key, 3/s without
    time.sleep(delay)

    resp = rate_limited_get(session, ESEARCH_URL, params=params,
                            delay_min=0, delay_max=0, logger=log)
    if resp is None:
        return []

    data = resp.json()
    pmids = data.get("esearchresult", {}).get("idlist", [])
    log.debug(f"  {drug_name}: {len(pmids)} PMIDs found")
    return pmids


def fetch_records_xml(session, pmids: list[str]) -> Optional[str]:
    """Fetch full PubMed records in XML for a batch of PMIDs."""
    if not pmids:
        return None

    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "abstract",
        "retmode": "xml",
        "email": NCBI_EMAIL,
    }
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY

    resp = rate_limited_get(session, EFETCH_URL, params=params,
                            delay_min=0.5, delay_max=1.5, logger=log)
    return resp.text if resp else None


def parse_pubmed_xml(xml_text: str, drug_name: str) -> list[PubMedRecord]:
    """Parse PubMed XML response into PubMedRecord objects."""
    records = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        log.error(f"XML parse error: {e}")
        return records

    for article in root.findall(".//PubmedArticle"):
        try:
            # ── PMID ──────────────────────────────────────────────────────
            pmid_el = article.find(".//PMID")
            pmid = pmid_el.text if pmid_el is not None else ""

            # ── Title ──────────────────────────────────────────────────────
            title_el = article.find(".//ArticleTitle")
            title = "".join(title_el.itertext()) if title_el is not None else ""

            # ── Abstract ──────────────────────────────────────────────────
            abstract_parts = article.findall(".//AbstractText")
            abstract = " ".join("".join(el.itertext()) for el in abstract_parts)

            # ── Authors ───────────────────────────────────────────────────
            authors = []
            for author in article.findall(".//Author"):
                last = author.findtext("LastName", "")
                fore = author.findtext("ForeName", "")
                if last:
                    authors.append(f"{last} {fore}".strip())
            authors_str = "; ".join(authors[:10])  # cap at 10

            # ── Journal ───────────────────────────────────────────────────
            journal = article.findtext(".//Journal/Title") or article.findtext(".//ISOAbbreviation")

            # ── Publication date ──────────────────────────────────────────
            pub_year = None
            year_el = article.find(".//PubDate/Year")
            if year_el is not None:
                try:
                    pub_year = int(year_el.text)
                except (ValueError, TypeError):
                    pass

            pub_date = None
            if pub_year:
                month = article.findtext(".//PubDate/Month", "01")
                pub_date = f"{pub_year}-{month}-01"

            # ── MeSH Keywords ─────────────────────────────────────────────
            mesh_terms = [mh.findtext("DescriptorName", "") for mh in article.findall(".//MeshHeading")]
            keywords_str = "; ".join(filter(None, mesh_terms[:15]))

            if not pmid or not title:
                continue

            records.append(PubMedRecord(
                drug_name=drug_name,
                pmid=pmid,
                title=title.strip(),
                abstract=abstract[:3000] if abstract else None,
                authors=authors_str or None,
                journal=journal,
                pub_year=pub_year,
                pub_date=pub_date,
                citation_count=None,  # not available from basic API
                keywords=keywords_str or None,
            ))

        except Exception as e:
            log.debug(f"Error parsing article: {e}")
            continue

    return records


def scrape_drug_pubmed(session, drug_name: str, writer: SchemaCSVWriter, max_articles: int = 200) -> int:
    """Scrape PubMed abstracts for one drug. Returns count written."""
    pmids = search_pmids(session, drug_name, max_results=max_articles)
    if not pmids:
        log.info(f"  {drug_name}: no PubMed results")
        return 0

    # Fetch in batches of 50 (NCBI recommended)
    batch_size = 50
    written = 0
    for i in range(0, len(pmids), batch_size):
        batch = pmids[i:i + batch_size]
        xml_text = fetch_records_xml(session, batch)
        if not xml_text:
            continue

        records = parse_pubmed_xml(xml_text, drug_name)
        written += writer.write_many(records)
        writer.flush()

    log.info(f"  → {drug_name}: {written} PubMed abstracts written")
    return written


def run(drugs: list = None, max_per_drug: int = 150):
    drug_list = drugs or TARGET_DRUGS
    output_path = RAW_DIR / "pubmed_abstracts.csv"
    checkpoint = Checkpoint("pubmed")
    session = make_session()

    log.info(f"Starting PubMed scraper | {len(drug_list)} drugs")

    with SchemaCSVWriter(output_path, PubMedRecord) as writer:
        for i, drug in enumerate(drug_list, 1):
            if checkpoint.is_done(drug):
                log.info(f"[{i}/{len(drug_list)}] Skipping {drug} (checkpoint)")
                continue
            log.info(f"[{i}/{len(drug_list)}] PubMed: {drug}")
            scrape_drug_pubmed(session, drug, writer, max_articles=max_per_drug)
            checkpoint.mark_done(drug)

        stats = writer.stats()

    log.info(f"PubMed scrape complete | written: {stats['written']}")
    return stats


if __name__ == "__main__":
    run()
