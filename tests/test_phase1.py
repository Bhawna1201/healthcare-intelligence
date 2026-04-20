"""
tests/test_phase1.py
──────────────────────
Unit tests for Phase 1 scrapers.
Uses `responses` library to mock HTTP calls — no real network needed.

Run:  pytest tests/test_phase1.py -v
"""

import sys
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from scrapers.utils.schemas import ReviewRecord, TrialRecord, ShortageRecord, PriceRecord
from scrapers.utils.base import SchemaCSVWriter, Checkpoint, clean_price_helper


# ─────────────────────────────────────────────────────────────────────────────
# Schema Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSchemas:
    def test_review_record_defaults(self):
        r = ReviewRecord(
            drug_name="metformin",
            rating=8.0,
            review_text="Works well for me",
            condition="Diabetes",
            date="2024-01-01",
            source="drugs.com",
            review_id="abc123",
            helpful_votes=5,
        )
        assert r.drug_name == "metformin"
        assert r.scraped_at is not None

    def test_trial_record_optional_fields(self):
        t = TrialRecord(
            drug_name="lisinopril",
            nct_id="NCT12345678",
            title="Test Trial",
            phase=None,
            status=None,
            start_date=None,
            completion_date=None,
            description=None,
            conditions=None,
            sponsor=None,
            enrollment=None,
        )
        assert t.nct_id == "NCT12345678"

    def test_price_record_types(self):
        p = PriceRecord(
            drug_name="atorvastatin",
            generic_name="atorvastatin",
            brand_name="Lipitor",
            price=12.49,
            pharmacy="CVS",
            dosage="20mg",
            quantity="30 tablets",
            with_coupon=True,
        )
        assert isinstance(p.price, float)
        assert p.with_coupon is True


# ─────────────────────────────────────────────────────────────────────────────
# SchemaCSVWriter Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSchemaCSVWriter:
    def test_write_and_read_back(self, tmp_path):
        import csv
        output = tmp_path / "test_reviews.csv"
        record = ReviewRecord(
            drug_name="aspirin",
            rating=7.0,
            review_text="Helps with headaches",
            condition="Pain",
            date="2024-03-15",
            source="drugs.com",
            review_id="test001",
            helpful_votes=2,
        )
        with SchemaCSVWriter(output, ReviewRecord) as writer:
            writer.write(record)

        assert output.exists()
        with open(output) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["drug_name"] == "aspirin"
        assert rows[0]["rating"] == "7.0"

    def test_stats_tracking(self, tmp_path):
        output = tmp_path / "test.csv"
        with SchemaCSVWriter(output, ReviewRecord) as writer:
            writer.write(ReviewRecord("metformin", 8.0, "good drug", None, None, "drugs.com", "id1", 0))
            writer.write(ReviewRecord("lisinopril", 6.0, "ok drug", None, None, "drugs.com", "id2", 1))
            stats = writer.stats()
        assert stats["written"] == 2
        assert stats["skipped"] == 0

    def test_appends_on_reopen(self, tmp_path):
        import csv
        output = tmp_path / "append_test.csv"
        r1 = ReviewRecord("drug1", 5.0, "text1", None, None, "drugs.com", "id1", 0)
        r2 = ReviewRecord("drug2", 7.0, "text2", None, None, "drugs.com", "id2", 0)

        with SchemaCSVWriter(output, ReviewRecord) as w:
            w.write(r1)
        with SchemaCSVWriter(output, ReviewRecord) as w:
            w.write(r2)

        with open(output) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckpoint:
    def test_mark_and_check(self, tmp_path):
        cp = Checkpoint("test", checkpoint_dir=tmp_path)
        assert not cp.is_done("metformin")
        cp.mark_done("metformin")
        assert cp.is_done("metformin")

    def test_persists_across_instances(self, tmp_path):
        cp1 = Checkpoint("persist_test", checkpoint_dir=tmp_path)
        cp1.mark_done("lisinopril")

        cp2 = Checkpoint("persist_test", checkpoint_dir=tmp_path)
        assert cp2.is_done("lisinopril")

    def test_reset(self, tmp_path):
        cp = Checkpoint("reset_test", checkpoint_dir=tmp_path)
        cp.mark_done("drug_a")
        cp.mark_done("drug_b")
        assert cp.completed == 2
        cp.reset()
        assert cp.completed == 0


# ─────────────────────────────────────────────────────────────────────────────
# ClinicalTrials Parser Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestClinicalTrialsParser:
    def test_parse_valid_trial(self):
        from scrapers.clinicaltrials_scraper import parse_trial

        mock_study = {
            "protocolSection": {
                "identificationModule": {
                    "nctId": "NCT04321234",
                    "briefTitle": "Metformin vs Placebo in Type 2 Diabetes",
                },
                "statusModule": {
                    "overallStatus": "Completed",
                    "startDateStruct": {"date": "2020-01-15"},
                    "primaryCompletionDateStruct": {"date": "2023-06-30"},
                },
                "descriptionModule": {
                    "briefSummary": "A randomized controlled trial of metformin.",
                },
                "designModule": {
                    "phases": ["Phase 3"],
                    "enrollmentInfo": {"count": 500},
                },
                "sponsorCollaboratorsModule": {
                    "leadSponsor": {"name": "NIH"},
                },
                "conditionsModule": {
                    "conditions": ["Type 2 Diabetes", "Hyperglycemia"],
                },
            }
        }

        record = parse_trial(mock_study, "metformin")
        assert record is not None
        assert record.nct_id == "NCT04321234"
        assert record.phase == "Phase 3"
        assert record.status == "Completed"
        assert record.enrollment == 500
        assert "randomized" in record.description.lower()

    def test_parse_missing_fields(self):
        from scrapers.clinicaltrials_scraper import parse_trial
        record = parse_trial({"protocolSection": {}}, "metformin")
        assert record is not None  # should not crash


# ─────────────────────────────────────────────────────────────────────────────
# OpenFDA Parser Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestOpenFDAParser:
    def test_parse_adverse_event(self):
        from scrapers.openfda_scraper import parse_adverse_event

        mock_result = {
            "safetyreportid": "REPORT001",
            "serious": "1",
            "seriousnesshospitalization": "1",
            "receivedate": "20230615",
            "patient": {
                "patientonsetage": "65",
                "patientonsetageunit": "years",
                "patientsex": "1",
                "reaction": [
                    {"reactionmeddrapt": "Nausea"},
                    {"reactionmeddrapt": "Vomiting"},
                ],
            },
        }

        records = parse_adverse_event(mock_result, "metformin")
        assert len(records) == 2
        assert records[0].event_type == "Nausea"
        assert records[0].severity == "serious"
        assert records[0].patient_sex == "male"
        assert records[0].report_date == "2023-06-15"


# ─────────────────────────────────────────────────────────────────────────────
# GoodRx Parser Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGoodRxParser:
    def test_clean_price(self):
        from scrapers.goodrx_scraper import clean_price
        assert clean_price("$12.49") == 12.49
        assert clean_price("12") == 12.0
        assert clean_price("$1,234.56") == 1234.56
        assert clean_price("") is None
        assert clean_price(None) is None

    def test_parse_json_ld_prices(self):
        from scrapers.goodrx_scraper import parse_goodrx_page

        html = """<html><head>
        <script type="application/ld+json">
        {"@type": "Drug", "name": "Metformin",
         "offers": [
           {"price": "4.00", "seller": {"name": "Walmart"}},
           {"price": "8.50", "seller": {"name": "CVS"}}
         ]}
        </script></head><body></body></html>"""

        records = parse_goodrx_page(html, "metformin")
        assert len(records) == 2
        prices = {r.pharmacy: r.price for r in records}
        assert prices.get("Walmart") == 4.0
        assert prices.get("CVS") == 8.5


# ─────────────────────────────────────────────────────────────────────────────
# PubMed XML Parser Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPubMedParser:
    def test_parse_xml(self):
        from scrapers.pubmed_scraper import parse_pubmed_xml

        xml = """<?xml version="1.0"?>
        <PubmedArticleSet>
          <PubmedArticle>
            <MedlineCitation>
              <PMID>12345678</PMID>
              <Article>
                <ArticleTitle>Metformin in Type 2 Diabetes</ArticleTitle>
                <Abstract>
                  <AbstractText>Metformin is the first-line treatment.</AbstractText>
                </Abstract>
                <AuthorList>
                  <Author><LastName>Smith</LastName><ForeName>John</ForeName></Author>
                </AuthorList>
                <Journal>
                  <Title>New England Journal of Medicine</Title>
                </Journal>
              </Article>
              <MeshHeadingList>
                <MeshHeading><DescriptorName>Diabetes Mellitus</DescriptorName></MeshHeading>
              </MeshHeadingList>
            </MedlineCitation>
            <PubmedData>
              <History>
                <PubMedPubDate PubStatus="pubmed">
                  <Year>2023</Year><Month>6</Month><Day>1</Day>
                </PubMedPubDate>
              </History>
            </PubmedData>
          </PubmedArticle>
        </PubmedArticleSet>"""

        records = parse_pubmed_xml(xml, "metformin")
        assert len(records) == 1
        assert records[0].pmid == "12345678"
        assert "Metformin" in records[0].title
        assert "Smith" in records[0].authors
        assert records[0].journal == "New England Journal of Medicine"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
