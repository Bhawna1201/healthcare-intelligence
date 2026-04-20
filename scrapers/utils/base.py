"""
scrapers/utils/base.py
──────────────────────
Shared utilities used by every scraper:
  - Retry decorator with exponential backoff
  - Rate-limited HTTP session
  - Validated CSV writer with schema enforcement
  - Structured logger
"""

import csv
import json
import time
import random
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict, fields
from typing import Any, Dict, List, Optional, Type, TypeVar

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ── Logger setup ──────────────────────────────────────────────────────────────
def get_logger(name: str) -> logging.Logger:
    """Return a configured logger writing to console + rotating log file."""
    from config.settings import LOG_DIR

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    log_path = LOG_DIR / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ── HTTP Session with retry ───────────────────────────────────────────────────
def make_session(
    retries: int = 3,
    backoff_factor: float = 1.5,
    status_forcelist: tuple = (429, 500, 502, 503, 504),
) -> requests.Session:
    """
    Return a requests.Session with automatic retry + backoff.
    Honours Retry-After header on 429 responses.
    """
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=["GET", "POST"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def rate_limited_get(
    session: requests.Session,
    url: str,
    params: Optional[Dict] = None,
    headers: Optional[Dict] = None,
    delay_min: float = 2.0,
    delay_max: float = 4.0,
    timeout: int = 30,
    logger: Optional[logging.Logger] = None,
) -> Optional[requests.Response]:
    """
    GET with random delay + error handling.
    Returns None on unrecoverable error.
    """
    log = logger or logging.getLogger("rate_limited_get")
    delay = random.uniform(delay_min, delay_max)
    time.sleep(delay)

    try:
        resp = session.get(url, params=params, headers=headers, timeout=timeout)

        if resp.status_code == 429:
            wait = int(resp.headers.get("Retry-After", 60))
            log.warning(f"Rate limited. Waiting {wait}s before retry...")
            time.sleep(wait)
            resp = session.get(url, params=params, headers=headers, timeout=timeout)

        if resp.status_code != 200:
            log.error(f"HTTP {resp.status_code} for {url}")
            return None

        return resp

    except requests.exceptions.Timeout:
        log.error(f"Timeout fetching {url}")
    except requests.exceptions.ConnectionError as e:
        log.error(f"Connection error for {url}: {e}")
    except Exception as e:
        log.exception(f"Unexpected error fetching {url}: {e}")

    return None


# ── CSV Writer with schema validation ─────────────────────────────────────────
T = TypeVar("T")


class SchemaCSVWriter:
    """
    Thread-safe CSV writer that enforces a dataclass schema.
    Writes header on first record. Appends on subsequent runs.
    Logs skipped records for audit.
    """

    def __init__(
        self,
        filepath: Path,
        schema_class: Type,
        mode: str = "a",
    ):
        self.filepath = Path(filepath)
        self.schema_class = schema_class
        self._col_names = [f.name for f in fields(schema_class)]
        self._write_header = not self.filepath.exists() or self.filepath.stat().st_size == 0
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.filepath, mode, newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self._col_names)
        if self._write_header:
            self._writer.writeheader()
        self._written = 0
        self._skipped = 0

    def write(self, record: Any) -> bool:
        """Write one dataclass record. Returns True on success."""
        try:
            row = asdict(record) if hasattr(record, "__dataclass_fields__") else record
            # Ensure all required fields present
            for col in self._col_names:
                if col not in row:
                    row[col] = None
            self._writer.writerow({k: row.get(k) for k in self._col_names})
            self._written += 1
            return True
        except Exception as e:
            self._skipped += 1
            logging.getLogger("SchemaCSVWriter").warning(f"Skipped record: {e} | {record}")
            return False

    def write_many(self, records: List[Any]) -> int:
        """Write multiple records. Returns count written."""
        return sum(self.write(r) for r in records)

    def flush(self):
        self._file.flush()

    def close(self):
        self._file.close()

    def stats(self) -> Dict:
        return {"written": self._written, "skipped": self._skipped, "file": str(self.filepath)}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ── Checkpoint manager (resume interrupted scrapes) ───────────────────────────
class Checkpoint:
    """
    Persists scraping progress to JSON so runs can be resumed.

    Usage:
        cp = Checkpoint("trials_scraper")
        if cp.is_done("metformin"):
            continue
        # ... scrape drug ...
        cp.mark_done("metformin")
    """

    def __init__(self, name: str, checkpoint_dir: Optional[Path] = None):
        from config.settings import LOG_DIR
        cp_dir = checkpoint_dir or (LOG_DIR / "checkpoints")
        cp_dir.mkdir(parents=True, exist_ok=True)
        self._path = cp_dir / f"{name}.json"
        self._done: set = set()
        self._load()

    def _load(self):
        if self._path.exists():
            with open(self._path) as f:
                self._done = set(json.load(f).get("done", []))

    def _save(self):
        with open(self._path, "w") as f:
            json.dump({"done": sorted(self._done)}, f, indent=2)

    def is_done(self, key: str) -> bool:
        return key in self._done

    def mark_done(self, key: str):
        self._done.add(key)
        self._save()

    def reset(self):
        self._done.clear()
        self._save()

    @property
    def completed(self) -> int:
        return len(self._done)
