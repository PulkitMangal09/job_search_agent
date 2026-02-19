# scraping/base_scraper.py
"""
Abstract base class that all job scrapers must implement.
Handles common functionality: rate limiting, retries, user-agent rotation,
and provides a consistent output schema for downstream processing.
"""

import time
import random
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Optional, Generator

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config.settings import SCRAPING

logger = logging.getLogger(__name__)


@dataclass
class RawJob:
    """
    Standardized raw job record produced by every scraper.
    Fields marked Optional may not be available from all sources.
    """
    # Required fields
    title: str
    company: str
    description: str
    location: str
    source: str                    # e.g., "indeed", "linkedin"
    source_url: str                # Original job posting URL
    scraped_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # Optional raw fields (cleaned/parsed later in the pipeline)
    salary_raw: Optional[str] = None         # e.g., "$120k - $150k / year"
    work_mode_raw: Optional[str] = None      # e.g., "Remote", "Hybrid"
    skills_raw: Optional[str] = None         # Raw skills string from listing
    posted_date: Optional[str] = None
    job_id: Optional[str] = None            # Source-specific ID for deduplication

    def to_dict(self) -> dict:
        return asdict(self)


class BaseScraper(ABC):
    """
    Abstract base class for all job scrapers.
    Subclasses implement `_fetch_jobs_page()` for source-specific logic.
    """

    def __init__(self, search_terms: List[str], location: str = "United States"):
        self.search_terms = search_terms
        self.location = location
        self.session = self._build_session()
        self._request_count = 0

    # ── Session Setup ─────────────────────────────────────────────────────────

    def _build_session(self) -> requests.Session:
        """Create a requests.Session with retry logic baked in."""
        session = requests.Session()
        retry_strategy = Retry(
            total=SCRAPING.max_retries,
            backoff_factor=1,                 # 1s, 2s, 4s exponential backoff
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def _get_headers(self) -> dict:
        """Rotate user agents on each request to reduce blocking risk."""
        return {
            "User-Agent": random.choice(SCRAPING.user_agents),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
        }

    # ── Rate Limiting ─────────────────────────────────────────────────────────

    def _polite_delay(self):
        """
        Wait between requests to respect the target site.
        Adds random jitter to avoid pattern detection.
        """
        delay = SCRAPING.request_delay_seconds + random.uniform(0.5, 1.5)
        logger.debug(f"Rate limiting: sleeping {delay:.1f}s")
        time.sleep(delay)

    # ── HTTP Helpers ──────────────────────────────────────────────────────────

    def _get(self, url: str, params: dict = None) -> Optional[requests.Response]:
        """
        Perform a GET request with automatic rate limiting and error handling.
        Returns None on unrecoverable failure.
        """
        self._polite_delay()
        try:
            resp = self.session.get(
                url,
                headers=self._get_headers(),
                params=params,
                timeout=SCRAPING.timeout_seconds,
            )
            resp.raise_for_status()
            self._request_count += 1
            logger.debug(f"[{self.source_name}] GET {url} → {resp.status_code}")
            return resp
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                # Respect rate limit: wait longer then retry once
                logger.warning(f"Rate limited on {url}. Waiting 60s...")
                time.sleep(60)
                return self._get(url, params)
            logger.error(f"HTTP error {e.response.status_code} for {url}: {e}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {url}: {e}")
        return None

    # ── Abstract Interface ────────────────────────────────────────────────────

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Identifier for this scraper, e.g. 'indeed'."""

    @abstractmethod
    def _fetch_jobs_page(self, search_term: str, page: int) -> List[RawJob]:
        """
        Fetch a single page of job results for the given search term.
        Must return a list of RawJob objects (empty list if no results).
        """

    # ── Public Interface ──────────────────────────────────────────────────────

    def scrape(self) -> Generator[RawJob, None, None]:
        """
        Main scraping loop. Iterates over all search terms and pages.
        Yields individual RawJob objects for streaming processing.
        """
        total = 0
        for term in self.search_terms:
            logger.info(f"[{self.source_name}] Scraping: '{term}' in '{self.location}'")
            for page in range(1, SCRAPING.max_pages_per_source + 1):
                jobs = self._fetch_jobs_page(term, page)
                if not jobs:
                    logger.info(f"[{self.source_name}] No more results on page {page}")
                    break
                for job in jobs:
                    total += 1
                    yield job
                logger.info(f"[{self.source_name}] Page {page}: {len(jobs)} jobs (total: {total})")
        logger.info(f"[{self.source_name}] Scraping complete. {total} total jobs, {self._request_count} requests.")
