# scraping/greenhouse_scraper.py
"""
Greenhouse ATS scraper — no API key, no auth, no bot detection.

Greenhouse powers hiring at: OpenAI, Anthropic, Stripe, Airbnb, Figma,
Notion, Coinbase, Reddit, Dropbox, and 500+ tech companies.

How it works:
  Every company on Greenhouse exposes a public JSON API at:
  https://boards-api.greenhouse.io/v1/boards/{company}/jobs?content=true

  This is intentionally public — Greenhouse designed it for job aggregators.
  No scraping, no HTML parsing, just clean JSON.

Usage:
    scraper = GreenhouseScraper(keywords=["devops", "machine learning"])
    for job in scraper.scrape():
        print(job.title, job.company)
"""

import logging
import re
from typing import List, Optional

import requests
from bs4 import BeautifulSoup

from scraping.base_scraper import BaseScraper, RawJob
from config.settings import SCRAPING

logger = logging.getLogger(__name__)

GREENHOUSE_API = "https://boards-api.greenhouse.io/v1/boards/{company}/jobs"

# Companies using Greenhouse — add more as needed
# Find more at: https://boards.greenhouse.io/{company}
GREENHOUSE_COMPANIES = [
    # AI/ML focused
    "openai", "anthropic", "cohere", "huggingface", "scale",
    "deepmind", "inflection", "mistral", "adept", "together",
    "perplexity", "characterai",
    # DevOps/Cloud heavy
    "hashicorp", "datadog", "confluent", "fastly", "cloudflare",
    "pagerduty", "newrelic", "sumo", "lightstep",
    # Big tech & unicorns
    "stripe", "airbnb", "notion", "figma", "dropbox",
    "reddit", "coinbase", "brex", "plaid", "robinhood",
    "doordash", "instacart", "lyft", "waymo",
    # Data & ML platforms
    "databricks", "dbtlabs", "fivetran", "astronomer",
    "weights-and-biases", "scale", "labelbox",
]


class GreenhouseScraper(BaseScraper):
    """
    Scrapes jobs from Greenhouse ATS public JSON API.
    Filters by keyword to return only DevOps/AI/ML relevant jobs.
    """

    def __init__(self, keywords: List[str], companies: List[str] = None):
        # Pass empty list as search_terms — we use keywords for filtering instead
        super().__init__(search_terms=["*"], location="United States")
        self.keywords = [k.lower() for k in keywords]
        self.companies = companies or GREENHOUSE_COMPANIES

    @property
    def source_name(self) -> str:
        return "greenhouse"

    def scrape(self):
        """
        Override base scrape() — iterate over companies not search terms.
        Yields RawJob objects for matching jobs.
        """
        total = 0
        for company in self.companies:
            jobs = self._fetch_company_jobs(company)
            for job in jobs:
                total += 1
                yield job
        logger.info(f"[greenhouse] Complete. {total} matching jobs across {len(self.companies)} companies.")

    def _fetch_jobs_page(self, search_term: str, page: int) -> List[RawJob]:
        """Not used — greenhouse iterates by company, not search term."""
        return []

    def _fetch_company_jobs(self, company: str) -> List[RawJob]:
        """
        Fetch all open jobs for a company and filter by keywords.
        Returns empty list if company not on Greenhouse or request fails.
        """
        url = GREENHOUSE_API.format(company=company)
        response = self._get(url, params={"content": "true"})
        if not response:
            return []

        try:
            data = response.json()
        except Exception:
            logger.debug(f"[greenhouse] Invalid JSON from {company}")
            return []

        all_jobs = data.get("jobs", [])
        if not all_jobs:
            return []

        # Filter to DevOps/AI/ML relevant jobs by keyword matching
        matching = []
        for item in all_jobs:
            job = self._parse_job(item, company)
            if job and self._is_relevant(job):
                matching.append(job)

        if matching:
            logger.info(f"[greenhouse] {company}: {len(matching)}/{len(all_jobs)} relevant jobs")
        return matching

    def _parse_job(self, item: dict, company: str) -> Optional[RawJob]:
        """Parse a single Greenhouse job JSON object into a RawJob."""
        try:
            title = item.get("title", "").strip()
            if not title:
                return None

            job_id = str(item.get("id", ""))
            job_url = item.get("absolute_url", "")

            # Location
            locations = item.get("offices", []) or item.get("locations", [])
            location = locations[0].get("name", "United States") if locations else "United States"

            # Full description (HTML) — strip tags
            content = item.get("content", "") or ""
            description = self._strip_html(content)

            # Metadata
            departments = item.get("departments", [])
            department = departments[0].get("name", "") if departments else ""

            # Posted date
            posted = item.get("updated_at", "") or item.get("created_at", "")

            # Work mode — detect from title, location, description
            work_mode = self._detect_work_mode_text(
                f"{title} {location} {description[:500]}"
            )

            return RawJob(
                title=title,
                company=self._format_company_name(company),
                description=description,
                location=location,
                source="greenhouse",
                source_url=job_url,
                work_mode_raw=work_mode,
                job_id=f"greenhouse_{job_id}",
                posted_date=posted,
                skills_raw=department,
            )

        except Exception as e:
            logger.debug(f"[greenhouse] Failed to parse job from {company}: {e}")
            return None

    def _is_relevant(self, job: RawJob) -> bool:
        """
        Check if a job matches our DevOps/AI/ML keywords.
        Matches against title and first 300 chars of description.
        """
        search_text = f"{job.title} {job.description[:300]}".lower()
        return any(kw in search_text for kw in self.keywords)

    def _strip_html(self, html: str) -> str:
        """Convert HTML job description to plain text."""
        if not html:
            return ""
        soup = BeautifulSoup(html, "html.parser")
        # Preserve line breaks from block elements
        for tag in soup.find_all(["p", "li", "br"]):
            tag.insert_after("\n")
        return re.sub(r"\n{3,}", "\n\n", soup.get_text()).strip()

    def _detect_work_mode_text(self, text: str) -> str:
        text_lower = text.lower()
        if any(w in text_lower for w in ["remote", "work from home", "wfh", "distributed"]):
            return "remote"
        if "hybrid" in text_lower:
            return "hybrid"
        return "onsite"

    def _format_company_name(self, slug: str) -> str:
        """Convert company slug to display name. e.g. 'dbtlabs' → 'dbt Labs'"""
        name_map = {
            "openai": "OpenAI",
            "anthropic": "Anthropic",
            "huggingface": "Hugging Face",
            "weights-and-biases": "Weights & Biases",
            "dbtlabs": "dbt Labs",
            "hashicorp": "HashiCorp",
            "datadog": "Datadog",
            "cloudflare": "Cloudflare",
            "pagerduty": "PagerDuty",
            "databricks": "Databricks",
            "coinbase": "Coinbase",
        }
        return name_map.get(slug, slug.replace("-", " ").title())