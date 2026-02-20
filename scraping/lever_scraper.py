# scraping/lever_scraper.py
"""
Lever ATS scraper — no API key, no auth, no bot detection.

Lever powers hiring at: Netflix, Uber, GitHub, Shopify, Spotify,
Twitter/X, Atlassian, Snowflake, MongoDB, and 1000+ tech companies.

How it works:
  Every company on Lever exposes a public JSON API at:
  https://api.lever.co/v0/postings/{company}?mode=json

  Like Greenhouse, this is intentionally public.
  Returns clean JSON with full job descriptions.

Usage:
    scraper = LeverScraper(keywords=["devops", "machine learning", "kubernetes"])
    for job in scraper.scrape():
        print(job.title, job.company)
"""

import logging
import re
from typing import List, Optional

from bs4 import BeautifulSoup

from scraping.base_scraper import BaseScraper, RawJob

logger = logging.getLogger(__name__)

LEVER_API = "https://api.lever.co/v0/postings/{company}"

# Companies using Lever — add more as needed
# Verify at: https://jobs.lever.co/{company}
LEVER_COMPANIES = [
    # AI/ML focused
    "mistralai", "anyscale", "modal", "replicate",
    "qdrant", "weaviate", "pinecone", "chroma",
    # DevOps/Infrastructure
    "grafana", "pulumi", "temporal", "earthly",
    "harness", "launchdarkly", "honeycomb-io",
    # Big tech & unicorns
    "netflix", "shopify", "atlassian", "mongodb",
    "snowflake", "twilio", "zendesk", "intercom",
    "canva", "miro", "airtable", "linear",
    # Data platforms
    "dbt-labs", "airbyte", "prefect", "dagster",
    "great-expectations", "monte-carlo-data",
    # Security/DevSecOps
    "snyk", "lacework", "orca-security",
]


class LeverScraper(BaseScraper):
    """
    Scrapes jobs from Lever ATS public JSON API.
    Filters by keyword to return only DevOps/AI/ML relevant jobs.
    """

    def __init__(self, keywords: List[str], companies: List[str] = None):
        super().__init__(search_terms=["*"], location="United States")
        self.keywords = [k.lower() for k in keywords]
        self.companies = companies or LEVER_COMPANIES

    @property
    def source_name(self) -> str:
        return "lever"

    def scrape(self):
        """Override base — iterate over companies not search terms."""
        total = 0
        for company in self.companies:
            jobs = self._fetch_company_jobs(company)
            for job in jobs:
                total += 1
                yield job
        logger.info(f"[lever] Complete. {total} matching jobs across {len(self.companies)} companies.")

    def _fetch_jobs_page(self, search_term: str, page: int) -> List[RawJob]:
        """Not used for Lever — company-based iteration."""
        return []

    def _fetch_company_jobs(self, company: str) -> List[RawJob]:
        """
        Fetch all open jobs for a company from Lever's public API.
        Lever returns all postings in a single paginated request.
        """
        url = LEVER_API.format(company=company)
        response = self._get(url, params={"mode": "json", "limit": 500})
        if not response:
            return []

        try:
            data = response.json()
        except Exception:
            logger.debug(f"[lever] Invalid JSON from {company}")
            return []

        # Lever returns either a list directly or {data: [...]}
        if isinstance(data, list):
            all_jobs = data
        else:
            all_jobs = data.get("data", [])

        if not all_jobs:
            return []

        matching = []
        for item in all_jobs:
            job = self._parse_job(item, company)
            if job and self._is_relevant(job):
                matching.append(job)

        if matching:
            logger.info(f"[lever] {company}: {len(matching)}/{len(all_jobs)} relevant jobs")
        return matching

    def _parse_job(self, item: dict, company: str) -> Optional[RawJob]:
        """Parse a single Lever posting into a RawJob."""
        try:
            title = item.get("text", "").strip()
            if not title:
                return None

            job_id = item.get("id", "")
            job_url = item.get("hostedUrl", "") or item.get("applyUrl", "")

            # Location
            categories = item.get("categories", {})
            location = categories.get("location", "United States") or "United States"
            work_mode_cat = categories.get("commitment", "")
            team = categories.get("team", "")
            department = categories.get("department", "")

            # Description — Lever splits into lists/additional sections
            description = self._build_description(item)

            # Work mode
            work_mode = self._detect_work_mode(
                work_mode_cat,
                location,
                title,
                description[:500],
            )

            # Posted timestamp (milliseconds)
            created_ms = item.get("createdAt", 0)
            posted = ""
            if created_ms:
                from datetime import datetime, timezone
                posted = datetime.fromtimestamp(
                    created_ms / 1000, tz=timezone.utc
                ).isoformat()

            return RawJob(
                title=title,
                company=self._format_company_name(company),
                description=description,
                location=location,
                source="lever",
                source_url=job_url,
                work_mode_raw=work_mode,
                job_id=f"lever_{job_id}",
                posted_date=posted,
                skills_raw=f"{team} {department}".strip(),
            )

        except Exception as e:
            logger.debug(f"[lever] Failed to parse job from {company}: {e}")
            return None

    def _build_description(self, item: dict) -> str:
        """
        Lever jobs have structured description sections.
        Combine them into a single text block.
        """
        parts = []

        # Main description
        desc_html = item.get("descriptionPlain", "") or item.get("description", "")
        if desc_html:
            parts.append(self._strip_html(desc_html))

        # Additional sections (requirements, nice-to-haves, etc.)
        for section in item.get("lists", []):
            heading = section.get("text", "")
            content_html = section.get("content", "")
            if heading:
                parts.append(f"\n{heading}:")
            if content_html:
                parts.append(self._strip_html(content_html))

        # Closing section
        closing = item.get("additional", "") or item.get("additionalPlain", "")
        if closing:
            parts.append(self._strip_html(closing))

        return "\n".join(parts).strip()

    def _strip_html(self, html: str) -> str:
        """Convert HTML to plain text, preserving structure."""
        if not html:
            return ""
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup.find_all(["p", "li", "br"]):
            tag.insert_after("\n")
        return re.sub(r"\n{3,}", "\n\n", soup.get_text()).strip()

    def _is_relevant(self, job: RawJob) -> bool:
        """Filter jobs by keyword match in title or description."""
        search_text = f"{job.title} {job.description[:300]}".lower()
        return any(kw in search_text for kw in self.keywords)

    def _detect_work_mode(self, commitment: str, location: str,
                          title: str, description: str) -> str:
        """Detect work mode from Lever's commitment field and text signals."""
        combined = f"{commitment} {location} {title} {description}".lower()
        if any(w in combined for w in ["remote", "distributed", "work from home", "wfh"]):
            return "remote"
        if "hybrid" in combined:
            return "hybrid"
        return "onsite"

    def _format_company_name(self, slug: str) -> str:
        """Convert Lever company slug to display name."""
        name_map = {
            "mistralai": "Mistral AI",
            "anyscale": "Anyscale",
            "grafana": "Grafana Labs",
            "pulumi": "Pulumi",
            "launchdarkly": "LaunchDarkly",
            "honeycomb-io": "Honeycomb",
            "dbt-labs": "dbt Labs",
            "airbyte": "Airbyte",
            "great-expectations": "Great Expectations",
            "monte-carlo-data": "Monte Carlo",
            "orca-security": "Orca Security",
        }
        return name_map.get(slug, slug.replace("-", " ").title())