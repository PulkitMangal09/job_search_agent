# scraping/workday_scraper.py
"""
Workday ATS scraper using their public Jobs API.

Workday powers hiring at: Microsoft, Amazon, Salesforce, Adobe,
VMware, ServiceNow, Nvidia, and most Fortune 500 companies.

How it works:
  Workday has a consistent internal API across all tenants:
  POST https://{tenant}.wd5.myworkdayjobs.com/wday/cxs/{tenant}/{board}/jobs

  Returns JSON with job listings. No auth required for public postings.

Usage:
    scraper = WorkdayScraper(keywords=["devops", "machine learning"])
    for job in scraper.scrape():
        print(job.title, job.company)
"""

import logging
import re
import json
from typing import List, Optional

import requests
from bs4 import BeautifulSoup

from scraping.base_scraper import BaseScraper, RawJob
from config.settings import SCRAPING

logger = logging.getLogger(__name__)

# Workday tenants: (tenant_id, board_name, display_name)
WORKDAY_COMPANIES = [
    # AI/ML/Cloud heavy
    ("nvidia",        "NVIDIAExternalCareerSite",  "NVIDIA"),
    ("salesforce",    "External_Career_Site",       "Salesforce"),
    ("adobe",         "External",                   "Adobe"),
    ("servicenow",    "External",                   "ServiceNow"),
    ("vmware",        "VMware",                     "VMware"),
    ("workday",       "External",                   "Workday"),
    ("splunk",        "splunk",                     "Splunk"),
    ("cloudera",      "External",                   "Cloudera"),
    ("nutanix",       "Nutanix",                    "Nutanix"),
    ("paloaltonetworks", "PaloAltoNetworks",        "Palo Alto Networks"),
]

WORKDAY_API = "https://{tenant}.wd5.myworkdayjobs.com/wday/cxs/{tenant}/{board}/jobs"


class WorkdayScraper(BaseScraper):
    """
    Scrapes jobs from Workday ATS via their internal JSON API.
    Filters results by keyword to DevOps/AI/ML jobs only.
    """

    def __init__(self, keywords: List[str], companies: List[tuple] = None):
        super().__init__(search_terms=["*"], location="United States")
        self.keywords = [k.lower() for k in keywords]
        self.companies = companies or WORKDAY_COMPANIES

    @property
    def source_name(self) -> str:
        return "workday"

    def scrape(self):
        """Override — iterate over companies not search terms."""
        total = 0
        for tenant, board, display_name in self.companies:
            jobs = self._fetch_company_jobs(tenant, board, display_name)
            for job in jobs:
                total += 1
                yield job
        logger.info(f"[workday] Complete. {total} matching jobs.")

    def _fetch_jobs_page(self, search_term: str, page: int) -> List[RawJob]:
        return []

    def _fetch_company_jobs(
        self, tenant: str, board: str, display_name: str
    ) -> List[RawJob]:
        """Fetch all matching jobs from a Workday tenant."""
        url = WORKDAY_API.format(tenant=tenant, board=board)
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            **self._get_headers(),
        }

        # Workday uses POST with search body
        # We search for each keyword separately to maximize results
        all_jobs = []
        seen_ids = set()

        for keyword in self.keywords[:5]:   # Limit to avoid too many requests
            payload = {
                "appliedFacets": {},
                "limit": 20,
                "offset": 0,
                "searchText": keyword,
            }
            try:
                response = requests.post(
                    url, headers=headers,
                    json=payload,
                    timeout=SCRAPING.timeout_seconds
                )
                if response.status_code != 200:
                    continue
                data = response.json()
            except Exception as e:
                logger.debug(f"[workday] {tenant} keyword '{keyword}': {e}")
                continue

            for item in data.get("jobPostings", []):
                job_id = item.get("externalPath", "")
                if job_id in seen_ids:
                    continue
                seen_ids.add(job_id)

                job = self._parse_job(item, tenant, board, display_name)
                if job:
                    all_jobs.append(job)

        if all_jobs:
            logger.info(f"[workday] {display_name}: {len(all_jobs)} matching jobs")
        return all_jobs

    def _parse_job(
        self, item: dict, tenant: str, board: str, display_name: str
    ) -> Optional[RawJob]:
        """Parse a Workday job posting into a RawJob."""
        try:
            title = item.get("title", "").strip()
            if not title:
                return None

            external_path = item.get("externalPath", "")
            job_url = (
                f"https://{tenant}.wd5.myworkdayjobs.com/en-US/{board}{external_path}"
            )

            location_parts = []
            for loc in item.get("locationsText", "").split(","):
                location_parts.append(loc.strip())
            location = ", ".join(location_parts) if location_parts else "United States"

            posted = item.get("postedOn", "")
            description = item.get("jobDescription", {}).get("descriptor", "") or ""
            description = self._strip_html(description)

            work_mode = self._detect_work_mode(f"{title} {location} {description[:300]}")

            return RawJob(
                title=title,
                company=display_name,
                description=description,
                location=location,
                source="workday",
                source_url=job_url,
                work_mode_raw=work_mode,
                job_id=f"workday_{tenant}_{external_path}",
                posted_date=posted,
            )

        except Exception as e:
            logger.debug(f"[workday] Parse error for {display_name}: {e}")
            return None

    def _strip_html(self, html: str) -> str:
        if not html:
            return ""
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup.find_all(["p", "li", "br"]):
            tag.insert_after("\n")
        return re.sub(r"\n{3,}", "\n\n", soup.get_text()).strip()

    def _detect_work_mode(self, text: str) -> str:
        text_lower = text.lower()
        if any(w in text_lower for w in ["remote", "work from home", "distributed"]):
            return "remote"
        if "hybrid" in text_lower:
            return "hybrid"
        return "onsite"