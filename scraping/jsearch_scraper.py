# scraping/jsearch_scraper.py
"""
JSearch API scraper via RapidAPI.

JSearch aggregates jobs from LinkedIn, Indeed, Glassdoor, and more.
Free tier: 200 requests/month. Paid: unlimited.

Setup:
  1. Go to https://rapidapi.com/letscrape-6bRBa3QguO5/api/jsearch
  2. Sign up (free) and subscribe to JSearch
  3. Copy your RapidAPI key
  4. Set env var: export RAPIDAPI_KEY=your_key_here

No Selenium, no 403s — clean JSON API responses.
"""

import logging
import os
from typing import List, Optional

import requests

from scraping.base_scraper import BaseScraper, RawJob

logger = logging.getLogger(__name__)

JSEARCH_URL = "https://jsearch.p.rapidapi.com/search"
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY", "")


class JSearchScraper(BaseScraper):
    """
    Scrapes jobs via the JSearch RapidAPI endpoint.
    Returns jobs from LinkedIn, Indeed, Glassdoor, and more in one call.
    """

    @property
    def source_name(self) -> str:
        return "jsearch"

    def _fetch_jobs_page(self, search_term: str, page: int) -> List[RawJob]:
        """Fetch one page of results from JSearch API."""
        if not RAPIDAPI_KEY:
            raise ValueError(
                "RAPIDAPI_KEY not set. "
                "Get a free key at https://rapidapi.com/letscrape-6bRBa3QguO5/api/jsearch"
            )

        headers = {
            "X-RapidAPI-Key": RAPIDAPI_KEY,
            "X-RapidAPI-Host": "jsearch.p.rapidapi.com",
        }
        params = {
            "query": f"{search_term} in {self.location}",
            "page": str(page),
            "num_pages": "1",
            "date_posted": "month",   # Jobs from last 30 days
        }

        try:
            response = requests.get(
                JSEARCH_URL, headers=headers, params=params, timeout=15
            )
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"JSearch API error: {e}")
            return []

        raw_results = data.get("data", [])
        if not raw_results:
            return []

        jobs = []
        for item in raw_results:
            job = self._parse_item(item)
            if job:
                jobs.append(job)

        logger.info(f"[jsearch] '{search_term}' page {page}: {len(jobs)} jobs")
        return jobs

    def _parse_item(self, item: dict) -> Optional[RawJob]:
        """Parse a single JSearch API result into a RawJob."""
        try:
            title = item.get("job_title", "").strip()
            company = item.get("employer_name", "Unknown").strip()
            if not title or not company:
                return None

            # Location
            city = item.get("job_city", "")
            state = item.get("job_state", "")
            country = item.get("job_country", "")
            location = ", ".join(p for p in [city, state, country] if p) or self.location

            # Description
            description = item.get("job_description", "")

            # Work mode
            is_remote = item.get("job_is_remote", False)
            work_mode_raw = "remote" if is_remote else "onsite"

            # Salary
            min_sal = item.get("job_min_salary")
            max_sal = item.get("job_max_salary")
            sal_currency = item.get("job_salary_currency", "USD")
            sal_period = item.get("job_salary_period", "")
            salary_raw = ""
            if min_sal and max_sal:
                salary_raw = f"{sal_currency} {min_sal}-{max_sal} {sal_period}"
            elif min_sal:
                salary_raw = f"{sal_currency} {min_sal} {sal_period}"

            # URL and ID
            job_url = item.get("job_apply_link", "") or item.get("job_url", "")
            job_id = item.get("job_id", "")

            # Posted date
            posted = item.get("job_posted_at_datetime_utc", "")

            return RawJob(
                title=title,
                company=company,
                description=description,
                location=location,
                source=f"jsearch:{item.get('job_publisher', 'unknown').lower()[:40]}",
                source_url=job_url,
                salary_raw=salary_raw,
                work_mode_raw=work_mode_raw,
                job_id=f"jsearch_{job_id}",
                posted_date=posted,
            )

        except Exception as e:
            logger.warning(f"Failed to parse JSearch item: {e}")
            return None