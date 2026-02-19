# scraping/indeed_scraper.py
"""
Indeed.com job scraper using BeautifulSoup.

NOTE: Indeed's HTML structure changes frequently. This scraper targets
the structure as of early 2024. If selectors break, inspect the page
HTML and update the CSS selectors in _parse_job_card().

IMPORTANT: For production use, consider the Indeed Publisher API
(requires approval) or a paid proxy/SERP API to avoid blocks.
"""

import logging
import re
from typing import List, Optional
from urllib.parse import urlencode, urljoin

from bs4 import BeautifulSoup

from scraping.base_scraper import BaseScraper, RawJob

logger = logging.getLogger(__name__)

BASE_URL = "https://www.indeed.com"
SEARCH_URL = f"{BASE_URL}/jobs"


class IndeedScraper(BaseScraper):
    """Scrapes job listings from Indeed.com."""

    @property
    def source_name(self) -> str:
        return "indeed"

    def _build_search_url(self, search_term: str, page: int) -> tuple[str, dict]:
        """Construct the Indeed search URL and query params for a given page."""
        params = {
            "q": search_term,
            "l": self.location,
            "start": (page - 1) * 10,     # Indeed uses offset-based pagination (10 per page)
            "fromage": 30,                  # Jobs posted in last 30 days
            "sort": "date",
        }
        return SEARCH_URL, params

    def _fetch_jobs_page(self, search_term: str, page: int) -> List[RawJob]:
        """Fetch one page of Indeed results and parse job cards."""
        url, params = self._build_search_url(search_term, page)
        response = self._get(url, params)
        if not response:
            return []

        soup = BeautifulSoup(response.text, "html.parser")
        job_cards = soup.find_all("div", class_=re.compile(r"job_seen_beacon|tapItem"))
        
        if not job_cards:
            logger.debug("No job cards found — page structure may have changed")
            return []

        jobs = []
        for card in job_cards:
            job = self._parse_job_card(card, soup)
            if job:
                jobs.append(job)
        return jobs

    def _parse_job_card(self, card: BeautifulSoup, page_soup: BeautifulSoup) -> Optional[RawJob]:
        """Extract fields from a single Indeed job card element."""
        try:
            # Job title
            title_el = (
                card.find("h2", class_=re.compile(r"jobTitle"))
                or card.find("a", {"data-jk": True})
            )
            title = title_el.get_text(strip=True) if title_el else None
            if not title:
                return None

            # Company name
            company_el = card.find("span", {"data-testid": "company-name"}) \
                          or card.find("span", class_=re.compile(r"companyName"))
            company = company_el.get_text(strip=True) if company_el else "Unknown"

            # Location
            location_el = card.find("div", {"data-testid": "text-location"}) \
                          or card.find("div", class_=re.compile(r"companyLocation"))
            location = location_el.get_text(strip=True) if location_el else self.location

            # Salary (often absent)
            salary_el = card.find("div", class_=re.compile(r"salary|compensation"))
            salary_raw = salary_el.get_text(strip=True) if salary_el else None

            # Job URL + ID
            link_el = card.find("a", {"id": re.compile(r"job_")}) \
                      or card.find("a", href=re.compile(r"/rc/clk"))
            job_url = urljoin(BASE_URL, link_el["href"]) if link_el else ""
            job_id = link_el.get("data-jk", "") if link_el else ""

            # Description — Indeed loads full desc on detail page;
            # grab the snippet from the card as a proxy
            desc_el = card.find("div", class_=re.compile(r"job-snippet|underShelfFooter"))
            description = desc_el.get_text(separator=" ", strip=True) if desc_el else ""

            # Work mode (often embedded in location text)
            work_mode_raw = self._detect_work_mode(location + " " + description)

            return RawJob(
                title=title,
                company=company,
                description=description,
                location=location,
                source=self.source_name,
                source_url=job_url,
                salary_raw=salary_raw,
                work_mode_raw=work_mode_raw,
                job_id=f"indeed_{job_id}",
            )

        except Exception as e:
            logger.warning(f"Failed to parse job card: {e}")
            return None

    def _detect_work_mode(self, text: str) -> Optional[str]:
        """Quick heuristic to detect work mode from location/description text."""
        text_lower = text.lower()
        if any(w in text_lower for w in ["remote", "work from home", "wfh"]):
            return "remote"
        if "hybrid" in text_lower:
            return "hybrid"
        return "onsite"

    def fetch_job_details(self, job_url: str) -> Optional[str]:
        """
        Fetch the full job description from the detail page.
        Call this for jobs where the card only shows a snippet.
        """
        response = self._get(job_url)
        if not response:
            return None
        soup = BeautifulSoup(response.text, "html.parser")
        desc_el = soup.find("div", id="jobDescriptionText") \
                  or soup.find("div", class_=re.compile(r"jobsearch-jobDescriptionText"))
        return desc_el.get_text(separator="\n", strip=True) if desc_el else None
