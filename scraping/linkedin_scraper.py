# scraping/linkedin_scraper.py
"""
LinkedIn job scraper using Selenium for dynamic content rendering.

LinkedIn is heavily JavaScript-driven and has strong anti-bot measures.
Selenium with a headless Chrome/Firefox browser is used here.

PRODUCTION ALTERNATIVES:
- LinkedIn Job Search API (requires LinkedIn partner access)
- RapidAPI LinkedIn endpoints (paid but reliable)
- Playwright (faster than Selenium, better stealth)

Setup: pip install selenium webdriver-manager
       brew install chromedriver   (or apt-get)
"""

import logging
import time
import re
from typing import List, Optional

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

try:
    from webdriver_manager.chrome import ChromeDriverManager
    WEBDRIVER_MANAGER_AVAILABLE = True
except ImportError:
    WEBDRIVER_MANAGER_AVAILABLE = False

from scraping.base_scraper import BaseScraper, RawJob
from config.settings import SCRAPING

logger = logging.getLogger(__name__)

BASE_URL = "https://www.linkedin.com/jobs/search"


class LinkedInScraper(BaseScraper):
    """
    Scrapes LinkedIn Jobs using Selenium.
    Uses the public (non-login) job search to avoid account requirements.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.driver = None

    @property
    def source_name(self) -> str:
        return "linkedin"

    # ── Browser Setup / Teardown ──────────────────────────────────────────────

    def _start_driver(self):
        """Initialize a headless Chrome browser with anti-detection options."""
        options = Options()
        if SCRAPING.headless_browser:
            options.add_argument("--headless=new")    # Modern headless mode
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)
        options.add_argument(f"user-agent={SCRAPING.user_agents[0]}")
        
        if WEBDRIVER_MANAGER_AVAILABLE:
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=options)
        else:
            self.driver = webdriver.Chrome(options=options)
        
        # Mask webdriver fingerprint
        self.driver.execute_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )

    def _stop_driver(self):
        """Clean up browser resources."""
        if self.driver:
            self.driver.quit()
            self.driver = None

    # ── Scraping Logic ────────────────────────────────────────────────────────

    def _build_search_url(self, search_term: str, page: int) -> str:
        """Build LinkedIn job search URL. LinkedIn uses page offset of 25."""
        params = {
            "keywords": search_term,
            "location": self.location,
            "start": (page - 1) * 25,
            "f_TPR": "r2592000",          # Posted within last 30 days
            "sortBy": "DD",                # Sort by date
        }
        query = "&".join(f"{k}={v}" for k, v in params.items())
        return f"{BASE_URL}?{query}"

    def _fetch_jobs_page(self, search_term: str, page: int) -> List[RawJob]:
        """Navigate to a search results page and extract all job cards."""
        if not self.driver:
            self._start_driver()

        url = self._build_search_url(search_term, page)
        logger.debug(f"[LinkedIn] Loading: {url}")
        self.driver.get(url)
        time.sleep(2)  # Let JavaScript render

        # Scroll to load lazy content
        self._scroll_to_load()

        # Find all job cards
        try:
            cards = WebDriverWait(self.driver, 10).until(
                EC.presence_of_all_elements_located(
                    (By.CSS_SELECTOR, "li.jobs-search-results__list-item, div.base-card")
                )
            )
        except TimeoutException:
            logger.info("[LinkedIn] No job cards found on page")
            return []

        jobs = []
        for card in cards[:25]:
            job = self._parse_job_card(card)
            if job:
                jobs.append(job)
        return jobs

    def _scroll_to_load(self, scrolls: int = 5):
        """Scroll the page to trigger lazy loading of job cards."""
        for _ in range(scrolls):
            self.driver.execute_script("window.scrollBy(0, 800)")
            time.sleep(0.5)

    def _parse_job_card(self, card) -> Optional[RawJob]:
        """Extract structured data from a LinkedIn job card element."""
        try:
            # Title
            title_el = card.find_element(By.CSS_SELECTOR, 
                "h3.base-search-card__title, span.sr-only")
            title = title_el.text.strip() if title_el else None
            if not title:
                return None

            # Company
            company_el = card.find_element(By.CSS_SELECTOR,
                "h4.base-search-card__subtitle, a.hidden-nested-link")
            company = company_el.text.strip() if company_el else "Unknown"

            # Location
            location_el = card.find_element(By.CSS_SELECTOR,
                "span.job-search-card__location")
            location = location_el.text.strip() if location_el else self.location

            # Job URL
            link_el = card.find_element(By.CSS_SELECTOR, "a.base-card__full-link")
            job_url = link_el.get_attribute("href") if link_el else ""
            # Strip tracking params
            job_url = job_url.split("?")[0] if job_url else ""

            # LinkedIn doesn't show salary/description in cards
            # Fetch detail page for full description
            description = self._fetch_job_description(job_url) if job_url else ""

            # Work mode from metadata
            work_mode_raw = self._extract_work_mode(card)

            return RawJob(
                title=title,
                company=company,
                description=description,
                location=location,
                source=self.source_name,
                source_url=job_url,
                work_mode_raw=work_mode_raw,
                job_id=f"linkedin_{hash(job_url)}",
            )

        except NoSuchElementException as e:
            logger.debug(f"Element not found in LinkedIn card: {e}")
            return None
        except Exception as e:
            logger.warning(f"Failed to parse LinkedIn card: {e}")
            return None

    def _fetch_job_description(self, job_url: str) -> str:
        """
        Open the job detail page and scrape the full description.
        NOTE: This makes an extra request per job — throttle accordingly.
        """
        try:
            self.driver.execute_script(f"window.open('{job_url}', '_blank')")
            self.driver.switch_to.window(self.driver.window_handles[-1])
            time.sleep(2)

            desc_el = WebDriverWait(self.driver, 5).until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, "div.description__text, div.show-more-less-html__markup")
                )
            )
            description = desc_el.text
        except Exception:
            description = ""
        finally:
            # Close detail tab, switch back to results
            if len(self.driver.window_handles) > 1:
                self.driver.close()
                self.driver.switch_to.window(self.driver.window_handles[0])
        return description

    def _extract_work_mode(self, card) -> Optional[str]:
        """Extract work mode badge from LinkedIn job card."""
        try:
            badge = card.find_element(By.CSS_SELECTOR,
                "span.job-search-card__benefits, li.job-criteria__item")
            text = badge.text.lower()
            if "remote" in text:
                return "remote"
            if "hybrid" in text:
                return "hybrid"
            return "onsite"
        except NoSuchElementException:
            return None

    def scrape(self):
        """Override to ensure driver cleanup after scraping."""
        try:
            yield from super().scrape()
        finally:
            self._stop_driver()
