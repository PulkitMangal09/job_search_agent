# cleaning/cleaner.py
"""
Data cleaning and preprocessing for raw job listings.

Handles:
- Text normalization (HTML stripping, whitespace, encoding)
- Deduplication (by job ID and fuzzy title+company matching)
- Skill name standardization using the taxonomy alias map
- Field extraction (work mode, experience level) from raw text
"""

import re
import hashlib
import logging
from typing import List, Optional, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime

from bs4 import BeautifulSoup

from scraping.base_scraper import RawJob
from config.taxonomy import SKILL_NORMALIZATION, DEVOPS_ALIAS_MAP, AIML_ALIAS_MAP
from config.settings import EXPERIENCE_KEYWORDS, WORK_MODE_KEYWORDS

logger = logging.getLogger(__name__)


@dataclass
class CleanJob:
    """
    Cleaned and normalized job record ready for NLP processing and storage.
    All fields are guaranteed non-null (fallback to empty string/list).
    """
    # Core fields
    title: str
    company: str
    description: str
    location: str
    source: str
    source_url: str
    scraped_at: str

    # Extracted / normalized fields
    work_mode: str = "unknown"              # remote | hybrid | onsite | unknown
    experience_level: str = "unknown"       # junior | mid | senior | unknown
    salary_min: Optional[float] = None      # Parsed minimum salary (USD/year)
    salary_max: Optional[float] = None      # Parsed maximum salary (USD/year)
    salary_currency: str = "USD"

    # Dedup fingerprint
    fingerprint: str = ""

    # Skills extracted from description (populated by NLP stage)
    skills_raw: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class JobCleaner:
    """
    Transforms a stream of RawJob objects into cleaned CleanJob objects.
    Maintains a seen-set in memory for deduplication within a batch.
    """

    def __init__(self):
        self._seen_fingerprints: Set[str] = set()   # In-memory dedup cache

    # ── Main Entry Point ──────────────────────────────────────────────────────

    def clean(self, raw_jobs: List[RawJob]) -> List[CleanJob]:
        """
        Clean a list of RawJob objects.
        Returns deduplicated, normalized CleanJob objects.
        """
        cleaned = []
        for raw in raw_jobs:
            try:
                job = self._clean_one(raw)
                if job:
                    cleaned.append(job)
            except Exception as e:
                logger.warning(f"Failed to clean job '{raw.title}': {e}")
        logger.info(f"Cleaned {len(cleaned)}/{len(raw_jobs)} jobs "
                    f"({len(raw_jobs) - len(cleaned)} duplicates/failures dropped)")
        return cleaned

    def _clean_one(self, raw: RawJob) -> Optional[CleanJob]:
        """Process a single RawJob into a CleanJob."""
        # 1. Clean text fields
        title = self._normalize_text(raw.title)
        company = self._normalize_text(raw.company)
        description = self._clean_description(raw.description)
        location = self._normalize_text(raw.location)

        if not title or not company:
            return None   # Skip jobs with no usable identity

        # 2. Deduplicate
        fingerprint = self._compute_fingerprint(title, company, location)
        if fingerprint in self._seen_fingerprints:
            logger.debug(f"Duplicate: '{title}' @ {company}")
            return None
        self._seen_fingerprints.add(fingerprint)

        # 3. Parse salary
        salary_min, salary_max, currency = self._parse_salary(raw.salary_raw or "")

        # 4. Detect work mode
        work_mode = self._detect_work_mode(
            raw.work_mode_raw or "",
            location,
            description
        )

        # 5. Detect experience level
        exp_level = self._detect_experience_level(title, description)

        return CleanJob(
            title=title,
            company=company,
            description=description,
            location=location,
            source=raw.source,
            source_url=raw.source_url,
            scraped_at=raw.scraped_at,
            work_mode=work_mode,
            experience_level=exp_level,
            salary_min=salary_min,
            salary_max=salary_max,
            salary_currency=currency,
            fingerprint=fingerprint,
            skills_raw=[],   # Populated by NLP stage
        )

    # ── Text Normalization ────────────────────────────────────────────────────

    def _normalize_text(self, text: str) -> str:
        """Strip HTML, normalize whitespace, fix encoding."""
        if not text:
            return ""
        # Remove HTML tags
        text = BeautifulSoup(text, "html.parser").get_text(separator=" ")
        # Normalize unicode (smart quotes, dashes, etc.)
        text = text.encode("ascii", errors="ignore").decode("ascii")
        # Collapse multiple whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _clean_description(self, text: str) -> str:
        """
        Clean a job description:
        - Strip HTML and markdown
        - Remove boilerplate phrases
        - Normalize whitespace
        """
        if not text:
            return ""
        text = BeautifulSoup(text, "html.parser").get_text(separator="\n")
        # Remove markdown-style formatting
        text = re.sub(r"[#*_`]", "", text)
        # Remove excessive blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Remove common boilerplate
        boilerplate_patterns = [
            r"equal opportunity employer.*",
            r"we are an equal.*",
            r"click apply.*",
            r"by submitting.*application.*",
        ]
        for pattern in boilerplate_patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)
        return text.strip()

    # ── Deduplication ─────────────────────────────────────────────────────────

    def _compute_fingerprint(self, title: str, company: str, location: str) -> str:
        """
        Create a hash-based fingerprint for deduplication.
        Two jobs with the same title + company + location are considered duplicates.
        """
        canonical = f"{title.lower()}|{company.lower()}|{location.lower()}"
        # Remove punctuation and extra spaces for fuzzy matching
        canonical = re.sub(r"[^\w|]", "", canonical)
        return hashlib.md5(canonical.encode()).hexdigest()

    # ── Salary Parsing ────────────────────────────────────────────────────────

    def _parse_salary(self, salary_raw: str) -> tuple[Optional[float], Optional[float], str]:
        """
        Parse raw salary strings into (min, max, currency).
        
        Examples handled:
          "$120,000 - $150,000 a year"  → (120000, 150000, "USD")
          "$80k - $100k"                → (80000, 100000, "USD")
          "£50,000"                     → (50000, 50000, "GBP")
          "$45/hr"                      → (93600, 93600, "USD")  # annualized
        """
        if not salary_raw:
            return None, None, "USD"

        # Detect currency
        currency = "USD"
        if "£" in salary_raw or "GBP" in salary_raw:
            currency = "GBP"
        elif "€" in salary_raw or "EUR" in salary_raw:
            currency = "EUR"

        # Normalize: remove commas, currency symbols
        cleaned = re.sub(r"[£€$,]", "", salary_raw.lower())

        # Extract numbers (with optional k suffix)
        numbers = re.findall(r"(\d+\.?\d*)\s*k?", cleaned)
        if not numbers:
            return None, None, currency

        def to_annual(val: float, text: str) -> float:
            """Annualize hourly/monthly rates."""
            if "hr" in text or "hour" in text:
                return val * 2080   # 40hrs * 52 weeks
            if "month" in text:
                return val * 12
            if val < 1000:          # Likely expressed in thousands (e.g., "120k")
                return val * 1000
            return val

        values = [to_annual(float(n), cleaned) for n in numbers[:2]]

        if len(values) == 1:
            return values[0], values[0], currency
        return min(values), max(values), currency

    # ── Work Mode Detection ───────────────────────────────────────────────────

    def _detect_work_mode(self, work_mode_raw: str, location: str, description: str) -> str:
        """
        Determine remote/hybrid/onsite from available text signals.
        Priority: explicit field > location text > description text.
        """
        combined = f"{work_mode_raw} {location} {description}".lower()
        
        for mode, keywords in WORK_MODE_KEYWORDS.items():
            if any(kw in combined for kw in keywords):
                return mode
        return "onsite"   # Default assumption

    # ── Experience Level Detection ────────────────────────────────────────────

    def _detect_experience_level(self, title: str, description: str) -> str:
        """
        Classify job experience level from title and description keywords.
        Returns: 'junior' | 'mid' | 'senior' | 'unknown'
        """
        combined = f"{title} {description}".lower()

        # Score each level by keyword matches
        scores = {level: 0 for level in EXPERIENCE_KEYWORDS}
        for level, keywords in EXPERIENCE_KEYWORDS.items():
            for kw in keywords:
                if kw in combined:
                    scores[level] += 1

        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else "unknown"

    # ── Skill Normalization ───────────────────────────────────────────────────

    @staticmethod
    def normalize_skill(skill: str) -> str:
        """
        Normalize a skill name to its canonical form.
        Example: "k8s" → "kubernetes", "sklearn" → "scikit-learn"
        """
        skill_lower = skill.lower().strip()
        return SKILL_NORMALIZATION.get(skill_lower, skill_lower)
