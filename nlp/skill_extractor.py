# nlp/skill_extractor.py
"""
NLP-based skill extraction from job descriptions.

Two-stage approach:
  1. Rule-based matching using spaCy's PhraseMatcher with skill taxonomy
     — Fast, deterministic, high precision for known skills
  2. Pattern-based extraction for skill variations and compound terms
     — Catches "AWS Lambda", "Kubernetes 1.28+", etc.

Usage:
    extractor = SkillExtractor()
    skills = extractor.extract("Experience with Kubernetes, Docker, and Terraform required.")
    # → ["kubernetes", "docker", "terraform"]
"""

import re
import logging
from typing import List, Set, Dict
from functools import lru_cache

import spacy
from spacy.matcher import PhraseMatcher

from config.settings import NLP as NLP_CONFIG
from config.taxonomy import (
    DEVOPS_SKILLS, AIML_SKILLS,
    DEVOPS_ALIAS_MAP, AIML_ALIAS_MAP,
    SKILL_NORMALIZATION
)
from cleaning.cleaner import JobCleaner
from typing import Optional 
logger = logging.getLogger(__name__)


class SkillExtractor:
    """
    Extracts and normalizes technology skills from job description text
    using spaCy PhraseMatcher with a predefined skill taxonomy.
    """

    def __init__(self):
        self.nlp = self._load_spacy_model()
        self.matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        self._build_matchers()
        self._all_skills = self._collect_all_skills()
        logger.info("SkillExtractor initialized with "
                    f"{len(self._all_skills)} tracked skills")

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _load_spacy_model(self):
        """Load spaCy model; fall back to blank English if not installed."""
        try:
            return spacy.load(NLP_CONFIG.spacy_model)
        except OSError:
            logger.warning(
                f"spaCy model '{NLP_CONFIG.spacy_model}' not found. "
                "Run: python -m spacy download en_core_web_sm"
            )
            return spacy.blank("en")

    def _build_matchers(self):
        """
        Register all skills from the taxonomy with the PhraseMatcher.
        Multi-word skills like "github actions" are handled natively.
        """
        # Combine DevOps and AI/ML skill lists
        all_skill_lists = {}
        for category, skills in {**DEVOPS_SKILLS, **AIML_SKILLS}.items():
            all_skill_lists[category] = skills

        for category, skills in all_skill_lists.items():
            patterns = [self.nlp.make_doc(skill) for skill in skills]
            self.matcher.add(category, patterns)

    def _collect_all_skills(self) -> Set[str]:
        """Flatten taxonomy into a set of all known skill strings."""
        skills = set()
        for skill_list in DEVOPS_SKILLS.values():
            skills.update(skill_list)
        for skill_list in AIML_SKILLS.values():
            skills.update(skill_list)
        return skills

    # ── Main Extraction Interface ─────────────────────────────────────────────

    def extract(self, text: str) -> List[str]:
        """
        Extract normalized skills from job description text.
        Returns a deduplicated, sorted list of skill strings.
        """
        if not text:
            return []

        found: Set[str] = set()

        # Stage 1: PhraseMatcher on full text (catches known multi-word skills)
        doc = self.nlp(text[:100_000])   # spaCy has a max doc size
        matches = self.matcher(doc)
        for match_id, start, end in matches:
            skill_text = doc[start:end].text.lower()
            normalized = JobCleaner.normalize_skill(skill_text)
            found.add(normalized)

        # Stage 2: Regex patterns for common formats the matcher might miss
        found.update(self._regex_extract(text))

        # Stage 3: Alias normalization pass
        found = {SKILL_NORMALIZATION.get(s, s) for s in found}

        return sorted(found)

    def extract_with_categories(self, text: str) -> Dict[str, List[str]]:
        """
        Extract skills AND group them by taxonomy category.
        
        Returns:
            {
                "orchestration": ["kubernetes", "helm"],
                "ml_frameworks": ["pytorch", "scikit-learn"],
                ...
            }
        """
        skills = self.extract(text)
        categorized: Dict[str, List[str]] = {}

        for skill in skills:
            category = (
                DEVOPS_ALIAS_MAP.get(skill)
                or AIML_ALIAS_MAP.get(skill)
                or "other"
            )
            categorized.setdefault(category, []).append(skill)

        return categorized

    # ── Regex Extraction ──────────────────────────────────────────────────────

    def _regex_extract(self, text: str) -> Set[str]:
        """
        Catch skills that the PhraseMatcher misses due to:
        - Version numbers: "Python 3.10", "Kubernetes 1.27+"
        - Casing variations: "PyTorch", "TensorFlow"  
        - Slash notation: "CI/CD"
        """
        found = set()
        text_lower = text.lower()

        # Version-suffixed skills: extract base name only
        version_pattern = re.compile(r"\b([a-z][a-z0-9\-\.]{1,20})\s*(?:v?\d+[\.\d]*\+?)\b")
        for match in version_pattern.finditer(text_lower):
            candidate = match.group(1)
            if candidate in self._all_skills:
                found.add(candidate)

        # Slash notation: "CI/CD", "ML/AI", "k8s/helm"
        slash_pattern = re.compile(r"\b([a-z][a-z0-9]+)/([a-z][a-z0-9]+)\b")
        for match in slash_pattern.finditer(text_lower):
            for part in [match.group(1), match.group(2)]:
                if part in self._all_skills:
                    found.add(part)

        # Common abbreviations that spaCy may not catch as tokens
        abbreviations = {
            r"\bk8s\b": "kubernetes",
            r"\bgh\s+actions\b": "github actions",
            r"\baws\s+eks\b": "kubernetes",
            r"\bci/cd\b": "ci_cd",
            r"\bml\b": "machine learning",
            r"\bai\b": "ai",
            r"\bnlp\b": "nlp",
            r"\bcv\b": "computer vision",
        }
        for pattern, canonical in abbreviations.items():
            if re.search(pattern, text_lower):
                found.add(canonical)

        return found

    # ── Experience Level from Description ─────────────────────────────────────

    @staticmethod
    def extract_years_of_experience(text: str) -> Optional[int]:
        """
        Extract the explicitly required years of experience, if stated.
        Returns the minimum mentioned, or None if not found.
        
        Examples:
          "5+ years of experience"     → 5
          "3-5 years experience"       → 3
          "minimum 2 years required"   → 2
        """
        pattern = re.compile(
            r"(\d+)\+?\s*(?:-\s*\d+)?\s*years?\s*(?:of\s*)?(?:experience|exp)",
            re.IGNORECASE
        )
        matches = pattern.findall(text)
        if matches:
            return min(int(m) for m in matches)
        return None


# ── Optional: Import guard for Optional type hint ─────────────────────────────
  # noqa: E402 (placed here to keep code readable)
