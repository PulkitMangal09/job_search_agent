# nlp/categorizer.py
"""
Classifies jobs into DevOps or AI/ML categories (or both, or Other).

Three-layer classification (most to least specific):
  1. Title matching — fastest, highest confidence
  2. Skill indicator matching — checks extracted skills against known sets
  3. Zero-shot classification — HuggingFace BART for ambiguous cases

The zero-shot model is loaded lazily to avoid startup overhead.
"""

import logging
import re
from typing import List, Optional
from dataclasses import dataclass

from config.taxonomy import (
    DEVOPS_TITLE_PATTERNS, AIML_TITLE_PATTERNS,
    DEVOPS_INDICATOR_SKILLS, AIML_INDICATOR_SKILLS,
)
from config.settings import NLP as NLP_CONFIG

logger = logging.getLogger(__name__)

# Zero-shot label candidates
CANDIDATE_LABELS = ["DevOps", "AI/ML", "Data Science", "Backend Engineering", "Other"]


@dataclass
class ClassificationResult:
    """Result of a job category classification."""
    primary_category: str         # "DevOps" | "AI/ML" | "Other"
    categories: List[str]         # May include both if job spans both fields
    confidence: float             # 0-1 confidence score
    method: str                   # "title" | "skills" | "zero_shot"


class JobCategorizer:
    """
    Classifies job listings into DevOps, AI/ML, or Other categories.
    Designed to be extended with additional categories.
    """

    def __init__(self, use_zero_shot: bool = True):
        self.use_zero_shot = use_zero_shot
        self._classifier = None   # Loaded lazily on first use

    # ── Main Interface ────────────────────────────────────────────────────────

    def classify(
        self,
        title: str,
        skills: List[str],
        description: str = "",
    ) -> ClassificationResult:
        """
        Classify a job using a tiered approach.
        
        Args:
            title: Job title
            skills: Extracted skill list from SkillExtractor
            description: Full job description (used for zero-shot fallback)
        
        Returns:
            ClassificationResult with category, confidence, and method used
        """
        # Tier 1: Title matching (fast, deterministic)
        result = self._classify_by_title(title)
        if result.confidence >= 0.9:
            return result

        # Tier 2: Skill indicator matching
        result = self._classify_by_skills(skills)
        if result.confidence >= 0.7:
            return result

        # Tier 3: Zero-shot classification on description (slowest, most flexible)
        if self.use_zero_shot and description:
            return self._classify_by_zero_shot(title, description)

        # Default: return best tier-2 result even if low confidence
        return result

    # ── Tier 1: Title Matching ────────────────────────────────────────────────

    def _classify_by_title(self, title: str) -> ClassificationResult:
        """Match title against known job title patterns."""
        title_lower = title.lower()

        is_devops = any(p in title_lower for p in DEVOPS_TITLE_PATTERNS)
        is_aiml = any(p in title_lower for p in AIML_TITLE_PATTERNS)

        if is_devops and is_aiml:
            return ClassificationResult(
                primary_category="DevOps",
                categories=["DevOps", "AI/ML"],
                confidence=0.85,
                method="title"
            )
        if is_devops:
            return ClassificationResult(
                primary_category="DevOps",
                categories=["DevOps"],
                confidence=0.95,
                method="title"
            )
        if is_aiml:
            return ClassificationResult(
                primary_category="AI/ML",
                categories=["AI/ML"],
                confidence=0.95,
                method="title"
            )
        return ClassificationResult(
            primary_category="Other",
            categories=[],
            confidence=0.4,
            method="title"
        )

    # ── Tier 2: Skill Indicator Matching ─────────────────────────────────────

    def _classify_by_skills(self, skills: List[str]) -> ClassificationResult:
        """
        Count how many extracted skills are DevOps or AI/ML indicators.
        The category with the most matches wins.
        """
        skill_set = set(skills)

        devops_matches = len(skill_set & DEVOPS_INDICATOR_SKILLS)
        aiml_matches = len(skill_set & AIML_INDICATOR_SKILLS)

        total = devops_matches + aiml_matches
        if total == 0:
            return ClassificationResult(
                primary_category="Other",
                categories=[],
                confidence=0.3,
                method="skills"
            )

        # Normalize confidence by match count
        devops_conf = devops_matches / max(total, 1)
        aiml_conf = aiml_matches / max(total, 1)

        categories = []
        if devops_matches > 0:
            categories.append("DevOps")
        if aiml_matches > 0:
            categories.append("AI/ML")

        if devops_conf > aiml_conf:
            primary = "DevOps"
            confidence = min(0.5 + devops_matches * 0.1, 0.9)
        elif aiml_conf > devops_conf:
            primary = "AI/ML"
            confidence = min(0.5 + aiml_matches * 0.1, 0.9)
        else:
            primary = "DevOps"   # Tie-break to DevOps
            confidence = 0.55

        return ClassificationResult(
            primary_category=primary,
            categories=categories,
            confidence=confidence,
            method="skills"
        )

    # ── Tier 3: Zero-Shot Classification ─────────────────────────────────────

    def _classify_by_zero_shot(self, title: str, description: str) -> ClassificationResult:
        """
        Use HuggingFace zero-shot classification for ambiguous jobs.
        The model is loaded lazily to avoid startup overhead.
        
        Model: facebook/bart-large-mnli (~1.5GB)
        Slower but handles novel job titles gracefully.
        """
        if self._classifier is None:
            self._load_classifier()

        if self._classifier is None:
            return ClassificationResult(
                primary_category="Other",
                categories=[],
                confidence=0.0,
                method="zero_shot_failed"
            )

        # Use title + first 500 chars of description as input
        input_text = f"{title}. {description[:500]}"
        
        try:
            result = self._classifier(
                input_text,
                candidate_labels=CANDIDATE_LABELS,
                multi_label=True,   # A job can be both DevOps and AI/ML
            )
        except Exception as e:
            logger.error(f"Zero-shot classification failed: {e}")
            return ClassificationResult(
                primary_category="Other", categories=[], confidence=0.0, method="zero_shot_error"
            )

        # Build category list for scores above threshold
        categories = []
        scores = dict(zip(result["labels"], result["scores"]))
        threshold = 0.4

        for label in ["DevOps", "AI/ML"]:
            if scores.get(label, 0) >= threshold:
                categories.append(label)

        primary = result["labels"][0] if result["labels"] else "Other"
        if primary not in ["DevOps", "AI/ML"]:
            primary = "Other"

        return ClassificationResult(
            primary_category=primary,
            categories=categories or ["Other"],
            confidence=result["scores"][0],
            method="zero_shot"
        )

    def _load_classifier(self):
        """Lazy load the HuggingFace zero-shot pipeline."""
        try:
            from transformers import pipeline
            logger.info(f"Loading zero-shot model: {NLP_CONFIG.classifier_model}")
            self._classifier = pipeline(
                "zero-shot-classification",
                model=NLP_CONFIG.classifier_model,
                device=-1,    # CPU; use device=0 for GPU
            )
        except Exception as e:
            logger.error(f"Failed to load zero-shot model: {e}")
            self._classifier = None
