# agent/recommender.py
"""
Personalized job recommendation engine.

Given a user's skill profile and preferences, recommends the most relevant
job listings using a combination of:
  1. Skill overlap scoring (hard match on required skills)
  2. Semantic similarity (embedding-based soft match)
  3. Preference alignment (work mode, location, salary range)

Usage:
    recommender = JobRecommender()
    
    user_profile = UserProfile(
        skills=["kubernetes", "terraform", "aws", "python"],
        preferred_category="DevOps",
        preferred_work_mode="remote",
        target_salary_min=130000,
        experience_level="senior",
    )
    
    recommendations = recommender.recommend(user_profile, top_k=10)
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from agent.embedder import JobEmbedder
from storage.vector_store import VectorStore
from storage.postgres_store import PostgresStore

logger = logging.getLogger(__name__)


@dataclass
class UserProfile:
    """
    Represents a user's professional profile and job search preferences.
    All fields are optional — more fields = better recommendations.
    """
    # Skills the user currently has
    skills: List[str] = field(default_factory=list)

    # Job preferences
    preferred_category: Optional[str] = None       # "DevOps" | "AI/ML"
    preferred_work_mode: Optional[str] = None      # "remote" | "hybrid" | "onsite"
    preferred_location: Optional[str] = None
    target_salary_min: Optional[float] = None
    target_salary_max: Optional[float] = None
    experience_level: Optional[str] = None         # "junior" | "mid" | "senior"

    # Open text description of desired role (used for semantic matching)
    role_description: Optional[str] = None

    def to_query_text(self) -> str:
        """
        Build a descriptive text string from the profile for embedding.
        This is what gets embedded and compared against job embeddings.
        """
        parts = []
        if self.experience_level:
            parts.append(self.experience_level)
        if self.preferred_category:
            parts.append(self.preferred_category)
        if self.skills:
            parts.append(f"skills: {', '.join(self.skills)}")
        if self.preferred_work_mode:
            parts.append(self.preferred_work_mode)
        if self.role_description:
            parts.append(self.role_description)
        return " ".join(parts)


class JobRecommender:
    """
    Multi-signal job recommendation engine.
    Combines semantic embeddings with explicit skill and preference matching.
    """

    def __init__(self):
        self.embedder = JobEmbedder()
        self.vector_store = VectorStore()
        self.pg_store = PostgresStore()

    # ── Primary Recommendation Interface ─────────────────────────────────────

    def recommend(
        self,
        profile: UserProfile,
        top_k: int = 10,
        candidate_pool_multiplier: int = 5,
    ) -> List[Dict]:
        """
        Generate personalized job recommendations for a user profile.
        
        Pipeline:
          1. Embed the user profile into a query vector
          2. Retrieve a large candidate pool via FAISS
          3. Score each candidate with multi-signal ranking
          4. Return top_k ranked results with explanation
        
        Args:
            profile: User's skills and preferences
            top_k: Number of recommendations to return
            candidate_pool_multiplier: How many candidates to retrieve per top_k
        
        Returns:
            List of job dicts with added "_score" and "_explanation" fields
        """
        # Step 1: Embed user profile
        query_text = profile.to_query_text()
        profile_embedding = self.embedder.embed_text(query_text)

        # Step 2: Retrieve candidate pool from FAISS
        candidates_raw = self.vector_store.search(
            profile_embedding,
            top_k=top_k * candidate_pool_multiplier,
            score_threshold=0.2,
        )

        if not candidates_raw:
            logger.warning("No candidates from FAISS. Index may be empty.")
            return self._fallback_recommend(profile, top_k)

        # Step 3: Fetch full job details
        candidate_ids = [jid for jid, _ in candidates_raw]
        semantic_scores = {jid: score for jid, score in candidates_raw}
        jobs = self._fetch_jobs(candidate_ids)

        # Step 4: Multi-signal scoring
        scored_jobs = []
        for job in jobs:
            scores = self._score_job(job, profile, semantic_scores.get(job["id"], 0.0))
            total = (
                0.40 * scores["semantic"] +
                0.35 * scores["skill_overlap"] +
                0.15 * scores["preference"] +
                0.10 * scores["salary"]
            )
            job["_score"] = round(total, 3)
            job["_score_breakdown"] = scores
            job["_explanation"] = self._explain(job, profile, scores)
            scored_jobs.append(job)

        # Step 5: Sort and return top-K
        scored_jobs.sort(key=lambda j: j["_score"], reverse=True)
        recommendations = scored_jobs[:top_k]

        logger.info(
            f"Generated {len(recommendations)} recommendations "
            f"(pool: {len(jobs)}, profile: '{query_text[:60]}')"
        )
        return recommendations

    # ── Scoring Signals ───────────────────────────────────────────────────────

    def _score_job(
        self, job: Dict, profile: UserProfile, semantic_score: float
    ) -> Dict[str, float]:
        """
        Compute individual signal scores for a single job.
        All scores are in [0, 1].
        """
        return {
            "semantic": semantic_score,
            "skill_overlap": self._skill_overlap_score(job, profile),
            "preference": self._preference_score(job, profile),
            "salary": self._salary_score(job, profile),
        }

    def _skill_overlap_score(self, job: Dict, profile: UserProfile) -> float:
        """
        Compute Jaccard-style skill overlap between user skills and job requirements.
        Score = (matching skills) / (total required skills)
        
        We use "coverage" not strict Jaccard to avoid penalizing users who
        have MORE skills than required (which is always good).
        """
        if not profile.skills or not job.get("skills"):
            return 0.0

        user_skills = set(s.lower() for s in profile.skills)
        job_skills = set(s.lower() for s in job["skills"])

        overlap = user_skills & job_skills
        if not job_skills:
            return 0.0

        coverage = len(overlap) / len(job_skills)    # Fraction of requirements met
        return min(coverage, 1.0)

    def _preference_score(self, job: Dict, profile: UserProfile) -> float:
        """
        Score based on how well the job matches stated preferences.
        Each matching preference contributes equally.
        """
        score = 0.0
        total_preferences = 0

        checks = [
            ("preferred_category", "category"),
            ("preferred_work_mode", "work_mode"),
            ("experience_level", "experience_level"),
        ]
        for profile_field, job_field in checks:
            pref = getattr(profile, profile_field, None)
            if pref:
                total_preferences += 1
                if job.get(job_field) == pref:
                    score += 1.0

        if profile.preferred_location and job.get("location"):
            total_preferences += 1
            if profile.preferred_location.lower() in job["location"].lower():
                score += 1.0

        return score / total_preferences if total_preferences > 0 else 0.5

    def _salary_score(self, job: Dict, profile: UserProfile) -> float:
        """
        Score based on salary alignment.
        - No salary data on either side: neutral (0.5)
        - Salary within target range: 1.0
        - Salary below target: scaled down
        - Salary above target: 1.0 (always good)
        """
        if not profile.target_salary_min or not job.get("salary_min"):
            return 0.5   # Neutral when no data

        job_mid = ((job.get("salary_min") or 0) + (job.get("salary_max") or 0)) / 2
        target_min = profile.target_salary_min

        if job_mid >= target_min:
            return 1.0
        # Linear decay below target
        return max(0.0, job_mid / target_min)

    # ── Explanation Generation ────────────────────────────────────────────────

    def _explain(self, job: Dict, profile: UserProfile, scores: Dict) -> str:
        """
        Generate a human-readable explanation of why this job was recommended.
        """
        reasons = []

        if scores["skill_overlap"] >= 0.7:
            matching = set(s.lower() for s in profile.skills) & set(s.lower() for s in (job.get("skills") or []))
            reasons.append(f"Strong skill match ({len(matching)} matching: {', '.join(list(matching)[:4])})")
        elif scores["skill_overlap"] >= 0.4:
            reasons.append(f"Partial skill overlap ({scores['skill_overlap']:.0%} of required skills)")

        if scores["semantic"] >= 0.7:
            reasons.append("Highly relevant role based on your profile")

        if job.get("work_mode") == profile.preferred_work_mode:
            reasons.append(f"{job['work_mode'].capitalize()} position (your preference)")

        if scores["salary"] == 1.0 and job.get("salary_min"):
            reasons.append(f"Salary ${job['salary_min']:,.0f}+ meets your target")

        return "; ".join(reasons) if reasons else "Semantically similar to your profile"

    # ── Fallback ──────────────────────────────────────────────────────────────

    def _fallback_recommend(self, profile: UserProfile, top_k: int) -> List[Dict]:
        """
        Fall back to structured PostgreSQL query when FAISS index is empty.
        """
        return self.pg_store.search_jobs(
            skills=profile.skills[:5] if profile.skills else None,
            category=profile.preferred_category,
            work_mode=profile.preferred_work_mode,
            experience_level=profile.experience_level,
            limit=top_k,
        )

    def _fetch_jobs(self, job_ids: List[int]) -> List[Dict]:
        """Batch-fetch jobs from PostgreSQL by ID list."""
        jobs = []
        for jid in job_ids:
            job = self.pg_store.get_job_by_id(jid)
            if job:
                jobs.append(job)
        return jobs
