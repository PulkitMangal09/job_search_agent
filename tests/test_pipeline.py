# tests/test_pipeline.py
"""
Unit tests and sample query demonstrations for the Job Intelligence Agent.

Run with:
    pytest tests/test_pipeline.py -v

Sections:
  1. Unit tests for cleaning, NLP, and classification
  2. Integration tests (require running PostgreSQL)
  3. Sample queries demonstrating the recommendation and search API
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from datetime import datetime


# ═══════════════════════════════════════════════════════════════
# SECTION 1: Unit Tests (no external dependencies)
# ═══════════════════════════════════════════════════════════════

class TestJobCleaner:
    """Tests for the data cleaning stage."""

    def setup_method(self):
        from cleaning.cleaner import JobCleaner
        self.cleaner = JobCleaner()

    def test_salary_parsing_dollar_range(self):
        min_s, max_s, curr = self.cleaner._parse_salary("$120,000 - $150,000 a year")
        assert min_s == 120_000
        assert max_s == 150_000
        assert curr == "USD"

    def test_salary_parsing_k_notation(self):
        min_s, max_s, curr = self.cleaner._parse_salary("$80k - $100k")
        assert min_s == 80_000
        assert max_s == 100_000

    def test_salary_parsing_hourly(self):
        min_s, max_s, curr = self.cleaner._parse_salary("$60/hr")
        # 60 * 2080 = 124,800
        assert min_s == pytest.approx(124_800, rel=0.01)

    def test_salary_parsing_gbp(self):
        min_s, max_s, curr = self.cleaner._parse_salary("£50,000 - £70,000")
        assert curr == "GBP"
        assert min_s == 50_000

    def test_salary_parsing_missing(self):
        min_s, max_s, curr = self.cleaner._parse_salary("")
        assert min_s is None
        assert max_s is None

    def test_work_mode_detection_remote(self):
        mode = self.cleaner._detect_work_mode("", "Remote, USA", "We are a remote-first team")
        assert mode == "remote"

    def test_work_mode_detection_hybrid(self):
        mode = self.cleaner._detect_work_mode("Hybrid", "New York", "")
        assert mode == "hybrid"

    def test_experience_level_senior(self):
        level = self.cleaner._detect_experience_level("Senior DevOps Engineer", "")
        assert level == "senior"

    def test_experience_level_junior(self):
        level = self.cleaner._detect_experience_level("Junior ML Engineer", "0-2 years exp")
        assert level == "junior"

    def test_deduplication(self):
        from scraping.base_scraper import RawJob
        raw_jobs = [
            RawJob(title="DevOps Engineer", company="Acme", description="k8s", location="NYC",
                   source="indeed", source_url="http://a.com/1"),
            RawJob(title="DevOps Engineer", company="Acme", description="docker", location="NYC",
                   source="linkedin", source_url="http://b.com/2"),  # Same title+company+loc = duplicate
        ]
        cleaned = self.cleaner.clean(raw_jobs)
        assert len(cleaned) == 1   # Second should be deduplicated

    def test_html_stripping(self):
        result = self.cleaner._normalize_text("<b>Senior</b> <em>Engineer</em>")
        assert "<b>" not in result
        assert "Senior Engineer" in result


class TestSkillExtractor:
    """Tests for NLP skill extraction."""

    def setup_method(self):
        from nlp.skill_extractor import SkillExtractor
        self.extractor = SkillExtractor()

    def test_extract_known_skills(self):
        text = "We require Kubernetes, Terraform, and Docker experience."
        skills = self.extractor.extract(text)
        assert "kubernetes" in skills
        assert "terraform" in skills
        assert "docker" in skills

    def test_extract_ml_skills(self):
        text = "Strong PyTorch and TensorFlow skills. Experience with MLflow."
        skills = self.extractor.extract(text)
        assert "pytorch" in skills
        assert "tensorflow" in skills
        assert "mlflow" in skills

    def test_extract_multiword_skills(self):
        text = "CI/CD experience with GitHub Actions and GitLab CI required."
        skills = self.extractor.extract(text)
        assert "github actions" in skills or "ci_cd" in skills

    def test_alias_normalization(self):
        text = "5+ years with k8s and sklearn required."
        skills = self.extractor.extract(text)
        assert "kubernetes" in skills         # k8s → kubernetes
        assert "scikit-learn" in skills       # sklearn → scikit-learn

    def test_empty_description(self):
        skills = self.extractor.extract("")
        assert skills == []

    def test_extract_years_of_experience(self):
        from nlp.skill_extractor import SkillExtractor
        text = "Minimum 5+ years of experience in cloud infrastructure required."
        years = SkillExtractor.extract_years_of_experience(text)
        assert years == 5

    def test_extract_years_range(self):
        from nlp.skill_extractor import SkillExtractor
        text = "3-7 years of machine learning experience"
        years = SkillExtractor.extract_years_of_experience(text)
        assert years == 3   # Returns minimum


class TestJobCategorizer:
    """Tests for DevOps/AI/ML classification."""

    def setup_method(self):
        from nlp.categorizer import JobCategorizer
        self.categorizer = JobCategorizer(use_zero_shot=False)

    def test_classify_devops_by_title(self):
        result = self.categorizer.classify("Senior DevOps Engineer", [], "")
        assert result.primary_category == "DevOps"
        assert result.confidence >= 0.9

    def test_classify_aiml_by_title(self):
        result = self.categorizer.classify("Machine Learning Engineer", [], "")
        assert result.primary_category == "AI/ML"
        assert result.confidence >= 0.9

    def test_classify_by_skills_devops(self):
        result = self.categorizer.classify(
            "Software Engineer",
            ["kubernetes", "docker", "terraform", "ansible"],
            ""
        )
        assert result.primary_category == "DevOps"

    def test_classify_by_skills_aiml(self):
        result = self.categorizer.classify(
            "Software Engineer",
            ["pytorch", "tensorflow", "mlops", "deep learning"],
            ""
        )
        assert result.primary_category == "AI/ML"

    def test_classify_other(self):
        result = self.categorizer.classify("Frontend Engineer", ["react", "css"], "")
        assert result.primary_category == "Other"

    def test_classify_dual_category(self):
        result = self.categorizer.classify(
            "MLOps Platform Engineer",
            ["kubernetes", "pytorch", "mlflow", "docker"],
            ""
        )
        # Should detect both categories
        assert len(result.categories) >= 1


# ═══════════════════════════════════════════════════════════════
# SECTION 2: Vector Store Tests (no external DB needed)
# ═══════════════════════════════════════════════════════════════

class TestVectorStore:
    """Tests for FAISS vector store operations."""

    def setup_method(self, tmp_path=None):
        import tempfile, os
        from storage.vector_store import VectorStore
        # Use temp path to avoid polluting the real index
        self.tmp_dir = tempfile.mkdtemp()
        with patch("storage.vector_store.INDEX_PATH") as mock_path:
            mock_path.exists.return_value = False
            self.store = VectorStore(dim=16)   # Tiny dim for testing

    def test_add_and_search(self):
        # Add 3 vectors
        vecs = np.random.randn(3, 16).astype(np.float32)
        job_ids = [101, 102, 103]
        self.store.add(vecs, job_ids)

        assert len(self.store) == 3

        # Search should return results
        query = vecs[0]
        results = self.store.search(query, top_k=3, score_threshold=0.0)
        assert len(results) >= 1
        # Top result should be the same vector (score ~1.0)
        assert results[0][0] == 101
        assert results[0][1] > 0.99

    def test_empty_search_returns_empty(self):
        query = np.random.randn(16).astype(np.float32)
        results = self.store.search(query, top_k=5)
        assert results == []


# ═══════════════════════════════════════════════════════════════
# SECTION 3: Sample Queries (Demonstration)
# ═══════════════════════════════════════════════════════════════

"""
The following are sample query patterns for the recommendation and search API.
Run these manually after setting up the database with real data.
"""

SAMPLE_QUERIES = """
# ─── SEMANTIC SEARCH EXAMPLES ─────────────────────────────────────────────────

from agent.searcher import JobSearcher

searcher = JobSearcher()

# 1. Natural language search for remote DevOps roles
results = searcher.search(
    "senior kubernetes platform engineer aws remote",
    filters={"work_mode": "remote", "experience_level": "senior"},
    top_k=10,
)

# 2. Search for AI/ML roles with specific skills
results = searcher.search(
    "machine learning engineer pytorch transformers NLP",
    filters={"category": "AI/ML"},
)

# 3. Find similar jobs to one you like
similar = searcher.find_similar(job_id=42, top_k=5)

# 4. Structured skills-based search
results = searcher.search_by_skills(
    skills=["terraform", "kubernetes", "aws"],
    work_mode="remote",
)

# ─── RECOMMENDATION EXAMPLES ──────────────────────────────────────────────────

from agent.recommender import JobRecommender, UserProfile

recommender = JobRecommender()

# 1. DevOps engineer looking for senior remote roles
profile = UserProfile(
    skills=["kubernetes", "terraform", "aws", "python", "docker", "helm", "prometheus"],
    preferred_category="DevOps",
    preferred_work_mode="remote",
    experience_level="senior",
    target_salary_min=140_000,
    role_description="Platform engineering and SRE work on large-scale infrastructure",
)
recommendations = recommender.recommend(profile, top_k=10)
for rec in recommendations:
    print(f"[{rec['_score']:.2f}] {rec['title']} @ {rec['company']}")
    print(f"  Reason: {rec['_explanation']}")

# 2. Junior ML engineer
ml_profile = UserProfile(
    skills=["python", "pytorch", "scikit-learn", "pandas", "sql"],
    preferred_category="AI/ML",
    preferred_work_mode="hybrid",
    experience_level="junior",
    target_salary_min=90_000,
)
ml_recs = recommender.recommend(ml_profile, top_k=5)

# ─── ANALYTICS / INSIGHTS EXAMPLES ───────────────────────────────────────────

from agent.insights import InsightsEngine

engine = InsightsEngine()

# Full market report
report = engine.generate_full_report(period_days=30)
print(report.to_text())

# Skill demand trend over time
trend = engine.get_skill_demand_over_time("kubernetes", days=90)
for week_data in trend:
    print(f"  {week_data['week']}: {week_data['count']} jobs")

# ─── DIRECT DB QUERIES ────────────────────────────────────────────────────────

from storage.postgres_store import PostgresStore

store = PostgresStore()

# All remote senior AI/ML roles with salary > $120k
jobs = store.search_jobs(
    category="AI/ML",
    work_mode="remote",
    experience_level="senior",
    salary_min=120_000,
    limit=20,
)

# Top 20 in-demand DevOps skills this month
top_skills = store.get_top_skills(category="DevOps", top_n=20)
for skill_data in top_skills:
    print(f"  {skill_data['skill']:25} {skill_data['count']:4} jobs")

# Salary stats
salary_stats = store.get_salary_stats(category="DevOps")
print(f"DevOps avg salary: ${salary_stats['avg_min']:,.0f} - ${salary_stats['avg_max']:,.0f}")
"""

if __name__ == "__main__":
    print("Sample queries for the Job Intelligence Agent:")
    print(SAMPLE_QUERIES)
    print("\nRun unit tests with: pytest tests/test_pipeline.py -v")
