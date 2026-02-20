# pipelines/full_pipeline.py
"""
Full pipeline orchestrator: Scrape → Clean → NLP → Store → Index.

This script ties all pipeline stages together and can be run directly
for a full data refresh or imported by the scheduler for periodic runs.

Execution order:
  1. Scrape raw jobs from configured sources
  2. Clean and deduplicate raw jobs
  3. Extract skills using NLP
  4. Classify job categories (DevOps / AI/ML)
  5. Upsert into PostgreSQL
  6. Generate embeddings and update FAISS index
  7. (Optional) Refresh Elasticsearch index

Usage:
    python pipelines/full_pipeline.py
    python pipelines/full_pipeline.py --sources indeed --limit 100
"""

import logging
import argparse
import time
from typing import List, Generator

import numpy as np

# Pipeline stages
from scraping.base_scraper import RawJob
from scraping.indeed_scraper import IndeedScraper
from scraping.linkedin_scraper import LinkedInScraper
from scraping.jsearch_scraper import JSearchScraper
from scraping.greenhouse_scraper import GreenhouseScraper
from scraping.lever_scraper import LeverScraper
from scraping.workday_scraper import WorkdayScraper

from cleaning.cleaner import JobCleaner, CleanJob
from nlp.skill_extractor import SkillExtractor
from nlp.categorizer import JobCategorizer

from storage.models import create_tables
from storage.postgres_store import PostgresStore
from storage.vector_store import VectorStore
from agent.embedder import JobEmbedder

from config.settings import TARGET_CATEGORIES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")

# ── Search terms for DevOps and AI/ML jobs ────────────────────────────────────
DEVOPS_SEARCH_TERMS = [
    "DevOps Engineer",
    "Site Reliability Engineer",
    "Platform Engineer",
    "Kubernetes Engineer",
    "Cloud Infrastructure Engineer",
    "MLOps Engineer",
]

AIML_SEARCH_TERMS = [
    "Machine Learning Engineer",
    "AI Engineer",
    "Data Scientist",
    "NLP Engineer",
    "Computer Vision Engineer",
    "Applied Scientist",
    "ML Research Engineer",
]

ALL_SEARCH_TERMS = DEVOPS_SEARCH_TERMS + AIML_SEARCH_TERMS


class JobPipeline:
    """
    Orchestrates the complete job intelligence pipeline.
    Modular design: each stage can be run independently.
    """

    def __init__(
        self,
        sources: List[str] = None,
        location: str = "United States",
        batch_size: int = 50,
    ):
        self.sources = sources or ["indeed"]
        self.location = location
        self.batch_size = batch_size

        # Initialize all components once (avoids repeated model loading)
        logger.info("Initializing pipeline components...")
        self.cleaner = JobCleaner()
        self.skill_extractor = SkillExtractor()
        self.categorizer = JobCategorizer(use_zero_shot=False)   # Skip zero-shot for speed
        self.pg_store = PostgresStore()
        self.vector_store = VectorStore()
        self.embedder = JobEmbedder()
        logger.info("Pipeline ready.")

    # ── Stage 1: Scraping ─────────────────────────────────────────────────────

    def stage_scrape(self) -> Generator[RawJob, None, None]:
        """
        Stage 1: Scrape raw jobs from all configured sources.
        Yields RawJob objects as they are scraped (streaming).
        """
        logger.info(f"=== STAGE 1: SCRAPING ({self.sources}) ===")
        scrapers = self._build_scrapers()

        for scraper in scrapers:
            logger.info(f"Starting {scraper.source_name} scraper...")
            for job in scraper.scrape():
                yield job

    def _build_scrapers(self):
        """Instantiate scrapers for all configured sources."""
        # Keywords used by ATS scrapers to filter relevant jobs
        devops_aiml_keywords = [
            "devops", "mlops", "machine learning", "deep learning",
            "kubernetes", "platform engineer", "site reliability",
            "data scientist", "ml engineer", "ai engineer",
            "infrastructure", "cloud engineer", "nlp", "llm",
        ]

        scrapers = []
        if "indeed" in self.sources:
            scrapers.append(IndeedScraper(ALL_SEARCH_TERMS, self.location))
        if "linkedin" in self.sources:
            scrapers.append(LinkedInScraper(ALL_SEARCH_TERMS, self.location))
        if "jsearch" in self.sources:
            scrapers.append(JSearchScraper(ALL_SEARCH_TERMS, self.location))
        if "greenhouse" in self.sources:
            scrapers.append(GreenhouseScraper(keywords=devops_aiml_keywords))
        if "lever" in self.sources:
            scrapers.append(LeverScraper(keywords=devops_aiml_keywords))
        if "workday" in self.sources:
            scrapers.append(WorkdayScraper(keywords=devops_aiml_keywords))
        return scrapers

    # ── Stage 2: Cleaning ─────────────────────────────────────────────────────

    def stage_clean(self, raw_jobs: List[RawJob]) -> List[CleanJob]:
        """
        Stage 2: Clean and deduplicate raw jobs.
        """
        logger.info(f"=== STAGE 2: CLEANING ({len(raw_jobs)} raw jobs) ===")
        return self.cleaner.clean(raw_jobs)

    # ── Stage 3: NLP Enrichment ───────────────────────────────────────────────

    def stage_enrich(self, clean_jobs: List[CleanJob]) -> List[dict]:
        """
        Stage 3: NLP skill extraction and job categorization.
        Returns list of job dicts ready for database storage.
        """
        logger.info(f"=== STAGE 3: NLP ENRICHMENT ({len(clean_jobs)} jobs) ===")
        enriched = []

        for i, job in enumerate(clean_jobs):
            try:
                # Extract skills from description
                skills = self.skill_extractor.extract(job.description)
                skills_by_category = self.skill_extractor.extract_with_categories(job.description)
                years = self.skill_extractor.extract_years_of_experience(job.description)

                # Classify job category
                result = self.categorizer.classify(
                    title=job.title,
                    skills=skills,
                    description=job.description,
                )

                # Refine experience level if years were explicitly mentioned
                exp_level = job.experience_level
                if years is not None and exp_level == "unknown":
                    if years <= 2:
                        exp_level = "junior"
                    elif years <= 5:
                        exp_level = "mid"
                    else:
                        exp_level = "senior"

                # Build the storage dict
                job_dict = job.to_dict()
                job_dict.update({
                    "skills": skills,
                    "skills_by_category": skills_by_category,
                    "category": result.primary_category,
                    "categories": result.categories,
                    "experience_level": exp_level,
                })
                enriched.append(job_dict)

                if (i + 1) % 50 == 0:
                    logger.info(f"Enriched {i+1}/{len(clean_jobs)} jobs")

            except Exception as e:
                logger.warning(f"NLP enrichment failed for '{job.title}': {e}")

        # Only keep jobs that match our target categories
        target_enriched = [
            j for j in enriched
            if j.get("category") in TARGET_CATEGORIES
        ]
        logger.info(
            f"NLP complete: {len(enriched)} enriched, "
            f"{len(target_enriched)} in target categories"
        )
        return target_enriched

    # ── Stage 4: Database Storage ─────────────────────────────────────────────

    def stage_store(self, enriched_jobs: List[dict]) -> List[int]:
        """
        Stage 4: Upsert enriched jobs into PostgreSQL.
        Returns list of new job IDs.
        """
        logger.info(f"=== STAGE 4: DATABASE STORAGE ({len(enriched_jobs)} jobs) ===")
        inserted = self.pg_store.upsert_jobs(enriched_jobs)
        logger.info(f"Stored {inserted} new jobs in PostgreSQL")
        return inserted

    # ── Stage 5: Vector Indexing ──────────────────────────────────────────────

    def stage_index(self, enriched_jobs: List[dict]):
        """
        Stage 5: Generate embeddings and add to FAISS vector index.
        Processes in batches to manage memory.
        """
        logger.info(f"=== STAGE 5: VECTOR INDEXING ({len(enriched_jobs)} jobs) ===")

        if not enriched_jobs:
            logger.info("No jobs to index.")
            return

        # Process in batches
        for batch_start in range(0, len(enriched_jobs), self.batch_size):
            batch = enriched_jobs[batch_start:batch_start + self.batch_size]

            # Only index jobs that have a database ID
            indexed_batch = [j for j in batch if j.get("id") or j.get("fingerprint")]

            # Generate embeddings
            embeddings = self.embedder.embed_jobs(indexed_batch)

            # Get corresponding job IDs (use fingerprint hash as proxy if no DB ID yet)
            job_ids = []
            for job in indexed_batch:
                if job.get("id"):
                    job_ids.append(job["id"])
                else:
                    # Fetch the DB-assigned ID using fingerprint
                    stored = self._get_id_by_fingerprint(job.get("fingerprint", ""))
                    job_ids.append(stored or 0)

            # Filter out jobs without IDs
            valid = [(emb, jid) for emb, jid in zip(embeddings, job_ids) if jid > 0]
            if valid:
                valid_embs = np.array([e for e, _ in valid], dtype=np.float32)
                valid_ids = [jid for _, jid in valid]
                self.vector_store.add(valid_embs, valid_ids)

            logger.info(
                f"Indexed batch {batch_start//self.batch_size + 1}: "
                f"{len(valid)} vectors added"
            )

        # Persist index to disk
        self.vector_store.save()
        logger.info(f"FAISS index updated: {len(self.vector_store)} total vectors")

    def _get_id_by_fingerprint(self, fingerprint: str) -> int:
        """Fetch the PostgreSQL ID for a job by fingerprint."""
        session = self.pg_store.get_session()
        try:
            from storage.models import Job
            job = session.query(Job).filter_by(fingerprint=fingerprint).first()
            return job.id if job else 0
        finally:
            session.close()

    # ── Full Pipeline Run ─────────────────────────────────────────────────────

    def run(self):
        """
        Execute the complete pipeline end-to-end.
        Reports timing for each stage.
        """
        start = time.time()
        logger.info("=" * 60)
        logger.info("🚀 JOB INTELLIGENCE PIPELINE STARTING")
        logger.info("=" * 60)

        # Ensure DB tables exist
        create_tables()

        # Stage 1: Scrape (stream → collect)
        t0 = time.time()
        raw_jobs = list(self.stage_scrape())
        logger.info(f"Stage 1 done: {len(raw_jobs)} raw jobs ({time.time()-t0:.0f}s)")

        if not raw_jobs:
            logger.warning("No raw jobs collected. Exiting pipeline.")
            return

        # Stage 2: Clean
        t0 = time.time()
        clean_jobs = self.stage_clean(raw_jobs)
        logger.info(f"Stage 2 done: {len(clean_jobs)} clean jobs ({time.time()-t0:.0f}s)")

        # Stage 3: NLP Enrichment
        t0 = time.time()
        enriched = self.stage_enrich(clean_jobs)
        logger.info(f"Stage 3 done: {len(enriched)} enriched jobs ({time.time()-t0:.0f}s)")

        # Stage 4: Store
        t0 = time.time()
        self.stage_store(enriched)
        logger.info(f"Stage 4 done ({time.time()-t0:.0f}s)")

        # Stage 5: Index
        t0 = time.time()
        self.stage_index(enriched)
        logger.info(f"Stage 5 done ({time.time()-t0:.0f}s)")

        total = time.time() - start
        logger.info("=" * 60)
        logger.info(f"✅ PIPELINE COMPLETE in {total:.0f}s")
        logger.info(f"   {len(raw_jobs)} raw → {len(clean_jobs)} clean → {len(enriched)} enriched")
        logger.info("=" * 60)


# ── CLI Entry Point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Job Intelligence Pipeline")
    parser.add_argument(
        "--sources", nargs="+", default=["greenhouse", "lever"],
        choices=["indeed", "linkedin", "jsearch", "greenhouse", "lever", "workday"],
        help="Job sources to scrape",
    )
    parser.add_argument("--location", default="United States")
    parser.add_argument("--batch-size", type=int, default=50)
    args = parser.parse_args()

    pipeline = JobPipeline(
        sources=args.sources,
        location=args.location,
        batch_size=args.batch_size,
    )
    pipeline.run()