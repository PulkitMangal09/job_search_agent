# agent/searcher.py
"""
Unified search interface combining:
  1. Semantic search (FAISS vector similarity)
  2. Structured filtering (PostgreSQL queries)
  3. Hybrid re-ranking (combine semantic score + recency + match quality)

This is the primary interface for user-facing job search queries.

Usage:
    searcher = JobSearcher()
    
    # Natural language query
    results = searcher.search("remote senior ML engineer pytorch kubernetes")
    
    # With explicit filters
    results = searcher.search(
        "machine learning infrastructure",
        filters={"work_mode": "remote", "experience_level": "senior"},
        top_k=10,
    )
    
    # Skills-focused structured query
    results = searcher.search_by_skills(["kubernetes", "terraform", "aws"])
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta

from agent.embedder import JobEmbedder
from storage.vector_store import VectorStore
from storage.postgres_store import PostgresStore

logger = logging.getLogger(__name__)


class JobSearcher:
    """
    Hybrid search engine: combines vector similarity with PostgreSQL filters.
    """

    def __init__(self):
        self.embedder = JobEmbedder()
        self.vector_store = VectorStore()
        self.pg_store = PostgresStore()

    # ── Primary Search Interface ──────────────────────────────────────────────

    def search(
        self,
        query: str,
        *,
        filters: Optional[Dict] = None,
        top_k: int = 20,
        semantic_weight: float = 0.7,    # Balance between semantic and structured
        min_score: float = 0.25,
    ) -> List[Dict]:
        """
        Hybrid search combining semantic similarity with structured filters.
        
        Algorithm:
          1. Embed the query
          2. Find top-K * 3 candidates via FAISS (semantic)
          3. Apply PostgreSQL filters to narrow candidates
          4. Re-rank by combined score
          5. Return top-K results
        
        Args:
            query: Natural language query, e.g. "remote senior DevOps AWS"
            filters: Dict with optional keys: category, work_mode,
                     experience_level, location, salary_min, salary_max
            top_k: Number of results to return
            semantic_weight: 0=pure structured, 1=pure semantic
            min_score: Minimum score threshold for results
        
        Returns:
            List of job dicts sorted by relevance score
        """
        filters = filters or {}

        # Step 1: Semantic search — retrieve candidate pool
        query_embedding = self.embedder.embed_query(query, filters)
        semantic_results = self.vector_store.search(
            query_embedding,
            top_k=top_k * 3,   # Over-retrieve for post-filtering
            score_threshold=min_score * 0.5,
        )

        if not semantic_results:
            logger.info("No semantic results; falling back to structured-only search")
            return self.pg_store.search_jobs(
                **{k: v for k, v in filters.items() if v},
                limit=top_k,
            )

        candidate_ids = [job_id for job_id, _ in semantic_results]
        semantic_scores = {job_id: score for job_id, score in semantic_results}

        # Step 2: Fetch full job records from PostgreSQL
        candidates = self._fetch_by_ids(candidate_ids)

        # Step 3: Apply structured filters
        if filters:
            candidates = self._apply_filters(candidates, filters)

        # Step 4: Re-rank with hybrid score
        for job in candidates:
            sem_score = semantic_scores.get(job["id"], 0.0)
            struct_score = self._compute_structured_score(job, query, filters)
            job["_score"] = semantic_weight * sem_score + (1 - semantic_weight) * struct_score
            job["_semantic_score"] = sem_score

        # Step 5: Sort and trim
        candidates.sort(key=lambda j: j["_score"], reverse=True)
        results = [j for j in candidates if j["_score"] >= min_score][:top_k]

        logger.info(f"Search '{query[:50]}' → {len(results)} results")
        return results

    def search_by_skills(
        self,
        skills: List[str],
        category: Optional[str] = None,
        work_mode: Optional[str] = None,
        top_k: int = 20,
    ) -> List[Dict]:
        """
        Structured search for jobs requiring specific skills.
        Skills are AND-matched (job must have ALL listed skills).
        """
        return self.pg_store.search_jobs(
            skills=skills,
            category=category,
            work_mode=work_mode,
            limit=top_k,
        )

    def find_similar(self, job_id: int, top_k: int = 5) -> List[Dict]:
        """
        Find jobs similar to a given job ("More like this").
        Uses FAISS vector similarity on the source job's embedding.
        """
        similar = self.vector_store.search_by_ids(job_id, top_k=top_k)
        if not similar:
            return []
        ids = [jid for jid, _ in similar]
        jobs = self._fetch_by_ids(ids)
        score_map = dict(similar)
        for job in jobs:
            job["_score"] = score_map.get(job["id"], 0.0)
        return sorted(jobs, key=lambda j: j["_score"], reverse=True)

    # ── Helper Methods ────────────────────────────────────────────────────────

    def _fetch_by_ids(self, job_ids: List[int]) -> List[Dict]:
        """Batch-fetch job records by ID from PostgreSQL."""
        jobs = []
        for jid in job_ids:
            job = self.pg_store.get_job_by_id(jid)
            if job:
                jobs.append(job)
        return jobs

    def _apply_filters(self, jobs: List[Dict], filters: Dict) -> List[Dict]:
        """Apply structured filters to an in-memory list of job dicts."""
        filtered = jobs

        if "category" in filters and filters["category"]:
            filtered = [j for j in filtered
                        if j.get("category") == filters["category"]]

        if "work_mode" in filters and filters["work_mode"]:
            filtered = [j for j in filtered
                        if j.get("work_mode") == filters["work_mode"]]

        if "experience_level" in filters and filters["experience_level"]:
            filtered = [j for j in filtered
                        if j.get("experience_level") == filters["experience_level"]]

        if "location" in filters and filters["location"]:
            loc = filters["location"].lower()
            filtered = [j for j in filtered
                        if loc in (j.get("location") or "").lower()]

        if "salary_min" in filters and filters["salary_min"]:
            filtered = [j for j in filtered
                        if (j.get("salary_max") or 0) >= filters["salary_min"]]

        return filtered

    def _compute_structured_score(self, job: Dict, query: str, filters: Dict) -> float:
        """
        Compute a 0-1 structured relevance score based on:
        - Keyword presence in title and description
        - Filter field matches (exact boosts)
        - Recency (fresher = higher)
        """
        score = 0.0
        query_lower = query.lower()
        title_lower = (job.get("title") or "").lower()
        desc_lower = (job.get("description") or "").lower()[:1000]

        # Keyword matches in title (high weight) and description (low weight)
        query_words = set(query_lower.split())
        title_hits = sum(1 for w in query_words if w in title_lower)
        desc_hits = sum(1 for w in query_words if w in desc_lower)
        score += min(title_hits * 0.2, 0.6)    # Cap title contribution at 0.6
        score += min(desc_hits * 0.05, 0.2)    # Cap desc contribution at 0.2

        # Filter match boosts
        if filters.get("work_mode") and job.get("work_mode") == filters["work_mode"]:
            score += 0.1
        if filters.get("experience_level") and job.get("experience_level") == filters["experience_level"]:
            score += 0.1

        # Recency boost (jobs from last 7 days get +0.1)
        scraped_at = job.get("scraped_at")
        if scraped_at:
            try:
                age = (datetime.utcnow() - datetime.fromisoformat(scraped_at)).days
                if age <= 7:
                    score += 0.1
            except (ValueError, TypeError):
                pass

        return min(score, 1.0)


# ── CLI Demo ──────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Job search CLI")
    parser.add_argument("--query", "-q", required=True, help="Search query")
    parser.add_argument("--category", "-c", choices=["DevOps", "AI/ML"])
    parser.add_argument("--mode", "-m", choices=["remote", "hybrid", "onsite"])
    parser.add_argument("--level", "-l", choices=["junior", "mid", "senior"])
    parser.add_argument("--top", "-n", type=int, default=10)
    args = parser.parse_args()

    searcher = JobSearcher()
    results = searcher.search(
        args.query,
        filters={
            "category": args.category,
            "work_mode": args.mode,
            "experience_level": args.level,
        },
        top_k=args.top,
    )

    print(f"\n{'='*65}")
    print(f"Results for: '{args.query}'  ({len(results)} found)")
    print(f"{'='*65}\n")

    for i, job in enumerate(results, 1):
        score = job.get("_score", 0)
        salary = ""
        if job.get("salary_min"):
            salary = f" | ${job['salary_min']:,.0f}–${job.get('salary_max', job['salary_min']):,.0f}"

        print(f"{i:2}. [{score:.2f}] {job['title']}")
        print(f"     {job['company']}   {job['location']}   {job['work_mode']}{salary}")
        if job.get("skills"):
            print(f"     Skills: {', '.join(job['skills'][:8])}")
        if job.get("source_url"):
            print(f"     Apply : {job['source_url']}")
        print()