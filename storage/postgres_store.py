# storage/postgres_store.py
"""
PostgreSQL storage layer for job listings.

Provides:
- Bulk upsert (insert-or-update on fingerprint)
- Flexible search queries (by skills, location, salary, category)
- Skill aggregation queries for the insights engine
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_, cast
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy import String

from storage.models import Job, SkillTrend, get_engine, get_session_factory, create_tables
from cleaning.cleaner import CleanJob

logger = logging.getLogger(__name__)


class PostgresStore:
    """
    PostgreSQL persistence layer.
    Use as a context manager or call close() when done.
    """

    def __init__(self, db_url: str = None):
        self.engine = get_engine(db_url)
        self.SessionFactory = get_session_factory(self.engine)
        create_tables(self.engine)

    def __enter__(self):
        self._session = self.SessionFactory()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self._session.rollback()
        else:
            self._session.commit()
        self._session.close()

    def get_session(self) -> Session:
        return self.SessionFactory()

    # ── Write Operations ──────────────────────────────────────────────────────

    def upsert_jobs(self, jobs: List[dict]) -> int:
        """
        Bulk upsert job records. Uses fingerprint as the conflict key.
        Returns count of new jobs inserted.
        
        Args:
            jobs: List of dicts with Job model fields (use job.to_dict())
        """
        if not jobs:
            return 0

        session = self.get_session()
        inserted = 0
        updated = 0

        try:
            for job_data in jobs:
                fingerprint = job_data.get("fingerprint")
                if not fingerprint:
                    continue

                existing = session.query(Job).filter_by(fingerprint=fingerprint).first()

                if existing:
                    # Update mutable fields (skills may have been re-extracted)
                    for field in ["skills", "category", "experience_level",
                                  "salary_min", "salary_max", "embedding_id"]:
                        if field in job_data and job_data[field] is not None:
                            setattr(existing, field, job_data[field])
                    existing.updated_at = datetime.utcnow()
                    updated += 1
                else:
                    job = Job(**{k: v for k, v in job_data.items()
                                 if hasattr(Job, k)})
                    session.add(job)
                    inserted += 1

            session.commit()
            logger.info(f"DB upsert complete: {inserted} new, {updated} updated")
            return inserted

        except Exception as e:
            session.rollback()
            logger.error(f"DB upsert failed: {e}")
            raise
        finally:
            session.close()

    # ── Read / Search Operations ──────────────────────────────────────────────

    def search_jobs(
        self,
        *,
        skills: Optional[List[str]] = None,
        category: Optional[str] = None,
        location: Optional[str] = None,
        work_mode: Optional[str] = None,
        experience_level: Optional[str] = None,
        salary_min: Optional[float] = None,
        salary_max: Optional[float] = None,
        company: Optional[str] = None,
        days_since_scraped: int = 30,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict]:
        """
        Flexible job search with multiple filter dimensions.
        All filters are combined with AND logic.
        
        Example:
            results = store.search_jobs(
                skills=["kubernetes", "terraform"],
                category="DevOps",
                work_mode="remote",
                salary_min=120000,
            )
        """
        session = self.get_session()
        try:
            query = session.query(Job).filter(Job.is_active == True)

            # Filter by age
            cutoff = datetime.utcnow() - timedelta(days=days_since_scraped)
            query = query.filter(Job.scraped_at >= cutoff)

            # Category filter
            if category:
                query = query.filter(Job.category == category)

            # Work mode filter
            if work_mode:
                query = query.filter(Job.work_mode == work_mode)

            # Experience level filter
            if experience_level:
                query = query.filter(Job.experience_level == experience_level)

            # Location filter (case-insensitive partial match)
            if location:
                query = query.filter(Job.location.ilike(f"%{location}%"))

            # Company filter
            if company:
                query = query.filter(Job.company.ilike(f"%{company}%"))

            # Salary filters
            if salary_min:
                query = query.filter(
                    or_(Job.salary_min >= salary_min, Job.salary_max >= salary_min)
                )
            if salary_max:
                query = query.filter(
                    or_(Job.salary_max <= salary_max, Job.salary_min <= salary_max)
                )

            # Skills filter — job must have ALL specified skills (PostgreSQL array contains)
            if skills:
                for skill in skills:
                    query = query.filter(
                        Job.skills.contains(cast([skill], ARRAY(String)))
                    )

            # Execute with pagination
            results = query.order_by(Job.scraped_at.desc()).offset(offset).limit(limit).all()
            return [job.to_dict() for job in results]

        finally:
            session.close()

    def get_job_by_id(self, job_id: int) -> Optional[Dict]:
        """Retrieve a single job by primary key."""
        session = self.get_session()
        try:
            job = session.query(Job).filter_by(id=job_id).first()
            return job.to_dict() if job else None
        finally:
            session.close()

    # ── Analytics Queries ─────────────────────────────────────────────────────

    def get_top_skills(
        self,
        category: Optional[str] = None,
        top_n: int = 20,
        days: int = 30,
    ) -> List[Dict]:
        """
        Return the most frequently required skills in recent job listings.
        Uses PostgreSQL's unnest() to flatten the skills arrays.
        
        Returns: [{"skill": "kubernetes", "count": 342}, ...]
        """
        session = self.get_session()
        try:
            cutoff = datetime.utcnow() - timedelta(days=days)

            # Raw SQL for unnest performance (SQLAlchemy ORM doesn't support this natively)
            category_filter = f"AND category = '{category}'" if category else ""
            sql = f"""
                SELECT skill, COUNT(*) as count
                FROM jobs,
                     LATERAL unnest(skills) AS skill
                WHERE scraped_at >= '{cutoff.isoformat()}'
                  AND is_active = true
                  {category_filter}
                GROUP BY skill
                ORDER BY count DESC
                LIMIT {top_n}
            """
            result = session.execute(sql)
            return [{"skill": row[0], "count": row[1]} for row in result]
        finally:
            session.close()

    def get_salary_stats(
        self, category: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Compute salary statistics for jobs with available salary data.
        Returns: {"avg_min", "avg_max", "median_min", "p25_min", "p75_min"}
        """
        session = self.get_session()
        try:
            query = session.query(
                func.avg(Job.salary_min).label("avg_min"),
                func.avg(Job.salary_max).label("avg_max"),
                func.min(Job.salary_min).label("min_salary"),
                func.max(Job.salary_max).label("max_salary"),
                func.count(Job.id).label("count"),
            ).filter(
                Job.salary_min.isnot(None),
                Job.is_active == True,
            )
            if category:
                query = query.filter(Job.category == category)

            row = query.first()
            return {
                "avg_min": round(row.avg_min or 0, 2),
                "avg_max": round(row.avg_max or 0, 2),
                "min_salary": row.min_salary,
                "max_salary": row.max_salary,
                "count": row.count,
            }
        finally:
            session.close()

    def get_work_mode_distribution(self, category: Optional[str] = None) -> Dict[str, int]:
        """Return counts of remote/hybrid/onsite jobs."""
        session = self.get_session()
        try:
            query = session.query(Job.work_mode, func.count(Job.id))\
                           .filter(Job.is_active == True)\
                           .group_by(Job.work_mode)
            if category:
                query = query.filter(Job.category == category)
            return {mode: count for mode, count in query.all()}
        finally:
            session.close()

    def count_jobs(self, category: Optional[str] = None) -> int:
        """Count total active jobs, optionally filtered by category."""
        session = self.get_session()
        try:
            query = session.query(func.count(Job.id)).filter(Job.is_active == True)
            if category:
                query = query.filter(Job.category == category)
            return query.scalar()
        finally:
            session.close()
