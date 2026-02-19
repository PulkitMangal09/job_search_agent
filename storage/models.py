# storage/models.py
"""
SQLAlchemy ORM models for structured job data storage.
PostgreSQL is the primary store for all structured queries.
"""

from datetime import datetime
from typing import List, Optional

from sqlalchemy import (
    create_engine, Column, String, Float, Integer,
    DateTime, Boolean, Text, ARRAY, Index
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base, sessionmaker

from config.settings import DB

Base = declarative_base()


class Job(Base):
    """
    Main job listing table. Stores all enriched job data.
    Indexed for common query patterns: skills, category, location, salary.
    """
    __tablename__ = "jobs"

    # Identity
    id = Column(Integer, primary_key=True, autoincrement=True)
    fingerprint = Column(String(32), unique=True, nullable=False, index=True)
    source = Column(String(200), nullable=False)          # "indeed", "linkedin"
    source_url = Column(Text, nullable=True)
    job_id_external = Column(String(200), nullable=True)  # Source's own ID

    # Core fields
    title = Column(String(300), nullable=False)
    company = Column(String(200), nullable=False)
    location = Column(String(200), nullable=True)
    description = Column(Text, nullable=True)

    # Classification (populated by NLP pipeline)
    category = Column(String(50), nullable=True, index=True)   # "DevOps" | "AI/ML" | "Other"
    categories = Column(ARRAY(String), nullable=True)          # May have multiple
    experience_level = Column(String(20), nullable=True, index=True)  # junior|mid|senior
    work_mode = Column(String(20), nullable=True, index=True)         # remote|hybrid|onsite

    # Skills (list of normalized skill strings)
    skills = Column(ARRAY(String), nullable=True)
    skills_by_category = Column(JSONB, nullable=True)    # {category: [skills]}

    # Salary
    salary_min = Column(Float, nullable=True)
    salary_max = Column(Float, nullable=True)
    salary_currency = Column(String(5), default="USD")

    # Embeddings (stored as JSON array; use pgvector for production)
    embedding_id = Column(Integer, nullable=True)   # Reference to FAISS index

    # Metadata
    posted_date = Column(DateTime, nullable=True)
    scraped_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)

    # Composite indexes for common query patterns
    __table_args__ = (
        Index("idx_jobs_skills", "skills",
              postgresql_using="gin"),           # GIN only on the array column
        Index("idx_jobs_category", "category"),  # Regular btree on category
        Index("idx_jobs_location_mode", "location", "work_mode"),
        Index("idx_jobs_salary_range", "salary_min", "salary_max"),
        Index("idx_jobs_company", "company"),
    )

    def __repr__(self) -> str:
        return f"<Job id={self.id} title='{self.title}' company='{self.company}'>"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "company": self.company,
            "location": self.location,
            "category": self.category,
            "experience_level": self.experience_level,
            "work_mode": self.work_mode,
            "skills": self.skills or [],
            "salary_min": self.salary_min,
            "salary_max": self.salary_max,
            "salary_currency": self.salary_currency,
            "source": self.source,
            "source_url": self.source_url,
            "scraped_at": self.scraped_at.isoformat() if self.scraped_at else None,
        }


class SkillTrend(Base):
    """
    Aggregated skill frequency data, refreshed periodically.
    Used for the insights/analytics dashboard.
    """
    __tablename__ = "skill_trends"

    id = Column(Integer, primary_key=True)
    skill = Column(String(100), nullable=False, index=True)
    category = Column(String(50), nullable=False)       # "DevOps" | "AI/ML"
    taxonomy_category = Column(String(100), nullable=True)  # e.g., "orchestration"
    job_count = Column(Integer, default=0)              # Jobs requiring this skill
    avg_salary_min = Column(Float, nullable=True)
    avg_salary_max = Column(Float, nullable=True)
    remote_job_count = Column(Integer, default=0)
    period_start = Column(DateTime, nullable=False)     # Aggregation window start
    period_end = Column(DateTime, nullable=False)
    computed_at = Column(DateTime, default=datetime.utcnow)


# ── Database Setup Utilities ──────────────────────────────────────────────────

def get_engine(db_url: str = None):
    """Create SQLAlchemy engine with connection pooling."""
    url = db_url or DB.url
    return create_engine(
        url,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,    # Reconnect on stale connections
        echo=False,
    )


def get_session_factory(engine=None):
    """Return a session factory for the given engine."""
    if engine is None:
        engine = get_engine()
    return sessionmaker(bind=engine, expire_on_commit=False)


def create_tables(engine=None):
    """Create all tables (idempotent — safe to call multiple times)."""
    if engine is None:
        engine = get_engine()
    Base.metadata.create_all(engine)
    print("✓ Database tables created/verified")
