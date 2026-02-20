# api/main.py
"""
FastAPI backend for the Job Intelligence Agent UI.
Exposes REST endpoints for search, recommendations, and insights.

Run with:
    uvicorn api.main:app --reload --port 8000
"""

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import logging

from agent.searcher import JobSearcher
from agent.recommender import JobRecommender, UserProfile
from agent.insights import InsightsEngine
from storage.postgres_store import PostgresStore

logger = logging.getLogger(__name__)

app = FastAPI(title="Job Intelligence Agent API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singletons — loaded once at startup
searcher = JobSearcher()
recommender = JobRecommender()
pg_store = PostgresStore()


# ── Request / Response Models ─────────────────────────────────────────────────

class RecommendRequest(BaseModel):
    skills: List[str] = []
    preferred_category: Optional[str] = None
    preferred_work_mode: Optional[str] = None
    experience_level: Optional[str] = None
    target_salary_min: Optional[float] = None
    role_description: Optional[str] = None
    top_k: int = 10


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/api/search")
def search_jobs(
    q: str = Query(..., description="Search query"),
    category: Optional[str] = None,
    work_mode: Optional[str] = None,
    experience_level: Optional[str] = None,
    salary_min: Optional[float] = None,
    top_k: int = 20,
):
    """Semantic + structured hybrid job search."""
    results = searcher.search(
        q,
        filters={
            "category": category,
            "work_mode": work_mode,
            "experience_level": experience_level,
            "salary_min": salary_min,
        },
        top_k=top_k,
    )
    return {"results": results, "count": len(results)}


@app.post("/api/recommend")
def recommend_jobs(req: RecommendRequest):
    """Personalized job recommendations based on user profile."""
    profile = UserProfile(
        skills=req.skills,
        preferred_category=req.preferred_category,
        preferred_work_mode=req.preferred_work_mode,
        experience_level=req.experience_level,
        target_salary_min=req.target_salary_min,
        role_description=req.role_description,
    )
    results = recommender.recommend(profile, top_k=req.top_k)
    return {"results": results, "count": len(results)}


@app.get("/api/insights")
def get_insights():
    """Get market insights and analytics."""
    engine = InsightsEngine()
    insights = engine.generate_full_report()
    return {
        "total_jobs": insights.total_jobs,
        "total_devops": insights.total_devops,
        "total_aiml": insights.total_aiml,
        "top_devops_skills": insights.top_devops_skills[:15],
        "top_aiml_skills": insights.top_aiml_skills[:15],
        "work_mode_distribution": insights.work_mode_distribution,
        "experience_distribution": insights.experience_distribution,
        "salary_by_category": insights.salary_by_category,
        "top_hiring_companies": insights.top_hiring_companies[:10],
        "skill_co_occurrences": [
            {"skill_a": a, "skill_b": b, "count": c}
            for a, b, c in insights.skill_co_occurrences[:8]
        ],
    }


@app.get("/api/jobs/{job_id}")
def get_job(job_id: int):
    """Get a single job by ID."""
    job = pg_store.get_job_by_id(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/api/stats")
def get_stats():
    """Quick stats for the dashboard header."""
    return {
        "total": pg_store.count_jobs(),
        "devops": pg_store.count_jobs(category="DevOps"),
        "aiml": pg_store.count_jobs(category="AI/ML"),
        "work_modes": pg_store.get_work_mode_distribution(),
    }


@app.get("/health")
def health():
    return {"status": "ok"}