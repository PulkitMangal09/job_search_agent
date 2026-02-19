# agent/insights.py
"""
Analytics and insights engine for the job market data.

Generates:
  1. Top in-demand skills by category (DevOps vs AI/ML)
  2. Salary trends by skill, level, and location
  3. Remote vs onsite distribution
  4. Experience level distribution
  5. Company hiring activity
  6. Skill co-occurrence analysis (skills that appear together)

Usage:
    insights = InsightsEngine()
    report = insights.generate_full_report()
    print(report.to_text())
"""

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta

from storage.postgres_store import PostgresStore

logger = logging.getLogger(__name__)


@dataclass
class MarketInsights:
    """Container for all computed market insights."""
    generated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    period_days: int = 30

    # Skill demand
    top_devops_skills: List[Dict] = field(default_factory=list)
    top_aiml_skills: List[Dict] = field(default_factory=list)
    skill_co_occurrences: List[Tuple] = field(default_factory=list)

    # Salary data
    salary_by_category: Dict = field(default_factory=dict)
    salary_by_level: Dict = field(default_factory=dict)

    # Distribution
    work_mode_distribution: Dict = field(default_factory=dict)
    experience_distribution: Dict = field(default_factory=dict)
    category_distribution: Dict = field(default_factory=dict)

    # Top companies
    top_hiring_companies: List[Dict] = field(default_factory=list)

    # Job volume
    total_jobs: int = 0
    total_devops: int = 0
    total_aiml: int = 0

    def to_text(self) -> str:
        """Format insights as a human-readable text report."""
        lines = [
            "=" * 65,
            " 🤖 DevOps & AI/ML Job Market Intelligence Report",
            f" Generated: {self.generated_at[:10]} | Period: Last {self.period_days} days",
            "=" * 65,
            "",
            f"📊 OVERVIEW",
            f"  Total Active Jobs : {self.total_jobs:,}",
            f"  DevOps Jobs       : {self.total_devops:,}",
            f"  AI/ML Jobs        : {self.total_aiml:,}",
            "",
        ]

        # Work Mode
        lines.append("🌍 WORK MODE DISTRIBUTION")
        for mode, count in self.work_mode_distribution.items():
            pct = (count / max(self.total_jobs, 1)) * 100
            bar = "█" * int(pct / 3)
            lines.append(f"  {mode:10} {bar:20} {count:5,} ({pct:.0f}%)")
        lines.append("")

        # Experience Distribution
        lines.append("👥 EXPERIENCE LEVEL DISTRIBUTION")
        for level, count in self.experience_distribution.items():
            pct = (count / max(self.total_jobs, 1)) * 100
            lines.append(f"  {level:10} {count:5,} ({pct:.0f}%)")
        lines.append("")

        # Top DevOps Skills
        lines.append("🔧 TOP DEVOPS SKILLS (by demand)")
        for i, skill_data in enumerate(self.top_devops_skills[:15], 1):
            lines.append(f"  {i:2}. {skill_data['skill']:25} {skill_data['count']:4,} jobs")
        lines.append("")

        # Top AI/ML Skills
        lines.append("🧠 TOP AI/ML SKILLS (by demand)")
        for i, skill_data in enumerate(self.top_aiml_skills[:15], 1):
            lines.append(f"  {i:2}. {skill_data['skill']:25} {skill_data['count']:4,} jobs")
        lines.append("")

        # Salary Data
        lines.append("💰 SALARY INSIGHTS (USD/year, annualized)")
        for category, stats in self.salary_by_category.items():
            if stats.get("count", 0) > 0:
                lines.append(
                    f"  {category:10} Avg: ${stats.get('avg_min', 0):>8,.0f}–"
                    f"${stats.get('avg_max', 0):>8,.0f} "
                    f"(n={stats['count']})"
                )
        lines.append("")

        # Salary by level
        if self.salary_by_level:
            lines.append("💰 SALARY BY EXPERIENCE LEVEL")
            for level, stats in self.salary_by_level.items():
                lines.append(
                    f"  {level:10} Avg: ${stats.get('avg_min', 0):>8,.0f}–"
                    f"${stats.get('avg_max', 0):>8,.0f}"
                )
            lines.append("")

        # Top Companies
        if self.top_hiring_companies:
            lines.append("🏢 TOP HIRING COMPANIES")
            for i, co in enumerate(self.top_hiring_companies[:10], 1):
                lines.append(f"  {i:2}. {co['company']:30} {co['count']:3,} open roles")
            lines.append("")

        # Skill Co-occurrences
        if self.skill_co_occurrences:
            lines.append("🔗 COMMON SKILL COMBINATIONS")
            for skill_a, skill_b, count in self.skill_co_occurrences[:8]:
                lines.append(f"  {skill_a} + {skill_b}: {count} jobs")

        lines.append("")
        lines.append("=" * 65)
        return "\n".join(lines)


class InsightsEngine:
    """Computes and formats job market insights from stored data."""

    def __init__(self):
        self.store = PostgresStore()

    def generate_full_report(self, period_days: int = 30) -> MarketInsights:
        """
        Generate a complete market insights report from the last N days.
        Runs all analytics queries and assembles the result.
        """
        logger.info(f"Generating market insights report (last {period_days} days)...")

        insights = MarketInsights(period_days=period_days)

        # Job volume counts
        insights.total_jobs = self.store.count_jobs()
        insights.total_devops = self.store.count_jobs(category="DevOps")
        insights.total_aiml = self.store.count_jobs(category="AI/ML")

        # Top skills by category
        insights.top_devops_skills = self.store.get_top_skills(
            category="DevOps", top_n=20, days=period_days
        )
        insights.top_aiml_skills = self.store.get_top_skills(
            category="AI/ML", top_n=20, days=period_days
        )

        # Work mode distribution
        insights.work_mode_distribution = self.store.get_work_mode_distribution()

        # Experience level distribution
        insights.experience_distribution = self._get_level_distribution()

        # Salary by category
        for cat in ["DevOps", "AI/ML"]:
            insights.salary_by_category[cat] = self.store.get_salary_stats(category=cat)

        # Salary by experience level
        for level in ["junior", "mid", "senior"]:
            stats = self._get_salary_by_level(level)
            if stats:
                insights.salary_by_level[level] = stats

        # Top hiring companies
        insights.top_hiring_companies = self._get_top_companies()

        # Skill co-occurrences (computed in Python for flexibility)
        insights.skill_co_occurrences = self._compute_skill_cooccurrences(top_n=10)

        logger.info("Market insights report generated successfully")
        return insights

    # ── Analytics Helpers ─────────────────────────────────────────────────────

    def _get_level_distribution(self) -> Dict[str, int]:
        """Count jobs by experience level."""
        session = self.store.get_session()
        try:
            from storage.models import Job
            from sqlalchemy import func
            result = session.query(
                Job.experience_level, func.count(Job.id)
            ).filter(Job.is_active == True).group_by(Job.experience_level).all()
            return {level: count for level, count in result if level}
        finally:
            session.close()

    def _get_salary_by_level(self, level: str) -> Optional[Dict]:
        """Compute salary stats for a specific experience level."""
        session = self.store.get_session()
        try:
            from storage.models import Job
            from sqlalchemy import func
            row = session.query(
                func.avg(Job.salary_min).label("avg_min"),
                func.avg(Job.salary_max).label("avg_max"),
                func.count(Job.id).label("count"),
            ).filter(
                Job.experience_level == level,
                Job.salary_min.isnot(None),
                Job.is_active == True,
            ).first()
            if row and row.count > 0:
                return {"avg_min": float(row.avg_min or 0), "avg_max": float(row.avg_max or 0)}
            return None
        finally:
            session.close()

    def _get_top_companies(self, top_n: int = 15) -> List[Dict]:
        """Find companies with the most active job listings."""
        session = self.store.get_session()
        try:
            from storage.models import Job
            from sqlalchemy import func
            result = session.query(
                Job.company, func.count(Job.id).label("count")
            ).filter(Job.is_active == True).group_by(Job.company)\
             .order_by(func.count(Job.id).desc()).limit(top_n).all()
            return [{"company": c, "count": n} for c, n in result]
        finally:
            session.close()

    def _compute_skill_cooccurrences(self, top_n: int = 10) -> List[Tuple]:
        """
        Find pairs of skills that frequently appear together.
        Computed in Python from stored jobs (not in SQL for portability).
        """
        session = self.store.get_session()
        try:
            from storage.models import Job
            jobs = session.query(Job.skills).filter(
                Job.skills.isnot(None), Job.is_active == True
            ).limit(5000).all()   # Sample for performance

            pair_counts: Counter = Counter()
            for (skills,) in jobs:
                if not skills or len(skills) < 2:
                    continue
                skill_list = sorted(set(skills))[:10]   # Cap per-job skills
                for i, s1 in enumerate(skill_list):
                    for s2 in skill_list[i + 1:]:
                        pair_counts[(s1, s2)] += 1

            return [
                (s1, s2, count)
                for (s1, s2), count in pair_counts.most_common(top_n)
            ]
        finally:
            session.close()

    def get_skill_demand_over_time(
        self, skill: str, days: int = 90
    ) -> List[Dict]:
        """
        Return weekly job count for a specific skill over the past N days.
        Useful for tracking skill growth/decline trends.
        """
        session = self.store.get_session()
        try:
            from storage.models import Job
            from sqlalchemy import func, cast, Date
            results = session.execute(f"""
                SELECT 
                    date_trunc('week', scraped_at) AS week,
                    COUNT(*) AS count
                FROM jobs, LATERAL unnest(skills) AS skill
                WHERE skill = '{skill}'
                  AND scraped_at >= NOW() - INTERVAL '{days} days'
                  AND is_active = true
                GROUP BY week
                ORDER BY week
            """)
            return [{"week": str(row[0]), "count": row[1]} for row in results]
        finally:
            session.close()
