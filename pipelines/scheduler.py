# pipelines/scheduler.py
"""
Periodic pipeline scheduler using APScheduler.

Runs the full scraping pipeline on a configurable schedule to keep
job data fresh. Designed to run as a long-lived process or Docker service.

Schedules:
  - Full pipeline: Every 12 hours (scrape + clean + NLP + store + index)
  - Insights refresh: Every 6 hours (fast, no scraping)

Usage:
    python pipelines/scheduler.py                    # Run scheduler daemon
    python pipelines/scheduler.py --run-now          # Run immediately then schedule
    python pipelines/scheduler.py --once             # Run once and exit (for cron)
"""

import logging
import argparse
from datetime import datetime

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger
from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED

from pipelines.full_pipeline import JobPipeline
from agent.insights import InsightsEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("scheduler")


# ── Job Functions ─────────────────────────────────────────────────────────────

def run_full_pipeline():
    """Scheduled job: execute the full scrape-to-index pipeline."""
    logger.info(f"⏰ Scheduled pipeline starting at {datetime.utcnow().isoformat()}")
    try:
        pipeline = JobPipeline(sources=["indeed"])
        pipeline.run()
        logger.info("✅ Scheduled pipeline completed successfully")
    except Exception as e:
        logger.error(f"❌ Scheduled pipeline failed: {e}", exc_info=True)


def run_insights_refresh():
    """Scheduled job: refresh the analytics/insights cache."""
    logger.info("📊 Refreshing market insights...")
    try:
        engine = InsightsEngine()
        report = engine.generate_full_report()
        # Save report to file for quick access (e.g., API endpoint caching)
        with open("/tmp/job_market_report.txt", "w") as f:
            f.write(report.to_text())
        logger.info("✅ Insights refresh complete")
    except Exception as e:
        logger.error(f"❌ Insights refresh failed: {e}", exc_info=True)


# ── Event Listeners ───────────────────────────────────────────────────────────

def job_executed_listener(event):
    """Log when a scheduled job completes."""
    logger.info(f"Job '{event.job_id}' executed at {datetime.utcnow().isoformat()}")


def job_error_listener(event):
    """Alert on scheduled job failure."""
    logger.error(
        f"Job '{event.job_id}' FAILED at {datetime.utcnow().isoformat()}: "
        f"{event.exception}"
    )
    # In production: send Slack/PagerDuty alert here


# ── Scheduler Setup ───────────────────────────────────────────────────────────

def build_scheduler() -> BlockingScheduler:
    """Build and configure the APScheduler instance."""
    scheduler = BlockingScheduler(timezone="UTC")

    # Full pipeline: every 12 hours at :00
    scheduler.add_job(
        run_full_pipeline,
        trigger=CronTrigger(hour="*/12", minute=0),    # 00:00 and 12:00 UTC
        id="full_pipeline",
        name="Full Job Intelligence Pipeline",
        max_instances=1,           # Prevent overlapping runs
        coalesce=True,             # Run once if multiple triggers missed
        misfire_grace_time=300,    # Allow 5 min late start
    )

    # Insights refresh: every 6 hours
    scheduler.add_job(
        run_insights_refresh,
        trigger=CronTrigger(hour="*/6", minute=30),
        id="insights_refresh",
        name="Market Insights Refresh",
        max_instances=1,
        coalesce=True,
    )

    # Register event listeners
    scheduler.add_listener(job_executed_listener, EVENT_JOB_EXECUTED)
    scheduler.add_listener(job_error_listener, EVENT_JOB_ERROR)

    return scheduler


def main():
    parser = argparse.ArgumentParser(description="Job Intelligence Pipeline Scheduler")
    parser.add_argument(
        "--run-now", action="store_true",
        help="Run full pipeline immediately before starting scheduler"
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Run pipeline once and exit (use with cron instead of APScheduler)"
    )
    args = parser.parse_args()

    if args.once:
        logger.info("Running pipeline once (--once mode)...")
        run_full_pipeline()
        return

    if args.run_now:
        logger.info("Running pipeline immediately before scheduling...")
        run_full_pipeline()

    scheduler = build_scheduler()
    logger.info("=" * 50)
    logger.info("🤖 Job Intelligence Scheduler Started")
    logger.info("   Full pipeline:    Every 12h at 00:00 and 12:00 UTC")
    logger.info("   Insights refresh: Every 6h at HH:30 UTC")
    logger.info("   Press Ctrl+C to stop")
    logger.info("=" * 50)

    try:
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user.")


if __name__ == "__main__":
    main()
