# DevOps & AI/ML Job Intelligence Agent

A complete end-to-end pipeline to collect, process, analyze, and semantically search job listings in **DevOps** and **AI/ML** fields.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        JOB AGENT PIPELINE                        │
│                                                                   │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌─────────────┐  │
│  │ SCRAPING │──▶│ CLEANING │──▶│   NLP    │──▶│   STORAGE   │  │
│  │          │   │          │   │          │   │             │  │
│  │ LinkedIn │   │Normalize │   │ Skills   │   │ PostgreSQL  │  │
│  │ Indeed   │   │Dedupe    │   │ Category │   │ Elasticsearch│ │
│  │ Glassdoor│   │Std skills│   │ Exp level│   │ FAISS       │  │
│  │ GH Jobs  │   │          │   │ Salary   │   │             │  │
│  └──────────┘   └──────────┘   └──────────┘   └─────────────┘  │
│                                                        │         │
│  ┌────────────────────────────────────────────────────▼──────┐  │
│  │                        AI AGENT                            │  │
│  │   Semantic Search │ Job Recommendations │ Insights/Reports │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                   │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │         AUTOMATION (APScheduler / Celery + Redis)          │   │
│  └───────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
job_agent/
├── config/
│   ├── settings.py          # Central config (DB URLs, API keys, etc.)
│   └── taxonomy.py          # Skill taxonomy & job category definitions
├── scraping/
│   ├── base_scraper.py      # Abstract base class for all scrapers
│   ├── indeed_scraper.py    # Indeed job scraper
│   ├── linkedin_scraper.py  # LinkedIn job scraper
│   ├── glassdoor_scraper.py # Glassdoor scraper
│   └── github_scraper.py    # GitHub Jobs API (deprecated but documented)
├── cleaning/
│   └── cleaner.py           # Text normalization, deduplication, skill standardization
├── nlp/
│   ├── skill_extractor.py   # spaCy + HuggingFace skill extraction
│   ├── categorizer.py       # DevOps vs AI/ML classification
│   └── salary_estimator.py  # Salary extraction and estimation
├── storage/
│   ├── models.py            # SQLAlchemy ORM models
│   ├── postgres_store.py    # PostgreSQL insert/query operations
│   ├── elasticsearch_store.py # Elasticsearch indexing & search
│   └── vector_store.py      # FAISS vector index for semantic search
├── agent/
│   ├── embedder.py          # Sentence embedding generation
│   ├── recommender.py       # Job recommendation engine
│   ├── searcher.py          # Unified search interface
│   └── insights.py          # Analytics & report generation
├── pipelines/
│   ├── full_pipeline.py     # Orchestrates the entire flow
│   └── scheduler.py         # APScheduler for periodic runs
├── tests/
│   └── test_pipeline.py     # Unit tests
├── requirements.txt
└── docker-compose.yml
```

## Quick Start

```bash
# 1. Clone and install
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. Start infrastructure
docker-compose up -d   # PostgreSQL + Elasticsearch + Redis

# 3. Configure
cp config/settings.py.example config/settings.py
# Fill in your API keys

# 4. Run full pipeline
python pipelines/full_pipeline.py

# 5. Query the agent
python agent/searcher.py --query "senior kubernetes engineer remote"
```
