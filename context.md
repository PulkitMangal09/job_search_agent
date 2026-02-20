# JobRadar — DevOps & AI/ML Job Intelligence Agent
## Comprehensive Technical Documentation

> **Purpose of this document:** Complete technical reference for the JobRadar project. Pass this entire document to any LLM to give it full context about the project architecture, tech stack, code structure, and design decisions.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Tech Stack](#3-tech-stack)
4. [Project Structure](#4-project-structure)
5. [Module Reference](#5-module-reference)
6. [Data Flow](#6-data-flow)
7. [Database Schema](#7-database-schema)
8. [API Reference](#8-api-reference)
9. [Configuration](#9-configuration)
10. [Skill Taxonomy](#10-skill-taxonomy)
11. [Running the Project](#11-running-the-project)
12. [Known Issues & Fixes Applied](#12-known-issues--fixes-applied)
13. [Deployment](#13-deployment)

---

## 1. Project Overview

JobRadar is an end-to-end AI-powered job intelligence platform focused on **DevOps** and **AI/ML** job markets. It:

- Scrapes job listings from multiple sources (Greenhouse ATS, Lever ATS, Workday, JSearch API)
- Cleans, deduplicates, and normalizes job data
- Extracts skills using NLP (spaCy PhraseMatcher)
- Classifies jobs into DevOps or AI/ML categories
- Stores structured data in PostgreSQL
- Indexes job embeddings in FAISS for semantic search
- Exposes a FastAPI REST backend
- Serves a single-file HTML/CSS/JS frontend (dark terminal aesthetic)
- Supports personalized job recommendations and market insights

**Target users:** Job seekers in DevOps and AI/ML, recruiters, and market analysts.

---

## 2. Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        JOBRADAR PIPELINE                             │
│                                                                       │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────┐   ┌──────────┐ │
│  │   SCRAPING   │──▶│   CLEANING   │──▶│   NLP    │──▶│ STORAGE  │ │
│  │              │   │              │   │          │   │          │ │
│  │ Greenhouse   │   │ Normalize    │   │ Skills   │   │PostgreSQL│ │
│  │ Lever        │   │ Deduplicate  │   │ Category │   │ FAISS    │ │
│  │ Workday      │   │ Salary parse │   │ Exp level│   │          │ │
│  │ JSearch API  │   │ Work mode    │   │          │   │          │ │
│  └──────────────┘   └──────────────┘   └──────────┘   └──────────┘ │
│                                                              │        │
│  ┌───────────────────────────────────────────────────────────▼────┐  │
│  │                         AI AGENT LAYER                          │  │
│  │   Semantic Search  │  Job Recommendations  │  Market Insights   │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │              FASTAPI BACKEND  +  HTML/JS FRONTEND                │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │              AUTOMATION  (APScheduler — every 12h)               │  │
│  └─────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────┘
```

### Search Architecture

```
User Query (text)
      │
      ▼
 Embed query          ← sentence-transformers/all-MiniLM-L6-v2
      │
      ▼
 FAISS ANN search     ← retrieves top-K*3 candidate job IDs
      │
      ▼
 PostgreSQL fetch     ← gets full job records for candidate IDs
      │
      ▼
 Apply filters        ← category, work_mode, experience_level, salary
      │
      ▼
 Re-rank              ← 70% semantic score + 30% structured score
      │                  structured = keyword hits + filter match + recency
      ▼
 Return top-K results
```

### Recommendation Architecture

```
UserProfile (skills, preferences)
      │
      ▼
 Build profile text   ← "senior DevOps kubernetes terraform remote"
      │
      ▼
 Embed profile        ← same embedding model as jobs
      │
      ▼
 FAISS search         ← top-K*5 candidates
      │
      ▼
 Multi-signal scoring:
   40% semantic similarity
   35% skill overlap (Jaccard coverage)
   15% preference match (category, mode, level, location)
   10% salary alignment
      │
      ▼
 Add explanation text
      │
      ▼
 Return ranked recommendations
```

---

## 3. Tech Stack

### Core Language
- **Python 3.12** — all backend, pipeline, and ML code

### Scraping
| Tool | Purpose |
|---|---|
| `requests` | HTTP client for API-based scrapers |
| `beautifulsoup4` | HTML parsing for job descriptions |
| `selenium` | Headless Chrome for LinkedIn (currently disabled) |
| `webdriver-manager` | Auto-manages ChromeDriver versions |
| `lxml` | Fast HTML/XML parser backend for BS4 |

### NLP & ML
| Tool | Purpose |
|---|---|
| `spacy` (en_core_web_sm) | PhraseMatcher for skill extraction from text |
| `sentence-transformers` (all-MiniLM-L6-v2) | 384-dim embeddings for semantic search |
| `transformers` (facebook/bart-large-mnli) | Zero-shot classification (lazy-loaded, optional) |
| `torch` | Backend for sentence-transformers (uses MPS on Apple Silicon) |
| `faiss-cpu` | Vector similarity index for semantic search |
| `numpy` | Vector math, embedding arrays |

### Database & Storage
| Tool | Purpose |
|---|---|
| `PostgreSQL 16` | Primary structured data store |
| `sqlalchemy` | ORM — models, queries, connection pooling |
| `psycopg2-binary` | PostgreSQL driver |
| `FAISS` | Local vector index (persisted as `storage/faiss_index.bin`) |

### API & Backend
| Tool | Purpose |
|---|---|
| `fastapi` | REST API framework |
| `uvicorn` | ASGI server |
| `pydantic` | Request/response validation |

### Frontend
| Tool | Purpose |
|---|---|
| Vanilla HTML/CSS/JS | Single-file UI (`ui/index.html`) — no build step |
| Google Fonts (Space Mono + Syne) | Typography |
| Fetch API | Communicates with FastAPI backend |

### Infrastructure
| Tool | Purpose |
|---|---|
| `docker-compose` | Local PostgreSQL + Elasticsearch + Redis |
| `APScheduler` | Periodic pipeline execution (every 12h) |
| Docker | Container for production deployment |

### External APIs (Optional)
| API | Purpose | Cost |
|---|---|---|
| JSearch (RapidAPI) | Aggregates LinkedIn/Indeed/Glassdoor | Free tier: 200 req/month |
| Greenhouse API | Public ATS — no key needed | Free |
| Lever API | Public ATS — no key needed | Free |
| Workday API | Enterprise ATS — no key needed | Free |

---

## 4. Project Structure

```
job_agent/
│
├── config/
│   ├── __init__.py
│   ├── settings.py          # All config (DB, scraping, NLP, API keys)
│   └── taxonomy.py          # Skill taxonomy for DevOps & AI/ML
│
├── scraping/
│   ├── __init__.py
│   ├── base_scraper.py      # Abstract base: rate limiting, retries, user-agent rotation
│   ├── greenhouse_scraper.py # Greenhouse ATS public JSON API (FREE)
│   ├── lever_scraper.py     # Lever ATS public JSON API (FREE)
│   ├── workday_scraper.py   # Workday enterprise ATS (FREE)
│   ├── jsearch_scraper.py   # JSearch via RapidAPI (paid, 200 req/month free)
│   ├── indeed_scraper.py    # Indeed (blocked by 403, not used)
│   └── linkedin_scraper.py  # LinkedIn Selenium (heavy, not used in default)
│
├── cleaning/
│   ├── __init__.py
│   └── cleaner.py           # Normalization, deduplication, salary/mode/level parsing
│
├── nlp/
│   ├── __init__.py
│   ├── skill_extractor.py   # spaCy PhraseMatcher + regex skill extraction
│   └── categorizer.py       # DevOps vs AI/ML classification (3-tier)
│
├── storage/
│   ├── __init__.py
│   ├── models.py            # SQLAlchemy ORM models (Job, SkillTrend)
│   ├── postgres_store.py    # DB operations: upsert, search, analytics queries
│   └── vector_store.py      # FAISS index wrapper (add, search, persist)
│
├── agent/
│   ├── __init__.py
│   ├── embedder.py          # Sentence-transformer embedding generation
│   ├── searcher.py          # Hybrid semantic + structured search
│   ├── recommender.py       # Personalized job recommendation engine
│   └── insights.py          # Market analytics and report generation
│
├── api/
│   ├── __init__.py
│   └── main.py              # FastAPI app with CORS, all REST endpoints
│
├── pipelines/
│   ├── __init__.py
│   ├── full_pipeline.py     # 5-stage orchestrator: scrape→clean→nlp→store→index
│   └── scheduler.py         # APScheduler daemon (pipeline every 12h)
│
├── ui/
│   └── index.html           # Complete frontend (single file, no build needed)
│
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py     # Unit tests for cleaner, NLP, categorizer, vector store
│
├── storage/                 # Runtime data directory (gitignore this)
│   ├── faiss_index.bin      # Persisted FAISS vector index
│   └── faiss_index.ids.json # Maps FAISS positions → PostgreSQL job IDs
│
├── requirements.txt
├── docker-compose.yml       # PostgreSQL + Elasticsearch + Redis
├── Dockerfile               # For Railway/production deployment
└── README.md
```

---

## 5. Module Reference

### `config/settings.py`
Central configuration using Python dataclasses. All secrets via environment variables.

**Key classes:**
- `DatabaseConfig` — PostgreSQL host/port/name/user/password, `.url` property
- `ElasticsearchConfig` — ES host/port/index (currently unused in active pipeline)
- `ScrapingConfig` — delays, retries, user agents, max pages, headless flag
- `NLPConfig` — spaCy model name, embedding model, classifier model, FAISS path
- `APIConfig` — RapidAPI key, SerpAPI key, OpenAI key (all from env vars)

**Global singletons:** `DB`, `ES`, `SCRAPING`, `NLP`, `API`

**Other constants:** `TARGET_CATEGORIES`, `EXPERIENCE_KEYWORDS`, `WORK_MODE_KEYWORDS`

---

### `config/taxonomy.py`
Predefined skill taxonomy used for extraction and categorization.

**DevOps skill categories:** `containers`, `orchestration`, `ci_cd`, `iac`, `cloud_aws`, `cloud_gcp`, `cloud_azure`, `monitoring`, `networking`, `security`, `version_control`, `scripting`, `databases`

**AI/ML skill categories:** `ml_frameworks`, `llm_genai`, `mlops`, `data_processing`, `data_science`, `visualization`, `languages`, `vector_db`, `cloud_ai`, `ml_infra`

**Key exports:**
- `DEVOPS_ALIAS_MAP` — `{skill_string: category}` for all DevOps skills
- `AIML_ALIAS_MAP` — `{skill_string: category}` for all AI/ML skills
- `SKILL_NORMALIZATION` — `{alias: canonical}` e.g. `"k8s" → "kubernetes"`
- `DEVOPS_INDICATOR_SKILLS` — set of strong DevOps signals
- `AIML_INDICATOR_SKILLS` — set of strong AI/ML signals
- `DEVOPS_TITLE_PATTERNS` / `AIML_TITLE_PATTERNS` — title keyword lists

---

### `scraping/base_scraper.py`
Abstract base class for all scrapers.

**`RawJob` dataclass fields:**
```python
title: str
company: str
description: str
location: str
source: str           # e.g. "greenhouse", "lever", "jsearch:linkedin"
source_url: str       # Direct apply link
scraped_at: str       # ISO timestamp (auto-set)
salary_raw: str       # Raw salary string e.g. "$120k-$150k"
work_mode_raw: str    # Raw work mode hint
skills_raw: str       # Department/team hint
posted_date: str      # ISO date when posted
job_id: str           # Source-specific ID for dedup
```

**`BaseScraper` methods:**
- `_build_session()` — requests.Session with retry (3x, exponential backoff)
- `_get_headers()` — rotates user agents
- `_polite_delay()` — base delay + random jitter
- `_get(url, params)` — GET with rate limiting and error handling
- `scrape()` — generator yielding RawJob objects

---

### `scraping/greenhouse_scraper.py`
Scrapes Greenhouse ATS. **No API key needed.**

- Endpoint: `https://boards-api.greenhouse.io/v1/boards/{company}/jobs?content=true`
- Covers: OpenAI, Anthropic, Stripe, Airbnb, Databricks, Coinbase, Cloudflare, Datadog, etc.
- Filters by keywords: devops, mlops, machine learning, kubernetes, etc.
- 34 companies configured by default in `GREENHOUSE_COMPANIES` list
- Strips HTML descriptions with BeautifulSoup

---

### `scraping/lever_scraper.py`
Scrapes Lever ATS. **No API key needed.**

- Endpoint: `https://api.lever.co/v0/postings/{company}?mode=json&limit=500`
- Covers: Netflix, Shopify, Atlassian, MongoDB, Grafana, Pulumi, Airbyte, Dagster, Snyk, etc.
- 31 companies configured in `LEVER_COMPANIES` list
- Parses structured description sections: `descriptionPlain`, `lists[]`, `additional`
- Converts millisecond timestamps to ISO dates

---

### `scraping/workday_scraper.py`
Scrapes Workday ATS. **No API key needed.**

- Endpoint: `POST https://{tenant}.wd5.myworkdayjobs.com/wday/cxs/{tenant}/{board}/jobs`
- Covers: NVIDIA, Salesforce, Adobe, ServiceNow, VMware, Splunk, Palo Alto Networks, etc.
- 10 companies configured in `WORKDAY_COMPANIES` list (tuples of tenant/board/name)
- Searches per keyword (not per company) due to Workday's search-first API

---

### `scraping/jsearch_scraper.py`
Scrapes via JSearch RapidAPI. **Requires `RAPIDAPI_KEY` env var.**

- Endpoint: `https://jsearch.p.rapidapi.com/search`
- Aggregates LinkedIn, Indeed, Glassdoor, ZipRecruiter
- 13 search terms in `ALL_SEARCH_TERMS` (DevOps + AI/ML)
- Parses salary fields: `job_min_salary`, `job_max_salary`, `job_salary_period`
- Free tier: 200 requests/month

---

### `cleaning/cleaner.py`
Cleans and deduplicates raw jobs.

**`CleanJob` dataclass fields:**
```python
title, company, description, location, source, source_url, scraped_at
work_mode: str        # remote | hybrid | onsite | unknown
experience_level: str # junior | mid | senior | unknown
salary_min: float     # Annualized USD
salary_max: float
salary_currency: str  # USD | GBP | EUR
fingerprint: str      # MD5(title+company+location) for dedup
skills_raw: list      # Populated later by NLP stage
```

**Key methods:**
- `clean(raw_jobs)` — main entry, returns deduplicated CleanJob list
- `_normalize_text(text)` — strips HTML, normalizes unicode and whitespace
- `_clean_description(text)` — removes markdown, collapses blank lines, strips boilerplate
- `_compute_fingerprint(title, company, location)` — MD5 dedup hash
- `_parse_salary(salary_raw)` — handles `$120k`, `$60/hr`, `£50,000`, ranges → `(min, max, currency)`
- `_detect_work_mode(raw, location, description)` — keyword matching across all text
- `_detect_experience_level(title, description)` — scores keyword matches per level
- `normalize_skill(skill)` — applies `SKILL_NORMALIZATION` alias map

---

### `nlp/skill_extractor.py`
Extracts technology skills from job descriptions.

**Two-stage approach:**
1. **spaCy PhraseMatcher** — fast, exact match on all taxonomy skills (including multi-word like "github actions")
2. **Regex patterns** — catches versioned skills ("Python 3.10"), slash notation ("CI/CD"), abbreviations

**Key methods:**
- `extract(text)` → `List[str]` — deduplicated, sorted, normalized skills
- `extract_with_categories(text)` → `Dict[str, List[str]]` — skills grouped by taxonomy category
- `extract_years_of_experience(text)` → `Optional[int]` — parses "5+ years of experience"

**Handles:** `k8s→kubernetes`, `sklearn→scikit-learn`, `gh actions→github actions`

---

### `nlp/categorizer.py`
Classifies jobs into DevOps, AI/ML, or Other.

**Three-tier classification (fastest to slowest):**
1. **Title matching** — regex against `DEVOPS_TITLE_PATTERNS` / `AIML_TITLE_PATTERNS` → confidence 0.95
2. **Skill indicator matching** — counts overlap with `DEVOPS_INDICATOR_SKILLS` / `AIML_INDICATOR_SKILLS` → confidence up to 0.9
3. **Zero-shot classification** — `facebook/bart-large-mnli` via HuggingFace pipeline → confidence from model score

**`ClassificationResult` fields:** `primary_category`, `categories` (list), `confidence` (0-1), `method`

Zero-shot model is **lazily loaded** — only instantiated if tiers 1 and 2 fail. Currently disabled in pipeline (`use_zero_shot=False`) for speed.

---

### `storage/models.py`
SQLAlchemy ORM models.

**`Job` table columns:**
```
id, fingerprint, source, source_url, job_id_external
title, company, location, description
category, categories (ARRAY), experience_level, work_mode
skills (ARRAY of VARCHAR), skills_by_category (JSONB)
salary_min, salary_max, salary_currency
embedding_id, posted_date, scraped_at, updated_at, is_active
```

**Indexes:**
- `idx_jobs_skills` — GIN on `skills` array (fast array containment queries)
- `idx_jobs_category` — btree on `category`
- `idx_jobs_location_mode` — composite on `location`, `work_mode`
- `idx_jobs_salary_range` — composite on `salary_min`, `salary_max`
- `idx_jobs_company` — btree on `company`

**`SkillTrend` table:** Aggregated skill demand stats per time period.

**Important fix:** The GIN index must be on the ARRAY column only (`skills`), not combined with a VARCHAR column. Mixed-type GIN indexes are not supported in PostgreSQL.

---

### `storage/postgres_store.py`
All database read/write operations.

**Write:**
- `upsert_jobs(jobs)` — insert-or-update on fingerprint, updates skills/category on re-run

**Read:**
- `search_jobs(skills, category, location, work_mode, experience_level, salary_min, salary_max, company, days, limit, offset)` — all filters optional, AND-combined
- `get_job_by_id(job_id)` → dict
- `count_jobs(category)` → int

**Analytics:**
- `get_top_skills(category, top_n, days)` — uses PostgreSQL `unnest()` on skills array
- `get_salary_stats(category)` → `{avg_min, avg_max, min_salary, max_salary, count}`
- `get_work_mode_distribution(category)` → `{mode: count}`

---

### `storage/vector_store.py`
FAISS vector index for semantic similarity.

**Index type:** `IndexFlatIP` (exact inner product = cosine similarity on L2-normalized vectors)

**Files:**
- `storage/faiss_index.bin` — FAISS binary index
- `storage/faiss_index.ids.json` — list mapping FAISS position → PostgreSQL `job.id`

**Key methods:**
- `add(embeddings, job_ids)` — add batch, normalizes vectors before adding
- `add_one(embedding, job_id)` → position
- `search(query_embedding, top_k, score_threshold)` → `[(job_id, score)]`
- `search_by_ids(source_job_id, top_k)` — "more like this" via vector reconstruction
- `save()` — persist both files to disk
- `_normalize(vectors)` — L2 normalization for cosine sim via inner product

**Limitation:** `IndexFlatIP` is exact search — fine up to ~100k vectors. For larger scale, switch to `IndexIVFFlat`.

**Apple Silicon note:** FAISS crashes inside pytest on macOS. Skip vector store tests on Darwin.

---

### `agent/embedder.py`
Generates sentence embeddings.

**Model:** `sentence-transformers/all-MiniLM-L6-v2`
- 384-dimensional output
- ~80MB download, fully local, no API cost
- Uses MPS (Metal) on Apple Silicon automatically

**Job text format for embedding:**
```
"Senior ML Engineer | OpenAI | Category: AI/ML | Level: senior | Skills: pytorch, transformers, python | We are looking for..."
```

**Methods:**
- `build_job_text(job)` — multi-field text for job embedding
- `build_query_text(query, filters)` — appends filter context to query
- `embed_text(text)` → `(384,) float32`
- `embed_texts(texts, batch_size=32)` → `(N, 384) float32`
- `embed_jobs(jobs)` → `(N, 384) float32`
- `embed_query(query, filters)` → `(384,) float32`

---

### `agent/searcher.py`
Unified hybrid search interface.

**`search(query, filters, top_k, semantic_weight, min_score)`**
- Embeds query → FAISS → PostgreSQL fetch → apply filters → re-rank → return
- `semantic_weight=0.7` (70% vector similarity, 30% structured score)
- Structured score = keyword hits in title/description + filter matches + recency boost

**`search_by_skills(skills, category, work_mode, top_k)`**
- Pure structured search via PostgreSQL array containment

**`find_similar(job_id, top_k)`**
- Reconstructs vector from FAISS index → searches for neighbors → returns "more like this"

**CLI usage:**
```bash
python agent/searcher.py --query "senior kubernetes remote" --category DevOps --mode remote --level senior --top 10
```

Output includes: match score, title, company, location, work mode, salary, skills, apply URL.

---

### `agent/recommender.py`
Personalized job recommendation engine.

**`UserProfile` dataclass:**
```python
skills: List[str]
preferred_category: Optional[str]     # "DevOps" | "AI/ML"
preferred_work_mode: Optional[str]    # "remote" | "hybrid" | "onsite"
preferred_location: Optional[str]
target_salary_min: Optional[float]
target_salary_max: Optional[float]
experience_level: Optional[str]       # "junior" | "mid" | "senior"
role_description: Optional[str]       # Free text for semantic matching
```

**Scoring weights:**
- 40% semantic similarity (FAISS cosine distance)
- 35% skill overlap (user skills ∩ job skills / job skills count)
- 15% preference match (category + work_mode + level + location)
- 10% salary alignment (linear decay below target)

**Output:** Ranked jobs with `_score`, `_score_breakdown`, `_explanation` fields added.

---

### `agent/insights.py`
Market analytics and report generation.

**`MarketInsights` dataclass fields:**
```python
total_jobs, total_devops, total_aiml
top_devops_skills: List[{skill, count}]
top_aiml_skills: List[{skill, count}]
skill_co_occurrences: List[(skill_a, skill_b, count)]
salary_by_category: Dict[category, {avg_min, avg_max, count}]
salary_by_level: Dict[level, {avg_min, avg_max}]
work_mode_distribution: Dict[mode, count]
experience_distribution: Dict[level, count]
top_hiring_companies: List[{company, count}]
```

**`InsightsEngine` methods:**
- `generate_full_report(period_days=30)` → `MarketInsights`
- `get_skill_demand_over_time(skill, days)` → weekly count trend
- `_compute_skill_cooccurrences(top_n)` — Python-side, samples 5000 jobs

---

### `api/main.py`
FastAPI REST backend.

**Endpoints:**

| Method | Path | Description |
|---|---|---|
| GET | `/api/search` | Semantic + structured job search |
| POST | `/api/recommend` | Personalized recommendations |
| GET | `/api/insights` | Full market insights report |
| GET | `/api/jobs/{id}` | Single job by ID |
| GET | `/api/stats` | Dashboard header stats |
| GET | `/health` | Health check |

CORS configured for all origins (tighten in production).

Singletons instantiated at startup: `JobSearcher`, `JobRecommender`, `PostgresStore`.

---

### `pipelines/full_pipeline.py`
5-stage end-to-end orchestrator.

**Search terms used:**
```python
DEVOPS: "DevOps Engineer", "Site Reliability Engineer", "Platform Engineer",
        "Kubernetes Engineer", "Cloud Infrastructure Engineer", "MLOps Engineer"

AIML:   "Machine Learning Engineer", "AI Engineer", "Data Scientist",
        "NLP Engineer", "Computer Vision Engineer", "Applied Scientist",
        "ML Research Engineer"
```

**Stages:**
1. **Scrape** — streams RawJob objects from all configured sources
2. **Clean** — deduplication + normalization → CleanJob list
3. **Enrich** — skill extraction + categorization + experience level refinement
4. **Store** — bulk upsert to PostgreSQL (fingerprint-based conflict resolution)
5. **Index** — batch embedding generation + FAISS index update + disk save

**CLI:**
```bash
python -m pipelines.full_pipeline --sources greenhouse lever workday jsearch --location "United States"
```

Available sources: `greenhouse`, `lever`, `workday`, `jsearch`, `indeed` (blocked), `linkedin` (heavy)

Default sources: `greenhouse lever`

---

### `pipelines/scheduler.py`
APScheduler daemon for periodic automation.

**Schedule:**
- Full pipeline: every 12 hours at 00:00 and 12:00 UTC
- Insights refresh: every 6 hours at HH:30 UTC

**CLI:**
```bash
python pipelines/scheduler.py            # Start daemon
python pipelines/scheduler.py --run-now # Run immediately then schedule
python pipelines/scheduler.py --once    # Run once (for cron)
```

---

### `ui/index.html`
Single-file frontend — no build step, no npm, no framework.

**Design:** Dark terminal aesthetic. Space Mono (monospace) + Syne (display) fonts. CSS grid layout. Cyan accent (#00e5ff) + purple (#7c3aed). Animated grid background with glow orbs.

**Three views:**
1. **Search** — text input, 3 filter dropdowns (category/mode/level), job cards with match score + Apply button
2. **Recommend** — profile form (skills as chips, dropdowns, salary input, text area), ranked results with explanation text
3. **Insights** — skill demand bars, work mode/level distribution, salary ranges, top companies, skill co-occurrence grid

**API base URL:** Hardcoded as `const API = 'http://localhost:8000/api'` — change for production.

---

## 6. Data Flow

### Scraping → Storage Flow

```
JSearch/Greenhouse/Lever/Workday API
         │
         ▼ RawJob (title, company, description, location, source, source_url,
         │         salary_raw, work_mode_raw, job_id, posted_date)
         │
    JobCleaner
         │ - Strip HTML, normalize whitespace
         │ - Compute fingerprint (MD5 of title+company+location)
         │ - Check fingerprint against in-memory seen set → deduplicate
         │ - Parse salary string → (min_float, max_float, currency)
         │ - Detect work mode → remote/hybrid/onsite/unknown
         │ - Detect experience level → junior/mid/senior/unknown
         │
         ▼ CleanJob
         │
    SkillExtractor + JobCategorizer
         │ - PhraseMatcher → matched skills
         │ - Regex → versioned/abbreviated skills
         │ - Normalize aliases (k8s→kubernetes)
         │ - Classify category (title → skills → zero-shot)
         │ - Refine experience level with years parsed from description
         │
         ▼ enriched dict (CleanJob fields + skills, skills_by_category,
         │                 category, categories)
         │
    PostgresStore.upsert_jobs()
         │ - INSERT ... ON CONFLICT (fingerprint) DO UPDATE
         │
    JobEmbedder.embed_jobs()
         │ - Build multi-field text per job
         │ - Batch encode (sentence-transformers, batch_size=32)
         │ - L2 normalize
         │
    VectorStore.add(embeddings, job_ids)
         │ - Add to FAISS IndexFlatIP
         │ - Append to id_map list
         │
    VectorStore.save()
         └─ Write faiss_index.bin + faiss_index.ids.json
```

---

## 7. Database Schema

### `jobs` table

```sql
CREATE TABLE jobs (
    id                  SERIAL PRIMARY KEY,
    fingerprint         VARCHAR(32) UNIQUE NOT NULL,    -- MD5 dedup hash
    source              VARCHAR(200) NOT NULL,           -- "greenhouse", "lever", etc.
    source_url          TEXT,                            -- Direct apply link
    job_id_external     VARCHAR(200),                    -- Source's own ID

    title               VARCHAR(300) NOT NULL,
    company             VARCHAR(200) NOT NULL,
    location            VARCHAR(200),
    description         TEXT,

    category            VARCHAR(50),                    -- "DevOps" | "AI/ML" | "Other"
    categories          VARCHAR[],                      -- May contain both
    experience_level    VARCHAR(20),                    -- junior | mid | senior | unknown
    work_mode           VARCHAR(20),                    -- remote | hybrid | onsite | unknown

    skills              VARCHAR[],                      -- Normalized skill list
    skills_by_category  JSONB,                          -- {category: [skills]}

    salary_min          FLOAT,                          -- Annualized USD
    salary_max          FLOAT,
    salary_currency     VARCHAR(5) DEFAULT 'USD',

    embedding_id        INTEGER,                        -- FAISS index position
    posted_date         TIMESTAMP,
    scraped_at          TIMESTAMP DEFAULT NOW(),
    updated_at          TIMESTAMP DEFAULT NOW(),
    is_active           BOOLEAN DEFAULT TRUE
);

-- Indexes
CREATE INDEX idx_jobs_skills ON jobs USING gin(skills);
CREATE INDEX idx_jobs_category ON jobs (category);
CREATE INDEX idx_jobs_location_mode ON jobs (location, work_mode);
CREATE INDEX idx_jobs_salary_range ON jobs (salary_min, salary_max);
CREATE INDEX idx_jobs_company ON jobs (company);
```

### `skill_trends` table

```sql
CREATE TABLE skill_trends (
    id                  SERIAL PRIMARY KEY,
    skill               VARCHAR(100) NOT NULL,
    category            VARCHAR(50) NOT NULL,
    taxonomy_category   VARCHAR(100),
    job_count           INTEGER DEFAULT 0,
    avg_salary_min      FLOAT,
    avg_salary_max      FLOAT,
    remote_job_count    INTEGER DEFAULT 0,
    period_start        TIMESTAMP NOT NULL,
    period_end          TIMESTAMP NOT NULL,
    computed_at         TIMESTAMP DEFAULT NOW()
);
```

---

## 8. API Reference

### GET `/api/search`

Query parameters:
```
q                string  required  Search query
category         string  optional  "DevOps" | "AI/ML"
work_mode        string  optional  "remote" | "hybrid" | "onsite"
experience_level string  optional  "junior" | "mid" | "senior"
salary_min       float   optional  Minimum salary filter
top_k            int     optional  Default: 20
```

Response:
```json
{
  "results": [
    {
      "id": 42,
      "title": "Senior ML Engineer",
      "company": "OpenAI",
      "location": "San Francisco, CA",
      "category": "AI/ML",
      "work_mode": "hybrid",
      "experience_level": "senior",
      "skills": ["pytorch", "python", "transformers"],
      "salary_min": 200000,
      "salary_max": 300000,
      "source": "greenhouse",
      "source_url": "https://boards.greenhouse.io/openai/jobs/...",
      "scraped_at": "2026-02-19T09:00:00",
      "_score": 0.847,
      "_semantic_score": 0.912
    }
  ],
  "count": 15
}
```

---

### POST `/api/recommend`

Request body:
```json
{
  "skills": ["kubernetes", "terraform", "aws", "python"],
  "preferred_category": "DevOps",
  "preferred_work_mode": "remote",
  "experience_level": "senior",
  "target_salary_min": 130000,
  "role_description": "Platform engineering at a fast-growing startup",
  "top_k": 10
}
```

Response: Same structure as search, plus `_explanation` and `_score_breakdown` fields per job.

---

### GET `/api/insights`

Response:
```json
{
  "total_jobs": 87,
  "total_devops": 45,
  "total_aiml": 42,
  "top_devops_skills": [{"skill": "kubernetes", "count": 32}],
  "top_aiml_skills": [{"skill": "pytorch", "count": 28}],
  "work_mode_distribution": {"remote": 41, "hybrid": 23, "onsite": 23},
  "experience_distribution": {"senior": 52, "mid": 21, "junior": 8, "unknown": 6},
  "salary_by_category": {
    "DevOps": {"avg_min": 118000, "avg_max": 155000, "count": 12}
  },
  "top_hiring_companies": [{"company": "OpenAI", "count": 5}],
  "skill_co_occurrences": [{"skill_a": "kubernetes", "skill_b": "docker", "count": 28}]
}
```

---

### GET `/api/stats`

```json
{
  "total": 87,
  "devops": 45,
  "aiml": 42,
  "work_modes": {"remote": 41, "hybrid": 23, "onsite": 23}
}
```

---

## 9. Configuration

### Environment Variables

```bash
# Database (required)
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=job_agent
POSTGRES_USER=postgres
POSTGRES_PASSWORD=password

# Elasticsearch (optional, not used in current active pipeline)
ES_HOST=localhost
ES_PORT=9200

# JSearch API (only needed for jsearch source)
RAPIDAPI_KEY=your_rapidapi_key_here

# Optional AI integrations
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
```

### Key Config Values in `config/settings.py`

```python
# Scraping behavior
request_delay_seconds: 2.0      # Polite delay between requests
max_retries: 3                  # HTTP retry attempts
timeout_seconds: 30             # Request timeout
max_pages_per_source: 1         # Set to 10 for production, 1 for testing

# NLP
spacy_model: "en_core_web_sm"
embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
embedding_dim: 384
faiss_index_path: "storage/faiss_index.bin"
```

---

## 10. Skill Taxonomy

### DevOps Skills (185 total across 13 categories)

| Category | Example Skills |
|---|---|
| `containers` | docker, podman, containerd |
| `orchestration` | kubernetes, k8s, helm, openshift, rancher |
| `ci_cd` | jenkins, github actions, gitlab ci, argocd, circleci |
| `iac` | terraform, pulumi, ansible, cloudformation, chef |
| `cloud_aws` | aws, ec2, s3, lambda, eks, ecs, rds |
| `cloud_gcp` | gcp, gke, bigquery, cloud run |
| `cloud_azure` | azure, aks, azure devops, blob storage |
| `monitoring` | prometheus, grafana, datadog, splunk, elk, opentelemetry |
| `networking` | nginx, haproxy, istio, envoy, consul |
| `security` | vault, sonarqube, snyk, trivy, falco |
| `version_control` | git, github, gitlab, bitbucket |
| `scripting` | bash, python, go, ruby, powershell |
| `databases` | postgresql, mysql, mongodb, redis, kafka |

### AI/ML Skills (across 10 categories)

| Category | Example Skills |
|---|---|
| `ml_frameworks` | pytorch, tensorflow, keras, jax, scikit-learn, xgboost |
| `llm_genai` | llm, gpt, langchain, rag, transformers, bert, hugging face |
| `mlops` | mlflow, kubeflow, airflow, sagemaker, vertex ai, wandb |
| `data_processing` | spark, pandas, numpy, dask, databricks, snowflake, dbt |
| `data_science` | machine learning, deep learning, nlp, computer vision, statistics |
| `visualization` | matplotlib, plotly, tableau, power bi |
| `languages` | python, r, julia, scala, sql |
| `vector_db` | faiss, pinecone, weaviate, chroma, milvus, pgvector |
| `cloud_ai` | sagemaker, vertex ai, bedrock, azure cognitive services |
| `ml_infra` | gpu, cuda, tensorrt, onnx, distributed training |

### Alias Normalization Map (sample)

```python
"k8s"         → "kubernetes"
"gh actions"  → "github actions"
"sklearn"     → "scikit-learn"
"tf"          → "terraform"
"postgres"    → "postgresql"
"mongo"       → "mongodb"
"gpt-4"       → "gpt"
"chatgpt"     → "gpt"
```

---

## 11. Running the Project

### First-Time Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download spaCy model
python -m spacy download en_core_web_sm

# 4. Start infrastructure
docker-compose up -d

# 5. Set environment variables
export POSTGRES_HOST=localhost
export POSTGRES_PASSWORD=password
export RAPIDAPI_KEY=your_key  # Only if using jsearch source
```

### Running the Pipeline

```bash
# Free sources (no API key) — recommended default
python -m pipelines.full_pipeline --sources greenhouse lever

# All free sources
python -m pipelines.full_pipeline --sources greenhouse lever workday

# Paid API source (requires RAPIDAPI_KEY)
python -m pipelines.full_pipeline --sources jsearch

# Testing mode (max_pages_per_source=1 in settings.py)
python -m pipelines.full_pipeline --sources greenhouse
```

### Running the API

```bash
uvicorn api.main:app --reload --port 8000
# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### Running the Frontend

```bash
open ui/index.html
# Or serve it:
python -m http.server 3000 --directory ui
```

### Running Tests

```bash
pytest tests/test_pipeline.py -v
# 24/26 tests pass
# 2 FAISS tests skipped on macOS (Apple Silicon crash in pytest context)
```

### Searching via CLI

```bash
python agent/searcher.py \
  --query "senior kubernetes platform engineer remote" \
  --category DevOps \
  --mode remote \
  --level senior \
  --top 10
```

### Generating Insights via Python

```python
from agent.insights import InsightsEngine
report = InsightsEngine().generate_full_report()
print(report.to_text())
```

### Automated Scheduling

```bash
python pipelines/scheduler.py --run-now   # Run now + schedule every 12h
python pipelines/scheduler.py --once      # Run once (for external cron)
```

---

## 12. Known Issues & Fixes Applied

### 1. GIN Index on Mixed Column Types
**Error:** `data type character varying has no default operator class for access method "gin"`  
**Cause:** Cannot create a single GIN index across both an array column and a varchar column.  
**Fix:** Split into two separate indexes — GIN on `skills` array only, regular btree on `category`.

### 2. Indeed / LinkedIn 403 Blocked
**Error:** `403 Client Error: Forbidden`  
**Cause:** Both sites aggressively block direct HTTP scrapers.  
**Fix:** Use Greenhouse + Lever + Workday instead (free, public, no blocking).

### 3. `source` Column Too Short
**Error:** `value too long for type character varying(50)`  
**Cause:** JSearch returns long publisher names like `"jsearch:usnlx ability jobs - national labor exchange"`.  
**Fix:** Increased `source` column to `VARCHAR(200)` and truncated source name to 40 chars in parser.

### 4. FAISS Crash in pytest on Apple Silicon
**Error:** `Fatal Python error: Aborted` in FAISS `.search()` during pytest  
**Cause:** Known incompatibility between FAISS and pytest on macOS with MPS.  
**Fix:** Skip vector store tests on Darwin using `pytest.skip()` in `setup_method`.

### 5. BeautifulSoup Filename Warning
**Warning:** `MarkupResemblesLocatorWarning: The input looks more like a filename than markup`  
**Cause:** Some job description strings happen to look like file paths.  
**Impact:** Cosmetic only — does not affect functionality.

### 6. `source` Field Truncation for JSearch
Applied `.lower()[:40]` to publisher name when building source string to prevent VARCHAR overflow.

---

## 13. Deployment

### Target Architecture (Production)

```
User Browser
    │
    ▼
Vercel (ui/index.html)          ← static frontend, free
    │
    │ fetch() to API URL
    ▼
Railway / Render (FastAPI)      ← backend + pipeline runner
    │
    ├── PostgreSQL (Railway managed)
    └── FAISS index (filesystem, persisted volume)
```

### Backend Dockerfile

```dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm
COPY . .
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Railway Deployment

```bash
npm install -g @railway/cli
railway login
railway init
railway add postgres          # Managed PostgreSQL, auto-injects DATABASE_URL
railway up                    # Deploy
railway domain                # Get public URL
railway run python -m pipelines.full_pipeline --sources greenhouse lever
```

### Vercel Deployment (Frontend)

```bash
# Update API URL in ui/index.html first:
# const API = 'https://your-railway-app.up.railway.app/api';

npm install -g vercel
vercel deploy ui/index.html
```

### CORS for Production

In `api/main.py`, restrict allowed origins:
```python
allow_origins=["https://your-app.vercel.app"]
```

---

## Appendix: Quick Reference Commands

```bash
# Pipeline
python -m pipelines.full_pipeline --sources greenhouse lever
python -m pipelines.full_pipeline --sources greenhouse lever workday jsearch

# API
uvicorn api.main:app --reload --port 8000

# CLI Search
python agent/searcher.py -q "ml engineer pytorch remote" -c "AI/ML" -m remote -n 10

# Tests
pytest tests/test_pipeline.py -v

# Scheduler
python pipelines/scheduler.py --run-now

# DB: Drop and recreate tables (after schema changes)
docker exec -it $(docker ps -qf "name=postgres") psql -U postgres -d job_agent \
  -c "DROP TABLE IF EXISTS jobs CASCADE; DROP TABLE IF EXISTS skill_trends CASCADE;"

# Check DB contents
docker exec -it $(docker ps -qf "name=postgres") psql -U postgres -d job_agent \
  -c "SELECT category, count(*) FROM jobs GROUP BY category;"
```