# config/settings.py
"""
Central configuration for the Job Intelligence Agent.
All secrets should be set via environment variables in production.
"""

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class DatabaseConfig:
    """PostgreSQL connection settings."""
    host: str = os.getenv("POSTGRES_HOST", "localhost")
    port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    name: str = os.getenv("POSTGRES_DB", "job_agent")
    user: str = os.getenv("POSTGRES_USER", "postgres")
    password: str = os.getenv("POSTGRES_PASSWORD", "password")

    @property
    def url(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


@dataclass
class ElasticsearchConfig:
    """Elasticsearch connection settings."""
    host: str = os.getenv("ES_HOST", "localhost")
    port: int = int(os.getenv("ES_PORT", "9200"))
    index_name: str = "job_listings"

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass
class ScrapingConfig:
    """Scraping behavior settings."""
    request_delay_seconds: float = 2.0       # Polite delay between requests
    max_retries: int = 3                      # Max retry attempts on failure
    timeout_seconds: int = 30                 # Request timeout
    user_agents: List[str] = field(default_factory=lambda: [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
    ])
    max_pages_per_source: int = 1 #10           # Cap pages to avoid excessive scraping
    headless_browser: bool = True            # Use headless Selenium


@dataclass
class NLPConfig:
    """NLP model configuration."""
    spacy_model: str = "en_core_web_sm"
    # Sentence transformer for embeddings (free, no API key needed)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Zero-shot classification for job categorization
    classifier_model: str = "facebook/bart-large-mnli"
    embedding_dim: int = 384                 # Dimension of all-MiniLM-L6-v2
    faiss_index_path: str = "storage/faiss_index.bin"


@dataclass
class APIConfig:
    """Third-party API keys."""
    # Set these in environment — never hardcode secrets
    rapidapi_key: str = os.getenv("RAPIDAPI_KEY", "")          # For LinkedIn/Indeed APIs
    serpapi_key: str = os.getenv("SERPAPI_KEY", "")            # For Google Jobs
    openai_key: str = os.getenv("OPENAI_API_KEY", "")          # Optional: GPT-4 for summaries
    anthropic_key: str = os.getenv("ANTHROPIC_API_KEY", "")    # Optional: Claude for insights


# Global config singletons
DB = DatabaseConfig()
ES = ElasticsearchConfig()
SCRAPING = ScrapingConfig()
NLP = NLPConfig()
API = APIConfig()

# Job categories to track
TARGET_CATEGORIES = ["DevOps", "AI/ML"]

# Experience level keywords for detection
EXPERIENCE_KEYWORDS = {
    "junior": ["junior", "entry level", "entry-level", "graduate", "intern", "0-2 years", "1 year"],
    "mid":    ["mid", "intermediate", "2-5 years", "3+ years", "experienced"],
    "senior": ["senior", "lead", "principal", "staff", "architect", "5+ years", "7+ years", "10+"],
}

# Work mode keywords
WORK_MODE_KEYWORDS = {
    "remote":  ["remote", "work from home", "wfh", "distributed", "fully remote"],
    "hybrid":  ["hybrid", "flexible", "part remote", "2 days"],
    "onsite":  ["on-site", "onsite", "in-office", "in office", "on site"],
}
