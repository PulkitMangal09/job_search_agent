# agent/embedder.py
"""
Generates sentence embeddings for job descriptions and user queries.

Model: sentence-transformers/all-MiniLM-L6-v2
  - 384-dimensional embeddings
  - ~80MB download, runs entirely locally (no API cost)
  - Strong performance on semantic similarity tasks

Usage:
    embedder = JobEmbedder()
    embedding = embedder.embed_text("Senior Kubernetes engineer, AWS experience required")
    job_embeddings = embedder.embed_jobs(list_of_job_dicts)
"""

import logging
from typing import List, Dict, Union

import numpy as np
from sentence_transformers import SentenceTransformer

from config.settings import NLP as NLP_CONFIG

logger = logging.getLogger(__name__)


class JobEmbedder:
    """
    Wraps a sentence-transformers model with job-specific text preparation.
    
    The text input to the model is crafted from multiple job fields to ensure
    the embedding captures: role, required skills, and seniority level.
    """

    def __init__(self):
        logger.info(f"Loading embedding model: {NLP_CONFIG.embedding_model}")
        self.model = SentenceTransformer(NLP_CONFIG.embedding_model)
        self.dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding model ready. Dimension: {self.dim}")

    # ── Text Preparation ──────────────────────────────────────────────────────

    def build_job_text(self, job: Dict) -> str:
        """
        Construct a condensed text representation of a job for embedding.
        
        We combine title, company, skills, category, and description excerpt
        because these signal semantics better than the full description alone.
        The description excerpt prevents noise from boilerplate text dominating.
        """
        parts = [
            job.get("title", ""),
            job.get("company", ""),
            f"Category: {job.get('category', '')}",
            f"Level: {job.get('experience_level', '')}",
            f"Skills: {', '.join(job.get('skills', [])[:15])}",  # Cap at 15 skills
            # First 300 chars of description (summary/intro tends to be most semantic)
            job.get("description", "")[:300],
        ]
        return " | ".join(p for p in parts if p)

    def build_query_text(self, query: str, filters: Dict = None) -> str:
        """
        Prepare a user's search query for embedding.
        Appending filter context improves retrieval precision.
        
        Example:
            query = "remote senior DevOps AWS Kubernetes"
            filters = {"work_mode": "remote", "experience_level": "senior"}
            → "remote senior DevOps AWS Kubernetes | remote senior"
        """
        if not filters:
            return query
        filter_context = " ".join(
            f"{v}" for v in filters.values() if v
        )
        return f"{query} | {filter_context}" if filter_context else query

    # ── Embedding Interface ───────────────────────────────────────────────────

    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a single text string.
        Returns: (dim,) float32 array.
        """
        return self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Embed a list of texts in batches.
        Returns: (N, dim) float32 array.
        """
        return self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100,
        )

    def embed_jobs(self, jobs: List[Dict], batch_size: int = 32) -> np.ndarray:
        """
        Embed a list of job dicts using the multi-field job text builder.
        Returns: (N, dim) float32 array.
        """
        texts = [self.build_job_text(job) for job in jobs]
        logger.info(f"Embedding {len(texts)} jobs...")
        embeddings = self.embed_texts(texts, batch_size=batch_size)
        logger.info(f"Generated {embeddings.shape} embeddings")
        return embeddings

    def embed_query(self, query: str, filters: Dict = None) -> np.ndarray:
        """
        Embed a search query with optional filter context.
        Returns: (dim,) float32 array.
        """
        text = self.build_query_text(query, filters)
        return self.embed_text(text)
