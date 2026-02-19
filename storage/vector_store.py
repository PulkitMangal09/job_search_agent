# storage/vector_store.py
"""
FAISS vector index for semantic job search.

Jobs are embedded using sentence-transformers (all-MiniLM-L6-v2).
The index maps embedding positions to PostgreSQL job IDs for retrieval.

Why FAISS?
  - Free, local, no API costs
  - Fast approximate nearest neighbor search (ANN)
  - Easy to persist to disk and reload

Production alternative: Pinecone, Weaviate, or pgvector (PostgreSQL extension)
"""

import os
import json
import logging
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path

import faiss

from config.settings import NLP as NLP_CONFIG

logger = logging.getLogger(__name__)

INDEX_PATH = Path(NLP_CONFIG.faiss_index_path)
ID_MAP_PATH = INDEX_PATH.with_suffix(".ids.json")


class VectorStore:
    """
    Wraps a FAISS FlatIP index for semantic nearest-neighbor job search.
    
    FlatIP = Flat (exact, not approximate) + Inner Product (cosine sim
    when vectors are L2-normalized).
    
    For >100k jobs, switch to IndexIVFFlat for ANN performance.
    """

    def __init__(self, dim: int = NLP_CONFIG.embedding_dim):
        self.dim = dim
        self.index: Optional[faiss.IndexFlatIP] = None
        self.id_map: List[int] = []     # Maps FAISS position → PostgreSQL job.id
        self._load_or_create()

    # ── Index Lifecycle ───────────────────────────────────────────────────────

    def _load_or_create(self):
        """Load existing FAISS index from disk, or create a fresh one."""
        if INDEX_PATH.exists() and ID_MAP_PATH.exists():
            self.index = faiss.read_index(str(INDEX_PATH))
            with open(ID_MAP_PATH) as f:
                self.id_map = json.load(f)
            logger.info(f"Loaded FAISS index: {self.index.ntotal} vectors")
        else:
            self.index = faiss.IndexFlatIP(self.dim)
            self.id_map = []
            logger.info(f"Created new FAISS index (dim={self.dim})")

    def save(self):
        """Persist the FAISS index and ID map to disk."""
        INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(INDEX_PATH))
        with open(ID_MAP_PATH, "w") as f:
            json.dump(self.id_map, f)
        logger.info(f"FAISS index saved: {self.index.ntotal} vectors → {INDEX_PATH}")

    # ── Adding Vectors ────────────────────────────────────────────────────────

    def add(self, embeddings: np.ndarray, job_ids: List[int]):
        """
        Add job embeddings to the index.
        
        Args:
            embeddings: (N, dim) float32 array of L2-normalized embeddings
            job_ids: List of PostgreSQL job.id values corresponding to rows
        """
        if len(embeddings) != len(job_ids):
            raise ValueError("embeddings and job_ids must have the same length")

        # Normalize to unit vectors for cosine similarity via inner product
        embeddings = self._normalize(embeddings)
        self.index.add(embeddings)
        self.id_map.extend(job_ids)
        logger.debug(f"Added {len(job_ids)} vectors. Total: {self.index.ntotal}")

    def add_one(self, embedding: np.ndarray, job_id: int) -> int:
        """
        Add a single job embedding. Returns its FAISS index position.
        """
        vec = self._normalize(embedding.reshape(1, -1))
        self.index.add(vec)
        self.id_map.append(job_id)
        return self.index.ntotal - 1

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        score_threshold: float = 0.3,
    ) -> List[Tuple[int, float]]:
        """
        Find the top_k most semantically similar jobs.
        
        Args:
            query_embedding: (dim,) or (1, dim) embedding of the query
            top_k: Number of results to return
            score_threshold: Minimum cosine similarity score (0-1)
        
        Returns:
            List of (job_id, score) tuples sorted by descending similarity
        """
        if self.index.ntotal == 0:
            logger.warning("FAISS index is empty — no jobs indexed yet")
            return []

        query = self._normalize(query_embedding.reshape(1, -1))
        scores, indices = self.index.search(query, min(top_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:   # FAISS uses -1 for empty slots
                continue
            if score < score_threshold:
                continue
            job_id = self.id_map[idx]
            results.append((job_id, float(score)))

        return results

    def search_by_ids(
        self,
        source_job_id: int,
        top_k: int = 5,
    ) -> List[Tuple[int, float]]:
        """
        Find jobs similar to a given job (by its PostgreSQL ID).
        Useful for "More like this" recommendations.
        """
        try:
            faiss_idx = self.id_map.index(source_job_id)
        except ValueError:
            logger.warning(f"Job ID {source_job_id} not found in FAISS index")
            return []

        # Reconstruct the vector for the source job
        source_vec = np.zeros((1, self.dim), dtype=np.float32)
        self.index.reconstruct(faiss_idx, source_vec[0])

        results = self.search(source_vec[0], top_k=top_k + 1)
        # Filter out the source job itself
        return [(jid, score) for jid, score in results if jid != source_job_id][:top_k]

    # ── Utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        """L2-normalize vectors so inner product equals cosine similarity."""
        vectors = vectors.astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)    # Avoid division by zero
        return vectors / norms

    def __len__(self) -> int:
        return self.index.ntotal if self.index else 0

    def stats(self) -> dict:
        return {
            "total_vectors": self.index.ntotal if self.index else 0,
            "dimension": self.dim,
            "index_path": str(INDEX_PATH),
        }
