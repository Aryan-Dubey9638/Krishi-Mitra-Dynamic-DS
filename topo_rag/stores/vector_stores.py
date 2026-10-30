"""
Vector Store Implementations
=============================
Route A: NarrativeStore — FAISS + sentence-transformer bi-encoder
Route B: TabularStore   — Multi-vector CALI store (ColBERT-style late interaction
                          approximated via per-cell FAISS with MaxSim aggregation)

Both stores share a common BaseVectorStore interface.
"""

import logging
import os
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports — only load heavy dependencies when actually used
# ---------------------------------------------------------------------------

def _load_faiss():
    try:
        import faiss
        return faiss
    except ImportError:
        raise ImportError("faiss-cpu is required: pip install faiss-cpu")


def _load_st():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer
    except ImportError:
        raise ImportError("sentence-transformers is required: pip install sentence-transformers")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

class SearchResult:
    def __init__(self, chunk_id: str, text: str, score: float, metadata: Dict):
        self.chunk_id = chunk_id
        self.text = text
        self.score = score
        self.metadata = metadata

    def __repr__(self):
        return f"SearchResult(chunk_id={self.chunk_id!r}, score={self.score:.4f})"


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------

class BaseVectorStore(ABC):
    @abstractmethod
    def add(self, chunk_ids: List[str], texts: List[str], metadatas: List[Dict]) -> None:
        ...

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        ...

    @abstractmethod
    def save(self, directory: str) -> None:
        ...

    @abstractmethod
    def load(self, directory: str) -> None:
        ...

    @property
    @abstractmethod
    def count(self) -> int:
        ...


# ---------------------------------------------------------------------------
# Route A: Narrative / Bi-encoder Store
# ---------------------------------------------------------------------------

class NarrativeStore(BaseVectorStore):
    """
    FAISS-backed bi-encoder store for unstructured narrative text.
    Uses sentence-transformers for dense embeddings (Inner Product / cosine).

    Default model: 'all-MiniLM-L6-v2' (fast, 22MB, good quality for English)
    Can be swapped to 'BAAI/bge-m3' for production multilingual use.
    """

    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model = None          # lazy-loaded
        self._index = None          # FAISS index
        self._chunk_ids: List[str] = []
        self._texts: List[str] = []
        self._metadatas: List[Dict] = []
        self._dim: Optional[int] = None
        logger.info("NarrativeStore initialised (model=%s)", model_name)

    @property
    def model(self):
        if self._model is None:
            SentenceTransformer = _load_st()
            logger.info("Loading bi-encoder model: %s …", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _embed(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 50,
            batch_size=32,
        )
        return embeddings.astype(np.float32)

    def add(self, chunk_ids: List[str], texts: List[str], metadatas: List[Dict]) -> None:
        if not texts:
            return
        faiss = _load_faiss()
        embeddings = self._embed(texts)
        if self._index is None:
            self._dim = embeddings.shape[1]
            self._index = faiss.IndexFlatIP(self._dim)   # Inner Product (cosine after normalize)
            logger.info("Created FAISS IndexFlatIP dim=%d", self._dim)

        self._index.add(embeddings)
        self._chunk_ids.extend(chunk_ids)
        self._texts.extend(texts)
        self._metadatas.extend(metadatas)
        logger.info("NarrativeStore: added %d vectors (total=%d)", len(texts), len(self._texts))

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        if self._index is None or self._index.ntotal == 0:
            return []
        q_emb = self._embed([query])
        scores, indices = self._index.search(q_emb, min(top_k, self._index.ntotal))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append(SearchResult(
                chunk_id=self._chunk_ids[idx],
                text=self._texts[idx],
                score=float(score),
                metadata=self._metadatas[idx],
            ))
        return results

    def save(self, directory: str) -> None:
        faiss = _load_faiss()
        Path(directory).mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(Path(directory) / "narrative.index"))
        with open(Path(directory) / "narrative_meta.pkl", "wb") as f:
            pickle.dump({
                "chunk_ids": self._chunk_ids,
                "texts": self._texts,
                "metadatas": self._metadatas,
                "dim": self._dim,
                "model_name": self.model_name,
            }, f)
        logger.info("NarrativeStore saved to %s", directory)

    def load(self, directory: str) -> None:
        faiss = _load_faiss()
        self._index = faiss.read_index(str(Path(directory) / "narrative.index"))
        with open(Path(directory) / "narrative_meta.pkl", "rb") as f:
            data = pickle.load(f)
        self._chunk_ids = data["chunk_ids"]
        self._texts = data["texts"]
        self._metadatas = data["metadatas"]
        self._dim = data["dim"]
        logger.info("NarrativeStore loaded from %s (%d vectors)", directory, self._index.ntotal)
        # Eager-load the sentence-transformer model now so the first real query
        # doesn't pay the 8-10 second PyTorch cold-start cost.
        logger.info("NarrativeStore: pre-warming bi-encoder model…")
        _ = self._embed(["warmup"])
        logger.info("NarrativeStore: bi-encoder ready.")

    @property
    def count(self) -> int:
        return self._index.ntotal if self._index else 0


# ---------------------------------------------------------------------------
# Route B: Tabular / CALI Multi-Vector Store
# ---------------------------------------------------------------------------

class TabularStore(BaseVectorStore):
    """
    Cell-Aware Late Interaction (CALI) tabular store.

    Each table cell is encoded as:
        "[COL: <header>] [ROW: <row_idx>] [VAL: <value>]"

    During retrieval, MaxSim aggregation over cell vectors is performed
    (ColBERT-style late interaction):
        Score(Q, D) = Σ_i max_j sim(q_i, d_j)

    Approximated here via:
        1. Dense embedding of each cell text.
        2. FAISS search over all cell embeddings.
        3. Group cell scores by source row → MaxSim aggregate per row.
        4. Return top-K rows sorted by aggregated score.
    """

    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model = None
        self._index = None
        self._chunk_ids: List[str] = []
        self._texts: List[str] = []
        self._metadatas: List[Dict] = []
        self._row_ids: List[str] = []    # maps cell → logical row key
        self._dim: Optional[int] = None
        logger.info("TabularStore (CALI) initialised (model=%s)", model_name)

    @property
    def model(self):
        if self._model is None:
            SentenceTransformer = _load_st()
            logger.info("Loading tabular encoder model: %s …", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _embed(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100,
            batch_size=64,
        ).astype(np.float32)

    @staticmethod
    def _extract_row_key(chunk_id: str, metadata: Dict) -> str:
        """Derive a logical row key from chunk_id or metadata."""
        # chunk_id format: <source>_r<row>_c<col>  or  <source>_row_<row>
        parts = chunk_id.rsplit("_c", 1)
        return parts[0] if len(parts) == 2 else chunk_id

    def add(self, chunk_ids: List[str], texts: List[str], metadatas: List[Dict]) -> None:
        if not texts:
            return
        faiss = _load_faiss()
        embeddings = self._embed(texts)

        if self._index is None:
            self._dim = embeddings.shape[1]
            self._index = faiss.IndexFlatIP(self._dim)
            logger.info("Created CALI FAISS IndexFlatIP dim=%d", self._dim)

        self._index.add(embeddings)
        for cid, meta in zip(chunk_ids, metadatas):
            self._row_ids.append(self._extract_row_key(cid, meta))
        self._chunk_ids.extend(chunk_ids)
        self._texts.extend(texts)
        self._metadatas.extend(metadatas)
        logger.info("TabularStore: added %d cells (total=%d)", len(texts), len(self._texts))

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        MaxSim late-interaction search.
        1. Embed query.
        2. Retrieve top-K * 20 candidate cells.
        3. Group by logical row, take max score per cell group (MaxSim).
        4. Return top-K rows.
        """
        if self._index is None or self._index.ntotal == 0:
            return []

        faiss = _load_faiss()
        q_emb = self._embed([query])
        candidate_k = min(top_k * 20, self._index.ntotal)
        scores, indices = self._index.search(q_emb, candidate_k)

        # Group scores by row_key (MaxSim aggregation)
        row_scores: Dict[str, float] = {}
        row_representative: Dict[str, int] = {}   # row_key → cell idx of max score

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            row_key = self._row_ids[idx]
            if row_key not in row_scores or score > row_scores[row_key]:
                row_scores[row_key] = float(score)
                row_representative[row_key] = idx

        # Sort rows by aggregated MaxSim score
        sorted_rows = sorted(row_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for row_key, agg_score in sorted_rows:
            rep_idx = row_representative[row_key]
            # Collect all cells for this row to form a rich context
            row_cells = [
                self._texts[i]
                for i, rk in enumerate(self._row_ids)
                if rk == row_key
            ]
            combined_text = " | ".join(row_cells) if row_cells else self._texts[rep_idx]
            results.append(SearchResult(
                chunk_id=row_key,
                text=combined_text,
                score=agg_score,
                metadata=self._metadatas[rep_idx],
            ))
        return results

    def save(self, directory: str) -> None:
        faiss = _load_faiss()
        Path(directory).mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(Path(directory) / "tabular.index"))
        with open(Path(directory) / "tabular_meta.pkl", "wb") as f:
            pickle.dump({
                "chunk_ids": self._chunk_ids,
                "texts": self._texts,
                "metadatas": self._metadatas,
                "row_ids": self._row_ids,
                "dim": self._dim,
                "model_name": self.model_name,
            }, f)
        logger.info("TabularStore saved to %s", directory)

    def load(self, directory: str) -> None:
        faiss = _load_faiss()
        self._index = faiss.read_index(str(Path(directory) / "tabular.index"))
        with open(Path(directory) / "tabular_meta.pkl", "rb") as f:
            data = pickle.load(f)
        self._chunk_ids = data["chunk_ids"]
        self._texts = data["texts"]
        self._metadatas = data["metadatas"]
        self._row_ids = data["row_ids"]
        self._dim = data["dim"]
        logger.info("TabularStore loaded from %s (%d cells)", directory, self._index.ntotal)
        # Eager-load the sentence-transformer model now so the first real query
        # doesn't pay the 8-10 second PyTorch cold-start cost.
        logger.info("TabularStore: pre-warming CALI encoder model…")
        _ = self._embed(["warmup"])
        logger.info("TabularStore: CALI encoder ready.")

    @property
    def count(self) -> int:
        return self._index.ntotal if self._index else 0
