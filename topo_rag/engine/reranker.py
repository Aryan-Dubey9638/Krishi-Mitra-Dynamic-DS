"""
Cross-Encoder Reranker
======================
Implements Min-Max normalized cross-encoder reranking using
sentence-transformers CrossEncoder (BGE-Reranker-v2-m3 or ms-marco fallback).

Architecture:
  1. Receive candidate SearchResult list from dual-path retrieval.
  2. Score (query, passage) pairs with CrossEncoder.
  3. Apply Min-Max normalization across batch scores.
  4. Return top-K results sorted by normalized score.
"""

import logging
from typing import List, Optional

import numpy as np

from topo_rag.stores.vector_stores import SearchResult

logger = logging.getLogger(__name__)

# Production model: BAAI/bge-reranker-v2-m3  (requires ~550MB, multilingual)
# Fast local alternative: cross-encoder/ms-marco-MiniLM-L-6-v2  (~80MB, English)
PRODUCTION_MODEL = "BAAI/bge-reranker-v2-m3"
FAST_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """
    Unified cross-encoder reranker with Min-Max normalization.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier for the cross-encoder.
    max_length : int
        Max token length for (query, passage) pair.
    use_fast : bool
        If True, automatically selects the lightweight ~80MB model
        instead of the production ~550MB model.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        max_length: int = 512,
        use_fast: bool = True,
    ):
        if model_name:
            self.model_name = model_name
        else:
            self.model_name = FAST_MODEL if use_fast else PRODUCTION_MODEL
        self.max_length = max_length
        self._model = None
        logger.info("CrossEncoderReranker initialised (model=%s)", self.model_name)
        # Eager-load the cross-encoder now so the first rerank() call is instant.
        # This moves the 3-4 second cost to startup rather than the first user query.
        try:
            _ = self.model
            logger.info("CrossEncoderReranker: model pre-loaded and ready.")
        except Exception as e:
            logger.warning("CrossEncoderReranker: pre-load failed (will retry on first use): %s", e)

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                logger.info("Loading cross-encoder: %s …", self.model_name)
                self._model = CrossEncoder(
                    self.model_name,
                    max_length=self.max_length,
                )
            except Exception as e:
                logger.error("CrossEncoder load failed: %s", e)
                raise
        return self._model

    @staticmethod
    def _minmax_normalize(scores: np.ndarray) -> np.ndarray:
        """
        Min-Max normalise scores to [0, 1].
        Handles edge case where all scores are identical.
        """
        mn, mx = scores.min(), scores.max()
        if mx - mn < 1e-9:
            return np.ones_like(scores) * 0.5
        return (scores - mn) / (mx - mn)

    def rerank(
        self,
        query: str,
        candidates: List[SearchResult],
        top_k: int = 5,
    ) -> List[SearchResult]:
        """
        Rerank candidates with cross-encoder and return top_k.

        Parameters
        ----------
        query      : The user query string.
        candidates : List of SearchResult from retrieval pipeline.
        top_k      : Number of top results to return.

        Returns
        -------
        Sorted list of SearchResult with updated .score (normalized).
        """
        if not candidates:
            return []

        pairs = [(query, r.text) for r in candidates]
        raw_scores = self.model.predict(pairs)                # np.ndarray
        raw_scores = np.array(raw_scores, dtype=np.float32)
        norm_scores = self._minmax_normalize(raw_scores)

        # Attach normalized scores back to results
        scored = []
        for result, norm_s, raw_s in zip(candidates, norm_scores, raw_scores):
            result.metadata["raw_reranker_score"] = float(raw_s)
            result.score = float(norm_s)
            scored.append(result)

        reranked = sorted(scored, key=lambda r: r.score, reverse=True)[:top_k]
        logger.info(
            "CrossEncoderReranker: %d → %d results (top score=%.4f)",
            len(candidates), len(reranked), reranked[0].score if reranked else 0.0,
        )
        return reranked
