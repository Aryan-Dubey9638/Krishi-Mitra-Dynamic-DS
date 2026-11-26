"""
SDS Router Module
=================
Implements the Structural Density Score (SDS) router described in the
Topo-RAG architecture proposal.

    SDS(b) = (N_num + N_sep + N_ent) / N_total

If SDS > 0.4  →  Route B (Tabular / CALI path)
Else           →  Route A (Narrative / bi-encoder path)
"""

import re
import math
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns for token classification
# ---------------------------------------------------------------------------

_NUM_PATTERN = re.compile(
    r"""
    (?:
        \b[\d०-९]{1,3}(?:,[\d०-९]{3})*(?:\.[\d०-९]+)?\b   # 1,000 / 1,000.50 / १०००
        | \b[\d०-९]+(?:\.[\d०-९]+)?%?\b                    # 95.5 / 12% / ५०%
        | ₹\s*[\d०-९]+(?:\.[\d०-९]+)?                      # ₹2500.00
        | Rs\.?\s*[\d०-९]+(?:\.[\d०-९]+)?                  # Rs. 2500
        | [\d०-९]{1,2}[/-][\d०-९]{1,2}[/-][\d०-९]{2,4}    # dates 25/09/2024
    )
    """,
    re.VERBOSE,
)

_SEP_PATTERN = re.compile(r"[|,\t;]")

# Regex for capitalized tokens (NE heuristic — works without spaCy)
_ENT_PATTERN = re.compile(r"\b[A-Z][a-z]{2,}\b")

# Known Indian state / city / commodity names used to boost NE count (English & Hindi)
_KNOWN_ENTITIES = {
    "Tomato", "Onion", "Potato", "Wheat", "Rice", "Maize", "Soybean",
    "Cotton", "Sugarcane", "Mango", "Banana", "Grapes", "Garlic",
    "Punjab", "Haryana", "Maharashtra", "Karnataka", "Gujarat", "Bihar",
    "Delhi", "Mumbai", "Chennai", "Kolkata", "Hyderabad", "Bengaluru",
    "Nasik", "Pune", "Indore", "Nagpur", "Agra", "Lucknow",
    "Azadpur", "Vashi", "Gultekdi", "Bowenpally", "Koyambedu",
    "Grade", "APMC", "Mandi", "Quintal", "Metric",
    # Hindi Commodity & Location Entities
    "टमाटर", "प्याज", "प्याज़", "आलू", "गेहूं", "गेंहू", "चावल", "धान", "मक्का", "सोयाबीन",
    "कपास", "गन्ना", "आम", "केला", "अंगूर", "लहसुन",
    "पंजाब", "हरियाणा", "महाराष्ट्र", "कर्नाटक", "गुजरात", "बिहार",
    "दिल्ली", "मुंबई", "चेन्नई", "कोलकाता", "हैदराबाद", "बेंगलुरु",
    "नासिक", "पुणे", "इंदौर", "नागपुर", "आगरा", "लखनऊ",
    "आजादपुर", "वाशी", "गुलटेकड़ी", "मंडी", "क्विंटल", "किलो", "किग्रा", "मुनाफा", "भाड़ा",
}


@dataclass
class SDSResult:
    """Full breakdown of an SDS computation."""
    text: str
    n_num: int = 0
    n_sep: int = 0
    n_ent: int = 0
    n_total: int = 1
    sds: float = 0.0
    route: str = "A"
    threshold: float = 0.4

    @property
    def is_tabular(self) -> bool:
        return self.route == "B"

    def __repr__(self) -> str:
        return (
            f"SDSResult(sds={self.sds:.3f}, route={self.route}, "
            f"n_num={self.n_num}, n_sep={self.n_sep}, n_ent={self.n_ent}, "
            f"n_total={self.n_total})"
        )


class SDSRouter:
    """
    Structural Density Score router.

    Parameters
    ----------
    threshold : float
        SDS value above which a chunk is routed to Route B.
        Default 0.4 per the Topo-RAG proposal.
    window_size : int
        Sliding window size in characters for segmented SDS computation.
        Set to 0 (default) to evaluate the whole block at once.
    """

    def __init__(self, threshold: float = 0.4, window_size: int = 0):
        self.threshold = threshold
        self.window_size = window_size

    # ------------------------------------------------------------------
    # Token-level helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Whitespace-level tokenize; preserve separators as standalone tokens."""
        # Split on whitespace but also split out pipe/tab/comma chars
        tokens = re.split(r"(\s+|[|,\t;])", text)
        return [t for t in tokens if t.strip()]

    @staticmethod
    def _count_numeric_tokens(tokens: List[str]) -> int:
        return sum(1 for t in tokens if _NUM_PATTERN.fullmatch(t) or _NUM_PATTERN.search(t))

    @staticmethod
    def _count_separator_tokens(tokens: List[str]) -> int:
        return sum(1 for t in tokens if _SEP_PATTERN.fullmatch(t))

    @staticmethod
    def _count_entity_tokens(tokens: List[str]) -> int:
        count = 0
        for t in tokens:
            if t in _KNOWN_ENTITIES:
                count += 1
            elif _ENT_PATTERN.fullmatch(t):
                count += 1
        return count

    # ------------------------------------------------------------------
    # Core SDS computation
    # ------------------------------------------------------------------

    def calculate_sds(self, text_block: str) -> float:
        """
        Compute SDS for a single text block.

            SDS(b) = (N_num + N_sep + N_ent) / N_total

        Returns
        -------
        float in [0.0, 1.0]
        """
        if not text_block or not text_block.strip():
            return 0.0

        tokens = self._tokenize(text_block)
        n_total = max(len(tokens), 1)
        n_num = self._count_numeric_tokens(tokens)
        n_sep = self._count_separator_tokens(tokens)
        n_ent = self._count_entity_tokens(tokens)
        sds = (n_num + n_sep + n_ent) / n_total
        return min(sds, 1.0)  # Clamp to [0,1]

    def compute_full(self, text_block: str) -> SDSResult:
        """Compute SDS and return full breakdown + routing decision."""
        tokens = self._tokenize(text_block) if text_block.strip() else []
        n_total = max(len(tokens), 1)
        n_num = self._count_numeric_tokens(tokens)
        n_sep = self._count_separator_tokens(tokens)
        n_ent = self._count_entity_tokens(tokens)
        sds = min((n_num + n_sep + n_ent) / n_total, 1.0)
        route = "B" if sds > self.threshold else "A"

        return SDSResult(
            text=text_block,
            n_num=n_num,
            n_sep=n_sep,
            n_ent=n_ent,
            n_total=n_total,
            sds=sds,
            route=route,
            threshold=self.threshold,
        )

    def route(self, text_block: str) -> str:
        """Return 'A' (Narrative) or 'B' (Tabular)."""
        return "B" if self.calculate_sds(text_block) > self.threshold else "A"

    def route_chunks(self, chunks: List[str]) -> Tuple[List[str], List[str]]:
        """
        Partition a list of text chunks into Route A and Route B lists.

        Returns
        -------
        (narrative_chunks, tabular_chunks)
        """
        narrative: List[str] = []
        tabular: List[str] = []
        for chunk in chunks:
            if self.route(chunk) == "A":
                narrative.append(chunk)
            else:
                tabular.append(chunk)
        logger.info(
            "SDSRouter partitioned %d chunks → %d narrative / %d tabular",
            len(chunks), len(narrative), len(tabular),
        )
        return narrative, tabular

    # ------------------------------------------------------------------
    # Sliding-window variant
    # ------------------------------------------------------------------

    def calculate_sds_windowed(self, text_block: str) -> float:
        """
        Compute SDS using a sliding window over characters.
        Returns the MAXIMUM window SDS (conservative / sensitive).
        """
        if self.window_size <= 0 or len(text_block) <= self.window_size:
            return self.calculate_sds(text_block)

        scores = []
        for start in range(0, len(text_block), self.window_size // 2):
            window = text_block[start: start + self.window_size]
            scores.append(self.calculate_sds(window))
        return max(scores) if scores else 0.0
