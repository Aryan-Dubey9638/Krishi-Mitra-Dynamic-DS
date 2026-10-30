"""
Topo-RAG Pipeline Orchestrator
================================
Ties all components together:
  1. SDSRouter         → Routes query to Route A or B (or both)
  2. NarrativeStore    → Retrieves narrative context (Route A)
  3. TabularStore      → Retrieves mandi price rows (Route B / CALI)
  4. CrossEncoderReranker → Unified reranking of merged candidates
  5. ProfitCalculator  → Deterministic math on retrieved prices
  6. LLM Synthesizer   → Optional LLM-based response generation

Usage
-----
    pipeline = TopoRAGPipeline()
    pipeline.index(tabular_chunks, narrative_chunks)
    result = pipeline.query("What is net profit for 500kg Grade-A Tomatoes from Pune to Azadpur?")
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from topo_rag.core.sds_router import SDSRouter, SDSResult
from topo_rag.stores.vector_stores import NarrativeStore, TabularStore, SearchResult
from topo_rag.engine.reranker import CrossEncoderReranker
from topo_rag.engine.profit_calculator import ProfitCalculator, ProfitBreakdown

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Query Parsing helpers
# ---------------------------------------------------------------------------

_QUANTITY_RE = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(?:kg|kilogram|kilo|quintal|q\b)", re.I)
_GRADE_RE = re.compile(r"grade[- ]?([abc])", re.I)
_CROP_KEYWORDS = [
    "tomato", "onion", "potato", "wheat", "rice", "maize", "soybean",
    "cotton", "mango", "banana", "grapes", "garlic", "sugarcane",
]
_CITY_KEYWORDS = [
    "pune", "mumbai", "delhi", "nashik", "indore", "agra", "lucknow",
    "hyderabad", "bengaluru", "bangalore", "chennai", "kolkata", "nagpur",
    "azadpur", "vashi", "gultekdi", "bowenpally", "koyambedu",
    # Central India
    "bhopal", "jabalpur", "gwalior", "raipur", "bilaspur",
    # Rajasthan
    "jaipur", "jodhpur", "kota", "udaipur", "ajmer",
    # UP / Bihar
    "varanasi", "kanpur", "allahabad", "prayagraj", "patna", "muzaffarpur",
    "gorakhpur", "meerut", "ghaziabad", "noida",
    # Punjab / Haryana
    "amritsar", "ludhiana", "chandigarh", "hisar", "karnal",
    # Gujarat
    "ahmedabad", "surat", "vadodara", "rajkot",
    # Maharashtra (more)
    "aurangabad", "solapur", "kolhapur", "sangli", "satara", "lasalgaon",
    # Andhra / Telangana
    "vijayawada", "visakhapatnam", "warangal", "tirupati",
    # Karnataka
    "mysuru", "hubli", "mangalore", "belagavi",
    # Tamil Nadu / Kerala
    "coimbatore", "madurai", "tiruchirappalli", "thiruvananthapuram", "kochi",
    # Odisha / WB
    "bhubaneswar", "cuttack", "siliguri", "asansol",
    # Himachal / J&K
    "shimla", "srinagar", "jammu",
]


def _parse_quantity(query: str) -> Optional[float]:
    match = _QUANTITY_RE.search(query)
    if not match:
        return None
    val_str = match.group(1).replace(",", "")
    val = float(val_str)
    unit = match.group(0).lower()
    if "quintal" in unit or unit.startswith("q"):
        val *= 100   # quintals → kg
    return val


def _parse_grade(query: str) -> str:
    match = _GRADE_RE.search(query)
    return match.group(1).upper() if match else "A"


def _parse_crop(query: str) -> Optional[str]:
    q_lower = query.lower()
    for crop in _CROP_KEYWORDS:
        if crop in q_lower:
            return crop.title()
    return None


def _parse_locations(query: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract up to 2 location mentions in the ORDER they appear in the query,
    so "from Bhopal to Pune" correctly gives (Bhopal, Pune) not (Pune, Bhopal).
    """
    q_lower = query.lower()
    # Collect (position_in_query, city_title) pairs for every city found
    hits = []
    for city in _CITY_KEYWORDS:
        idx = q_lower.find(city)
        if idx != -1:
            hits.append((idx, city.title()))
    # Sort by position in the original query string
    hits.sort(key=lambda x: x[0])
    cities = [h[1] for h in hits]

    if len(cities) >= 2:
        return cities[0], cities[1]
    elif len(cities) == 1:
        return cities[0], "Azadpur"   # sensible default destination mandi
    return None, None


# ---------------------------------------------------------------------------
# Pipeline Response
# ---------------------------------------------------------------------------

@dataclass
class PipelineResponse:
    query: str
    route_used: str                          # 'A', 'B', or 'AB'
    sds_score: float
    retrieved_results: List[SearchResult]
    reranked_results: List[SearchResult]
    context_text: str                        # merged context for LLM
    profit_breakdown: Optional[ProfitBreakdown] = None
    llm_response: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# LLM Synthesizer (template-based when no LLM API configured)
# ---------------------------------------------------------------------------

class TemplateSynthesizer:
    """
    Generates a final advisory response from reranked context +
    deterministic profit data. Falls back to template-based generation
    when no external LLM API is configured.

    To plug in a real LLM (Gemini/GPT-4/Ollama/LLaMA), subclass this
    and override the `generate()` method.
    """

    def generate(
        self,
        query: str,
        context_chunks: List[SearchResult],
        profit: Optional[ProfitBreakdown] = None,
    ) -> str:
        lines = [
            "═" * 55,
            "  Topo-RAG Agricultural Advisory Response",
            "═" * 55,
            f"  Query: {query}",
            "",
        ]

        if profit:
            lines.append(profit.summary())
            lines.append("")

        if context_chunks:
            lines.append("  Relevant Context Retrieved:")
            lines.append("  " + "─" * 50)
            for i, res in enumerate(context_chunks[:3], 1):
                snippet = res.text[:200].replace("\n", " ")
                lines.append(f"  [{i}] (score={res.score:.3f}) {snippet} …")
                lines.append("")

        lines.append(
            "  Advisory: Based on the retrieved mandi data and supply-chain costs,\n"
            "  the calculations above represent deterministic estimates. For precise\n"
            "  results, validate modal prices against live Agmarknet data."
        )
        lines.append("═" * 55)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

class TopoRAGPipeline:
    """
    Full Topo-RAG pipeline orchestrator.

    Parameters
    ----------
    narrative_model  : Sentence-transformer model for Route A.
    tabular_model    : Sentence-transformer model for Route B.
    reranker_model   : CrossEncoder model for reranking.
    sds_threshold    : SDS threshold for routing (default 0.4).
    top_k_retrieval  : Number of candidates per store.
    top_k_rerank     : Final top-K after reranking.
    use_fast_reranker: Use lightweight reranker model.
    synthesizer      : Optional custom synthesizer (uses TemplateSynthesizer by default).
    """

    def __init__(
        self,
        narrative_model: str = "all-MiniLM-L6-v2",
        tabular_model: str = "all-MiniLM-L6-v2",
        reranker_model: Optional[str] = None,
        sds_threshold: float = 0.4,
        top_k_retrieval: int = 10,
        top_k_rerank: int = 5,
        use_fast_reranker: bool = True,
        synthesizer: Optional[Any] = None,
    ):
        self.sds_router = SDSRouter(threshold=sds_threshold)
        self.narrative_store = NarrativeStore(model_name=narrative_model)
        self.tabular_store = TabularStore(model_name=tabular_model)
        self.reranker = CrossEncoderReranker(
            model_name=reranker_model,
            use_fast=use_fast_reranker,
        )
        self.profit_calculator = ProfitCalculator()
        self.synthesizer = synthesizer or TemplateSynthesizer()
        self.top_k_retrieval = top_k_retrieval
        self.top_k_rerank = top_k_rerank
        logger.info("TopoRAGPipeline initialised (sds_threshold=%.2f)", sds_threshold)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index(
        self,
        tabular_chunks=None,
        narrative_chunks=None,
    ) -> Dict[str, int]:
        """
        Index chunks into the appropriate stores.

        Parameters
        ----------
        tabular_chunks  : List[Chunk] for Route B (cell-level)
        narrative_chunks: List[Chunk] for Route A

        Returns
        -------
        dict with counts: {'narrative': N, 'tabular': M}
        """
        counts = {"narrative": 0, "tabular": 0}

        if narrative_chunks:
            ids = [c.chunk_id for c in narrative_chunks]
            texts = [c.text for c in narrative_chunks]
            metas = [c.metadata for c in narrative_chunks]
            self.narrative_store.add(ids, texts, metas)
            counts["narrative"] = len(narrative_chunks)
            logger.info("Indexed %d narrative chunks", len(narrative_chunks))

        if tabular_chunks:
            ids = [c.chunk_id for c in tabular_chunks]
            texts = [c.text for c in tabular_chunks]
            metas = [c.metadata for c in tabular_chunks]
            self.tabular_store.add(ids, texts, metas)
            counts["tabular"] = len(tabular_chunks)
            logger.info("Indexed %d tabular chunks", len(tabular_chunks))

        return counts

    def index_with_sds_routing(self, all_chunks) -> Dict[str, int]:
        """
        Auto-route chunks via SDS and index to appropriate stores.
        Useful when chunks are mixed (e.g., from a single document).
        """
        narrative_chunks, tabular_chunks = self.sds_router.route_chunks(
            [c.text for c in all_chunks]
        )
        n_chunks = [c for c in all_chunks if c.text in set(narrative_chunks)]
        t_chunks = [c for c in all_chunks if c.text in set(tabular_chunks)]
        return self.index(tabular_chunks=t_chunks, narrative_chunks=n_chunks)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def _retrieve(self, query: str, sds_result: SDSResult) -> Tuple[List[SearchResult], str]:
        """
        Execute dual-path retrieval based on SDS routing.
        Returns (merged_candidates, route_label)
        """
        candidates: List[SearchResult] = []
        route = "A"

        if sds_result.is_tabular:
            # Route B: Tabular CALI retrieval
            tab_results = self.tabular_store.search(query, top_k=self.top_k_retrieval)
            candidates.extend(tab_results)
            route = "B"
            logger.info("Route B: retrieved %d tabular results", len(tab_results))

            # Also fetch narrative context for qualitative grounding
            if self.narrative_store.count > 0:
                nar_results = self.narrative_store.search(query, top_k=self.top_k_retrieval // 2)
                candidates.extend(nar_results)
                route = "AB"
        else:
            # Route A: Narrative bi-encoder retrieval
            nar_results = self.narrative_store.search(query, top_k=self.top_k_retrieval)
            candidates.extend(nar_results)
            route = "A"
            logger.info("Route A: retrieved %d narrative results", len(nar_results))

            # Supplement with tabular if available
            if self.tabular_store.count > 0:
                tab_results = self.tabular_store.search(query, top_k=self.top_k_retrieval // 2)
                candidates.extend(tab_results)
                if tab_results:
                    route = "AB"

        return candidates, route

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        query: str,
        calculate_profit: bool = True,
        use_reranker: bool = True,
    ) -> PipelineResponse:
        """
        Full Topo-RAG query execution.

        Steps:
          1. Compute SDS on the query text → routing decision.
          2. Dual-path retrieval.
          3. Cross-encoder reranking.
          4. (Optional) Deterministic profit calculation.
          5. LLM/template synthesis.

        Returns
        -------
        PipelineResponse with all intermediate outputs.
        """
        logger.info("Pipeline query: %r", query)

        # Step 1: SDS routing
        sds_result = self.sds_router.compute_full(query)
        logger.info("SDS=%.3f → Route %s", sds_result.sds, sds_result.route)

        # Step 2: Retrieval
        candidates, route_used = self._retrieve(query, sds_result)

        # Step 3: Reranking
        if use_reranker and candidates:
            try:
                reranked = self.reranker.rerank(query, candidates, top_k=self.top_k_rerank)
            except Exception as e:
                logger.warning("Reranker failed, using retrieval order: %s", e)
                reranked = candidates[:self.top_k_rerank]
        else:
            reranked = candidates[:self.top_k_rerank]

        # Build merged context text
        context_text = "\n\n---\n\n".join(r.text for r in reranked)

        # Step 4: Profit calculation (if query is profit-related)
        profit_breakdown: Optional[ProfitBreakdown] = None
        if calculate_profit:
            profit_breakdown = self._try_profit_calculation(query, reranked)

        # Step 5: Synthesis
        llm_response = self.synthesizer.generate(query, reranked, profit_breakdown)

        return PipelineResponse(
            query=query,
            route_used=route_used,
            sds_score=sds_result.sds,
            retrieved_results=candidates,
            reranked_results=reranked,
            context_text=context_text,
            profit_breakdown=profit_breakdown,
            llm_response=llm_response,
        )

    def query_for_stream(
        self,
        query: str,
        calculate_profit: bool = True,
        use_reranker: bool = True,
    ) -> PipelineResponse:
        """
        Runs steps 1–4 of the pipeline (SDS → retrieval → reranking → profit)
        WITHOUT calling the synthesizer (step 5).

        Used by the streaming endpoint in app.py, which calls
        synthesizer.generate_stream() separately so LLM tokens can be forwarded
        to the browser as they arrive rather than buffering the full response.

        Returns a PipelineResponse with llm_response set to "" (empty string).
        """
        logger.info("Pipeline query_for_stream: %r", query)

        # Step 1: SDS routing
        sds_result = self.sds_router.compute_full(query)
        logger.info("SDS=%.3f → Route %s", sds_result.sds, sds_result.route)

        # Step 2: Retrieval
        candidates, route_used = self._retrieve(query, sds_result)

        # Step 3: Reranking
        if use_reranker and candidates:
            try:
                reranked = self.reranker.rerank(query, candidates, top_k=self.top_k_rerank)
            except Exception as e:
                logger.warning("Reranker failed, using retrieval order: %s", e)
                reranked = candidates[:self.top_k_rerank]
        else:
            reranked = candidates[:self.top_k_rerank]

        # Build merged context text
        context_text = "\n\n---\n\n".join(r.text for r in reranked)

        # Step 4: Profit calculation
        profit_breakdown: Optional[ProfitBreakdown] = None
        if calculate_profit:
            profit_breakdown = self._try_profit_calculation(query, reranked)

        # Step 5 intentionally skipped — caller streams it
        return PipelineResponse(
            query=query,
            route_used=route_used,
            sds_score=sds_result.sds,
            retrieved_results=candidates,
            reranked_results=reranked,
            context_text=context_text,
            profit_breakdown=profit_breakdown,
            llm_response="",
        )

    def _try_profit_calculation(
        self,
        query: str,
        reranked: List[SearchResult],
    ) -> Optional[ProfitBreakdown]:
        """
        Attempt to extract query parameters and compute profit.
        Returns None if query doesn't look like a profit query.
        """
        crop = _parse_crop(query)
        quantity_kg = _parse_quantity(query)
        grade = _parse_grade(query)
        source, destination = _parse_locations(query)

        if not crop or not quantity_kg:
            logger.debug("Profit calc skipped: crop=%s, qty=%s", crop, quantity_kg)
            return None

        # Retrieve modal price from top tabular result
        modal_price = None
        for res in reranked:
            mp = res.metadata.get("modal_price", "")
            if mp:
                try:
                    modal_price = float(str(mp).replace(",", "").replace("₹", "").strip())
                    logger.info("Using modal price ₹%.2f/q from retrieved data", modal_price)
                    break
                except (ValueError, TypeError):
                    pass

        # If reranked was mostly narrative and didn't contain modal_price, search tabular store for destination + crop
        if modal_price is None and destination and crop and hasattr(self, 'tabular_store') and self.tabular_store.count > 0:
            try:
                dest_results = self.tabular_store.search(f"{destination} {crop}", top_k=3)
                for res in dest_results:
                    mp = res.metadata.get("modal_price", "")
                    if mp:
                        modal_price = float(str(mp).replace(",", "").replace("₹", "").strip())
                        logger.info("Retrieved modal price ₹%.2f/q for %s in %s from TabularStore", modal_price, crop, destination)
                        break
            except Exception as e:
                logger.debug("Tabular fallback search failed: %s", e)

        if modal_price is None:
            # Use sensible defaults per commodity
            defaults = {
                "Tomato": 2200.0, "Onion": 1800.0, "Potato": 1200.0,
                "Wheat": 2100.0, "Rice": 3000.0, "Maize": 1800.0,
            }
            modal_price = defaults.get(crop, 2000.0)
            logger.info("No modal price from retrieval; using default ₹%.2f/q for %s", modal_price, crop)

        grade_premium = 5.0 if grade == "A" else (0.0 if grade == "B" else -5.0)

        return self.profit_calculator.calculate(
            commodity=crop,
            quantity_kg=quantity_kg,
            modal_price_per_qtl=modal_price,
            source_market=source or "Pune",
            destination_market=destination or "Azadpur",
            grade=grade,
            grade_premium_pct=grade_premium,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str) -> None:
        """Save both stores to disk."""
        import os
        self.narrative_store.save(os.path.join(directory, "narrative"))
        self.tabular_store.save(os.path.join(directory, "tabular"))
        logger.info("Pipeline saved to %s", directory)

    def load(self, directory: str) -> None:
        """Load both stores from disk."""
        import os
        self.narrative_store.load(os.path.join(directory, "narrative"))
        self.tabular_store.load(os.path.join(directory, "tabular"))
        logger.info("Pipeline loaded from %s", directory)
