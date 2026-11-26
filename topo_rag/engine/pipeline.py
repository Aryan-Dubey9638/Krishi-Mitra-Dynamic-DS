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
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()

from topo_rag.core.sds_router import SDSRouter, SDSResult
from topo_rag.stores.vector_stores import NarrativeStore, TabularStore, SearchResult
from topo_rag.engine.reranker import CrossEncoderReranker
from topo_rag.engine.profit_calculator import ProfitCalculator, ProfitBreakdown

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Query Parsing helpers (English & Hindi Devanagari)
# ---------------------------------------------------------------------------

_DEV_NUM_MAP = str.maketrans("०१२३४५६७८९", "0123456789")

def _normalize_numerals(text: str) -> str:
    return text.translate(_DEV_NUM_MAP)

def _is_devanagari(text: str) -> bool:
    return bool(re.search(r'[\u0900-\u097F]', text))

_QUANTITY_RE = re.compile(
    r"(\d[\d,]*(?:\.\d+)?)\s*(?:kg|kilogram|kilo|quintal|q\b|किलो|किग्रा|किलोग्राम|क्विंटल|कुंतल|टन|tonne|ton)",
    re.I
)
_GRADE_RE = re.compile(r"grade[- ]?([abc])", re.I)

_CROP_MAPPING = {
    # English
    "tomato": "Tomato", "onion": "Onion", "potato": "Potato", "wheat": "Wheat",
    "rice": "Rice", "paddy": "Paddy", "maize": "Maize", "soybean": "Soybean",
    "cotton": "Cotton", "mango": "Mango", "banana": "Banana", "grapes": "Grapes",
    "garlic": "Garlic", "sugarcane": "Sugarcane", "mustard": "Mustard",
    "groundnut": "Groundnut", "gram": "Gram", "tur": "Tur", "arhar": "Tur",
    "moong": "Moong", "urad": "Urad", "sunflower": "Sunflower",
    # Hindi (Devanagari)
    "टमाटर": "Tomato", "तमाटर": "Tomato",
    "प्याज": "Onion", "प्याज़": "Onion", "कांदा": "Onion",
    "आलू": "Potato", "आलु": "Potato", "बटाटा": "Potato",
    "गेहूं": "Wheat", "गेंहू": "Wheat", "गेहुं": "Wheat", "कनक": "Wheat",
    "चावल": "Rice", "धान": "Paddy",
    "मक्का": "Maize", "भुट्टा": "Maize",
    "सोयाबीन": "Soybean", "सोयाबिन": "Soybean",
    "कपास": "Cotton", "रुई": "Cotton",
    "गन्ना": "Sugarcane", "ईख": "Sugarcane",
    "सरसों": "Mustard", "मूंगफली": "Groundnut", "चना": "Gram",
    "अरहर": "Tur", "तूर": "Tur", "मूंग": "Moong", "उड़द": "Urad",
    "आम": "Mango",
    "केला": "Banana",
    "अंगूर": "Grapes",
    "लहसुन": "Garlic", "लहसून": "Garlic",
    # Hinglish
    "tamatar": "Tomato", "pyaj": "Onion", "pyaz": "Onion", "kanda": "Onion",
    "aaloo": "Potato", "alu": "Potato", "gehun": "Wheat", "chawal": "Rice",
    "dhan": "Paddy", "makka": "Maize", "bhutta": "Maize", "soyabean": "Soybean",
    "kapas": "Cotton", "sarson": "Mustard", "moongfali": "Groundnut",
    "chana": "Gram", "aam": "Mango", "kela": "Banana", "angoor": "Grapes",
    "lahsun": "Garlic", "ganna": "Sugarcane",
}

_CITY_MAPPING = {
    # English
    "pune": "Pune", "mumbai": "Mumbai", "delhi": "Delhi", "nashik": "Nashik",
    "nasik": "Nashik", "indore": "Indore", "agra": "Agra", "lucknow": "Lucknow",
    "hyderabad": "Hyderabad", "bengaluru": "Bengaluru", "bangalore": "Bengaluru",
    "chennai": "Chennai", "kolkata": "Kolkata", "nagpur": "Nagpur",
    "azadpur": "Azadpur", "vashi": "Vashi", "gultekdi": "Gultekdi",
    "bowenpally": "Bowenpally", "koyambedu": "Koyambedu", "bhopal": "Bhopal",
    "jabalpur": "Jabalpur", "gwalior": "Gwalior", "raipur": "Raipur",
    "jaipur": "Jaipur", "jodhpur": "Jodhpur", "kota": "Kota", "udaipur": "Udaipur",
    "varanasi": "Varanasi", "kanpur": "Kanpur", "patna": "Patna",
    "lasalgaon": "Lasalgaon", "amritsar": "Amritsar", "ludhiana": "Ludhiana",
    "chandigarh": "Chandigarh", "ahmedabad": "Ahmedabad", "surat": "Surat",
    "vadodara": "Vadodara",
    # Hindi
    "पुणे": "Pune", "मुंबई": "Mumbai", "बम्बई": "Mumbai", "दिल्ली": "Delhi",
    "नई दिल्ली": "Delhi", "नासिक": "Nashik", "नाशिक": "Nashik", "इंदौर": "Indore",
    "आगरा": "Agra", "लखनऊ": "Lucknow", "हैदराबाद": "Hyderabad", "बेंगलुरु": "Bengaluru",
    "बैंगलोर": "Bengaluru", "चेन्नई": "Chennai", "कोलकाता": "Kolkata", "नागपुर": "Nagpur",
    "आजादपुर": "Azadpur", "वाशी": "Vashi", "गुलटेकड़ी": "Gultekdi", "लासलगांव": "Lasalgaon",
    "भोपाल": "Bhopal", "जबलपुर": "Jabalpur", "ग्वालियर": "Gwalior", "रायपुर": "Raipur",
    "जयपुर": "Jaipur", "जोधपुर": "Jodhpur", "कोटा": "Kota", "उदयपुर": "Udaipur",
    "वाराणसी": "Varanasi", "बनारस": "Varanasi", "कानपुर": "Kanpur", "पटना": "Patna",
    "अहमदाबाद": "Ahmedabad", "सूरत": "Surat", "अमृतसर": "Amritsar", "लुधियाना": "Ludhiana",
    "चंडीगढ़": "Chandigarh",
}


def _parse_quantity(query: str) -> Optional[float]:
    normalized_q = _normalize_numerals(query)
    match = _QUANTITY_RE.search(normalized_q)
    if not match:
        return None
    val_str = match.group(1).replace(",", "")
    val = float(val_str)
    unit = match.group(0).lower()
    if any(q in unit for q in ("quintal", "क्विंटल", "कुंतल")) or unit.endswith("q"):
        val *= 100   # quintals → kg
    elif any(t in unit for t in ("टन", "tonne", "ton")):
        val *= 1000  # tonnes → kg
    return val


def _parse_grade(query: str) -> str:
    match = _GRADE_RE.search(query)
    if match:
        return match.group(1).upper()
    hi_match = re.search(r"ग्रेड[- ]?([एबीसीabcABC])", query)
    if hi_match:
        g = hi_match.group(1).upper()
        if g in ('ए', 'A'): return "A"
        if g in ('बी', 'B'): return "B"
        if g in ('सी', 'C'): return "C"
    return "A"


def _parse_crop(query: str) -> Optional[str]:
    q_lower = query.lower()
    for crop_word, std_crop in _CROP_MAPPING.items():
        if crop_word in q_lower:
            return std_crop
    return None


def _parse_locations(query: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract up to 2 location mentions (English & Hindi).
    Tries:
      1. 'from <src> to <dst>'
      2. '<src> से <dst>' or '<src> se <dst>'
      3. Positional keyword matching across _CITY_MAPPING.
    """
    from_to_match = re.search(
        r"\bfrom\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)\s+to\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)",
        query,
        re.I,
    )
    if from_to_match:
        src_cand = from_to_match.group(1).strip().title()
        dst_cand = from_to_match.group(2).strip().title()
        trailing_stop = {
            "what", "is", "how", "which", "with", "the", "cost", "profit",
            "transport", "market", "mandi", "via", "by", "for", "under",
            "on", "in", "through", "using", "fpo",
        }
        
        dst_words = dst_cand.split()
        if len(dst_words) > 1 and dst_words[-1].lower() in trailing_stop:
            dst_cand = dst_words[0].title()

        src_words = src_cand.split()
        if len(src_words) > 1 and src_words[-1].lower() in trailing_stop:
            src_cand = src_words[0].title()

        stopwords = {"what", "how", "the", "net", "profit", "grade", "tomato", "onion", "potato", "wheat", "rice"}
        if src_cand.lower() not in stopwords and dst_cand.lower() not in stopwords and len(src_cand) >= 3 and len(dst_cand) >= 3:
            return src_cand, dst_cand

    # Match Hindi pattern: "<src> से <dst>" or "<src> se <dst>"
    se_match = re.search(
        r"([A-Za-z\u0900-\u097F]+)\s*(?:से|se)\s*([A-Za-z\u0900-\u097F]+)",
        query,
        re.I,
    )
    if se_match:
        c1 = se_match.group(1).strip().lower()
        c2 = se_match.group(2).strip().lower()
        src_mapped = _CITY_MAPPING.get(c1)
        dst_mapped = _CITY_MAPPING.get(c2)
        if src_mapped and dst_mapped:
            return src_mapped, dst_mapped
        if src_mapped and not dst_mapped:
            for k, v in _CITY_MAPPING.items():
                if c2.startswith(k):
                    return src_mapped, v
        if not src_mapped and dst_mapped:
            for k, v in _CITY_MAPPING.items():
                if c1.endswith(k):
                    return v, dst_mapped

    q_lower = query.lower()
    hits = []
    for city_key, city_val in _CITY_MAPPING.items():
        idx = q_lower.find(city_key)
        if idx != -1:
            hits.append((idx, city_val))
    hits.sort(key=lambda x: x[0])
    
    seen = set()
    cities = []
    for _, c in hits:
        if c not in seen:
            seen.add(c)
            cities.append(c)

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
            lines.append(profit.summary(lang="hi" if _is_devanagari(query) else "en"))
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

        # Cross-lingual retrieval helper: if Devanagari is used, augment query for FAISS/reranker with English keywords
        search_query = query
        if _is_devanagari(query):
            crop_cand = _parse_crop(query)
            src_cand, dst_cand = _parse_locations(query)
            keywords = [k for k in [crop_cand, src_cand, dst_cand] if k]
            if keywords:
                search_query = f"{' '.join(keywords)} {query}"

        # Step 1: SDS routing
        sds_result = self.sds_router.compute_full(search_query)
        logger.info("SDS=%.3f → Route %s", sds_result.sds, sds_result.route)

        # Step 2: Retrieval
        candidates, route_used = self._retrieve(search_query, sds_result)

        # Step 3: Reranking
        if use_reranker and candidates:
            try:
                reranked = self.reranker.rerank(search_query, candidates, top_k=self.top_k_rerank)
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

        # Cross-lingual retrieval helper: if Devanagari is used, augment query for FAISS/reranker with English keywords
        search_query = query
        if _is_devanagari(query):
            crop_cand = _parse_crop(query)
            src_cand, dst_cand = _parse_locations(query)
            keywords = [k for k in [crop_cand, src_cand, dst_cand] if k]
            if keywords:
                search_query = f"{' '.join(keywords)} {query}"

        # Step 1: SDS routing
        sds_result = self.sds_router.compute_full(search_query)
        logger.info("SDS=%.3f → Route %s", sds_result.sds, sds_result.route)

        # Step 2: Retrieval
        candidates, route_used = self._retrieve(search_query, sds_result)

        # Step 3: Reranking
        if use_reranker and candidates:
            try:
                reranked = self.reranker.rerank(search_query, candidates, top_k=self.top_k_rerank)
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

        # ------------------------------------------------------------------
        # Mandi Price Discovery Hierarchy:
        # Tier 1: Real-time Live Agmarknet OGD API (data.gov.in)
        # Tier 2: Offline TabularStore Vector Search (historical Agmarknet CSV)
        # Tier 3: Hardcoded Commodity Baseline Defaults
        # ------------------------------------------------------------------
        modal_price = None
        price_provenance_note = None
        price_provenance_note_hi = None
        dest_market = destination or "Azadpur"

        # Tier 1: Live Agmarknet API via data.gov.in
        if self.profit_calculator.use_dynamic and self.profit_calculator.resolver:
            try:
                _, _, dest_state = self.profit_calculator.resolver._geocode_city(dest_market)
                live_price_res = self.profit_calculator.resolver.resolve_mandi_price(
                    commodity=crop,
                    destination_market=dest_market,
                    destination_state=dest_state,
                )
                if live_price_res:
                    modal_price = live_price_res.modal_price
                    price_provenance_note = live_price_res.provenance_note
                    price_provenance_note_hi = live_price_res.provenance_note_hi
                    logger.info("Using LIVE Mandi Price ₹%.2f/q from data.gov.in (%s)", modal_price, live_price_res.market)
            except Exception as e:
                logger.warning("Live mandi price resolution failed: %s", e)

        # Tier 2: Offline TabularStore (CSV Index) Fallback
        if modal_price is None:
            for res in reranked:
                mp = res.metadata.get("modal_price", "")
                if mp:
                    try:
                        modal_price = float(str(mp).replace(",", "").replace("₹", "").strip())
                        price_provenance_note = f"Mandi Price: ₹{modal_price:,.2f}/qtl [Historical baseline from TabularStore (Agmarknet CSV)]"
                        price_provenance_note_hi = f"मंडी भाव: ₹{modal_price:,.2f}/क्विंटल [ऐतिहासिक एगमार्कनेट CSV रिकॉर्ड से]"
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
                        price_provenance_note = f"Mandi Price: ₹{modal_price:,.2f}/qtl [Historical baseline from TabularStore (Agmarknet CSV)]"
                        price_provenance_note_hi = f"मंडी भाव: ₹{modal_price:,.2f}/क्विंटल [ऐतिहासिक एगमार्कनेट CSV रिकॉर्ड से]"
                        logger.info("Retrieved modal price ₹%.2f/q for %s in %s from TabularStore", modal_price, crop, destination)
                        break
            except Exception as e:
                logger.debug("Tabular fallback search failed: %s", e)

        # Tier 3: Baseline Commodity Defaults
        if modal_price is None:
            defaults = {
                "Tomato": 2200.0, "Onion": 1800.0, "Potato": 1200.0,
                "Wheat": 2100.0, "Rice": 3000.0, "Paddy": 2200.0,
                "Maize": 1800.0, "Soybean": 4800.0, "Mustard": 5600.0,
                "Groundnut": 6100.0, "Gram": 5200.0, "Cotton": 7000.0,
            }
            modal_price = defaults.get(crop, 2000.0)
            price_provenance_note = f"Mandi Price: ₹{modal_price:,.2f}/qtl [Benchmark Default]"
            price_provenance_note_hi = f"मंडी भाव: ₹{modal_price:,.2f}/क्विंटल [मानक आधार मूल्य]"
            logger.info("No modal price from retrieval; using default ₹%.2f/q for %s", modal_price, crop)

        grade_premium = 5.0 if grade == "A" else (0.0 if grade == "B" else -5.0)
        q_lower = query.lower()
        is_fpo = any(
            kw in q_lower
            for kw in (
                "fpo", "fpc", "farmer producer", "direct marketing",
                "shetkari bazar", "kisan bazaar", "एफपीओ", "किसान उत्पादक", "सहकारी",
            )
        )
        if any(kw in q_lower for kw in ("kisan rail", "किसान रेल", "railway", "train")):
            transport_mode = "kisan_rail"
        elif any(kw in q_lower for kw in ("by road", "road only", "truck only")):
            transport_mode = "road"
        else:
            transport_mode = "auto"

        return self.profit_calculator.calculate(
            commodity=crop,
            quantity_kg=quantity_kg,
            modal_price_per_qtl=modal_price,
            source_market=source or "Pune",
            destination_market=destination or "Azadpur",
            grade=grade,
            grade_premium_pct=grade_premium,
            price_provenance_note=price_provenance_note,
            price_provenance_note_hi=price_provenance_note_hi,
            is_fpo_member=is_fpo,
            transport_mode=transport_mode,
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
