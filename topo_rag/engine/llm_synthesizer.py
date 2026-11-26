"""
Ollama Local LLM Synthesizer
============================
Replaces the cloud Groq/Gemini synthesizer with a fully local Ollama-backed
model (krishi-mitra, based on qwen2.5:1.5b).

Key design decisions (preserved from original):
  - The deterministic profit calculation block is ALWAYS preserved verbatim.
    The LLM only generates the ADVISORY paragraph that comes after it.
  - Context chunks are cleaned before being sent to the model (strips
    [COL:][ROW:][VAL:] CALI encoding noise and markdown headers).
  - Multi-turn chat history is maintained for the lifetime of the session.
  - Falls back to a static template if Ollama is unreachable, so the app
    never crashes even if the local server is not running.

Model Stack
-----------
  Registered model : krishi-mitra  (custom Modelfile persona)
  Base model       : qwen2.5:1.5b  (~986 MB disk, ~1.0 GB RAM on M2)
  Server           : Ollama running at localhost:11434 (no internet needed)

Usage
-----
    synth = OllamaSynthesizer()
    response = synth.generate(query, context_chunks, profit_breakdown)

    # Kept for backward compat — same public API as the old GeminiSynthesizer
    synth = GeminiSynthesizer()   # alias to OllamaSynthesizer
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from typing import List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public constants (unchanged — app.py depends on BLOCK_SEPARATOR)
# ---------------------------------------------------------------------------

BLOCK_SEPARATOR = "<<<ADVISORY>>>"

# ---------------------------------------------------------------------------
# Internal defaults
# ---------------------------------------------------------------------------

_OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
_DEFAULT_MODEL:   str = os.getenv("KRISHI_MODEL", "krishi-mitra")
_FALLBACK_MODEL:  str = "qwen2.5:1.5b"   # used if krishi-mitra not yet registered

# Request timeout (seconds).  Local inference on M2 Pro is fast; 45 s is
# generous enough for worst-case cold-start.
_TIMEOUT: int = int(os.getenv("OLLAMA_TIMEOUT", "45"))


# ---------------------------------------------------------------------------
# Language & Intent Detection Helpers (Hindi Devanagari)
# ---------------------------------------------------------------------------

def _is_devanagari(text: str) -> bool:
    """Check if the text contains Devanagari script characters."""
    return bool(re.search(r'[\u0900-\u097F]', text))


def _is_hindi_translation_request(text: str) -> bool:
    """Check if the user is asking to translate into Hindi."""
    patterns = [
        r"translate\s+(?:it\s+)?(?:in|to|into)?\s*hindi",
        r"hindi\s*me(?:i?n)?\s*(?:batao|karo|bhejo|translate|likho|samjhao)",
        r"\bin\s+hindi\b",
        r"\binto\s+hindi\b",
        r"हिंदी\s*में\s*(?:अनुवाद|बताएं|बताओ|करो|लिखो|दीजिये|समझाओ)",
    ]
    return any(re.search(p, text.strip(), re.I) for p in patterns)


# ---------------------------------------------------------------------------
# Context cleaning  (identical logic to the original)
# ---------------------------------------------------------------------------

def _clean_context(chunks) -> str:
    """
    Strip CALI encoding noise from retrieved chunks so the model sees clean
    market knowledge instead of raw cell-encoding strings.
    """
    clean_parts = []
    for chunk in chunks:
        text = chunk.text
        text = re.sub(r"\[COL:[^\]]*\]", "", text)
        text = re.sub(r"\[ROW:[^\]]*\]", "", text)
        text = re.sub(r"\[VAL:[^\]]*\]", "", text)
        text = re.sub(r"^#{1,4}\s*", "", text, flags=re.MULTILINE)
        text = text.replace("…", "").strip()
        text = re.sub(r"\s+", " ", text)
        if text:
            clean_parts.append(text)
    return "\n\n".join(clean_parts[:3])   # top-3 chunks only, keeps prompt tight


# ---------------------------------------------------------------------------
# Low-level Ollama HTTP helpers  (no third-party libs required)
# ---------------------------------------------------------------------------

def _ollama_list_models(base_url: str) -> list[str]:
    """
    Return list of model names available in Ollama.

    Ollama always stores tags with an explicit ``:latest`` suffix
    (e.g. ``krishi-mitra:latest``).  We return both the full name AND a
    shorthand without the ``:latest`` suffix so callers can match either.
    """
    try:
        url = f"{base_url}/api/tags"
        req  = urllib.request.Request(url, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        names = []
        for m in data.get("models", []):
            full = m["name"]          # e.g. "krishi-mitra:latest"
            names.append(full)
            # also expose the bare name without :latest for convenient matching
            if full.endswith(":latest"):
                names.append(full[: -len(":latest")])
        return names
    except Exception:
        return []


def _ollama_chat(
    base_url:    str,
    model:       str,
    messages:    List[dict],
    max_tokens:  int = 420,
    temperature: float = 0.3,
) -> str:
    """
    Send a chat request to Ollama's /api/chat endpoint using stdlib only.
    Returns the full assistant content string (non-streaming).
    """
    payload = json.dumps({
        "model":      model,
        "messages":   messages,
        "stream":     False,
        # keep_alive: -1 keeps the model resident in Ollama memory indefinitely.
        # Without this, Ollama evicts the model after 5 minutes of inactivity,
        # causing a 3-4 second cold-reload on the next request.
        "keep_alive": -1,
        "options": {
            "temperature":    temperature,
            "num_predict":    max_tokens,
            "num_ctx":        4096,
            "repeat_penalty": 1.15,
        },
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{base_url}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        data = json.loads(resp.read())

    return data.get("message", {}).get("content", "").strip()


def _ollama_chat_stream(
    base_url:    str,
    model:       str,
    messages:    List[dict],
    max_tokens:  int = 420,
    temperature: float = 0.3,
):
    """
    Stream tokens from Ollama's /api/chat endpoint using stdlib only.
    Yields each token string as it arrives from the model.

    This is a generator — callers iterate over it to get incremental output.
    Identical output to _ollama_chat; only delivery mechanism differs.
    """
    payload = json.dumps({
        "model":      model,
        "messages":   messages,
        "stream":     True,   # <-- streaming enabled
        "keep_alive": -1,
        "options": {
            "temperature":    temperature,
            "num_predict":    max_tokens,
            "num_ctx":        4096,
            "repeat_penalty": 1.15,
        },
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{base_url}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        for raw_line in resp:
            line = raw_line.strip()
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            token = chunk.get("message", {}).get("content", "")
            if token:
                yield token
            if chunk.get("done", False):
                break


# ---------------------------------------------------------------------------
# Main synthesizer
# ---------------------------------------------------------------------------

class OllamaSynthesizer:
    """
    Local LLM synthesizer backed by Ollama + krishi-mitra (qwen2.5:1.5b).

    Drops in as a 1-to-1 replacement for the old GeminiSynthesizer:
    same constructor signature, same ``generate()`` / ``reset_chat()`` API.

    Parameters
    ----------
    model_name : str
        Ollama model tag to use. Defaults to ``krishi-mitra`` (the custom
        Modelfile persona).  Falls back to ``qwen2.5:1.5b`` automatically
        if the custom model has not been registered yet.
    base_url : str
        Ollama server URL.  Override via OLLAMA_BASE_URL env var.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        base_url:   Optional[str] = None,
        # ── Kept for API compatibility with old GeminiSynthesizer calls ──
        api_key:    Optional[str] = None,   # intentionally ignored
    ):
        self._base_url   = base_url or _OLLAMA_BASE_URL
        self._available  = False
        self._history:   List[dict] = []
        self._last_profit = None
        self._last_advisory: str = ""
        self._last_calc_block: str = ""

        # Resolve which model to actually use
        preferred = model_name or _DEFAULT_MODEL
        available = _ollama_list_models(self._base_url)

        if not available:
            logger.warning(
                "Ollama is not reachable at %s — falling back to static responses. "
                "Start it with: ollama serve",
                self._base_url,
            )
            self._model_name = preferred
            return

        if preferred in available:
            self._model_name = preferred
            self._available  = True
            logger.info("OllamaSynthesizer ready (model=%s)", self._model_name)
        elif _FALLBACK_MODEL in available:
            logger.warning(
                "Model '%s' not found in Ollama. Falling back to '%s'. "
                "Register the persona with: ollama create krishi-mitra -f KrishiMitra.Modelfile",
                preferred,
                _FALLBACK_MODEL,
            )
            self._model_name = _FALLBACK_MODEL
            self._available  = True
        else:
            logger.warning(
                "Neither '%s' nor '%s' found in Ollama. "
                "Pull the base model first: ollama pull qwen2.5:1.5b",
                preferred, _FALLBACK_MODEL,
            )
            self._model_name = preferred

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset_chat(self) -> None:
        """Clear multi-turn conversation history (call on new user session)."""
        self._history = []
        self._last_profit = None
        self._last_advisory = ""
        self._last_calc_block = ""
        logger.info("OllamaSynthesizer: chat history cleared")

    def generate(
        self,
        query:          str,
        context_chunks: list,
        profit=None,
    ) -> str:
        """
        Generate a response combining:
          1. The deterministic profit calculation block (verbatim, untouched)
          2. A locally-generated advisory paragraph from the Ollama model

        Returns the full response as a single string (non-streaming).
        Use generate_stream() for token-by-token delivery to the frontend.
        """
        if profit is not None:
            self._last_profit = profit

        is_hi = _is_devanagari(query)
        is_trans = _is_hindi_translation_request(query)

        if is_hi:
            calc_block = profit.summary(lang="hi") if profit else ""
        elif is_trans:
            if profit:
                calc_block = profit.summary(lang="hi")
            elif self._last_profit:
                calc_block = self._last_profit.summary(lang="hi")
            else:
                calc_block = ""
        else:
            calc_block = profit.summary(lang="en") if profit else ""

        if not self._available:
            return self._fallback(calc_block, "Ollama not reachable — run 'ollama serve'")

        try:
            return self._local_response(query, context_chunks, calc_block, profit=profit)
        except urllib.error.URLError as e:
            logger.error("Ollama connection error: %s", e)
            return self._fallback(calc_block, "Ollama server unreachable")
        except Exception as e:
            logger.error("OllamaSynthesizer error: %s", e)
            return self._fallback(calc_block, str(e))

    def generate_stream(
        self,
        query:          str,
        context_chunks: list,
        profit=None,
    ):
        """
        Streaming version of generate().

        Yields a sequence of SSE-formatted strings for Flask to forward to
        the browser as a Server-Sent Events stream.

        Protocol (each yield is one SSE ``data:`` line):
          - First event  : {"type": "calc",    "content": "<full calc block>"}
          - Middle events: {"type": "token",   "content": "<token>"}
          - Final event  : {"type": "done",    "content": ""}
        """
        import json as _json

        if profit is not None:
            self._last_profit = profit

        is_hi = _is_devanagari(query)
        is_trans = _is_hindi_translation_request(query)

        if is_hi:
            calc_block = profit.summary(lang="hi") if profit else ""
        elif is_trans:
            if profit:
                calc_block = profit.summary(lang="hi")
            elif self._last_profit:
                calc_block = self._last_profit.summary(lang="hi")
            else:
                calc_block = ""
        else:
            calc_block = profit.summary(lang="en") if profit else ""

        # Always emit the calc block immediately
        yield f"data: {_json.dumps({'type': 'calc', 'content': calc_block})}\n\n"

        if not self._available:
            fallback_msg = self._deterministic_advisory(query, profit or self._last_profit, is_hi or is_trans)
            yield f"data: {_json.dumps({'type': 'token', 'content': fallback_msg})}\n\n"
            yield f"data: {_json.dumps({'type': 'done',  'content': ''})}\n\n"
            return

        try:
            messages, history_user_msg = self._build_messages(
                query, context_chunks, calc_block, profit=profit
            )

            full_advisory = []
            for token in _ollama_chat_stream(
                base_url=self._base_url,
                model=self._model_name,
                messages=messages,
            ):
                full_advisory.append(token)
                yield f"data: {_json.dumps({'type': 'token', 'content': token})}\n\n"

            complete = "".join(full_advisory).strip()
            if not complete:
                # Guarantee non-empty advisory if local 1.5B model hits stop token immediately
                complete = self._deterministic_advisory(query, profit or self._last_profit, is_hi or is_trans)
                yield f"data: {_json.dumps({'type': 'token', 'content': complete})}\n\n"

            self._last_advisory = complete
            self._last_calc_block = calc_block
            self._record_history_turn(history_user_msg, complete, is_hi=is_hi or is_trans)

        except Exception as e:
            logger.error("OllamaSynthesizer stream error: %s", e)
            fallback_msg = self._deterministic_advisory(query, profit or self._last_profit, is_hi or is_trans)
            yield f"data: {_json.dumps({'type': 'token', 'content': fallback_msg})}\n\n"

        yield f"data: {_json.dumps({'type': 'done', 'content': ''})}\n\n"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compact_profit_line(profit, is_hi: bool) -> str:
        """Compact 1-line profit summary so 1.5B model does not echo huge box-art tables."""
        if not profit:
            return ""
        if is_hi:
            parts = [
                f"{profit.commodity} (ग्रेड {profit.grade})",
                f"{profit.quantity_kg:.0f} किग्रा",
                f"{profit.source_market} से {profit.destination_market} ({profit.distance_km:.0f} किमी)",
                f"खराबी {profit.wastage_rate * 100:.1f}%",
            ]
            if profit.govt_total_subsidy > 0:
                parts.append(f"सरकारी सब्सिडी ₹{profit.govt_total_subsidy:,.0f}")
            parts.append(f"शुद्ध लाभ ₹{profit.net_profit:,.0f}")
            return ", ".join(parts)

        parts = [
            f"{profit.commodity} (Grade {profit.grade})",
            f"{profit.quantity_kg:.0f} kg",
            f"{profit.source_market} to {profit.destination_market} ({profit.distance_km:.0f} km)",
            f"wastage {profit.wastage_rate * 100:.1f}%",
        ]
        if profit.govt_total_subsidy > 0:
            parts.append(f"govt subsidy ₹{profit.govt_total_subsidy:,.0f}")
        parts.append(f"net profit ₹{profit.net_profit:,.0f}")
        return ", ".join(parts)

    @staticmethod
    def _deterministic_advisory(query: str, profit, is_hi: bool) -> str:
        """Reliable fallback advisory if Ollama returns empty output or errors."""
        if not profit:
            if is_hi:
                return (
                    "वर्तमान मंडी जानकारी के आधार पर फसल बेचने से पहले अपने नजदीकी और बड़े मंडी के ताजा भाव की तुलना जरूर करें। "
                    "मंडी शुल्क, भाड़ा और रास्ते में होने वाले नुकसान को ध्यान में रखकर ही वाहन रवाना करें।"
                )
            return (
                "Compare current modal rates across nearby and destination mandis before dispatching your produce, "
                "and verify local APMC charges and freight availability with your commission agent."
            )

        if is_hi:
            if profit.net_profit <= 0:
                advice = (
                    "मौजूदा भाव और परिवहन खर्च के अनुसार इस मंडी में माल भेजना लाभदायक नहीं दिख रहा है। "
                    "बेहतर होगा कि आप पास की स्थानीय मंडी में भाव पता करें या यदि फसल रखने योग्य है तो कुछ समय इंतजार करें।"
                )
            elif profit.distance_km >= 400 and profit.wastage_rate >= 0.05:
                advice = (
                    "इस मार्ग पर शुद्ध लाभ सकारात्मक है, इसलिए माल भेजना फायदेमंद रहेगा। "
                    "दूरी लंबी होने से रास्ते में फसल खराब होने का जोखिम रहता है, इसलिए हवादार प्लास्टिक क्रेट्स में पैक करें और रात में सफर करें।"
                )
            else:
                advice = (
                    "इस मंडी में बिक्री से आपको अच्छा शुद्ध मुनाफा मिल रहा है, इसलिए सही ग्रेडिंग के साथ माल भेजना फायदेमंद रहेगा। "
                    "रवानगी से पहले अपने आढ़ती से आज का ताजा भाव जरूर पक्का कर लें।"
                )
            if profit.govt_total_subsidy > 0:
                advice += " सरकारी योजना की सब्सिडी का लाभ लेने के लिए मंडी और परिवहन की पक्की रसीदें व किसान पंजीकरण कागजात सुरक्षित रखें।"
            return advice

        if profit.net_profit <= 0:
            advice = (
                "At current modal prices and logistics costs, shipping on this route is loss-making. "
                "Check closer regional mandis or hold stock if the crop can be stored safely."
            )
        elif profit.distance_km >= 400 and profit.wastage_rate >= 0.05:
            advice = (
                "This route yields a positive net profit, making dispatch worthwhile despite transit costs. "
                "Because perishability increases over long distances, pack in ventilated crates and dispatch during cooler night hours."
            )
        else:
            advice = (
                "Transporting this lot yields a positive net margin at current market rates. "
                "Confirm today's arrival rate with your destination mandi agent before loading."
            )
        if profit.govt_total_subsidy > 0:
            advice += " Keep transport receipts and registration documents ready to claim the eligible government scheme subsidy."
        return advice

    def _record_history_turn(self, user_msg: str, assistant_msg: str, is_hi: bool) -> None:
        """Append completed turn cleanly without mixing Hindi/English history streams."""
        lang_tag = "hi" if is_hi else "en"
        if getattr(self, "_history_lang", lang_tag) != lang_tag:
            self._history = []
        self._history_lang = lang_tag
        self._history.append({"role": "user", "content": user_msg})
        self._history.append({"role": "assistant", "content": assistant_msg})
        if len(self._history) > 4:
            self._history = self._history[-4:]

    def _build_messages(
        self, query: str, context_chunks: list, calc_block: str, profit=None
    ) -> tuple:
        """
        Build the messages list for an Ollama chat request and return
        (messages, compact_history_user_msg).

        Note: Does NOT mutate self._history prematurely so interrupted streams
        never corrupt future turns.
        """
        clean_ctx = _clean_context(context_chunks)
        is_hi = _is_devanagari(query)
        is_trans = _is_hindi_translation_request(query)

        # ── Case 1: Translation Request (Rule 2) ──────────────────────
        if is_trans:
            last_advisory = self._last_advisory or ""
            if not last_advisory:
                for turn in reversed(self._history):
                    if turn.get("role") == "assistant":
                        last_advisory = turn.get("content", "")
                        break

            if last_advisory:
                user_msg = (
                    f"Translate into simple, natural Hindi (देवनागरी लिपि) in 3-4 sentences:\n"
                    f"\"{last_advisory}\""
                )
            else:
                user_msg = (
                    f"Question: {query}\n"
                    f"Answer in simple, natural Hindi (देवनागरी लिपि) in 3-4 sentences."
                )

            translation_system = [
                {
                    "role": "system",
                    "content": (
                        "You are Krishi Mitra (कृषि मित्र), an agricultural market advisor for Indian farmers. "
                        "Translate accurately into pure, natural Hindi (देवनागरी लिपि) in 3-4 sentences of plain prose. "
                        "Never output English words or mixed Marathi."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Translate into simple Hindi:\n"
                        "\"Azadpur mandi offers higher prices than local mandis. Transporting is profitable, "
                        "but tomatoes have a 12% wastage risk over 1400 km. Pack them in ventilated crates and confirm rates with your agent before dispatch.\""
                    ),
                },
                {
                    "role": "assistant",
                    "content": (
                        "आजादपुर मंडी में स्थानीय मंडियों की तुलना में टमाटर के बेहतर दाम मिल रहे हैं। "
                        "भाड़े के बाद भी माल भेजना फायदेमंद रहेगा, लेकिन 1400 किमी के सफर में टमाटर खराब होने का जोखिम रहता है। "
                        "इसलिए इन्हें हवादार प्लास्टिक क्रेट्स में ही पैक करें और गाड़ी रवाना करने से पहले अपने आढ़ती से आज का भाव जरूर पक्का कर लें।"
                    ),
                },
                {"role": "user", "content": user_msg},
            ]
            return translation_system, user_msg

        # ── Case 2: Hindi Devanagari Prompt (Rule 1) ─────────────────
        elif is_hi:
            compact_ctx = self._compact_profit_line(profit, is_hi=True)
            if compact_ctx:
                user_msg = f"Context: {compact_ctx}\nQuestion: {query}"
            elif clean_ctx:
                user_msg = f"Context: {clean_ctx[:400]}\nQuestion: {query}"
            else:
                user_msg = f"Question: {query}"

            prior_hist = self._history if getattr(self, "_history_lang", "hi") == "hi" else []
            hindi_system = [
                {
                    "role": "system",
                    "content": (
                        "You are Krishi Mitra (कृषि मित्र), an expert Indian agricultural market advisor for Indian farmers. "
                        "Respond strictly in pure, natural Hindi (देवनागरी लिपि) in 3-4 sentences of plain conversational prose. "
                        "Never use bullet points, numbered lists, or headings. "
                        "Never repeat prompt instructions or mix Marathi/English words."
                    ),
                },
                {
                    "role": "user",
                    "content": "Context: टमाटर (ग्रेड A), 500 किग्रा, पुणे से आजादपुर (1341 किमी), खराबी 11.0%, सरकारी सब्सिडी ₹2,916, शुद्ध लाभ ₹4,761\nQuestion: क्या मुझे अपने 500 किलो टमाटर आजादपुर मंडी भेजने चाहिए?",
                },
                {
                    "role": "assistant",
                    "content": (
                        "आजादपुर मंडी में इस समय टमाटर के भाव स्थानीय बाजार से काफी बेहतर मिल रहे हैं और सरकारी भाड़ा सब्सिडी से खर्च भी घट जाता है, इसलिए माल भेजना फायदेमंद रहेगा। "
                        "चूंकि पुणे से दिल्ली का सफर लंबा है और टमाटर जल्दी खराब होते हैं, इसलिए हवादार प्लास्टिक क्रेट्स में ही अच्छी तरह पैक करें। "
                        "माल रवाना करने से पहले आजादपुर में अपने आढ़ती से आज का ताजा भाव पक्का कर लें और सब्सिडी के लिए पक्की रसीदें संभाल कर रखें।"
                    ),
                },
            ]
            return hindi_system + prior_hist + [{"role": "user", "content": user_msg}], user_msg

        # ── Case 3: Standard English Prompt ──────────────────────────
        else:
            compact_ctx = self._compact_profit_line(profit, is_hi=False)
            if compact_ctx:
                user_msg = f"Context: {compact_ctx}\nQuestion: {query}"
            elif clean_ctx:
                user_msg = f"Context: {clean_ctx[:500]}\nQuestion: {query}"
            else:
                user_msg = f"Question: {query}"

            prior_hist = self._history if getattr(self, "_history_lang", "en") == "en" else []
            english_system = [
                {
                    "role": "system",
                    "content": (
                        "You are Krishi Mitra, an expert Indian agricultural market advisor. "
                        "Give short, practical advice in 3-4 sentences of plain prose. "
                        "Respond in English only. Never use bullet points or headings."
                    ),
                },
                {
                    "role": "user",
                    "content": "Context: Tomato (Grade A), 500 kg, Pune to Azadpur (1341 km), wastage 11.0%, govt subsidy ₹2,916, net profit ₹4,761\nQuestion: Should I ship these tomatoes to Azadpur?",
                },
                {
                    "role": "assistant",
                    "content": (
                        "Shipping to Azadpur is profitable right now because destination rates and the freight subsidy comfortably cover your logistics costs. "
                        "Since tomatoes perish quickly over a long haul, pack them in ventilated plastic crates and travel at night to reduce heat damage. "
                        "Confirm today's arrival rate with your mandi agent before dispatch and keep your transport receipts ready for the subsidy claim."
                    ),
                },
            ]
            return english_system + prior_hist + [{"role": "user", "content": user_msg}], user_msg

    def _local_response(
        self,
        query:          str,
        context_chunks: list,
        calc_block:     str,
        profit=None,
    ) -> str:
        """Build the message list and call Ollama (non-streaming)."""
        is_hi = _is_devanagari(query)
        is_trans = _is_hindi_translation_request(query)
        messages, history_user_msg = self._build_messages(
            query, context_chunks, calc_block, profit=profit
        )

        advisory = _ollama_chat(
            base_url=self._base_url,
            model=self._model_name,
            messages=messages,
        )
        if not advisory:
            advisory = self._deterministic_advisory(query, profit or self._last_profit, is_hi or is_trans)

        self._last_advisory = advisory
        self._last_calc_block = calc_block
        self._record_history_turn(history_user_msg, advisory, is_hi=is_hi or is_trans)

        return f"{calc_block}{BLOCK_SEPARATOR}{advisory}" if calc_block else advisory

    def _fallback(self, calc_block: str, reason: str) -> str:
        """Safe fallback when Ollama is unavailable."""
        advisory = (
            f"*(Local AI advisory unavailable: {reason}. "
            f"Please run `ollama serve` and ensure the krishi-mitra model is registered.)*"
        )
        return f"{calc_block}{BLOCK_SEPARATOR}{advisory}" if calc_block else advisory


# ---------------------------------------------------------------------------
# Backward-compatibility alias
# ---------------------------------------------------------------------------
# app.py and any other module import GeminiSynthesizer by name.
# This alias means zero changes are needed in any other file.

GeminiSynthesizer = OllamaSynthesizer
