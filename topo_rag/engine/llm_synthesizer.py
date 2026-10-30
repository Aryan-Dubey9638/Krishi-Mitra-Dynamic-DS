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
    max_tokens:  int = 280,
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
            "num_ctx":        2048,
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
    max_tokens:  int = 280,
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
            "num_ctx":        2048,
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
        calc_block = profit.summary() if profit else ""

        if not self._available:
            return self._fallback(calc_block, "Ollama not reachable — run 'ollama serve'")

        try:
            return self._local_response(query, context_chunks, calc_block)
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

        The frontend accumulates "token" events into the advisory text area,
        so the user sees words appearing in real-time rather than waiting for
        the full response.  Total output is identical to generate().
        """
        import json as _json

        calc_block = profit.summary() if profit else ""

        # Always emit the calc block immediately — it's deterministic and ready
        # before the LLM even starts.  The user sees numbers in < 200 ms.
        yield f"data: {_json.dumps({'type': 'calc', 'content': calc_block})}\n\n"

        if not self._available:
            err_msg = "*(Local AI advisory unavailable: Ollama not reachable — run 'ollama serve'.)*"
            yield f"data: {_json.dumps({'type': 'token', 'content': err_msg})}\n\n"
            yield f"data: {_json.dumps({'type': 'done',  'content': ''})}\n\n"
            return

        try:
            messages = self._build_messages(query, context_chunks, calc_block)

            full_advisory = []
            for token in _ollama_chat_stream(
                base_url=self._base_url,
                model=self._model_name,
                messages=messages,
            ):
                full_advisory.append(token)
                yield f"data: {_json.dumps({'type': 'token', 'content': token})}\n\n"

            # Store the complete advisory in history for multi-turn context
            complete = "".join(full_advisory).strip()
            self._history.append({"role": "assistant", "content": complete})

        except Exception as e:
            logger.error("OllamaSynthesizer stream error: %s", e)
            err_msg = f"*(Advisory generation error: {e})*"
            yield f"data: {_json.dumps({'type': 'token', 'content': err_msg})}\n\n"

        yield f"data: {_json.dumps({'type': 'done', 'content': ''})}\n\n"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_messages(self, query: str, context_chunks: list, calc_block: str) -> List[dict]:
        """
        Build the messages list for an Ollama chat request and append the
        new user turn to the rolling conversation history.

        Shared by both generate() (non-streaming) and generate_stream().
        """
        clean_ctx = _clean_context(context_chunks)

        if calc_block:
            user_msg = (
                f"The farmer asked: {query}\n\n"
                f"I have already shown them this profit calculation:\n"
                f"{calc_block}\n\n"
                f"Additional market knowledge retrieved from the database:\n{clean_ctx}\n\n"
                f"Give a short advisory (3-5 sentences). "
                f"Do NOT repeat any numbers from the calculation block above."
            )
        else:
            user_msg = (
                f"The farmer asked: {query}\n\n"
                f"Relevant market knowledge:\n{clean_ctx}\n\n"
                f"Answer conversationally in 3-5 sentences."
            )

        self._history.append({"role": "user", "content": user_msg})

        # Keep history bounded to last 6 turns (3 Q&A pairs)
        if len(self._history) > 6:
            self._history = self._history[-6:]

        # For the krishi-mitra Modelfile the system prompt is baked in.
        # For the raw qwen2.5:1.5b fallback, inject a lightweight system msg.
        if self._model_name == _FALLBACK_MODEL:
            return [
                {
                    "role":    "system",
                    "content": (
                        "You are Krishi Mitra, an expert Indian agricultural market "
                        "advisor. Give short, practical advice (3-5 sentences). "
                        "Never repeat numbers already shown in the profit block."
                    ),
                }
            ] + self._history
        return self._history

    def _local_response(
        self,
        query:          str,
        context_chunks: list,
        calc_block:     str,
    ) -> str:
        """Build the message list and call Ollama (non-streaming)."""
        messages = self._build_messages(query, context_chunks, calc_block)

        advisory = _ollama_chat(
            base_url=self._base_url,
            model=self._model_name,
            messages=messages,
        )

        # Store assistant reply in history for next turn
        self._history.append({"role": "assistant", "content": advisory})

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
