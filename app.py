from flask import Flask, render_template, request, jsonify, session, Response, stream_with_context
from flask_cors import CORS
from pathlib import Path
from dotenv import load_dotenv
import logging
import os
import secrets
import threading

load_dotenv()  # Load env vars from .env file

from topo_rag.engine.pipeline import TopoRAGPipeline
from topo_rag.engine.llm_synthesizer import GeminiSynthesizer, BLOCK_SEPARATOR

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Initialize pipeline with local Ollama synthesizer
# ---------------------------------------------------------------------------
INDEX_DIR = Path(__file__).parent / "topo_rag" / "data" / "index"

synthesizer = GeminiSynthesizer(api_key=os.getenv("GROQ_API_KEY"))
pipeline = TopoRAGPipeline(use_fast_reranker=True, synthesizer=synthesizer)

# _pipeline_ready gates user requests until the background warmup finishes.
_pipeline_ready = threading.Event()

try:
    if (INDEX_DIR / "narrative" / "narrative.index").exists():
        logger.info(f"Loading index from {INDEX_DIR}...")
        pipeline.load(str(INDEX_DIR))
        # NOTE: pipeline.load() now eager-loads SentenceTransformer models
        # inside NarrativeStore.load() and TabularStore.load().
        logger.info("Index loaded successfully.")
    else:
        logger.warning("No index found. Run 'python main.py build' first.")
        _pipeline_ready.set()
except Exception as e:
    logger.error(f"Error loading index: {e}")
    _pipeline_ready.set()


# ---------------------------------------------------------------------------
# Background warmup thread
# ---------------------------------------------------------------------------

def _warmup_pipeline() -> None:
    """
    Run a single dummy query at startup to:
      1. Verify all models (SentenceTransformer, CrossEncoder) loaded correctly.
      2. Force Ollama to load krishi-mitra into Metal/GPU memory with keep_alive=-1.
      3. Set _pipeline_ready so all subsequent requests are instant (~100ms retrieval
         + ~2s Ollama generation, streamed token-by-token to the user).

    Runs concurrently — Flask serves requests immediately; the first user
    query that arrives during warmup gets a friendly "warming up" message.
    """
    logger.info("Pipeline warmup started in background thread…")
    try:
        _ = pipeline.query(
            "warmup tomato 100kg pune to azadpur",
            calculate_profit=False,
            use_reranker=True,
        )
        logger.info("Pipeline warmup complete — all models hot and ready.")
    except Exception as e:
        logger.warning("Warmup query failed (non-fatal): %s", e)
    finally:
        _pipeline_ready.set()


threading.Thread(target=_warmup_pipeline, daemon=True, name="pipeline-warmup").start()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/api/ready', methods=['GET'])
def ready():
    """Health-check endpoint the frontend can poll to know when warmup is done."""
    return jsonify({"ready": _pipeline_ready.is_set()})


@app.route('/api/chat/stream', methods=['POST'])
def chat_stream():
    """
    Streaming chat endpoint using Server-Sent Events (SSE).

    The pipeline runs fully (retrieval → reranking → profit calc) then
    streams the LLM advisory token-by-token to the browser.  The calc block
    is emitted as the very first event so the user sees numbers immediately,
    before a single LLM token has been generated.

    SSE event format:
        data: {"type": "calc",  "content": "<full profit block>"}
        data: {"type": "token", "content": "<word or token>"}
        ...
        data: {"type": "done",  "content": ""}
    """
    data = request.json
    query = data.get('query', '').strip()

    if not query:
        return jsonify({"error": "Empty query"}), 400

    if not _pipeline_ready.is_set():
        def warming_up():
            import json
            yield f"data: {json.dumps({'type': 'calc',  'content': ''})}\n\n"
            yield f"data: {json.dumps({'type': 'token', 'content': 'Krishi Mitra is still warming up its models — this only happens once after server restart. Please try again in a few seconds.'})}\n\n"
            yield f"data: {json.dumps({'type': 'done',  'content': ''})}\n\n"
        return Response(stream_with_context(warming_up()), mimetype='text/event-stream')

    def event_stream():
        try:
            # Run steps 1-4 only (SDS -> retrieval -> reranking -> profit calc).
            # Synthesis (step 5) is skipped so LLM tokens can stream live.
            resp = pipeline.query_for_stream(query, calculate_profit=True, use_reranker=True)

            # Hand off to the synthesizer's streaming generator which:
            #   1. Immediately yields the calc block as event type "calc"
            #   2. Streams each token from Ollama as event type "token"
            #   3. Emits a final event type "done"
            yield from synthesizer.generate_stream(
                query=query,
                context_chunks=resp.reranked_results,
                profit=resp.profit_breakdown,
            )
        except Exception as e:
            import json
            logger.error(f"Streaming error: {e}")
            yield f"data: {json.dumps({'type': 'token', 'content': f'Error: {e}'})}\n\n"
            yield f"data: {json.dumps({'type': 'done',  'content': ''})}\n\n"

    return Response(
        stream_with_context(event_stream()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',   # disables nginx proxy buffering
        },
    )


@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Non-streaming fallback endpoint (kept for backward compat / CLI clients).
    The browser UI uses /api/chat/stream instead.
    """
    data = request.json
    query = data.get('query', '').strip()

    if not query:
        return jsonify({"error": "Empty query"}), 400

    if not _pipeline_ready.is_set():
        return jsonify({
            "calculation": "",
            "advisory": (
                "Krishi Mitra is still warming up its models — "
                "this only happens once after server restart. "
                "Please try again in a few seconds."
            ),
            "route_used": "warmup",
            "sds_score": 0.0,
        }), 200

    try:
        resp = pipeline.query(query, calculate_profit=True, use_reranker=True)
        full_response = resp.llm_response

        if BLOCK_SEPARATOR in full_response:
            parts = full_response.split(BLOCK_SEPARATOR, 1)
            calculation = parts[0].strip()
            advisory = parts[1].strip()
        else:
            calculation = ""
            advisory = full_response.strip()

        return jsonify({
            "calculation": calculation,
            "advisory": advisory,
            "route_used": resp.route_used,
            "sds_score": resp.sds_score
        })
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/reset', methods=['POST'])
def reset_conversation():
    """Reset chat history to start a fresh conversation context."""
    try:
        if hasattr(synthesizer, 'reset_chat'):
            synthesizer.reset_chat()
        return jsonify({"status": "ok", "message": "Conversation history cleared."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)
