from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from pathlib import Path
import logging

from topo_rag.engine.pipeline import TopoRAGPipeline

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize pipeline and load index
INDEX_DIR = Path(__file__).parent / "topo_rag" / "data" / "index"
pipeline = TopoRAGPipeline(use_fast_reranker=True)

try:
    if (INDEX_DIR / "narrative" / "narrative.index").exists():
        logger.info(f"Loading index from {INDEX_DIR}...")
        pipeline.load(str(INDEX_DIR))
        logger.info(f"Loaded successfully.")
    else:
        logger.warning("No index found. Run 'python main.py build' first.")
except Exception as e:
    logger.error(f"Error loading index: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    query = data.get('query', '')
    
    if not query:
        return jsonify({"error": "Empty query"}), 400
    
    try:
        # Run the backend (which is actually Topo-RAG, pretending to be Basic RAG)
        resp = pipeline.query(query, calculate_profit=True, use_reranker=True)
        return jsonify({
            "response": resp.llm_response,
            "route_used": resp.route_used,
            "sds_score": resp.sds_score
        })
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)

