"""
Topo-RAG Main Entry Point
=========================
CLI for building and querying the Topo-RAG pipeline.

Usage:
    python main.py build          # Generate data + build index
    python main.py query          # Interactive query loop
    python main.py validate       # Run full validation suite
    python main.py demo           # Run hardcoded demo queries
    python main.py sds <text>     # Test SDS router on a text snippet
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("topo_rag.main")

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "topo_rag" / "data"
INDEX_DIR = BASE_DIR / "topo_rag" / "data" / "index"


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

BANNER = r"""
  ████████╗ ██████╗ ██████╗  ██████╗       ██████╗  █████╗  ██████╗
     ██╔══╝██╔═══██╗██╔══██╗██╔═══██╗      ██╔══██╗██╔══██╗██╔════╝
     ██║   ██║   ██║██████╔╝██║   ██║█████╗██████╔╝███████║██║  ███╗
     ██║   ██║   ██║██╔═══╝ ██║   ██║╚════╝██╔══██╗██╔══██║██║   ██║
     ██║   ╚██████╔╝██║     ╚██████╔╝      ██║  ██║██║  ██║╚██████╔╝
     ╚═╝    ╚═════╝ ╚═╝      ╚═════╝       ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝

     Topology-Aware RAG for Agricultural Price Discovery & Logistics
     MVP v1.0  |  Dual-Path CALI + Bi-Encoder  |  SDS Router
"""


# ---------------------------------------------------------------------------
# Build command
# ---------------------------------------------------------------------------

def cmd_build(args):
    """Generate mock data → load chunks → index into both stores."""
    from topo_rag.utils.mock_data_generator import (
        generate_agmarknet_csv,
        generate_transport_matrix_csv,
    )
    from topo_rag.core.data_loader import AgmarknetLoader, NarrativeLoader
    from topo_rag.engine.pipeline import TopoRAGPipeline

    print(BANNER)
    print("=" * 60)
    print("  Phase 1: Data Generation")
    print("=" * 60)

    raw_dir = DATA_DIR / "raw"
    real_csv = raw_dir / "agmarknet_sample.csv"
    
    if real_csv.exists():
        print(f"  Found real dataset: {real_csv}")
        csv_path = str(real_csv)
    else:
        print("  Generating synthetic dataset as fallback...")
        csv_path = generate_agmarknet_csv(str(raw_dir), num_days=90)
        
    generate_transport_matrix_csv(str(raw_dir))

    print("\n" + "=" * 60)
    print("  Phase 2: Loading & Chunking")
    print("=" * 60)

    # Load tabular (cell-level for Route B)
    tab_loader = AgmarknetLoader(cell_mode=True)
    tabular_chunks = tab_loader.load_csv(csv_path)
    print(f"  Loaded {len(tabular_chunks):,} cell-level chunks (Route B)")

    # Load narrative documents (Route A)
    nar_loader = NarrativeLoader(chunk_size=512, chunk_overlap=64)
    narrative_dir = DATA_DIR / "narrative"
    narrative_chunks = nar_loader.load_directory(str(narrative_dir))
    print(f"  Loaded {len(narrative_chunks):,} narrative chunks (Route A)")

    # Limit tabular chunks for fast MVP (first 5000 cells)
    if len(tabular_chunks) > 5000:
        print(f"  Limiting to 5,000 tabular cells for fast MVP indexing …")
        tabular_chunks = tabular_chunks[:5000]

    print("\n" + "=" * 60)
    print("  Phase 3: Building Vector Index")
    print("=" * 60)

    pipeline = TopoRAGPipeline(use_fast_reranker=True)
    t0 = time.time()
    counts = pipeline.index(
        tabular_chunks=tabular_chunks,
        narrative_chunks=narrative_chunks,
    )
    elapsed = time.time() - t0
    print(f"  Indexed {counts['tabular']:,} tabular + {counts['narrative']:,} narrative chunks")
    print(f"  Indexing time: {elapsed:.1f}s")

    print("\n  Saving index to disk …")
    pipeline.save(str(INDEX_DIR))
    print(f"  ✓ Index saved to {INDEX_DIR}")
    print("\n  Build complete! Run:  python main.py query")


# ---------------------------------------------------------------------------
# Query command
# ---------------------------------------------------------------------------

def cmd_query(args):
    """Interactive query loop."""
    from topo_rag.engine.pipeline import TopoRAGPipeline

    print(BANNER)

    # Try loading existing index
    pipeline = TopoRAGPipeline(use_fast_reranker=True)
    index_path = INDEX_DIR
    if (index_path / "narrative" / "narrative.index").exists():
        print(f"  Loading index from {index_path} …")
        pipeline.load(str(index_path))
        print(f"  ✓ Loaded (narrative={pipeline.narrative_store.count}, tabular={pipeline.tabular_store.count})")
    else:
        print("  ⚠ No index found. Running without retrieval (profit calc only).")
        print("    Run 'python main.py build' first to build the index.\n")

    print("\n  Type your query (or 'quit' to exit, 'help' for examples):\n")

    HELP_EXAMPLES = [
        "What is my net profit for 500kg Grade-A Tomatoes from Pune to Azadpur?",
        "Calculate profit for 200kg Grade-B Onion from Nashik to Mumbai.",
        "Tell me about APMC commission fees and mandi charges.",
        "What are the quality grading standards for export tomatoes?",
        "Transport cost for 1000kg Potato from Indore to Delhi?",
    ]

    while True:
        try:
            query = input("\n  Query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("  Goodbye!")
            break
        if query.lower() == "help":
            print("\n  Example queries:")
            for ex in HELP_EXAMPLES:
                print(f"    • {ex}")
            continue

        t0 = time.time()
        resp = pipeline.query(query, calculate_profit=True, use_reranker=True)
        elapsed = time.time() - t0

        print(f"\n  Route: {resp.route_used} | SDS: {resp.sds_score:.3f} | Time: {elapsed:.2f}s")
        print(resp.llm_response)


# ---------------------------------------------------------------------------
# Demo command
# ---------------------------------------------------------------------------

def cmd_demo(args):
    """Run hardcoded demo queries with full results."""
    from topo_rag.tests.validate import run_integration_demo
    print(BANNER)
    run_integration_demo()


# ---------------------------------------------------------------------------
# Validate command
# ---------------------------------------------------------------------------

def cmd_validate(args):
    """Run full unit + integration test suite."""
    from topo_rag.tests.validate import main as validate_main
    validate_main()


# ---------------------------------------------------------------------------
# SDS command
# ---------------------------------------------------------------------------

def cmd_sds(args):
    """Test SDS router on a text snippet."""
    from topo_rag.core.sds_router import SDSRouter
    text = " ".join(args.text)
    router = SDSRouter(threshold=0.4)
    result = router.compute_full(text)
    print(f"\n  Text    : {text[:120]}")
    print(f"  N_num   : {result.n_num}")
    print(f"  N_sep   : {result.n_sep}")
    print(f"  N_ent   : {result.n_ent}")
    print(f"  N_total : {result.n_total}")
    print(f"  SDS     : {result.sds:.4f}  (threshold=0.4)")
    print(f"  Route   : {'B (Tabular)' if result.is_tabular else 'A (Narrative)'}")


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Topo-RAG Agricultural Advisory Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  build       Generate mock data and build vector index
  query       Interactive query loop
  demo        Run demonstration queries
  validate    Run unit and integration tests
  sds <text>  Compute SDS score for a text snippet
        """,
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("build", help="Build the pipeline index")
    subparsers.add_parser("query", help="Interactive query loop")
    subparsers.add_parser("demo", help="Run demo queries")
    subparsers.add_parser("validate", help="Run validation tests")
    sds_parser = subparsers.add_parser("sds", help="Test SDS routing on text")
    sds_parser.add_argument("text", nargs="+", help="Text to evaluate")

    args = parser.parse_args()

    if args.command == "build":
        cmd_build(args)
    elif args.command == "query":
        cmd_query(args)
    elif args.command == "demo":
        cmd_demo(args)
    elif args.command == "validate":
        cmd_validate(args)
    elif args.command == "sds":
        cmd_sds(args)
    else:
        print(BANNER)
        parser.print_help()


if __name__ == "__main__":
    main()
