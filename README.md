# Topo-RAG: Topology-Aware RAG for Agricultural Price Discovery & Logistics

**MVP v1.0** — Dual-Path CALI + Bi-Encoder Pipeline with Deterministic Profit Engine

---

## Architecture Overview

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  SDS Router  (Structural Density Score)                 │
│                                                         │
│  SDS(b) = (N_num + N_sep + N_ent) / N_total            │
│                                                         │
│  SDS > 0.4  →  Route B (Tabular / CALI)                │
│  SDS ≤ 0.4  →  Route A (Narrative / Bi-Encoder)        │
└─────────────┬───────────────────┬───────────────────────┘
              │                   │
     Route A  │                   │  Route B
              ▼                   ▼
   ┌─────────────────┐   ┌─────────────────────────┐
   │ Narrative Store │   │ Tabular Store (CALI)     │
   │ FAISS + BGE-M3  │   │ FAISS + MaxSim Aggregate │
   │ (Bi-encoder)    │   │ (Multi-vector late inter.)│
   └────────┬────────┘   └──────────┬──────────────┘
            │                       │
            └──────────┬────────────┘
                       │ Merged candidates
                       ▼
          ┌────────────────────────────┐
          │  Cross-Encoder Reranker    │
          │  BGE-Reranker-v2-m3        │
          │  Min-Max Score Normalization│
          └────────────┬───────────────┘
                       │ Top-K reranked results
                       ▼
          ┌────────────────────────────┐
          │  Deterministic Math Engine │
          │                           │
          │  Net Profit = (Y × P) -   │
          │  (Transport + Tax + Waste) │
          └────────────┬───────────────┘
                       │
                       ▼
          ┌────────────────────────────┐
          │  LLM Synthesis Module      │
          │  (Template / Gemini / GPT) │
          └────────────────────────────┘
```

---

## Project Structure

```
IRSW PBL/
├── main.py                          ← CLI entry point
├── requirements.txt
├── venv/                            ← Python virtual environment
└── topo_rag/
    ├── core/
    │   ├── sds_router.py            ← SDS calculation & routing (Route A/B)
    │   └── data_loader.py           ← Agmarknet CSV + Narrative loaders
    ├── stores/
    │   └── vector_stores.py         ← NarrativeStore (FAISS) + TabularStore (CALI)
    ├── engine/
    │   ├── reranker.py              ← CrossEncoder reranker + Min-Max normalization
    │   ├── profit_calculator.py     ← Deterministic Net Profit formula
    │   └── pipeline.py             ← Full pipeline orchestrator
    ├── utils/
    │   └── mock_data_generator.py   ← Synthetic Agmarknet CSV generator
    ├── data/
    │   ├── raw/                     ← Generated CSV files (Agmarknet schema)
    │   ├── processed/
    │   ├── narrative/               ← Knowledge base documents
    │   │   ├── apmc_regulations.md  ← APMC Act, fees, grading, transport rules
    │   │   └── quality_standards.md ← Export specs, contract farming, grading
    │   └── index/                   ← Saved FAISS indexes (created after build)
    └── tests/
        └── validate.py              ← Unit tests + integration demo
```

---

## Quick Start

### 1. Activate environment
```bash
source venv/bin/activate
```

### 2. Generate data & build vector index
```bash
python main.py build
```
Generates ~90-day synthetic Agmarknet CSV (12 markets × 8 commodities), loads 5,000 tabular cell chunks into the CALI store, and indexes narrative documents into the FAISS bi-encoder store.

### 3. Run interactive queries
```bash
python main.py query
```

### 4. Run validation suite
```bash
python main.py validate
```

### 5. Test SDS routing on any text
```bash
python main.py sds "Tomato | Azadpur | 2400 | Grade A | Maharashtra"
python main.py sds "The APMC Act governs agricultural markets across India"
```

---

## Core Modules

### SDS Router (`topo_rag/core/sds_router.py`)

Computes Structural Density Score for query/document routing:

```python
from topo_rag.core.sds_router import SDSRouter

router = SDSRouter(threshold=0.4)
result = router.compute_full("Tomato | Azadpur | 2400 | Grade A | 1200 | 3200")
# SDSResult(sds=0.524, route='B', n_num=3, n_sep=5, n_ent=2, n_total=19)
```

Token classification:
- **N_num**: Numeric tokens (prices, dates, percentages, ₹ values)
- **N_sep**: Separator tokens (`|`, `,`, `\t`, `;`)
- **N_ent**: Named entity tokens (capitalized / known market/commodity names)

### Dual Vector Stores (`topo_rag/stores/vector_stores.py`)

**Route A — NarrativeStore (Bi-Encoder):**
```python
store = NarrativeStore(model_name="all-MiniLM-L6-v2")
store.add(chunk_ids, texts, metadatas)
results = store.search("APMC commission rates Maharashtra", top_k=5)
```

**Route B — TabularStore (CALI / MaxSim):**
- Each table cell encoded: `[COL: COMMODITY] [ROW: 42] [VAL: Tomato]`
- MaxSim aggregation groups cell scores by logical row
- Returns richly contextualized row strings
```python
store = TabularStore(model_name="all-MiniLM-L6-v2")
results = store.search("Tomato modal price Azadpur Grade A", top_k=5)
```

### Profit Calculator (`topo_rag/engine/profit_calculator.py`)

```python
from topo_rag.engine.profit_calculator import ProfitCalculator

calc = ProfitCalculator()
result = calc.calculate(
    commodity="Tomato",
    quantity_kg=500.0,
    modal_price_per_qtl=2200.0,    # Retrieved from mandi data
    source_market="Pune",
    destination_market="Azadpur",
    grade="A",
    grade_premium_pct=5.0,
)
print(result.summary())
```

**Output:**
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Topo-RAG Net Profit Calculation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Commodity   : Tomato (Grade A)
  Quantity    : 500.0 kg (5.000 quintals)
  Route       : Pune → Azadpur
  Distance    : 1420 km
  Modal Price : ₹2,310.00/quintal

  GROSS REVENUE        ₹    9,817.50
  ─────────────────────────────────────────────
  Transport Cost       ₹    5,680.00
  APMC Commission      ₹      196.35
  Mandi Fees           ₹       75.00
  Wastage Loss         ₹    1,732.50  (15.0%)
  ─────────────────────────────────────────────
  TOTAL COSTS          ₹    7,683.85

  NET PROFIT           ₹    2,133.65
  Net Profit / kg      ₹        4.27
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Validated Test Results (15/15 passing)

| Test | Result |
|------|--------|
| SDS: Tabular text routes to B | ✅ PASS |
| SDS: Narrative text routes to A | ✅ PASS |
| SDS: Score in [0, 1] range | ✅ PASS |
| SDS: Empty string safe | ✅ PASS |
| SDS: Chunk partitioning correct | ✅ PASS |
| Profit: Formula exact match (manual) | ✅ PASS |
| Profit: Wastage reduces net profit | ✅ PASS |
| Profit: Longer distance reduces profit | ✅ PASS |
| Profit: Grade A > Grade B profit | ✅ PASS |
| Profit: Zero quantity safe | ✅ PASS |
| Profit: All breakdown fields ≥ 0 | ✅ PASS |
| DataLoader: Agmarknet cell encoding | ✅ PASS |
| DataLoader: Narrative chunking | ✅ PASS |
| Reranker: Min-Max normalization | ✅ PASS |
| Reranker: Uniform scores = 0.5 | ✅ PASS |

---

## Upgrading to Production

| Component | MVP | Production |
|-----------|-----|------------|
| Bi-encoder | `all-MiniLM-L6-v2` (22MB) | `BAAI/bge-m3` (1.1GB) |
| Reranker | `ms-marco-MiniLM-L-6-v2` | `BAAI/bge-reranker-v2-m3` |
| Vector store | FAISS CPU | Qdrant / Weaviate |
| LLM synthesis | Template | Gemini Pro / GPT-4o / LLaMA-3 |
| Data source | Synthetic CSV | Live Agmarknet API |

---

## Dataset Schema (Agmarknet)

| Column | Description | Example |
|--------|-------------|---------|
| State | Indian state | Maharashtra |
| District | District name | Pune |
| Market | Mandi name | Gultekdi |
| Commodity | Crop name | Tomato |
| Variety | Crop variety | Hybrid |
| Grade | Quality grade | Grade A |
| Arrival_Date | Trading date | 15/09/2024 |
| Min_Price | Minimum price (₹/q) | 1200 |
| Max_Price | Maximum price (₹/q) | 3200 |
| Modal_Price | Modal price (₹/q) | 2400 |
| Arrivals_Tonnes | Volume arrived | 145 |
