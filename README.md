# Topo-RAG: Topology-Aware RAG for Agricultural Price Discovery & Logistics

**MVP v1.0** — Dual-Path CALI + Bi-Encoder Pipeline with Deterministic Profit Engine

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
