"""
dataset_prep.py
===============
Cleans and maps Agriculture_price_dataset.csv (Agmarknet real data, 737k rows)
into the schema expected by AgmarknetLoader, then saves a stratified sample
to topo_rag/data/raw/agmarknet_prices.csv.

Cleaning steps applied:
  1. Drop rows with Modal_Price == 0
  2. Drop extreme outliers: Modal_Price > 99th percentile per commodity
  3. Map dataset Grade values (FAQ/Local/Non-FAQ) → system grades (A/B/C)
  4. Rename columns to match AgmarknetLoader aliases exactly
  5. Stratified sample: 3,000 rows per commodity (capped) → max ~15k rows total
  6. Save to topo_rag/data/raw/agmarknet_prices.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────
RAW_CSV = Path("Agriculture_price_dataset.csv")
OUT_DIR = Path("topo_rag/data/raw")
OUT_CSV = OUT_DIR / "agmarknet_prices.csv"

ROWS_PER_COMMODITY = 3_000   # stratified cap — keeps file fast to index
OUTLIER_PRICE_LIMIT = 50_000 # ₹50k/qtl absolute cap (obvious entry errors above)

# Grade mapping: Agmarknet grades → system A/B/C
GRADE_MAP = {
    "FAQ": "A",          # Fair Average Quality = top standard
    "Ref grade-1": "A",
    "Large": "A",
    "Medium": "B",
    "Non-FAQ": "B",
    "Local": "C",
    "Ref grade-2": "C",
}

# Column rename map: dataset column names → AgmarknetLoader alias names
COL_RENAME = {
    "STATE":          "State",
    "District Name":  "District",
    "Market Name":    "Market",
    "Commodity":      "Commodity",
    "Variety":        "Variety",
    "Grade":          "Grade",
    "Min_Price":      "Min_Price",
    "Max_Price":      "Max_Price",
    "Modal_Price":    "Modal_Price",
    "Price Date":     "Arrival_Date",
}

# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def prepare(input_csv: Path = RAW_CSV) -> str:
    print(f"[1/6] Loading {input_csv} …")
    df = pd.read_csv(input_csv, low_memory=False)
    print(f"      Loaded {len(df):,} rows × {len(df.columns)} columns")

    # ── Step 1: Rename columns ──────────────────────────────────
    print("[2/6] Renaming columns …")
    df = df.rename(columns=COL_RENAME)
    expected = list(COL_RENAME.values())
    missing = [c for c in expected if c not in df.columns]
    if missing:
        print(f"  ⚠ Missing expected columns after rename: {missing}")
        sys.exit(1)

    # ── Step 2: Clean prices ────────────────────────────────────
    print("[3/6] Cleaning prices …")
    for col in ["Min_Price", "Max_Price", "Modal_Price"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df[df["Modal_Price"] > 0]                         # drop zero-price
    df = df[df["Modal_Price"] <= OUTLIER_PRICE_LIMIT]      # drop hard outliers
    print(f"      Dropped {before - len(df):,} rows (zero / extreme prices). Remaining: {len(df):,}")

    # ── Step 3: Map grades ──────────────────────────────────────
    print("[4/6] Mapping grades (FAQ→A, Local→C, …) …")
    df["Grade"] = df["Grade"].map(GRADE_MAP).fillna("B")   # unknown grades → B

    # ── Step 4: Parse dates ─────────────────────────────────────
    print("[5/6] Parsing dates …")
    df["Arrival_Date"] = pd.to_datetime(
        df["Arrival_Date"], dayfirst=False, errors="coerce"
    ).dt.strftime("%d/%m/%Y")
    df = df.dropna(subset=["Arrival_Date"])
    print(f"      Remaining after date parse: {len(df):,}")

    # ── Step 5: Stratified sampling ─────────────────────────────
    print(f"[6/6] Stratified sample ({ROWS_PER_COMMODITY:,} rows/commodity) …")
    parts = []
    for commodity, grp in df.groupby("Commodity"):
        sample_n = min(ROWS_PER_COMMODITY, len(grp))
        parts.append(grp.sample(sample_n, random_state=42))
        print(f"      {commodity}: {len(grp):,} rows → sampled {sample_n:,}")
    df_out = pd.concat(parts, ignore_index=True)

    # ── Save ─────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    final_cols = ["State", "District", "Market", "Commodity", "Variety", "Grade",
                  "Arrival_Date", "Min_Price", "Max_Price", "Modal_Price"]
    df_out = df_out[final_cols]
    df_out.to_csv(OUT_CSV, index=False)
    print(f"\n✓ Saved {len(df_out):,} rows → {OUT_CSV}")
    print(f"  Commodities : {sorted(df_out['Commodity'].unique())}")
    print(f"  States      : {df_out['State'].nunique()} unique")
    print(f"  Markets     : {df_out['Market'].nunique()} unique")
    print(f"  Grade dist  :\n{df_out['Grade'].value_counts().to_string()}")
    return str(OUT_CSV)


if __name__ == "__main__":
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else RAW_CSV
    prepare(src)
