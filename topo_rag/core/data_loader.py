"""
Data Loader & Chunker
=====================
Handles loading of:
  - Agmarknet-schema CSV files (daily mandi prices)
  - Narrative text/markdown documents (APMC policy, quality standards)

Generates text chunks tagged with metadata for downstream routing.
"""

import csv
import json
import logging
import os
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A text chunk with provenance metadata."""
    chunk_id: str
    text: str
    source: str              # filename / table name
    chunk_type: str          # 'narrative' | 'cell' | 'row' | 'header'
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Agmarknet CSV Loader
# ---------------------------------------------------------------------------

AGMARKNET_COLUMNS = {
    "state": ["State", "state", "STATE"],
    "district": ["District", "district", "DISTRICT"],
    "market": ["Market", "market", "MARKET", "Mandi", "mandi"],
    "commodity": ["Commodity", "commodity", "COMMODITY", "Crop", "crop"],
    "variety": ["Variety", "variety", "VARIETY"],
    "arrival_date": ["Arrival_Date", "arrival_date", "ArrivalDate", "Date", "date"],
    "min_price": ["Min_x0020_Price", "Min Price", "min_price", "MinPrice", "Min"],
    "max_price": ["Max_x0020_Price", "Max Price", "max_price", "MaxPrice", "Max"],
    "modal_price": ["Modal_x0020_Price", "Modal Price", "modal_price", "ModalPrice", "Modal"],
}


def _resolve_column(df: pd.DataFrame, aliases: List[str]) -> Optional[str]:
    """Return the first matching column alias found in df."""
    for alias in aliases:
        if alias in df.columns:
            return alias
    return None


def _build_column_map(df: pd.DataFrame) -> Dict[str, str]:
    """Build canonical → actual column name mapping."""
    col_map = {}
    for canonical, aliases in AGMARKNET_COLUMNS.items():
        actual = _resolve_column(df, aliases)
        if actual:
            col_map[canonical] = actual
    return col_map


class AgmarknetLoader:
    """
    Load Agmarknet-style CSVs and produce Chunk objects for both
    cell-level and row-level representations.

    Cell encoding (Route B ready):
        "[COL: <header>] [VAL: <value>]"

    Row encoding (compact row-level context):
        "<Market> | <Commodity> | <Date> | Min:₹<x> Max:₹<y> Modal:₹<z>"
    """

    def __init__(self, cell_mode: bool = True):
        """
        Parameters
        ----------
        cell_mode : bool
            If True, generate one Chunk per cell (positional CALI encoding).
            If False, generate one Chunk per row (compact representation).
        """
        self.cell_mode = cell_mode

    def load_csv(self, filepath: str) -> List[Chunk]:
        """Load a single CSV file and return all chunks."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {filepath}")

        logger.info("Loading Agmarknet CSV: %s", path.name)
        df = pd.read_csv(filepath, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        col_map = _build_column_map(df)

        if self.cell_mode:
            return list(self._generate_cell_chunks(df, col_map, path.stem))
        else:
            return list(self._generate_row_chunks(df, col_map, path.stem))

    def _generate_cell_chunks(
        self, df: pd.DataFrame, col_map: Dict[str, str], source: str
    ) -> Generator[Chunk, None, None]:
        """Positional cell-level chunk encoding for CALI / late interaction."""
        for row_idx, row in enumerate(df.itertuples(index=False)):
            row_dict = row._asdict()
            for col_idx, (col_name, col_actual) in enumerate(col_map.items()):
                val = row_dict.get(col_actual, "")
                if pd.isna(val):
                    val = ""
                text = f"[COL: {col_name.upper()}] [ROW: {row_idx}] [VAL: {str(val).strip()}]"
                meta = {
                    "row": row_idx,
                    "col": col_idx,
                    "col_name": col_name,
                    "source_file": source,
                }
                # Attach row-level context fields
                for canonical, actual in col_map.items():
                    meta[canonical] = str(row_dict.get(actual, "")).strip()

                yield Chunk(
                    chunk_id=f"{source}_r{row_idx}_c{col_idx}",
                    text=text,
                    source=source,
                    chunk_type="cell",
                    metadata=meta,
                )

    def _generate_row_chunks(
        self, df: pd.DataFrame, col_map: Dict[str, str], source: str
    ) -> Generator[Chunk, None, None]:
        """Compact row-level chunk for bi-encoder / narrative path."""
        def _get(row_dict, canonical):
            actual = col_map.get(canonical)
            if actual:
                val = row_dict.get(actual, "")
                return "" if pd.isna(val) else str(val).strip()
            return ""

        for row_idx, row in enumerate(df.itertuples(index=False)):
            row_dict = row._asdict()
            market = _get(row_dict, "market")
            commodity = _get(row_dict, "commodity")
            variety = _get(row_dict, "variety")
            date = _get(row_dict, "arrival_date")
            min_p = _get(row_dict, "min_price")
            max_p = _get(row_dict, "max_price")
            modal_p = _get(row_dict, "modal_price")
            state = _get(row_dict, "state")
            district = _get(row_dict, "district")

            text = (
                f"Market: {market} | State: {state} | District: {district} | "
                f"Commodity: {commodity} ({variety}) | Date: {date} | "
                f"Min: ₹{min_p}/q | Max: ₹{max_p}/q | Modal: ₹{modal_p}/q"
            )

            meta = {
                "row": row_idx,
                "market": market,
                "commodity": commodity,
                "variety": variety,
                "date": date,
                "min_price": min_p,
                "max_price": max_p,
                "modal_price": modal_p,
                "state": state,
                "district": district,
                "source_file": source,
            }

            yield Chunk(
                chunk_id=f"{source}_row_{row_idx}",
                text=text,
                source=source,
                chunk_type="row",
                metadata=meta,
            )

    def load_directory(self, directory: str) -> List[Chunk]:
        """Load all CSV files in a directory."""
        chunks = []
        for fpath in Path(directory).glob("*.csv"):
            try:
                chunks.extend(self.load_csv(str(fpath)))
            except Exception as e:
                logger.warning("Failed to load %s: %s", fpath.name, e)
        return chunks


# ---------------------------------------------------------------------------
# Narrative Document Loader
# ---------------------------------------------------------------------------

class NarrativeLoader:
    """
    Load text / markdown documents and split into overlapping chunks
    for Route A (bi-encoder / FAISS).

    Parameters
    ----------
    chunk_size   : target chars per chunk
    chunk_overlap: character overlap between consecutive chunks
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_file(self, filepath: str) -> List[Chunk]:
        path = Path(filepath)
        text = path.read_text(encoding="utf-8", errors="ignore")
        return self._split(text, source=path.name)

    def load_text(self, text: str, source: str = "inline") -> List[Chunk]:
        return self._split(text, source=source)

    def _split(self, text: str, source: str) -> List[Chunk]:
        """Sliding-window character-level chunking with overlap."""
        chunks = []
        step = self.chunk_size - self.chunk_overlap
        idx = 0
        chunk_idx = 0
        while idx < len(text):
            end = min(idx + self.chunk_size, len(text))
            snippet = text[idx:end].strip()
            if snippet:
                chunks.append(
                    Chunk(
                        chunk_id=f"{source}_chunk_{chunk_idx}",
                        text=snippet,
                        source=source,
                        chunk_type="narrative",
                        metadata={"source": source, "char_start": idx, "char_end": end},
                    )
                )
            idx += step
            chunk_idx += 1
        logger.info("NarrativeLoader: %d chunks from '%s'", len(chunks), source)
        return chunks

    def load_directory(self, directory: str) -> List[Chunk]:
        """Load all .txt and .md files in a directory."""
        chunks = []
        for fpath in Path(directory).rglob("*.txt"):
            chunks.extend(self.load_file(str(fpath)))
        for fpath in Path(directory).rglob("*.md"):
            chunks.extend(self.load_file(str(fpath)))
        return chunks
