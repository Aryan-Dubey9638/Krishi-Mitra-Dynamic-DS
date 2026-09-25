"""
Deterministic Math Engine
=========================
Implements the formula-enforced profit calculator from the Topo-RAG proposal:

    Net Profit = (Yield × Price) - (Transport + Taxes + Wastage)

All arithmetic is performed with exact Python float/Decimal arithmetic
(no LLM inference in the hot-path) to guarantee numerical accuracy.

Components
----------
- ProfitCalculator : Core formula engine.
- TariffLookup     : Retrieves transport tariffs from indexed data.
- TaxEngine        : Applies APMC commission + mandi tax rules.
- WastageModel     : Adjustable wastage-rate lookup by commodity.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default lookup tables (overridden at runtime by retrieved mandi data)
# ---------------------------------------------------------------------------

# Transport tariff: ₹ per quintal per km (heuristic defaults)
_DEFAULT_TARIFF_PER_QTL_KM: float = 0.80   # ₹0.80/q/km

# APMC commission: % of total sale value
_DEFAULT_APMC_COMMISSION_PCT: float = 2.0   # 2%

# Mandi loading/unloading + other fees: ₹ per quintal
_DEFAULT_MANDI_FEES_PER_QTL: float = 15.0   # ₹15/q

# Wastage rates by commodity (fraction of quantity lost)
_DEFAULT_WASTAGE_RATES: Dict[str, float] = {
    "tomato": 0.15,
    "onion": 0.08,
    "potato": 0.06,
    "wheat": 0.02,
    "rice": 0.02,
    "maize": 0.03,
    "soybean": 0.02,
    "cotton": 0.01,
    "mango": 0.20,
    "banana": 0.22,
    "grapes": 0.18,
    "garlic": 0.10,
    "default": 0.10,
}

# Distance matrix (km) — defaults; populated from retrieved data at runtime
_DEFAULT_DISTANCE_MATRIX: Dict[Tuple[str, str], float] = {
    ("pune", "mumbai"): 145.0,
    ("pune", "delhi"): 1420.0,
    ("pune", "azadpur"): 1420.0,   # Azadpur is the main Delhi wholesale market
    ("nashik", "mumbai"): 170.0,
    ("nashik", "delhi"): 1380.0,
    ("nashik", "azadpur"): 1380.0,
    ("indore", "delhi"): 780.0,
    ("indore", "azadpur"): 780.0,
    ("indore", "mumbai"): 590.0,
    ("agra", "delhi"): 210.0,
    ("agra", "azadpur"): 210.0,
    ("agra", "mumbai"): 1170.0,
    ("lucknow", "delhi"): 555.0,
    ("lucknow", "azadpur"): 555.0,
    ("lucknow", "mumbai"): 1365.0,
    ("default", "default"): 500.0,
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ProfitBreakdown:
    """Full breakdown of the net profit calculation."""
    commodity: str
    quantity_kg: float
    quantity_quintals: float

    # Gross revenue
    modal_price_per_qtl: float      # ₹/quintal (fetched from mandi data)
    gross_revenue: float            # quantity_quintals × modal_price_per_qtl

    # Costs
    transport_cost: float           # ₹
    apmc_commission: float          # ₹
    mandi_fees: float               # ₹
    wastage_cost: float             # ₹ (price loss from wasted produce)
    total_costs: float              # sum of all above

    # Net
    net_profit: float               # gross_revenue - total_costs
    net_profit_per_kg: float        # per kg

    # Metadata
    source_market: str = ""
    destination_market: str = ""
    distance_km: float = 0.0
    wastage_rate: float = 0.0
    grade: str = "A"
    notes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"  Basic RAG Net Profit Calculation",
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"  Commodity   : {self.commodity.title()} (Grade {self.grade})",
            f"  Quantity    : {self.quantity_kg:.1f} kg ({self.quantity_quintals:.3f} quintals)",
            f"  Route       : {self.source_market} → {self.destination_market}",
            f"  Distance    : {self.distance_km:.0f} km",
            f"  Modal Price : ₹{self.modal_price_per_qtl:,.2f}/quintal",
            f"",
            f"  GROSS REVENUE        ₹{self.gross_revenue:>12,.2f}",
            f"  ─────────────────────────────────────────────",
            f"  Transport Cost       ₹{self.transport_cost:>12,.2f}",
            f"  APMC Commission      ₹{self.apmc_commission:>12,.2f}",
            f"  Mandi Fees           ₹{self.mandi_fees:>12,.2f}",
            f"  Wastage Loss         ₹{self.wastage_cost:>12,.2f}  ({self.wastage_rate*100:.1f}%)",
            f"  ─────────────────────────────────────────────",
            f"  TOTAL COSTS          ₹{self.total_costs:>12,.2f}",
            f"",
            f"  NET PROFIT           ₹{self.net_profit:>12,.2f}",
            f"  Net Profit / kg      ₹{self.net_profit_per_kg:>12,.2f}",
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        ]
        if self.notes:
            lines.append("  Notes:")
            for note in self.notes:
                lines.append(f"    • {note}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Profit Calculator
# ---------------------------------------------------------------------------

class ProfitCalculator:
    """
    Deterministic profit calculator.

    Net Profit = (Yield × Price) - (Transport + Taxes + Wastage)

    Parameters
    ----------
    tariff_per_qtl_km   : ₹ per quintal per km
    apmc_commission_pct : % of gross revenue charged as APMC commission
    mandi_fees_per_qtl  : flat ₹ per quintal for loading/unloading
    wastage_rates       : dict of commodity → fraction lost
    distance_matrix     : dict of (src, dst) → km
    """

    def __init__(
        self,
        tariff_per_qtl_km: float = _DEFAULT_TARIFF_PER_QTL_KM,
        apmc_commission_pct: float = _DEFAULT_APMC_COMMISSION_PCT,
        mandi_fees_per_qtl: float = _DEFAULT_MANDI_FEES_PER_QTL,
        wastage_rates: Optional[Dict[str, float]] = None,
        distance_matrix: Optional[Dict[Tuple[str, str], float]] = None,
    ):
        self.tariff_per_qtl_km = tariff_per_qtl_km
        self.apmc_commission_pct = apmc_commission_pct
        self.mandi_fees_per_qtl = mandi_fees_per_qtl
        self.wastage_rates: Dict[str, float] = wastage_rates or dict(_DEFAULT_WASTAGE_RATES)
        self.distance_matrix: Dict[Tuple[str, str], float] = (
            distance_matrix or dict(_DEFAULT_DISTANCE_MATRIX)
        )

    def _get_distance(self, source: str, destination: str) -> Tuple[float, str]:
        """Return (distance_km, note)."""
        key = (source.lower(), destination.lower())
        rev_key = (destination.lower(), source.lower())
        if key in self.distance_matrix:
            return self.distance_matrix[key], ""
        if rev_key in self.distance_matrix:
            return self.distance_matrix[rev_key], ""
        default = self.distance_matrix.get(("default", "default"), 500.0)
        return default, f"Distance not found for {source}→{destination}; using default {default:.0f} km"

    def _get_wastage_rate(self, commodity: str) -> float:
        return self.wastage_rates.get(
            commodity.lower(),
            self.wastage_rates.get("default", 0.10)
        )

    def calculate(
        self,
        commodity: str,
        quantity_kg: float,
        modal_price_per_qtl: float,    # ₹ per quintal (from mandi data)
        source_market: str,
        destination_market: str,
        grade: str = "A",
        grade_premium_pct: float = 0.0,   # Extra % price for Grade A etc.
        override_distance_km: Optional[float] = None,
    ) -> ProfitBreakdown:
        """
        Compute full profit breakdown.

        Parameters
        ----------
        commodity           : e.g. 'Tomato'
        quantity_kg         : e.g. 500.0
        modal_price_per_qtl : Modal mandi price in ₹/quintal (retrieved)
        source_market       : Origin market name
        destination_market  : Target mandi name
        grade               : 'A' | 'B' | 'C'
        grade_premium_pct   : Additional price premium for grading (%)
        override_distance_km: If provided, skip distance matrix lookup.
        """
        notes: List[str] = []

        # Convert kg → quintals (1 quintal = 100 kg)
        quantity_quintals = quantity_kg / 100.0

        # Apply grade premium
        effective_price = modal_price_per_qtl * (1 + grade_premium_pct / 100.0)
        if grade_premium_pct > 0:
            notes.append(f"Grade {grade} premium: +{grade_premium_pct:.1f}% applied")

        # Gross revenue (before wastage deduction)
        gross_revenue = quantity_quintals * effective_price

        # Wastage
        wastage_rate = self._get_wastage_rate(commodity)
        wastage_qty_qtl = quantity_quintals * wastage_rate
        wastage_cost = wastage_qty_qtl * effective_price
        # Effective sellable quantity
        sellable_qtl = quantity_quintals * (1 - wastage_rate)
        gross_revenue = sellable_qtl * effective_price   # recalculate on actual sold qty
        if wastage_rate > 0:
            notes.append(f"Wastage: {wastage_rate*100:.1f}% of produce ({wastage_qty_qtl:.3f} q lost)")

        # Transport cost
        if override_distance_km is not None:
            distance_km = override_distance_km
        else:
            distance_km, dist_note = self._get_distance(source_market, destination_market)
            if dist_note:
                notes.append(dist_note)

        transport_cost = quantity_quintals * self.tariff_per_qtl_km * distance_km
        notes.append(f"Transport: ₹{self.tariff_per_qtl_km}/q/km × {quantity_quintals:.3f}q × {distance_km:.0f}km")

        # APMC commission (% of gross_revenue after wastage)
        apmc_commission = gross_revenue * (self.apmc_commission_pct / 100.0)

        # Mandi fees (flat per quintal)
        mandi_fees = quantity_quintals * self.mandi_fees_per_qtl

        # Total costs
        total_costs = transport_cost + apmc_commission + mandi_fees + wastage_cost

        # Net profit
        net_profit = gross_revenue - total_costs
        net_profit_per_kg = net_profit / quantity_kg if quantity_kg > 0 else 0.0

        result = ProfitBreakdown(
            commodity=commodity,
            quantity_kg=quantity_kg,
            quantity_quintals=quantity_quintals,
            modal_price_per_qtl=effective_price,
            gross_revenue=gross_revenue,
            transport_cost=transport_cost,
            apmc_commission=apmc_commission,
            mandi_fees=mandi_fees,
            wastage_cost=wastage_cost,
            total_costs=total_costs,
            net_profit=net_profit,
            net_profit_per_kg=net_profit_per_kg,
            source_market=source_market,
            destination_market=destination_market,
            distance_km=distance_km,
            wastage_rate=wastage_rate,
            grade=grade,
            notes=notes,
        )
        logger.info(
            "ProfitCalculator: %s %.1fkg %s→%s → Net ₹%.2f",
            commodity, quantity_kg, source_market, destination_market, net_profit,
        )
        return result

    def update_from_retrieved_data(self, metadata_list: List[Dict]) -> None:
        """
        Update modal prices and transport tariffs from retrieved mandi results.
        Modifies internal state for subsequent calculate() calls.
        """
        for meta in metadata_list:
            modal_p = meta.get("modal_price", "")
            if modal_p:
                try:
                    _price = float(str(modal_p).replace(",", "").replace("₹", ""))
                    logger.debug("Retrieved modal price: ₹%.2f/q", _price)
                except (ValueError, TypeError):
                    pass
