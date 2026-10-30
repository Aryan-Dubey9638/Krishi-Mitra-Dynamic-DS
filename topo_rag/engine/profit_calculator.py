"""
Deterministic Math Engine (with Dynamic Government & Logistics Resolvers)
========================================================================
Implements the formula-enforced profit calculator from the Topo-RAG proposal:

    Net Profit = (Yield × Price) - (Transport + Taxes + Wastage)

All arithmetic is performed with exact Python float arithmetic
(no LLM inference in the calculation hot-path) to guarantee numerical accuracy.

Dynamic Variable Integration:
----------------------------
- Highway Routing & Distance : Live OSRM National Highway Network
- APMC Market Taxes & Fees   : Resolved by destination State APMC Act Schedule
- Transport Tariffs          : Fuel-indexed via PPAC diesel rates & distance tiers
- Wastage Norms              : ICAR-CIPHET baseline + Open-Meteo ambient weather
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from topo_rag.engine.dynamic_resolvers import (
    DynamicVariableResolver,
    DynamicRouteResult,
    DynamicTaxResult,
    DynamicTariffResult,
    DynamicWastageResult,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fallback Defaults (only used if dynamic resolution is explicitly disabled)
# ---------------------------------------------------------------------------

_DEFAULT_TARIFF_PER_QTL_KM: float = 0.80   # ₹0.80/q/km
_DEFAULT_APMC_COMMISSION_PCT: float = 2.0   # 2%
_DEFAULT_MANDI_FEES_PER_QTL: float = 15.0   # ₹15/q

_DEFAULT_WASTAGE_RATES: Dict[str, float] = {
    "tomato": 0.12,
    "onion": 0.06,
    "potato": 0.04,
    "wheat": 0.015,
    "rice": 0.02,
    "maize": 0.02,
    "soybean": 0.02,
    "cotton": 0.01,
    "mango": 0.15,
    "banana": 0.15,
    "grapes": 0.12,
    "garlic": 0.06,
    "default": 0.08,
}

_DEFAULT_DISTANCE_MATRIX: Dict[Tuple[str, str], float] = {
    ("pune", "mumbai"): 148.0,
    ("pune", "delhi"): 1420.0,
    ("pune", "azadpur"): 1420.0,
    ("nashik", "mumbai"): 166.0,
    ("nashik", "delhi"): 1380.0,
    ("nashik", "azadpur"): 1380.0,
    ("indore", "delhi"): 815.0,
    ("indore", "azadpur"): 815.0,
    ("indore", "mumbai"): 585.0,
    ("bhopal", "mumbai"): 775.6,
    ("bhopal", "delhi"): 785.0,
    ("bhopal", "pune"): 788.0,
    ("agra", "delhi"): 215.0,
    ("agra", "azadpur"): 215.0,
    ("agra", "mumbai"): 1190.0,
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
            f"  Net Profit Calculation",
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"  Commodity   : {self.commodity.title()} (Grade {self.grade})",
            f"  Quantity    : {self.quantity_kg:.1f} kg ({self.quantity_quintals:.3f} quintals)",
            f"  Route       : {self.source_market} → {self.destination_market}",
            f"  Distance    : {self.distance_km:.1f} km",
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
# Profit Calculator Engine
# ---------------------------------------------------------------------------

class ProfitCalculator:
    """
    Deterministic profit calculator with live government & logistics resolvers.

    Net Profit = (Yield × Price) - (Transport + Taxes + Wastage)
    """

    def __init__(
        self,
        tariff_per_qtl_km: Optional[float] = None,
        apmc_commission_pct: Optional[float] = None,
        mandi_fees_per_qtl: Optional[float] = None,
        wastage_rates: Optional[Dict[str, float]] = None,
        distance_matrix: Optional[Dict[Tuple[str, str], float]] = None,
        use_dynamic_resolvers: bool = True,
    ):
        self.tariff_per_qtl_km = tariff_per_qtl_km
        self.apmc_commission_pct = apmc_commission_pct
        self.mandi_fees_per_qtl = mandi_fees_per_qtl
        self.wastage_rates: Dict[str, float] = wastage_rates or dict(_DEFAULT_WASTAGE_RATES)
        self.distance_matrix: Dict[Tuple[str, str], float] = (
            distance_matrix or dict(_DEFAULT_DISTANCE_MATRIX)
        )
        self.use_dynamic = use_dynamic_resolvers
        self.resolver = DynamicVariableResolver() if use_dynamic_resolvers else None
        logger.info("ProfitCalculator initialized (dynamic_resolvers=%s)", self.use_dynamic)

    def _get_static_distance(self, source: str, destination: str) -> Tuple[float, str]:
        key = (source.lower(), destination.lower())
        rev_key = (destination.lower(), source.lower())
        if key in self.distance_matrix:
            return self.distance_matrix[key], ""
        if rev_key in self.distance_matrix:
            return self.distance_matrix[rev_key], ""
        default = self.distance_matrix.get(("default", "default"), 500.0)
        return default, f"Distance not found for {source}→{destination}; using default {default:.0f} km"

    def calculate(
        self,
        commodity: str,
        quantity_kg: float,
        modal_price_per_qtl: float,    # ₹ per quintal (from mandi data)
        source_market: str,
        destination_market: str,
        grade: str = "A",
        grade_premium_pct: float = 0.0,
        override_distance_km: Optional[float] = None,
    ) -> ProfitBreakdown:
        """
        Compute full profit breakdown with dynamic resolution.
        """
        notes: List[str] = []

        # Convert kg → quintals (1 quintal = 100 kg)
        quantity_quintals = quantity_kg / 100.0

        # Apply grade premium
        effective_price = modal_price_per_qtl * (1 + grade_premium_pct / 100.0)
        if grade_premium_pct > 0:
            notes.append(f"Grade {grade} premium: +{grade_premium_pct:.1f}% applied")
        elif grade_premium_pct < 0:
            notes.append(f"Grade {grade} discount: {grade_premium_pct:.1f}% applied")

        # ------------------------------------------------------------------
        # 1. Resolve Highway Route & Distance (OSRM / OpenStreetMap)
        # ------------------------------------------------------------------
        if self.use_dynamic and self.resolver:
            route_res = self.resolver.resolve_route(source_market, destination_market)
            distance_km = (
                override_distance_km if override_distance_km is not None else route_res.distance_km
            )
            dest_state = route_res.destination_state
            notes.append(route_res.provenance_note)
        else:
            if override_distance_km is not None:
                distance_km = override_distance_km
            else:
                distance_km, dist_note = self._get_static_distance(source_market, destination_market)
                if dist_note:
                    notes.append(dist_note)
            dest_state = "default"
            route_res = None

        # ------------------------------------------------------------------
        # 2. Resolve Dynamic Fuel-Indexed Transport Tariff
        # ------------------------------------------------------------------
        if self.use_dynamic and self.resolver and self.tariff_per_qtl_km is None:
            tariff_res = self.resolver.resolve_tariff(distance_km, dest_state)
            tariff_rate = tariff_res.tariff_per_qtl_km
            notes.append(tariff_res.provenance_note)
        else:
            tariff_rate = self.tariff_per_qtl_km or _DEFAULT_TARIFF_PER_QTL_KM
            notes.append(f"Transport: ₹{tariff_rate:.2f}/q/km × {quantity_quintals:.3f}q × {distance_km:.1f}km")

        transport_cost = quantity_quintals * tariff_rate * distance_km

        # ------------------------------------------------------------------
        # 3. Resolve Statutory APMC Tax & Handling Charges
        # ------------------------------------------------------------------
        if self.use_dynamic and self.resolver and self.apmc_commission_pct is None:
            tax_res = self.resolver.resolve_apmc_tax(dest_state, destination_market)
            apmc_tax_pct = tax_res.total_tax_pct
            mandi_fee_rate = (
                self.mandi_fees_per_qtl if self.mandi_fees_per_qtl is not None else tax_res.handling_per_qtl
            )
            notes.append(tax_res.provenance_note)
        else:
            apmc_tax_pct = self.apmc_commission_pct or _DEFAULT_APMC_COMMISSION_PCT
            mandi_fee_rate = self.mandi_fees_per_qtl or _DEFAULT_MANDI_FEES_PER_QTL

        mandi_fees = quantity_quintals * mandi_fee_rate

        # ------------------------------------------------------------------
        # 4. Resolve ICAR-CIPHET Wastage & Live Ambient Weather
        # ------------------------------------------------------------------
        if self.use_dynamic and self.resolver and route_res:
            wastage_res = self.resolver.resolve_wastage(
                commodity=commodity,
                distance_km=distance_km,
                duration_hours=route_res.duration_hours,
                src_lat=route_res.src_lat,
                src_lon=route_res.src_lon,
                dst_lat=route_res.dst_lat,
                dst_lon=route_res.dst_lon,
            )
            wastage_rate = wastage_res.wastage_rate
            notes.append(wastage_res.provenance_note)
        else:
            wastage_rate = self.wastage_rates.get(
                commodity.lower(), self.wastage_rates.get("default", 0.08)
            )

        wastage_qty_qtl = quantity_quintals * wastage_rate
        wastage_cost = wastage_qty_qtl * effective_price

        # Effective sellable quantity at destination market
        sellable_qtl = quantity_quintals - wastage_qty_qtl
        gross_revenue = sellable_qtl * effective_price

        # APMC commission levied on actual produce traded in the mandi
        apmc_commission = gross_revenue * (apmc_tax_pct / 100.0)

        # Total costs
        total_costs = transport_cost + apmc_commission + mandi_fees + wastage_cost

        # Net profit
        net_profit = gross_revenue - total_costs
        net_profit_per_kg = net_profit / quantity_kg if quantity_kg > 0 else 0.0

        return ProfitBreakdown(
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
