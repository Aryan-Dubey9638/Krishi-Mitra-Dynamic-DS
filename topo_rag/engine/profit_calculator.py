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
    DynamicMandiPriceResult,
    DynamicRouteResult,
    DynamicTaxResult,
    DynamicTariffResult,
    DynamicWastageResult,
    DynamicSubsidyResult,
    GovernmentSubsidyResolver,
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


_COMMODITY_HI: Dict[str, str] = {
    "tomato": "टमाटर",
    "onion": "प्याज",
    "potato": "आलू",
    "wheat": "गेहूं",
    "rice": "चावल / धान",
    "paddy": "धान",
    "maize": "मक्का",
    "soybean": "सोयाबीन",
    "cotton": "कपास",
    "sugarcane": "गन्ना",
    "mustard": "सरसों",
    "groundnut": "मूंगफली",
    "gram": "चना",
    "tur": "अरहर / तूर",
    "moong": "मूंग",
    "urad": "उड़द",
    "sunflower": "सूरजमुखी",
    "mango": "आम",
    "banana": "केला",
    "grapes": "अंगूर",
    "garlic": "लहसुन",
}

_CITY_HI: Dict[str, str] = {
    "pune": "पुणे",
    "mumbai": "मुंबई",
    "delhi": "दिल्ली",
    "azadpur": "आजादपुर",
    "nashik": "नासिक",
    "nasik": "नासिक",
    "indore": "इंदौर",
    "agra": "आगरा",
    "lucknow": "लखनऊ",
    "bhopal": "भोपाल",
    "vashi": "वाशी",
    "gultekdi": "गुलटेकड़ी",
    "lasalgaon": "लासलगांव",
    "hyderabad": "हैदराबाद",
    "bengaluru": "बेंगलुरु",
    "bangalore": "बेंगलुरु",
    "chennai": "चेन्नई",
    "kolkata": "कोलकाता",
    "nagpur": "नागपुर",
    "jaipur": "जयपुर",
    "varanasi": "वाराणसी",
    "kanpur": "कानपुर",
    "patna": "पटना",
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
    net_profit: float               # effective_gross_revenue - effective_total_costs
    net_profit_per_kg: float        # per kg

    # Metadata
    source_market: str = ""
    destination_market: str = ""
    distance_km: float = 0.0
    wastage_rate: float = 0.0
    grade: str = "A"
    notes: List[str] = field(default_factory=list)
    notes_hi: List[str] = field(default_factory=list)

    # Explicit subsidized revenue, cost & per-quintal tracking (V6)
    net_profit_per_qtl: float = 0.0            # ₹ per quintal (subsidized)
    effective_gross_revenue: float = 0.0       # gross_revenue + govt_price_support
    effective_total_costs: float = 0.0         # total_costs - cost offsets (freight & fee waiver)
    govt_transport_subsidy: float = 0.0        # ₹ freight relief
    govt_mandi_fee_waiver: float = 0.0         # ₹ APMC tax/cess waiver
    govt_price_support: float = 0.0            # ₹ PM-AASHA PDPS deficiency
    govt_total_subsidy: float = 0.0            # sum of all scheme benefits
    govt_scheme_name: str = ""                 # applied scheme(s)
    baseline_net_profit: float = 0.0           # net profit without subsidies

    def summary(self, lang: str = "en") -> str:
        if lang == "hi":
            comm_hi = _COMMODITY_HI.get(self.commodity.lower(), self.commodity.title())
            src_hi = _CITY_HI.get(self.source_market.lower(), self.source_market)
            dst_hi = _CITY_HI.get(self.destination_market.lower(), self.destination_market)
            lines = [
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                f"  शुद्ध लाभ गणना (Net Profit Calculation)",
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                f"  फसल / जिंस   : {comm_hi} (ग्रेड {self.grade})",
                f"  मात्रा        : {self.quantity_kg:.1f} किग्रा ({self.quantity_quintals:.3f} क्विंटल)",
                f"  रूट (मार्ग)   : {src_hi} → {dst_hi}",
                f"  दूरी          : {self.distance_km:.1f} किमी",
                f"  मंडी भाव      : ₹{self.modal_price_per_qtl:,.2f} / क्विंटल",
                f"",
                f"  कुल अनुमानित आय      ₹{self.gross_revenue:>12,.2f}",
                f"  ─────────────────────────────────────────────",
                f"  परिवहन खर्च (भाड़ा)   ₹{self.transport_cost:>12,.2f}",
                f"  मंडी आढ़त (कमीशन)     ₹{self.apmc_commission:>12,.2f}",
                f"  मंडी शुल्क            ₹{self.mandi_fees:>12,.2f}",
                f"  खराबी / नुकसान         ₹{self.wastage_cost:>12,.2f}  ({self.wastage_rate*100:.1f}%)",
                f"  ─────────────────────────────────────────────",
                f"  कुल लागत खर्च         ₹{self.total_costs:>12,.2f}",
                f"",
            ]
            if self.govt_total_subsidy > 0:
                lines.extend([
                    f"  ═══════════════════════════════════════════════",
                    f"  सरकारी सब्सिडी / योजना लाभ (V6):",
                ])
                if self.govt_transport_subsidy > 0:
                    lines.append(f"  [-] भाड़ा राहत (ऑप ग्रीन्स)  ₹{self.govt_transport_subsidy:>12,.2f}")
                if self.govt_mandi_fee_waiver > 0:
                    lines.append(f"  [-] मंडी शुल्क छूट           ₹{self.govt_mandi_fee_waiver:>12,.2f}")
                if self.govt_price_support > 0:
                    lines.append(f"  [+] मूल्य सहायता (PM-AASHA)  ₹{self.govt_price_support:>12,.2f}")
                lines.append(f"  कुल सब्सिडी लाभ              ₹{self.govt_total_subsidy:>12,.2f}")
                lines.append(f"  प्रभावी कुल लागत (छूट बाद)   ₹{self.effective_total_costs:>12,.2f}")
                lines.append(f"  ─────────────────────────────────────────────")
                lines.append(f"  बिना सब्सिडी मुनाफा          ₹{self.baseline_net_profit:>12,.2f}")
            lines.extend([
                f"  शुद्ध मुनाफा (लाभ)    ₹{self.net_profit:>12,.2f}",
                f"  प्रति क्विंटल मुनाफा  ₹{self.net_profit_per_qtl:>12,.2f}",
                f"  प्रति किलो मुनाफा     ₹{self.net_profit_per_kg:>12,.2f}",
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            ])
            hi_notes = self.notes_hi if self.notes_hi else self.notes
            if hi_notes:
                lines.append("  विशेष विवरण:")
                for note in hi_notes:
                    lines.append(f"    • {note}")
            return "\n".join(lines)

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
        ]
        if self.govt_total_subsidy > 0:
            lines.extend([
                f"  ═══════════════════════════════════════════════",
                f"  GOVERNMENT SCHEME BENEFITS (V6):",
            ])
            if self.govt_transport_subsidy > 0:
                lines.append(f"  [-] Freight Subsidy    ₹{self.govt_transport_subsidy:>12,.2f}  (Op. Greens 50%)")
            if self.govt_mandi_fee_waiver > 0:
                lines.append(f"  [-] Mandi Fee Waiver   ₹{self.govt_mandi_fee_waiver:>12,.2f}  (FPO/State)")
            if self.govt_price_support > 0:
                lines.append(f"  [+] Price Support      ₹{self.govt_price_support:>12,.2f}  (PM-AASHA PDPS)")
            lines.append(f"  TOTAL SUBSIDY BENEFIT  ₹{self.govt_total_subsidy:>12,.2f}")
            if self.govt_price_support > 0:
                lines.append(f"  EFFECTIVE GROSS REV.   ₹{self.effective_gross_revenue:>12,.2f}")
            lines.append(f"  EFFECTIVE TOTAL COSTS  ₹{self.effective_total_costs:>12,.2f}")
            lines.append(f"  ─────────────────────────────────────────────")
            lines.append(f"  BASELINE PROFIT        ₹{self.baseline_net_profit:>12,.2f}  (without schemes)")
        lines.extend([
            f"  NET PROFIT           ₹{self.net_profit:>12,.2f}",
            f"  Net Profit / quintal ₹{self.net_profit_per_qtl:>12,.2f}",
            f"  Net Profit / kg      ₹{self.net_profit_per_kg:>12,.2f}",
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        ])
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

    Net Profit = (Yield × Price) - (Transport + Taxes + Wastage) + Subsidies

    Variables resolved dynamically:
      V1: Live Mandi Price (Agmarknet data.gov.in)
      V2: Highway Route & Distance (OSRM / OpenStreetMap)
      V3: Statutory APMC Taxes & Cesses (State APMC Acts)
      V4: Fuel-Indexed Freight Tariff (PPAC Diesel Index)
      V5: Weather-Aware Transit Wastage (ICAR-CIPHET & Open-Meteo)
      V6: Government Subsidies & Cost Offsets (MoFPI, PM-AASHA, State FPO)
    """

    def __init__(
        self,
        tariff_per_qtl_km: Optional[float] = None,
        apmc_commission_pct: Optional[float] = None,
        mandi_fees_per_qtl: Optional[float] = None,
        wastage_rates: Optional[Dict[str, float]] = None,
        distance_matrix: Optional[Dict[Tuple[str, str], float]] = None,
        use_dynamic_resolvers: bool = True,
        use_subsidy_resolver: Optional[bool] = None,
    ):
        self.tariff_per_qtl_km = tariff_per_qtl_km
        self.apmc_commission_pct = apmc_commission_pct
        self.mandi_fees_per_qtl = mandi_fees_per_qtl
        self.wastage_rates: Dict[str, float] = wastage_rates or dict(_DEFAULT_WASTAGE_RATES)
        self.distance_matrix: Dict[Tuple[str, str], float] = (
            distance_matrix or dict(_DEFAULT_DISTANCE_MATRIX)
        )
        self.use_dynamic = use_dynamic_resolvers
        self.use_subsidies = (
            use_dynamic_resolvers if use_subsidy_resolver is None else use_subsidy_resolver
        )
        self.resolver = DynamicVariableResolver() if use_dynamic_resolvers else None
        self.subsidy_resolver = GovernmentSubsidyResolver() if self.use_subsidies else None
        logger.info(
            "ProfitCalculator initialized (dynamic_resolvers=%s, subsidy_resolver=%s)",
            self.use_dynamic,
            self.use_subsidies,
        )

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
        price_provenance_note: Optional[str] = None,
        price_provenance_note_hi: Optional[str] = None,
        is_fpo_member: bool = False,
        transport_mode: str = "auto",
    ) -> ProfitBreakdown:
        """
        Compute full profit breakdown with dynamic resolution of all 6 variables.

        Variables resolved:
          V1: modal_price_per_qtl (passed in, from Agmarknet API)
          V2: Highway route & distance (OSRM live)
          V3: APMC taxes & cess (State APMC statutory schedule)
          V4: Transport tariff (PPAC fuel-indexed)
          V5: Transit wastage (ICAR-CIPHET + Open-Meteo weather)
          V6: Government subsidies (Operation Greens, FPO waivers, PM-AASHA)
        """
        notes: List[str] = []
        notes_hi: List[str] = []
        if price_provenance_note:
            notes.append(price_provenance_note)
            notes_hi.append(price_provenance_note_hi or price_provenance_note)

        # Convert kg → quintals (1 quintal = 100 kg)
        quantity_quintals = quantity_kg / 100.0

        # Apply grade premium
        effective_price = modal_price_per_qtl * (1 + grade_premium_pct / 100.0)
        if grade_premium_pct > 0:
            notes.append(f"Grade {grade} premium: +{grade_premium_pct:.1f}% applied")
            notes_hi.append(f"ग्रेड {grade} गुणवत्ता प्रीमियम: +{grade_premium_pct:.1f}% लागू")
        elif grade_premium_pct < 0:
            notes.append(f"Grade {grade} discount: {grade_premium_pct:.1f}% applied")
            notes_hi.append(f"ग्रेड {grade} छूट: {grade_premium_pct:.1f}% लागू")

        # ------------------------------------------------------------------
        # V2. Resolve Highway Route & Distance (OSRM / OpenStreetMap)
        # ------------------------------------------------------------------
        if self.use_dynamic and self.resolver:
            route_res = self.resolver.resolve_route(source_market, destination_market)
            distance_km = (
                override_distance_km if override_distance_km is not None else route_res.distance_km
            )
            dest_state = route_res.destination_state
            notes.append(route_res.provenance_note)
            notes_hi.append(route_res.provenance_note_hi or route_res.provenance_note)
        else:
            if override_distance_km is not None:
                distance_km = override_distance_km
            else:
                distance_km, dist_note = self._get_static_distance(source_market, destination_market)
                if dist_note:
                    notes.append(dist_note)
                    notes_hi.append(
                        f"{source_market}→{destination_market} दूरी उपलब्ध नहीं; मानक दूरी {distance_km:.0f} किमी उपयोग की गई"
                    )
            dest_state = "default"
            route_res = None

        # ------------------------------------------------------------------
        # V4. Resolve Dynamic Fuel-Indexed Transport Tariff
        # ------------------------------------------------------------------
        if self.use_dynamic and self.resolver and self.tariff_per_qtl_km is None:
            tariff_res = self.resolver.resolve_tariff(
                distance_km=distance_km,
                dest_state=dest_state,
                quantity_quintals=quantity_quintals,
            )
            tariff_rate = tariff_res.tariff_per_qtl_km
            notes.append(tariff_res.provenance_note)
            notes_hi.append(tariff_res.provenance_note_hi or tariff_res.provenance_note)
        else:
            tariff_rate = self.tariff_per_qtl_km or _DEFAULT_TARIFF_PER_QTL_KM
            notes.append(f"Transport: ₹{tariff_rate:.2f}/q/km × {quantity_quintals:.3f}q × {distance_km:.1f}km")
            notes_hi.append(f"परिवहन दर: ₹{tariff_rate:.2f}/क्विंटल/किमी × {quantity_quintals:.3f} क्विंटल × {distance_km:.1f} किमी")

        transport_cost = quantity_quintals * tariff_rate * distance_km

        # ------------------------------------------------------------------
        # V3. Resolve Statutory APMC Tax & Handling Charges
        # ------------------------------------------------------------------
        if self.use_dynamic and self.resolver and self.apmc_commission_pct is None:
            tax_res = self.resolver.resolve_apmc_tax(dest_state, destination_market)
            apmc_tax_pct = tax_res.total_tax_pct
            mandi_fee_rate = (
                self.mandi_fees_per_qtl if self.mandi_fees_per_qtl is not None else tax_res.handling_per_qtl
            )
            notes.append(tax_res.provenance_note)
            notes_hi.append(tax_res.provenance_note_hi or tax_res.provenance_note)
        else:
            apmc_tax_pct = self.apmc_commission_pct or _DEFAULT_APMC_COMMISSION_PCT
            mandi_fee_rate = self.mandi_fees_per_qtl or _DEFAULT_MANDI_FEES_PER_QTL

        mandi_fees = quantity_quintals * mandi_fee_rate

        # ------------------------------------------------------------------
        # V5. Resolve ICAR-CIPHET Wastage & Live Ambient Weather
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
            notes_hi.append(wastage_res.provenance_note_hi or wastage_res.provenance_note)
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

        # Total costs (before subsidies)
        total_costs = transport_cost + apmc_commission + mandi_fees + wastage_cost

        # Baseline net profit (before government schemes)
        baseline_net_profit = gross_revenue - total_costs

        # ------------------------------------------------------------------
        # V6. Resolve Government Subsidies & Cost Offsets
        # ------------------------------------------------------------------
        if self.use_subsidies and self.subsidy_resolver:
            subsidy_res = self.subsidy_resolver.resolve_subsidies(
                commodity=commodity,
                quantity_quintals=quantity_quintals,
                distance_km=distance_km,
                modal_price=effective_price,
                gross_transport_cost=transport_cost,
                gross_apmc_tax=apmc_commission,
                destination_state=dest_state,
                is_fpo_member=is_fpo_member,
                transport_mode=transport_mode,
            )
            if subsidy_res.is_eligible or subsidy_res.channel_details:
                notes.append(subsidy_res.provenance_note)
                notes_hi.append(subsidy_res.provenance_note_hi or subsidy_res.provenance_note)
            if subsidy_res.is_eligible and subsidy_res.msp_advisory_note:
                notes.append(subsidy_res.msp_advisory_note)
                notes_hi.append(subsidy_res.msp_advisory_note_hi or subsidy_res.msp_advisory_note)

            govt_transport_sub = subsidy_res.transport_subsidy_amount
            govt_fee_waiver = subsidy_res.mandi_fee_waiver_amount
            govt_price_support = subsidy_res.price_support_amount
            govt_total_sub = subsidy_res.total_subsidy_benefit
            govt_scheme_name = subsidy_res.scheme_name
        else:
            govt_transport_sub = 0.0
            govt_fee_waiver = 0.0
            govt_price_support = 0.0
            govt_total_sub = 0.0
            govt_scheme_name = "Disabled"

        # Effective revenue & costs after government scheme relief
        effective_gross_revenue = gross_revenue + govt_price_support
        effective_total_costs = max(total_costs - govt_transport_sub - govt_fee_waiver, 0.0)

        # Final net profit = effective_gross_revenue - effective_total_costs
        # (equivalently: baseline_net_profit + govt_total_sub)
        net_profit = effective_gross_revenue - effective_total_costs
        net_profit_per_qtl = net_profit / quantity_quintals if quantity_quintals > 0 else 0.0
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
            notes_hi=notes_hi,
            # V6 subsidy fields
            net_profit_per_qtl=net_profit_per_qtl,
            effective_gross_revenue=effective_gross_revenue,
            effective_total_costs=effective_total_costs,
            govt_transport_subsidy=govt_transport_sub,
            govt_mandi_fee_waiver=govt_fee_waiver,
            govt_price_support=govt_price_support,
            govt_total_subsidy=govt_total_sub,
            govt_scheme_name=govt_scheme_name,
            baseline_net_profit=baseline_net_profit,
        )
