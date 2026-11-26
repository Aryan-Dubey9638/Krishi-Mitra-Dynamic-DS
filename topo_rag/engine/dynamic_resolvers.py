"""
Dynamic Variable Resolvers for Agricultural Logistics
=====================================================
Eliminates static constants by dynamically fetching and resolving variables
from authoritative government and open public data sources:

1. Road Driving Distance & Duration:
   - Live Highway Routing via OSRM (Open Source Routing Machine) & OpenStreetMap
   - Authoritative road network geometry across Indian National & State Highways

2. State APMC Taxes, Mandi Cess & Handling Charges:
   - Resolved dynamically based on the destination state detected by the geocoder
   - Grounded in State APMC Acts (e.g. Maharashtra MSAMB, Delhi DAMB, Punjab PSAMB, UP Mandi Parishad)

3. Fuel-Indexed Road Freight Tariff:
   - Ministry of Petroleum & Natural Gas (PPAC) diesel indices by state
   - Distance economy-of-scale tiering (short-haul LCV vs long-haul multi-axle)

4. Perishability & Transit Wastage:
   - ICAR - CIPHET (Central Institute of Post-Harvest Engineering & Technology) baseline loss norms
   - Live ambient weather (temperature & humidity) fetched from Open-Meteo satellite feed
   - Dynamic thermal-transit deterioration model
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from dotenv import load_dotenv
import requests

load_dotenv()

logger = logging.getLogger(__name__)

# Request timeout for external calls (seconds)
_HTTP_TIMEOUT: float = 3.0

# User-Agent header required by Nominatim / OSM policy
_HEADERS: Dict[str, str] = {
    "User-Agent": "TopoRAG-KrishiPlatform/2.0 (Agricultural Logistics Engine)"
}

# In-memory caches to avoid redundant network calls during a session
_GEOCODE_CACHE: Dict[str, Tuple[float, float, str]] = {}
_ROUTE_CACHE: Dict[Tuple[str, str], Tuple[float, float]] = {}
_WEATHER_CACHE: Dict[Tuple[float, float], Tuple[float, float]] = {}
_MANDI_PRICE_CACHE: Dict[Tuple[str, str], DynamicMandiPriceResult] = {}


# ---------------------------------------------------------------------------
# 1. Comprehensive State APMC Statutory Schedules (State Mandi Acts & Gazettes)
# ---------------------------------------------------------------------------

STATE_APMC_SCHEDULE = {
    "maharashtra": {
        "state_name": "Maharashtra",
        "market_fee_pct": 1.05,
        "dev_cess_pct": 0.00,
        "handling_per_qtl": 15.0,
        "citation": "Maharashtra APMC Act Sec 31(1) — MSAMB Gazette",
    },
    "delhi": {
        "state_name": "Delhi (NCT)",
        "market_fee_pct": 1.00,
        "dev_cess_pct": 1.00,
        "handling_per_qtl": 14.0,
        "citation": "Delhi DAMB (Azadpur APMC) Fee Schedule Sec 28",
    },
    "madhya pradesh": {
        "state_name": "Madhya Pradesh",
        "market_fee_pct": 1.50,
        "dev_cess_pct": 0.20,
        "handling_per_qtl": 15.0,
        "citation": "MP Krishi Upaj Mandi Adhiniyam Sec 19",
    },
    "uttar pradesh": {
        "state_name": "Uttar Pradesh",
        "market_fee_pct": 1.50,
        "dev_cess_pct": 0.50,
        "handling_per_qtl": 12.0,
        "citation": "UP Krishi Utpadan Mandi Adhiniyam — Mandi Parishad Notified Rates",
    },
    "punjab": {
        "state_name": "Punjab",
        "market_fee_pct": 2.00,
        "dev_cess_pct": 2.00,  # Rural Development Fund (RDF)
        "handling_per_qtl": 18.0,
        "citation": "Punjab APMC Act & Rural Development Fund Norms (PSAMB)",
    },
    "haryana": {
        "state_name": "Haryana",
        "market_fee_pct": 2.00,
        "dev_cess_pct": 2.00,  # Haryana Rural Development Fund (HRDF)
        "handling_per_qtl": 18.0,
        "citation": "Haryana Agricultural Produce Markets Act & HRDF Rules",
    },
    "karnataka": {
        "state_name": "Karnataka",
        "market_fee_pct": 1.50,
        "dev_cess_pct": 0.00,
        "handling_per_qtl": 16.0,
        "citation": "Karnataka APMC Act Sec 65 — KSAMB Notified Rates",
    },
    "rajasthan": {
        "state_name": "Rajasthan",
        "market_fee_pct": 1.60,
        "dev_cess_pct": 0.50,  # Krishak Kalyan Cess
        "handling_per_qtl": 14.0,
        "citation": "Rajasthan Agricultural Produce Markets Act & Krishak Kalyan Kosh",
    },
    "gujarat": {
        "state_name": "Gujarat",
        "market_fee_pct": 1.00,
        "dev_cess_pct": 0.00,
        "handling_per_qtl": 12.0,
        "citation": "Gujarat APMC Act Gazette Schedule — GAPMC Rules",
    },
    "tamil nadu": {
        "state_name": "Tamil Nadu",
        "market_fee_pct": 1.00,
        "dev_cess_pct": 0.00,
        "handling_per_qtl": 15.0,
        "citation": "Tamil Nadu Agricultural Produce Marketing Act — TNAMB Schedule",
    },
    "andhra pradesh": {
        "state_name": "Andhra Pradesh",
        "market_fee_pct": 1.00,
        "dev_cess_pct": 0.00,
        "handling_per_qtl": 14.0,
        "citation": "AP (Agricultural Produce & Livestock) Markets Act — APAMB",
    },
    "telangana": {
        "state_name": "Telangana",
        "market_fee_pct": 1.00,
        "dev_cess_pct": 0.00,
        "handling_per_qtl": 14.0,
        "citation": "Telangana Agricultural Produce Markets Act — TGAMB Schedule",
    },
    "bihar": {
        "state_name": "Bihar",
        "market_fee_pct": 0.00,  # APMC abolished in 2006, direct trade
        "dev_cess_pct": 0.00,
        "handling_per_qtl": 10.0,
        "citation": "Bihar Agriculture Produce Market (Repealing) Act 2006 (Deregulated Private Trade)",
    },
    "west bengal": {
        "state_name": "West Bengal",
        "market_fee_pct": 1.00,
        "dev_cess_pct": 0.00,
        "handling_per_qtl": 14.0,
        "citation": "West Bengal Agricultural Produce Marketing (Regulation) Act — WBSAMB",
    },
    "odisha": {
        "state_name": "Odisha",
        "market_fee_pct": 1.00,
        "dev_cess_pct": 0.00,
        "handling_per_qtl": 12.0,
        "citation": "Odisha APMC Act — OSAMB Statutory Schedule",
    },
    "kerala": {
        "state_name": "Kerala",
        "market_fee_pct": 0.00,  # Non-APMC state, cooperative frameworks (VFPCK/Horticorp)
        "dev_cess_pct": 0.00,
        "handling_per_qtl": 12.0,
        "citation": "Kerala Agricultural Produce Marketing Cooperative Framework (Non-APMC State)",
    },
    "himachal pradesh": {
        "state_name": "Himachal Pradesh",
        "market_fee_pct": 1.00,
        "dev_cess_pct": 0.00,
        "handling_per_qtl": 15.0,
        "citation": "HP Agricultural Produce Markets Act — HPSAMB Schedule",
    },
    "uttarakhand": {
        "state_name": "Uttarakhand",
        "market_fee_pct": 1.50,
        "dev_cess_pct": 0.50,
        "handling_per_qtl": 12.0,
        "citation": "Uttarakhand Agricultural Produce Marketing Act — UKAMB Schedule",
    },
    "default": {
        "state_name": "National Benchmark",
        "market_fee_pct": 1.50,
        "dev_cess_pct": 0.00,
        "handling_per_qtl": 15.0,
        "citation": "Model APMC Act (National Average Schedule)",
    },
}

# ---------------------------------------------------------------------------
# 2. State-wise Retail Diesel Rates (Ministry of Petroleum & PPAC Indices)
# ---------------------------------------------------------------------------

STATE_DIESEL_PRICES_PER_LITRE: Dict[str, float] = {
    "maharashtra": 92.50,
    "delhi": 87.62,
    "madhya pradesh": 93.44,
    "uttar pradesh": 89.80,
    "punjab": 87.80,
    "haryana": 88.35,
    "karnataka": 89.20,
    "rajasthan": 93.72,
    "gujarat": 90.10,
    "tamil nadu": 92.35,
    "andhra pradesh": 96.40,
    "telangana": 95.65,
    "bihar": 93.20,
    "west bengal": 91.76,
    "odisha": 93.05,
    "kerala": 94.75,
    "himachal pradesh": 88.50,
    "uttarakhand": 88.90,
    "assam": 89.50,
    "chhattisgarh": 93.30,
    "jharkhand": 93.00,
    "goa": 88.50,
    "jammu and kashmir": 88.20,
    "default": 90.00,
}

_PPAC_INDEX_FILE = (
    Path(__file__).parent.parent / "data" / "raw" / "ppac_diesel_state_index.json"
)


def _load_ppac_diesel_index() -> Dict[str, Dict[str, Any]]:
    """Load state-wise diesel retail selling prices directly from parsed PPAC official report."""
    state_dict: Dict[str, Dict[str, Any]] = {}
    if _PPAC_INDEX_FILE.exists():
        try:
            with open(_PPAC_INDEX_FILE, "r") as f:
                data = json.load(f)
                for state_key, state_data in data.items():
                    state_dict[state_key.lower().strip()] = state_data
            logger.info("Loaded %d state diesel rates from PPAC document: %s", len(state_dict), _PPAC_INDEX_FILE.name)
        except Exception as e:
            logger.warning("Could not load PPAC diesel index file: %s", e)
    return state_dict

# ---------------------------------------------------------------------------
# 3. ICAR - CIPHET Empirical Post-Harvest & Transit Loss Norms
# ---------------------------------------------------------------------------

CIPHET_BASE_WASTAGE_RATES: Dict[str, float] = {
    # Perishable Fruits & Vegetables
    "tomato": 0.085,       # 8.5% CIPHET baseline transit/handling loss
    "onion": 0.045,        # 4.5%
    "potato": 0.035,       # 3.5%
    "mango": 0.120,        # 12.0%
    "banana": 0.115,       # 11.5%
    "grapes": 0.100,       # 10.0%
    "guava": 0.110,        # 11.0%
    "apple": 0.080,        # 8.0%
    "orange": 0.095,       # 9.5%
    "cauliflower": 0.090,  # 9.0%
    "cabbage": 0.080,      # 8.0%
    "chilli": 0.065,       # 6.5%
    "garlic": 0.050,       # 5.0%
    "ginger": 0.055,       # 5.5%
    # Food Grains & Cereals (CIPHET Study on Harvest & Post-Harvest Losses)
    "wheat": 0.015,        # 1.5%
    "rice": 0.015,         # 1.5%
    "paddy": 0.018,        # 1.8%
    "maize": 0.020,        # 2.0%
    # Oilseeds & Cash Crops
    "soybean": 0.020,      # 2.0%
    "soyabean": 0.020,     # 2.0%
    "cotton": 0.010,       # 1.0%
    "mustard": 0.018,      # 1.8%
    "sugarcane": 0.030,    # 3.0%
    "gram": 0.015,         # 1.5%
    "default": 0.050,      # 5.0% national benchmark
}


# ---------------------------------------------------------------------------
# Result Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DynamicMandiPriceResult:
    modal_price: float
    min_price: float
    max_price: float
    commodity: str
    market: str
    state: str
    arrival_date: str
    is_live: bool
    provenance_note: str
    provenance_note_hi: str = ""


@dataclass
class DynamicRouteResult:
    distance_km: float
    duration_hours: float
    source_state: str
    destination_state: str
    src_lat: float
    src_lon: float
    dst_lat: float
    dst_lon: float
    is_live: bool
    provenance_note: str
    provenance_note_hi: str = ""


@dataclass
class DynamicTaxResult:
    state_name: str
    total_tax_pct: float
    market_fee_pct: float
    dev_cess_pct: float
    handling_per_qtl: float
    citation: str
    provenance_note: str
    provenance_note_hi: str = ""


@dataclass
class DynamicTariffResult:
    tariff_per_qtl_km: float
    diesel_rate_used: float
    tier: str
    provenance_note: str
    provenance_note_hi: str = ""


@dataclass
class DynamicWastageResult:
    wastage_rate: float
    ambient_temp_c: float
    ambient_humidity_pct: float
    is_perishable: bool
    provenance_note: str
    provenance_note_hi: str = ""


@dataclass
class DynamicSubsidyResult:
    """Full breakdown of applicable government scheme benefits for a transaction."""
    scheme_name: str
    transport_subsidy_amount: float       # ₹ offset against freight cost
    mandi_fee_waiver_amount: float        # ₹ offset against APMC tax + handling
    price_support_amount: float           # ₹ deficiency payment (PM-AASHA PDPS)
    total_subsidy_benefit: float          # sum of all three
    is_eligible: bool
    citation: str
    provenance_note: str
    msp_benchmark_per_qtl: float = 0.0    # Statutory MSP if crop has one
    msp_advisory_note: str = ""           # PSS/FCI advisory if non-PDPS below MSP
    channel_details: str = ""             # Kisan Rail / Road / FPO eligibility note
    provenance_note_hi: str = ""
    msp_advisory_note_hi: str = ""
    channel_details_hi: str = ""


_STATE_HI_NAMES: Dict[str, str] = {
    "maharashtra": "महाराष्ट्र",
    "delhi": "दिल्ली (NCT)",
    "nct of delhi": "दिल्ली (NCT)",
    "delhi (nct)": "दिल्ली (NCT)",
    "madhya pradesh": "मध्य प्रदेश",
    "uttar pradesh": "उत्तर प्रदेश",
    "punjab": "पंजाब",
    "haryana": "हरियाणा",
    "karnataka": "कर्नाटक",
    "rajasthan": "राजस्थान",
    "gujarat": "गुजरात",
    "tamil nadu": "तमिलनाडु",
    "andhra pradesh": "आंध्र प्रदेश",
    "telangana": "तेलंगाना",
    "bihar": "बिहार",
    "west bengal": "पश्चिम बंगाल",
    "odisha": "ओडिशा",
    "kerala": "केरल",
    "keralam": "केरल",
    "himachal pradesh": "हिमाचल प्रदेश",
    "uttarakhand": "उत्तराखंड",
    "national benchmark": "राष्ट्रीय मानक",
    "default": "राष्ट्रीय मानक",
}

_CROP_HI_NAMES: Dict[str, str] = {
    "tomato": "टमाटर",
    "onion": "प्याज",
    "potato": "आलू",
    "wheat": "गेहूं",
    "rice": "चावल",
    "paddy": "धान",
    "maize": "मक्का",
    "soybean": "सोयाबीन",
    "soyabean": "सोयाबीन",
    "cotton": "कपास",
    "sugarcane": "गन्ना",
    "mustard": "सरसों",
    "groundnut": "मूंगफली",
    "gram": "चना",
    "tur": "अरहर / तूर",
    "arhar": "अरहर / तूर",
    "moong": "मूंग",
    "urad": "उड़द",
    "sunflower": "सूरजमुखी",
    "mango": "आम",
    "banana": "केला",
    "grapes": "अंगूर",
    "garlic": "लहसुन",
}

_METRO_REGION_DISTRICTS: Dict[str, Tuple[str, str, Tuple[str, ...]]] = {
    "delhi": (
        "Delhi-NCR",
        "दिल्ली-एनसीआर",
        (
            "gurgaon", "gurugram", "faridabad", "sonipat", "sonepat",
            "ghaziabad", "gautam budh", "noida", "bahadurgarh",
            "jhajjar", "baghpat", "palwal", "meerut",
        ),
    ),
    "mumbai": (
        "Mumbai Metropolitan",
        "मुंबई महानगर",
        ("mumbai", "thane", "raigad", "palghar", "navi mumbai", "vashi", "kalyan", "panvel"),
    ),
    "bengaluru": (
        "Bengaluru Metro",
        "बेंगलुरु क्षेत्र",
        ("bengaluru", "bangalore", "kolar", "ramanagara", "chikkaballapur"),
    ),
    "hyderabad": (
        "Hyderabad Metro",
        "हैदराबाद क्षेत्र",
        ("hyderabad", "rangareddy", "medchal", "sangareddy"),
    ),
}

_NEIGHBORING_STATES: Dict[str, Tuple[str, ...]] = {
    "delhi": ("haryana", "uttar pradesh", "rajasthan"),
    "maharashtra": ("madhya pradesh", "gujarat", "karnataka", "telangana"),
    "madhya pradesh": ("maharashtra", "uttar pradesh", "rajasthan", "gujarat"),
    "uttar pradesh": ("haryana", "delhi", "rajasthan", "madhya pradesh", "bihar"),
    "punjab": ("haryana", "himachal pradesh", "rajasthan"),
    "haryana": ("punjab", "uttar pradesh", "rajasthan", "delhi"),
    "karnataka": ("maharashtra", "andhra pradesh", "telangana", "tamil nadu"),
    "gujarat": ("maharashtra", "rajasthan", "madhya pradesh"),
}


def _state_to_hi(state_name: str) -> str:
    clean = (state_name or "default").strip().lower()
    if clean in _STATE_HI_NAMES:
        return _STATE_HI_NAMES[clean]
    for k, v in _STATE_HI_NAMES.items():
        if k in clean or clean in k:
            return v
    return state_name.title()


def _crop_to_hi(crop_name: str) -> str:
    return _CROP_HI_NAMES.get((crop_name or "").strip().lower(), crop_name.title())


# ---------------------------------------------------------------------------
# Resolver Class
# ---------------------------------------------------------------------------

class DynamicVariableResolver:
    """
    Central resolver for live agricultural market & logistics parameters.
    """

    def __init__(self):
        self.ppac_state_data = _load_ppac_diesel_index()
        logger.info("DynamicVariableResolver initialized (PPAC states loaded: %d).", len(self.ppac_state_data))

    # -----------------------------------------------------------------------
    # 0. Live Mandi Prices (Agmarknet via data.gov.in)
    # -----------------------------------------------------------------------

    def _fetch_ogd_records(
        self, base_url: str, params: Dict[str, Any], timeout_sec: float = 10.0
    ) -> list:
        """Query data.gov.in with browser User-Agent and one retry on transient timeout."""
        req_headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
        }
        for attempt in range(2):
            try:
                resp = requests.get(base_url, params=params, headers=req_headers, timeout=timeout_sec)
                if resp.status_code == 200:
                    return resp.json().get("records", [])
            except Exception as e:
                if attempt == 0:
                    logger.debug("data.gov.in query attempt 1 failed (%s); retrying…", e)
                else:
                    logger.warning("data.gov.in query failed: %s", e)
        return []

    @staticmethod
    def _extract_valid_prices(subset: list) -> list:
        prices = []
        for r in subset:
            try:
                p = float(str(r.get("modal_price", 0)).replace(",", ""))
                if p > 0:
                    prices.append(p)
            except (ValueError, TypeError):
                continue
        return prices

    def _build_live_benchmark(
        self,
        subset: list,
        norm_commodity: str,
        market_label: str,
        state_label: str,
        scope_en: str,
        scope_hi: str,
        cache_key: Tuple[str, str],
    ) -> Optional[DynamicMandiPriceResult]:
        valid_prices = self._extract_valid_prices(subset)
        if not valid_prices:
            return None
        avg_modal = round(sum(valid_prices) / len(valid_prices), 2)
        arr_date = subset[0].get("arrival_date", "Today")
        n_mandis = len(valid_prices)
        note = (
            f"Mandi Price: ₹{avg_modal:,.2f}/qtl [LIVE Agmarknet {scope_en} "
            f"via data.gov.in ({n_mandis} mandis reporting), Date: {arr_date}]"
        )
        note_hi = (
            f"मंडी भाव: ₹{avg_modal:,.2f}/क्विंटल [लाइव एगमार्कनेट {scope_hi} "
            f"— data.gov.in ({n_mandis} मंडियों का औसत), दिनांक: {arr_date}]"
        )
        res = DynamicMandiPriceResult(
            modal_price=avg_modal,
            min_price=min(valid_prices),
            max_price=max(valid_prices),
            commodity=norm_commodity,
            market=market_label,
            state=state_label,
            arrival_date=arr_date,
            is_live=True,
            provenance_note=note,
            provenance_note_hi=note_hi,
        )
        _MANDI_PRICE_CACHE[cache_key] = res
        return res

    def resolve_mandi_price(
        self,
        commodity: str,
        destination_market: str,
        destination_state: Optional[str] = None,
    ) -> Optional[DynamicMandiPriceResult]:
        """
        Fetch real-time daily mandi modal price from Agmarknet API on data.gov.in.

        Hierarchy within LIVE Agmarknet (before falling back to historical CSV):
          Tier 1A: Exact / substring market or district match in live records
          Tier 1B: Destination state live average
          Tier 1C: Metro/NCR or neighboring-state live regional benchmark
          Tier 1D: Live national Agmarknet benchmark across reporting mandis today
        """
        cache_key = (commodity.strip().lower(), destination_market.strip().lower())
        if cache_key in _MANDI_PRICE_CACHE:
            return _MANDI_PRICE_CACHE[cache_key]

        api_key = os.getenv("DATA_GOV_API_KEY")
        if not api_key:
            logger.debug("DATA_GOV_API_KEY not found in environment; skipping live mandi price fetch.")
            return None

        # Canonical commodity mapping for data.gov.in
        synonyms = {
            "soybean": "Soyabean",
            "soyabean": "Soyabean",
            "paddy": "Paddy(Dhan)(Common)",
            "chana": "Bengal Gram(Gram)(Whole)",
            "gram": "Bengal Gram(Gram)(Whole)",
            "tur": "Red gram/Arhar/Tur(whole)",
            "arhar": "Red gram/Arhar/Tur(whole)",
            "moong": "Green Gram(Moong)(Whole)",
            "urad": "Black Gram(Urd Beans)(Whole)",
            "chilli": "Green Chilli",
        }
        c_clean = commodity.strip().lower()
        norm_commodity = synonyms.get(c_clean, commodity.strip().title())
        m_clean = destination_market.strip().lower()
        m_stem = m_clean.replace("apmc", "").strip()

        resource_id = "9ef84268-d588-465a-a308-a864a43d0070"
        base_url = f"https://api.data.gov.in/resource/{resource_id}"

        state_aliases = {
            "Delhi": "NCT of Delhi",
            "Kerala": "Keralam",
            "Orissa": "Odisha",
            "Chhattisgarh": "Chattisgarh",
        }
        records: list = []

        # Strategy A: Query commodity + state first for speed when state matches
        if destination_state and destination_state.lower() != "default":
            s_title = destination_state.strip().title()
            query_state = state_aliases.get(s_title, s_title)
            records = self._fetch_ogd_records(
                base_url,
                {
                    "api-key": api_key,
                    "format": "json",
                    "limit": 100,
                    "filters[commodity]": norm_commodity,
                    "filters[state]": query_state,
                },
                timeout_sec=8.0,
            )

        # Strategy B: If state query had no exact records, fetch nationwide live records (limit=500)
        if not records:
            records = self._fetch_ogd_records(
                base_url,
                {
                    "api-key": api_key,
                    "format": "json",
                    "limit": 500,
                    "filters[commodity]": norm_commodity,
                },
                timeout_sec=10.0,
            )

        if not records:
            logger.info("No live mandi records found on data.gov.in for commodity=%s, market=%s", norm_commodity, destination_market)
            return None

        # Tier 1A: Exact or substring market/district match
        matched_record = None
        for r in records:
            r_market = r.get("market", "").lower()
            r_dist = r.get("district", "").lower()
            if (
                m_clean in r_market
                or r_market in m_clean
                or (m_stem and m_stem in r_market)
                or (m_stem and m_stem in r_dist)
            ):
                matched_record = r
                break

        if matched_record:
            try:
                modal_p = float(str(matched_record.get("modal_price", 0)).replace(",", ""))
                min_p = float(str(matched_record.get("min_price", 0)).replace(",", ""))
                max_p = float(str(matched_record.get("max_price", 0)).replace(",", ""))
                arr_date = matched_record.get("arrival_date", "Today")
                mkt_name = matched_record.get("market", destination_market).strip()
                st_name = matched_record.get("state", destination_state or "").strip()

                if modal_p > 0:
                    note = (
                        f"Mandi Price: ₹{modal_p:,.2f}/qtl [LIVE Agmarknet OGD API (data.gov.in) "
                        f"— {mkt_name}, {st_name}, Date: {arr_date}]"
                    )
                    note_hi = (
                        f"मंडी भाव: ₹{modal_p:,.2f}/क्विंटल [लाइव एगमार्कनेट (data.gov.in) "
                        f"— {mkt_name}, {_state_to_hi(st_name)}, दिनांक: {arr_date}]"
                    )
                    res = DynamicMandiPriceResult(
                        modal_price=modal_p,
                        min_price=min_p,
                        max_price=max_p,
                        commodity=norm_commodity,
                        market=mkt_name,
                        state=st_name,
                        arrival_date=arr_date,
                        is_live=True,
                        provenance_note=note,
                        provenance_note_hi=note_hi,
                    )
                    _MANDI_PRICE_CACHE[cache_key] = res
                    return res
            except (ValueError, TypeError) as e:
                logger.warning("Error parsing live mandi price values: %s", e)

        # Tier 1B: Destination state live benchmark
        s_clean = (destination_state or "").strip().lower()
        if s_clean and s_clean != "default":
            state_records = [
                r for r in records
                if s_clean in r.get("state", "").lower() or r.get("state", "").lower() in s_clean
            ]
            if state_records:
                st_name = state_records[0].get("state", destination_state).strip()
                res = self._build_live_benchmark(
                    subset=state_records,
                    norm_commodity=norm_commodity,
                    market_label=f"{destination_market} (Regional {st_name})",
                    state_label=st_name,
                    scope_en=f"Regional Benchmark for {st_name}",
                    scope_hi=f"{_state_to_hi(st_name)} क्षेत्रीय औसत",
                    cache_key=cache_key,
                )
                if res:
                    return res

        # Tier 1C: Metro / NCR or neighboring-state live regional benchmark
        metro_key = None
        if m_stem in ("azadpur", "delhi", "new delhi", "keshopur", "narela") or s_clean == "delhi":
            metro_key = "delhi"
        elif m_stem in _METRO_REGION_DISTRICTS:
            metro_key = m_stem

        if metro_key and metro_key in _METRO_REGION_DISTRICTS:
            label_en, label_hi, dist_tokens = _METRO_REGION_DISTRICTS[metro_key]
            metro_records = [
                r for r in records
                if any(
                    tok in (r.get("district", "") + " " + r.get("market", "")).lower()
                    for tok in dist_tokens
                )
            ]
            if metro_records:
                res = self._build_live_benchmark(
                    subset=metro_records,
                    norm_commodity=norm_commodity,
                    market_label=f"{destination_market} ({label_en} Live)",
                    state_label=label_en,
                    scope_en=f"{label_en} Benchmark (nearest live NCR/Metro mandis)",
                    scope_hi=f"{label_hi} की निकटतम लाइव मंडियों का औसत",
                    cache_key=cache_key,
                )
                if res:
                    return res

        if s_clean in _NEIGHBORING_STATES:
            nbrs = _NEIGHBORING_STATES[s_clean]
            nbr_records = [
                r for r in records
                if any(n in r.get("state", "").lower() for n in nbrs)
            ]
            if nbr_records:
                res = self._build_live_benchmark(
                    subset=nbr_records,
                    norm_commodity=norm_commodity,
                    market_label=f"{destination_market} (Adjacent States)",
                    state_label=destination_state or "Regional",
                    scope_en=f"Neighboring States Benchmark (nearest live around {destination_market.title()})",
                    scope_hi=f"{destination_market.title()} के निकटवर्ती राज्यों की लाइव मंडियों का औसत",
                    cache_key=cache_key,
                )
                if res:
                    return res

        # Tier 1D: Live national Agmarknet benchmark across all live records today
        return self._build_live_benchmark(
            subset=records,
            norm_commodity=norm_commodity,
            market_label=f"{destination_market} (All-India Live Benchmark)",
            state_label=destination_state or "India",
            scope_en="National Benchmark",
            scope_hi="राष्ट्रीय लाइव मंडी औसत",
            cache_key=cache_key,
        )

    # -----------------------------------------------------------------------
    # Geocoding & Highway Routing
    # -----------------------------------------------------------------------

    def _geocode_city(self, city_name: str) -> Tuple[float, float, str]:
        """Geocode city using OpenStreetMap Nominatim with state extraction."""
        key = city_name.strip().lower()
        if key in _GEOCODE_CACHE:
            return _GEOCODE_CACHE[key]

        try:
            url = f"https://nominatim.openstreetmap.org/search?q={requests.utils.quote(city_name)},India&format=json&limit=1"
            resp = requests.get(url, headers=_HEADERS, timeout=_HTTP_TIMEOUT)
            if resp.status_code == 200:
                data = resp.json()
                if data:
                    lat = float(data[0]["lat"])
                    lon = float(data[0]["lon"])
                    display = data[0].get("display_name", "")
                    # Extract state from address components (usually 2nd or 3rd from right)
                    parts = [p.strip() for p in display.split(",")]
                    state = ""
                    for p in reversed(parts[:-1]):  # exclude country "India"
                        p_lower = p.lower()
                        if p_lower in STATE_APMC_SCHEDULE:
                            state = p_lower
                            break
                    if not state and len(parts) >= 2:
                        state = parts[-2].lower()

                    _GEOCODE_CACHE[key] = (lat, lon, state)
                    return lat, lon, state
        except Exception as e:
            logger.warning("Geocoding failed for '%s': %s", city_name, e)

        # Fallback coordinates for key Indian mandis
        curated_fallbacks = {
            "bhopal": (23.2599, 77.4126, "madhya pradesh"),
            "mumbai": (19.0760, 72.8777, "maharashtra"),
            "pune": (18.5204, 73.8567, "maharashtra"),
            "delhi": (28.7041, 77.1025, "delhi"),
            "azadpur": (28.7077, 77.1758, "delhi"),
            "nashik": (19.9975, 73.7898, "maharashtra"),
            "lasalgaon": (20.1472, 74.2255, "maharashtra"),
            "indore": (22.7196, 75.8577, "madhya pradesh"),
            "lucknow": (26.8467, 80.9462, "uttar pradesh"),
            "agra": (27.1767, 78.0081, "uttar pradesh"),
            "amritsar": (31.6340, 74.8723, "punjab"),
            "ludhiana": (30.9010, 75.8573, "punjab"),
            "jaipur": (26.9124, 75.7873, "rajasthan"),
            "ahmedabad": (23.0225, 72.5714, "gujarat"),
            "bengaluru": (12.9716, 77.5946, "karnataka"),
            "hyderabad": (17.3850, 78.4867, "telangana"),
            "chennai": (13.0827, 80.2707, "tamil nadu"),
            "kolkata": (22.5726, 88.3639, "west bengal"),
            "patna": (25.5941, 85.1376, "bihar"),
            "guntur": (16.3067, 80.4365, "andhra pradesh"),
            "nagpur": (21.1458, 79.0882, "maharashtra"),
            "mandsaur": (24.0723, 75.0682, "madhya pradesh"),
            "ganaur": (29.1306, 77.0194, "haryana"),
            "shimla": (31.1048, 77.1734, "himachal pradesh"),
        }
        res = curated_fallbacks.get(key, (20.5937, 78.9629, "default"))
        _GEOCODE_CACHE[key] = res
        return res

    def resolve_route(self, source_market: str, destination_market: str) -> DynamicRouteResult:
        """
        Dynamically calculate road driving distance (km) and commercial truck travel duration
        via Open Source Routing Machine (OSRM) highway network.
        """
        cache_key = (source_market.strip().lower(), destination_market.strip().lower())
        src_lat, src_lon, src_state = self._geocode_city(source_market)
        dst_lat, dst_lon, dst_state = self._geocode_city(destination_market)

        if cache_key in _ROUTE_CACHE:
            dist_km, dur_hrs = _ROUTE_CACHE[cache_key]
            return DynamicRouteResult(
                distance_km=dist_km,
                duration_hours=dur_hrs,
                source_state=src_state,
                destination_state=dst_state,
                src_lat=src_lat,
                src_lon=src_lon,
                dst_lat=dst_lat,
                dst_lon=dst_lon,
                is_live=True,
                provenance_note=f"Road Distance: {dist_km:.1f} km ({dur_hrs:.1f}h truck transit via OSRM National Highway Network)",
                provenance_note_hi=f"सड़क दूरी: {dist_km:.1f} किमी ({dur_hrs:.1f} घंटे ट्रक यात्रा — OSRM राष्ट्रीय राजमार्ग नेटवर्क)",
            )

        try:
            route_url = (
                f"http://router.project-osrm.org/route/v1/driving/"
                f"{src_lon},{src_lat};{dst_lon},{dst_lat}?overview=false"
            )
            resp = requests.get(route_url, timeout=_HTTP_TIMEOUT)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("routes"):
                    r = data["routes"][0]
                    dist_km = round(r["distance"] / 1000.0, 1)
                    raw_dur_hrs = round(r["duration"] / 3600.0, 1)
                    # Commercial truck transit calibration (avg 42-48 km/h on Indian NH/SH including toll plazas)
                    dur_hrs = round(max(raw_dur_hrs * 1.35, dist_km / 48.0), 1)
                    _ROUTE_CACHE[cache_key] = (dist_km, dur_hrs)
                    return DynamicRouteResult(
                        distance_km=dist_km,
                        duration_hours=dur_hrs,
                        source_state=src_state,
                        destination_state=dst_state,
                        src_lat=src_lat,
                        src_lon=src_lon,
                        dst_lat=dst_lat,
                        dst_lon=dst_lon,
                        is_live=True,
                        provenance_note=f"Road Distance: {dist_km:.1f} km ({dur_hrs:.1f}h truck transit via OSRM National Highway Network)",
                        provenance_note_hi=f"सड़क दूरी: {dist_km:.1f} किमी ({dur_hrs:.1f} घंटे ट्रक यात्रा — OSRM राष्ट्रीय राजमार्ग नेटवर्क)",
                    )
        except Exception as e:
            logger.warning("OSRM highway routing query failed: %s", e)

        # Haversine approximate road distance (straight line * 1.3 road curvature factor)
        dlat = math.radians(dst_lat - src_lat)
        dlon = math.radians(dst_lon - src_lon)
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(math.radians(src_lat))
            * math.cos(math.radians(dst_lat))
            * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        straight_km = 6371.0 * c
        road_km = round(max(straight_km * 1.30, 50.0), 1)
        dur_hrs = round(road_km / 42.0, 1)

        return DynamicRouteResult(
            distance_km=road_km,
            duration_hours=dur_hrs,
            source_state=src_state,
            destination_state=dst_state,
            src_lat=src_lat,
            src_lon=src_lon,
            dst_lat=dst_lat,
            dst_lon=dst_lon,
            is_live=False,
            provenance_note=f"Road Distance: {road_km:.1f} km (~{dur_hrs:.1f}h truck transit estimated via geospatial coordinate network)",
            provenance_note_hi=f"सड़क दूरी: {road_km:.1f} किमी (~{dur_hrs:.1f} घंटे ट्रक यात्रा — भौगोलिक निर्देशांक अनुमान)",
        )

    # -----------------------------------------------------------------------
    # 2. Statutory APMC Tax & Handling Charges
    # -----------------------------------------------------------------------

    def resolve_apmc_tax(self, dest_state: str, dest_market: str) -> DynamicTaxResult:
        """
        Dynamically determine APMC market fee, development cess, and handling
        charges based on the destination state's notified legislative APMC schedule.
        """
        clean_state = dest_state.strip().lower()
        matched_sched = STATE_APMC_SCHEDULE.get(clean_state)

        if not matched_sched:
            for s_key, s_data in STATE_APMC_SCHEDULE.items():
                if s_key in clean_state or clean_state in s_key:
                    matched_sched = s_data
                    break

        if not matched_sched:
            matched_sched = STATE_APMC_SCHEDULE["default"]

        mkt_fee = matched_sched["market_fee_pct"]
        dev_cess = matched_sched["dev_cess_pct"]
        total_tax_pct = round(mkt_fee + dev_cess, 2)
        handling = matched_sched["handling_per_qtl"]
        citation = matched_sched["citation"]

        note = (
            f"APMC Tax: {total_tax_pct:.2f}% ({mkt_fee:.2f}% Market Fee + "
            f"{dev_cess:.2f}% Cess) as notified under {citation}"
        )
        note_hi = (
            f"मंडी कर (APMC): {total_tax_pct:.2f}% ({mkt_fee:.2f}% मंडी शुल्क + "
            f"{dev_cess:.2f}% उपकर) — {_state_to_hi(matched_sched['state_name'])} मंडी अधिनियम के अनुसार"
        )

        return DynamicTaxResult(
            state_name=matched_sched["state_name"],
            total_tax_pct=total_tax_pct,
            market_fee_pct=mkt_fee,
            dev_cess_pct=dev_cess,
            handling_per_qtl=handling,
            citation=citation,
            provenance_note=note,
            provenance_note_hi=note_hi,
        )

    # -----------------------------------------------------------------------
    # 3. Dynamic Fuel-Indexed Transport Tariff
    # -----------------------------------------------------------------------

    def resolve_tariff(
        self,
        distance_km: float,
        dest_state: str,
        quantity_quintals: float = 10.0,
    ) -> DynamicTariffResult:
        """
        Calculates transport tariff (₹/quintal/km) dynamically based on:
        - Ministry of Petroleum & Natural Gas (PPAC) diesel index in destination state
        - Payload tonnage vehicle class (SCV / Bolero Pickup vs MCV vs Heavy FTL Multi-Axle)
        - Distance economy-of-scale tiering (long haul vs short haul)
        """
        clean_state = dest_state.strip().lower()
        
        # Check parsed official PPAC document report first
        diesel_price = None
        source_citation = "PPAC Ready Reckoner (IOCL)"
        if hasattr(self, "ppac_state_data") and self.ppac_state_data:
            state_entry = self.ppac_state_data.get(clean_state)
            if not state_entry:
                for k, v in self.ppac_state_data.items():
                    if k in clean_state or clean_state in k:
                        state_entry = v
                        break
            if state_entry:
                diesel_price = float(state_entry.get("diesel_price_per_litre", 0))
                source_citation = "PPAC Table 8.14 PDF (IOCL)"

        if not diesel_price or diesel_price <= 0:
            diesel_price = STATE_DIESEL_PRICES_PER_LITRE.get(
                clean_state, STATE_DIESEL_PRICES_PER_LITRE["default"]
            )
            source_citation = "PPAC Benchmark Index"

        base_diesel = 90.0  # ₹90/L national reference baseline

        # Vehicle selection & payload scale
        if quantity_quintals >= 80:     # >= 8 MT: Heavy Full Truckload (10/12-wheel multi-axle)
            vehicle_type = "Heavy FTL Multi-Axle Truck (>8MT)"
            vehicle_type_hi = "भारी मल्टी-एक्सल ट्रक (>8 टन)"
            base_tariff = 0.68 if distance_km >= 600 else 0.78
        elif quantity_quintals >= 25:   # 2.5 - 8 MT: Medium Commercial Vehicle (6-wheel Eicher)
            vehicle_type = "Medium Commercial Vehicle (MCV 3-8MT)"
            vehicle_type_hi = "मध्यम मालवाहक वाहन (3-8 टन)"
            base_tariff = 0.76 if distance_km >= 600 else 0.86
        else:                           # < 2.5 MT: Light Commercial Vehicle (Tata Ace / Bolero Pickup)
            vehicle_type = "Short-haul LCV / Pickup (<2.5MT)"
            vehicle_type_hi = "छोटा पिकअप वाहन (<2.5 टन)"
            base_tariff = 0.85 if distance_km >= 600 else 0.94

        # Fuel weighting in road logistics = ~45% of running cost
        fuel_multiplier = 1.0 + 0.45 * ((diesel_price - base_diesel) / base_diesel)
        adjusted_tariff = round(base_tariff * fuel_multiplier, 2)

        note = (
            f"Transport Tariff: ₹{adjusted_tariff:.2f}/q/km [{vehicle_type}, "
            f"{distance_km:.0f}km haul, indexed to {source_citation} HSD diesel @ ₹{diesel_price:.2f}/L in {clean_state.title()}]"
        )
        note_hi = (
            f"परिवहन दर: ₹{adjusted_tariff:.2f}/क्विंटल/किमी [{vehicle_type_hi}, "
            f"{distance_km:.0f} किमी दूरी, {_state_to_hi(clean_state)} में PPAC डीजल ₹{diesel_price:.2f}/लीटर पर आधारित]"
        )

        return DynamicTariffResult(
            tariff_per_qtl_km=adjusted_tariff,
            diesel_rate_used=diesel_price,
            tier=vehicle_type,
            provenance_note=note,
            provenance_note_hi=note_hi,
        )

    # -----------------------------------------------------------------------
    # 4. Perishability & Weather-Aware Transit Wastage
    # -----------------------------------------------------------------------

    def _fetch_weather_conditions(self, lat: float, lon: float) -> Tuple[float, float]:
        """Fetch current ambient temperature (°C) and relative humidity (%) via Open-Meteo API."""
        rounded_key = (round(lat, 2), round(lon, 2))
        if rounded_key in _WEATHER_CACHE:
            return _WEATHER_CACHE[rounded_key]

        try:
            url = (
                f"https://api.open-meteo.com/v1/forecast?"
                f"latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m"
            )
            resp = requests.get(url, timeout=_HTTP_TIMEOUT)
            if resp.status_code == 200:
                data = resp.json()
                curr = data.get("current", {})
                temp = float(curr.get("temperature_2m", 25.0))
                humidity = float(curr.get("relative_humidity_2m", 55.0))
                _WEATHER_CACHE[rounded_key] = (temp, humidity)
                return temp, humidity
        except Exception as e:
            logger.debug("Open-Meteo weather query failed: %s", e)

        return 26.0, 55.0  # sensible ambient defaults

    def resolve_wastage(
        self,
        commodity: str,
        distance_km: float,
        duration_hours: float,
        src_lat: float,
        src_lon: float,
        dst_lat: float,
        dst_lon: float,
    ) -> DynamicWastageResult:
        """
        Calculates expected post-harvest transit loss dynamically using:
        - ICAR - CIPHET statutory baseline loss rates
        - Real-time Open-Meteo ambient temperature & relative humidity across the transit corridor
        - Transit duration thermal-respiratory degradation factor
        """
        c_clean = commodity.strip().lower()
        base_rate = CIPHET_BASE_WASTAGE_RATES.get(
            c_clean, CIPHET_BASE_WASTAGE_RATES["default"]
        )

        perishable_crops = {
            "tomato", "banana", "mango", "grapes", "guava",
            "apple", "orange", "cauliflower", "cabbage", "chilli", "vegetables"
        }
        is_perishable = c_clean in perishable_crops

        # Fetch ambient temperature and humidity at transit endpoints
        t_src, h_src = self._fetch_weather_conditions(src_lat, src_lon)
        t_dst, h_dst = self._fetch_weather_conditions(dst_lat, dst_lon)
        avg_temp = round((t_src + t_dst) / 2.0, 1)
        avg_humidity = round((h_src + h_dst) / 2.0, 1)

        if is_perishable:
            # Thermal multiplier: produce respiration accelerates above 25°C
            temp_ratio = max(avg_temp / 25.0, 0.8)
            thermal_factor = temp_ratio ** 0.55

            # Humidity factor: extremes accelerate rot or desiccation
            if avg_humidity > 80.0:
                humidity_factor = 1.0 + 0.12 * ((avg_humidity - 80.0) / 20.0)
            elif avg_humidity < 40.0:
                humidity_factor = 1.0 + 0.08 * ((40.0 - avg_humidity) / 40.0)
            else:
                humidity_factor = 1.0

            # Distance duration factor: longer transit exposes delicate cells to vibration/heat
            duration_factor = 1.0 + 0.10 * (duration_hours / 12.0)
            adjusted_rate = round(min(base_rate * thermal_factor * humidity_factor * duration_factor, 0.28), 3)

            reefer_tip = " [Reefer recommended for >400km]" if distance_km >= 400 else ""
            reefer_tip_hi = " [400 किमी से अधिक दूरी के लिए शीत/रीफर वाहन की सलाह]" if distance_km >= 400 else ""
            note = (
                f"Transit Wastage: {adjusted_rate * 100:.1f}% [ICAR-CIPHET base {base_rate * 100:.1f}% + "
                f"weather adjustment for {duration_hours:.1f}h transit at {avg_temp:.1f}°C, {avg_humidity:.0f}% RH via Open-Meteo]{reefer_tip}"
            )
            note_hi = (
                f"यात्रा में खराबी/नुकसान: {adjusted_rate * 100:.1f}% [ICAR-CIPHET मानक {base_rate * 100:.1f}% + "
                f"{duration_hours:.1f} घंटे सफर व मौसम ({avg_temp:.1f}°C तापमान, {avg_humidity:.0f}% नमी — Open-Meteo) के आधार पर]{reefer_tip_hi}"
            )
        else:
            # Grains / Non-perishable have minimal distance degradation
            distance_factor = 1.0 + 0.02 * (distance_km / 500.0)
            adjusted_rate = round(base_rate * distance_factor, 3)
            note = (
                f"Transit Wastage: {adjusted_rate * 100:.1f}% [ICAR-CIPHET post-harvest loss norm for {commodity.title()}]"
            )
            note_hi = (
                f"यात्रा में नुकसान: {adjusted_rate * 100:.1f}% [{_crop_to_hi(commodity)} के लिए ICAR-CIPHET कटाई-उपरांत मानक]"
            )

        return DynamicWastageResult(
            wastage_rate=adjusted_rate,
            ambient_temp_c=avg_temp,
            ambient_humidity_pct=avg_humidity,
            is_perishable=is_perishable,
            provenance_note=note,
            provenance_note_hi=note_hi,
        )


# ===========================================================================
# V6: Government Subsidies & Cost-Reduction Schemes
# ===========================================================================
# Data Sources & Provenance:
#   1. MoFPI Operation Greens (TOP to TOTAL) — Official Guidelines & PIB releases
#      URL: https://mofpi.gov.in/schemes/operation-greens
#      Legal basis: MoFPI Operational Guidelines under PM Kisan SAMPADA Yojana
#   2. Kisan Rail — 50% freight concession for notified perishables
#      URL: https://vikaspedia.in/agriculture/market-information/kisan-rail
#      Legal basis: MoFPI Operation Greens + Indian Railways notification
#   3. PM-AASHA (PDPS) — Price Deficiency Payment Scheme for oilseeds/pulses
#      URL: https://agriwelfare.gov.in & https://pib.gov.in CCEA approvals
#      Legal basis: DA&FW PM-AASHA Operational Scheme Guidelines
#   4. MSP Rates — CACP/CCEA approved rates from desagri.gov.in
#      Kharif 2025-26 & Rabi 2025-26 (Marketing Season)
#   5. State APMC FPO Exemptions — State APMC Amendment Acts & DMI Gazette
#      URL: https://dmi.gov.in & respective State SAMB portals
# ===========================================================================

# ---------------------------------------------------------------------------
# 5a. CACP/CCEA Minimum Support Prices (₹ per quintal) — 2024-25 / 2025-26
# ---------------------------------------------------------------------------
# Source: Directorate of Economics & Statistics (desagri.gov.in)
#         Press Information Bureau (pib.gov.in) CCEA approval notifications

_GOVT_SUBSIDIES_INDEX_FILE = (
    Path(__file__).parent.parent / "data" / "raw" / "govt_subsidies_index.json"
)


def _load_govt_subsidies_index() -> Dict[str, Any]:
    """Load structured government subsidy schedules from JSON dataset (like PPAC)."""
    if _GOVT_SUBSIDIES_INDEX_FILE.exists():
        try:
            with open(_GOVT_SUBSIDIES_INDEX_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info(
                "Loaded statutory subsidy dataset from %s (version=%s)",
                _GOVT_SUBSIDIES_INDEX_FILE.name,
                data.get("metadata", {}).get("version", "unknown"),
            )
            return data
        except Exception as e:
            logger.warning("Could not load govt subsidies index file: %s", e)
    return {}


GOVT_MSP_BENCHMARKS: Dict[str, Dict[str, Any]] = {
    # ── Kharif Crops (KMS 2025-26) ──
    "paddy": {
        "msp": 2369.0, "grade_a_msp": 2389.0, "season": "Kharif",
        "category": "cereal", "intervention_type": "MSP_PROCUREMENT",
        "citation": "CCEA KMS 2025-26 (desagri.gov.in)",
    },
    "rice": {
        "msp": 2369.0, "grade_a_msp": 2389.0, "season": "Kharif",
        "category": "cereal", "intervention_type": "MSP_PROCUREMENT",
        "citation": "CCEA KMS 2025-26 (desagri.gov.in)",
    },
    "maize": {
        "msp": 2225.0, "season": "Kharif",
        "category": "cereal", "intervention_type": "MSP_PROCUREMENT",
        "citation": "CCEA KMS 2024-25 (desagri.gov.in)",
    },
    "soybean": {
        "msp": 5328.0, "season": "Kharif",
        "category": "oilseed", "intervention_type": "PDPS",
        "citation": "CCEA KMS 2025-26 Yellow Soybean (pib.gov.in)",
    },
    "soyabean": {
        "msp": 5328.0, "season": "Kharif",
        "category": "oilseed", "intervention_type": "PDPS",
        "citation": "CCEA KMS 2025-26 Yellow Soybean (pib.gov.in)",
    },
    "cotton": {
        "msp": 7710.0, "long_staple_msp": 8110.0, "season": "Kharif",
        "category": "fiber", "intervention_type": "MSP_PROCUREMENT",
        "citation": "CCEA KMS 2025-26 Medium Staple (pib.gov.in)",
    },
    # ── Rabi Crops (RMS 2025-26) ──
    "wheat": {
        "msp": 2425.0, "season": "Rabi",
        "category": "cereal", "intervention_type": "MSP_PROCUREMENT",
        "citation": "CCEA RMS 2025-26 (desagri.gov.in)",
    },
    "gram": {
        "msp": 5650.0, "season": "Rabi",
        "category": "pulse", "intervention_type": "PSS",
        "citation": "CCEA RMS 2025-26 Chana (desagri.gov.in)",
    },
    "mustard": {
        "msp": 6200.0, "season": "Rabi",
        "category": "oilseed", "intervention_type": "PDPS",
        "citation": "CCEA RMS 2025-26 Rapeseed & Mustard (desagri.gov.in)",
    },
    # ── Kharif Pulses (KMS 2024-25) ──
    "tur": {
        "msp": 7550.0, "season": "Kharif",
        "category": "pulse", "intervention_type": "PSS",
        "citation": "CCEA KMS 2024-25 (pib.gov.in)",
    },
    "arhar": {
        "msp": 7550.0, "season": "Kharif",
        "category": "pulse", "intervention_type": "PSS",
        "citation": "CCEA KMS 2024-25 (pib.gov.in)",
    },
    "moong": {
        "msp": 8682.0, "season": "Kharif",
        "category": "pulse", "intervention_type": "PSS",
        "citation": "CCEA KMS 2024-25 (pib.gov.in)",
    },
    "urad": {
        "msp": 7400.0, "season": "Kharif",
        "category": "pulse", "intervention_type": "PSS",
        "citation": "CCEA KMS 2024-25 (pib.gov.in)",
    },
    # ── Kharif Oilseeds (KMS 2024-25) ──
    "groundnut": {
        "msp": 6783.0, "season": "Kharif",
        "category": "oilseed", "intervention_type": "PDPS",
        "citation": "CCEA KMS 2024-25 (pib.gov.in)",
    },
    "sunflower": {
        "msp": 7280.0, "season": "Kharif",
        "category": "oilseed", "intervention_type": "PDPS",
        "citation": "CCEA KMS 2024-25 (pib.gov.in)",
    },
    "sesamum": {
        "msp": 9267.0, "season": "Kharif",
        "category": "oilseed", "intervention_type": "PDPS",
        "citation": "CCEA KMS 2024-25 (pib.gov.in)",
    },
    "nigerseed": {
        "msp": 8717.0, "season": "Kharif",
        "category": "oilseed", "intervention_type": "PDPS",
        "citation": "CCEA KMS 2024-25 (pib.gov.in)",
    },
}

# ---------------------------------------------------------------------------
# 5b. MoFPI Operation Greens — Notified Eligible Crops (22 TOTAL Perishables)
# ---------------------------------------------------------------------------
# Source: PIB press release on Operation Greens TOTAL expansion
#         MoFPI Operational Guidelines for TOP to TOTAL

OPERATION_GREENS_CROPS: set = {
    # Fruits (10 categories)
    "mango", "banana", "apple", "pineapple", "orange", "citrus",
    "kinnow", "mosambi", "lime", "lemon", "grapes", "aonla", "amla",
    "pomegranate", "guava", "litchi", "kiwi", "papaya", "jackfruit",
    # Vegetables (11+ categories)
    "tomato", "onion", "potato", "green peas", "carrot", "cauliflower",
    "beans", "french beans", "cabbage", "capsicum",
    "bottle gourd", "bitter gourd", "ridge gourd", "sponge gourd",
    "pointed gourd", "ash gourd",
    "okra", "bhindi", "garlic", "ginger",
    "green chillies", "chilli", "cucumber", "turmeric",
}

OPERATION_GREENS_CONFIG: Dict[str, Any] = {
    "freight_subsidy_pct": 0.50,          # 50% of transport cost
    "storage_subsidy_pct": 0.50,          # 50% of cold storage rental
    "road_min_distance_km": 100.0,        # Individual farmer / FPO road minimum
    "federation_min_distance_km": 250.0,  # State federation / retailer minimum
    "kisan_rail_corridor_min_km": 250.0,  # Long-haul rail corridor benchmark
    "road_min_qty_farmer_qtl": 90.0,      # 9 MT = 90 qtl for individual road claims
    "road_min_qty_fpo_qtl": 1000.0,       # 100 MT = 1000 qtl for FPO road claims
    "kisan_rail_min_qty_qtl": 0.0,        # No quantity floor for Kisan Rail
    "citation": (
        "MoFPI Operation Greens (TOP to TOTAL) Operational Guidelines "
        "under PM Kisan SAMPADA Yojana & Kisan Rail Convergence (mofpi.gov.in / pib.gov.in)"
    ),
}

# ---------------------------------------------------------------------------
# 5c. State APMC FPO / Direct Marketing Fee Exemption Rules
# ---------------------------------------------------------------------------
# Source: State APMC Amendment Acts, DMI (dmi.gov.in), State SAMB portals
# Note: Distinguishes inside-APMC exemptions (e.g. Karnataka F&V) from
#       outside-APMC / FPO direct-marketing waivers (e.g. Maharashtra 2016 Ord.).

STATE_FPO_EXEMPTIONS: Dict[str, Dict[str, Any]] = {
    "maharashtra": {
        "fpo_mandi_fee_waiver_pct": 1.00,
        "direct_marketing_allowed": True,
        "fruits_veg_exempt_inside_apmc": False,
        "fruits_veg_delisted_outside_apmc": True,
        "citation": "Maharashtra APMC (Amendment) Ordinance 2016 & MSAMB Direct Marketing / Shetkari Bazar Rules (dmi.gov.in)",
    },
    "karnataka": {
        "fpo_mandi_fee_waiver_pct": 1.00,
        "direct_marketing_allowed": True,
        "fruits_veg_exempt_inside_apmc": True,
        "fruits_veg_delisted_outside_apmc": True,
        "citation": "Karnataka APMC Amendment Act & KSAMB Horticulture Market Fee Exemption (dmi.gov.in)",
    },
    "bihar": {
        "fpo_mandi_fee_waiver_pct": 1.00,
        "direct_marketing_allowed": True,
        "fruits_veg_exempt_inside_apmc": True,
        "fruits_veg_delisted_outside_apmc": True,
        "citation": "Bihar Agriculture Produce Market (Repealing) Act 2006 (0% APMC fee statewide)",
    },
    "kerala": {
        "fpo_mandi_fee_waiver_pct": 1.00,
        "direct_marketing_allowed": True,
        "fruits_veg_exempt_inside_apmc": True,
        "fruits_veg_delisted_outside_apmc": True,
        "citation": "Kerala Non-APMC Cooperative Marketing Framework (VFPCK / Horticorp)",
    },
    "uttar pradesh": {
        "fpo_mandi_fee_waiver_pct": 1.00,
        "direct_marketing_allowed": True,
        "fruits_veg_exempt_inside_apmc": False,
        "fruits_veg_delisted_outside_apmc": False,
        "citation": "UP Krishi Utpadan Mandi Parishad — FPO / Kisan Bazaar Direct Purchase Rules",
    },
    "madhya pradesh": {
        "fpo_mandi_fee_waiver_pct": 0.50,
        "direct_marketing_allowed": True,
        "fruits_veg_exempt_inside_apmc": False,
        "fruits_veg_delisted_outside_apmc": False,
        "citation": "MP Krishi Upaj Mandi Adhiniyam — FPO Direct Procurement Concession",
    },
    "rajasthan": {
        "fpo_mandi_fee_waiver_pct": 1.00,
        "direct_marketing_allowed": True,
        "fruits_veg_exempt_inside_apmc": False,
        "fruits_veg_delisted_outside_apmc": False,
        "citation": "Rajasthan APMC Amendment Act — Direct Purchase & FPO Exemption",
    },
    "gujarat": {
        "fpo_mandi_fee_waiver_pct": 1.00,
        "direct_marketing_allowed": True,
        "fruits_veg_exempt_inside_apmc": False,
        "fruits_veg_delisted_outside_apmc": False,
        "citation": "Gujarat APMC (Amendment) Rules — GAPMC Direct Marketing License",
    },
    "punjab": {
        "fpo_mandi_fee_waiver_pct": 0.00,
        "direct_marketing_allowed": False,
        "fruits_veg_exempt_inside_apmc": False,
        "fruits_veg_delisted_outside_apmc": False,
        "citation": "Punjab APMC Act & PSAMB — Full Market Fee + RDF Applicable",
    },
    "haryana": {
        "fpo_mandi_fee_waiver_pct": 0.00,
        "direct_marketing_allowed": False,
        "fruits_veg_exempt_inside_apmc": False,
        "fruits_veg_delisted_outside_apmc": False,
        "citation": "Haryana APMC Act & HRDF — Full Market Fee + HRDF Applicable",
    },
    "tamil nadu": {
        "fpo_mandi_fee_waiver_pct": 1.00,
        "direct_marketing_allowed": True,
        "fruits_veg_exempt_inside_apmc": False,
        "fruits_veg_delisted_outside_apmc": False,
        "citation": "Tamil Nadu APMC Amendment — TNAMB Direct Marketing License",
    },
    "andhra pradesh": {
        "fpo_mandi_fee_waiver_pct": 1.00,
        "direct_marketing_allowed": True,
        "fruits_veg_exempt_inside_apmc": False,
        "fruits_veg_delisted_outside_apmc": False,
        "citation": "AP (Agricultural Produce & Livestock) Markets Amendment Act",
    },
    "telangana": {
        "fpo_mandi_fee_waiver_pct": 1.00,
        "direct_marketing_allowed": True,
        "fruits_veg_exempt_inside_apmc": False,
        "fruits_veg_delisted_outside_apmc": False,
        "citation": "Telangana APMC Amendment Act — TGAMB Direct Marketing Rules",
    },
    "delhi": {
        "fpo_mandi_fee_waiver_pct": 0.00,
        "direct_marketing_allowed": False,
        "fruits_veg_exempt_inside_apmc": False,
        "fruits_veg_delisted_outside_apmc": False,
        "citation": "Delhi DAMB (Azadpur APMC) Fee Schedule — Standard 1% Fee + 1% Cess",
    },
    "default": {
        "fpo_mandi_fee_waiver_pct": 0.00,
        "direct_marketing_allowed": False,
        "fruits_veg_exempt_inside_apmc": False,
        "fruits_veg_delisted_outside_apmc": False,
        "citation": "Model APMC Act (Conservative Benchmark — No Exemption Assumed)",
    },
}


# ---------------------------------------------------------------------------
# Government Subsidy Resolver
# ---------------------------------------------------------------------------

class GovernmentSubsidyResolver:
    """
    Evaluates farmer eligibility for statutory cost-offsetting government
    schemes loaded from `topo_rag/data/raw/govt_subsidies_index.json`
    (with curated statutory fallbacks):

    Scheme 1 — MoFPI Operation Greens (TOP→TOTAL) & Kisan Rail:
        50% freight subsidy on notified perishable produce:
          • Road transport: ≥ 100 km and ≥ 9 MT (90 qtl) lot threshold.
          • Kisan Rail convergence: zero minimum quantity cap on long-haul
            interstate corridors (≥ 250 km or explicit Kisan Rail mode).

    Scheme 2 — State APMC FPO / Direct-Marketing / Horticulture Waivers:
        Up to 100% waiver of APMC market fee + cess via FPO/direct marketing
        or statutory horticulture exemptions (e.g., Karnataka F&V inside APMC).

    Scheme 3 — PM-AASHA PDPS (Oilseeds) & PSS/MSP Floor Advisory:
        • Oilseeds (PDPS): Direct DBT payment of (MSP − Modal Price), capped
          at 15% of MSP per quintal (Union Cabinet Sept 2024 revamp).
        • Pulses/Cereals/Cotton (PSS / MSP Procurement): Flags exact MSP gap
          advisory directing farmers to NAFED/NCCF/FCI/CCI procurement centers.
    """

    def __init__(self):
        index_data = _load_govt_subsidies_index()

        op_greens = index_data.get("operation_greens", {})
        self.op_greens_config: Dict[str, Any] = {
            **OPERATION_GREENS_CONFIG,
            **{k: v for k, v in op_greens.items() if k != "eligible_crops"},
        }
        crops_list = op_greens.get("eligible_crops")
        self.op_greens_crops: set = (
            {str(c).strip().lower() for c in crops_list}
            if crops_list else set(OPERATION_GREENS_CROPS)
        )

        pm_aasha = index_data.get("pm_aasha_msp", {})
        self.pdps_cap_pct: float = float(pm_aasha.get("pdps_max_cap_pct", 0.15))
        msp_crops = pm_aasha.get("crops")
        self.msp_benchmarks: Dict[str, Dict[str, Any]] = (
            {k.strip().lower(): v for k, v in msp_crops.items()}
            if msp_crops else dict(GOVT_MSP_BENCHMARKS)
        )

        fpo_rules = index_data.get("state_fpo_exemptions")
        self.state_fpo_exemptions: Dict[str, Dict[str, Any]] = (
            {k.strip().lower(): v for k, v in fpo_rules.items()}
            if fpo_rules else dict(STATE_FPO_EXEMPTIONS)
        )

    def _match_state_fpo_rule(self, destination_state: str) -> Dict[str, Any]:
        clean_state = destination_state.strip().lower()
        if clean_state in self.state_fpo_exemptions:
            return self.state_fpo_exemptions[clean_state]
        for s_key, s_rule in self.state_fpo_exemptions.items():
            if s_key != "default" and (s_key in clean_state or clean_state in s_key):
                return s_rule
        return self.state_fpo_exemptions.get("default", STATE_FPO_EXEMPTIONS["default"])

    def resolve_subsidies(
        self,
        commodity: str,
        quantity_quintals: float,
        distance_km: float,
        modal_price: float,
        gross_transport_cost: float,
        gross_apmc_tax: float,
        destination_state: str = "default",
        is_fpo_member: bool = False,
        transport_mode: str = "auto",
    ) -> DynamicSubsidyResult:
        """
        Compute applicable government subsidy benefits for a single transaction.

        Parameters
        ----------
        commodity : str
            Crop name (e.g. "Tomato", "Soybean", "Wheat").
        quantity_quintals : float
            Total quantity being transported (in quintals).
        distance_km : float
            Road driving distance from source to destination mandi (km).
        modal_price : float
            Effective mandi modal price (₹ per quintal) from V1 (+ grade adj).
        gross_transport_cost : float
            Gross transport freight (₹) before any subsidy (V2 × V4).
        gross_apmc_tax : float
            Gross APMC commission + cess (₹) before any waiver (V3).
        destination_state : str
            Destination state for FPO/APMC waiver lookup.
        is_fpo_member : bool
            Whether the farmer sells via a registered FPO / direct-marketing channel.
        transport_mode : str
            "auto" (Kisan Rail convergence for >=250km small lots; Road for >=9 MT),
            "kisan_rail" (50% rail freight regardless of qty), or
            "road" (strictly enforces 9 MT / 90 qtl road threshold).
        """
        c_clean = commodity.strip().lower()
        mode_clean = (transport_mode or "auto").strip().lower()

        transport_sub = 0.0
        tax_waiver = 0.0
        price_support = 0.0
        msp_benchmark = 0.0
        msp_advisory = ""
        msp_advisory_hi = ""
        channel_notes: list = []
        channel_notes_hi: list = []
        applied_schemes: list = []
        applied_schemes_hi: list = []
        citations: list = []

        # ── Scheme 1: MoFPI Operation Greens (TOP→TOTAL) / Kisan Rail ───────
        if c_clean in self.op_greens_crops and gross_transport_cost > 0:
            sub_pct = float(self.op_greens_config.get("freight_subsidy_pct", 0.50))
            road_min_km = float(self.op_greens_config.get("road_min_distance_km", 100.0))
            road_min_qtl = float(self.op_greens_config.get("road_min_qty_farmer_qtl", 90.0))
            rail_min_km = float(self.op_greens_config.get("kisan_rail_corridor_min_km", 250.0))
            op_citation = self.op_greens_config.get("citation", OPERATION_GREENS_CONFIG["citation"])

            if mode_clean == "kisan_rail":
                transport_sub = round(gross_transport_cost * sub_pct, 2)
                applied_schemes.append(
                    "MoFPI Operation Greens via Kisan Rail (50% Freight Concession — zero min qty)"
                )
                applied_schemes_hi.append(
                    "MoFPI ऑपरेशन ग्रीन्स / किसान रेल (50% भाड़ा सब्सिडी — न्यूनतम मात्रा सीमा नहीं)"
                )
                citations.append(op_citation)
                channel_notes.append("Kisan Rail 50% upfront freight subsidy active")
                channel_notes_hi.append("किसान रेल पर 50% भाड़ा छूट सक्रिय")
            elif quantity_quintals >= road_min_qtl and distance_km >= road_min_km:
                transport_sub = round(gross_transport_cost * sub_pct, 2)
                applied_schemes.append(
                    f"MoFPI Operation Greens TOTAL Road Subsidy "
                    f"(50% Freight Concession, {quantity_quintals:.1f} qtl ≥ 90 qtl & {distance_km:.0f} km ≥ {road_min_km:.0f} km)"
                )
                applied_schemes_hi.append(
                    f"MoFPI ऑपरेशन ग्रीन्स सड़क सब्सिडी (50% भाड़ा छूट, {quantity_quintals:.1f} क्विंटल ≥ 90 क्विंटल व {distance_km:.0f} किमी दूरी)"
                )
                citations.append(op_citation)
                channel_notes.append("Qualifies for standard MoFPI road claim (≥9 MT)")
                channel_notes_hi.append("मानक MoFPI सड़क सब्सिडी के लिए पात्र (≥9 टन)")
            elif mode_clean == "auto" and distance_km >= rail_min_km:
                transport_sub = round(gross_transport_cost * sub_pct, 2)
                applied_schemes.append(
                    f"MoFPI Operation Greens / Kisan Rail Convergence "
                    f"(50% Freight Concession, {distance_km:.0f} km corridor; no min qty on Kisan Rail)"
                )
                applied_schemes_hi.append(
                    f"MoFPI ऑपरेशन ग्रीन्स / किसान रेल योजना (50% भाड़ा राहत, {distance_km:.0f} किमी मार्ग; किसान रेल पर न्यूनतम मात्रा सीमा नहीं)"
                )
                citations.append(op_citation)
                channel_notes.append(
                    "Small lot (<9 MT) routed under Kisan Rail 50% freight convergence"
                )
                channel_notes_hi.append(
                    "कम मात्रा (<9 टन) के लिए किसान रेल 50% भाड़ा राहत योजना लागू"
                )
            elif distance_km >= road_min_km:
                potential = round(gross_transport_cost * sub_pct, 2)
                channel_notes.append(
                    f"OpGreens Tip: Lot {quantity_quintals:.1f} qtl is below 90 qtl (9 MT) road minimum; "
                    f"booking via Kisan Rail or pooling to ≥9 MT saves ₹{potential:,.2f} (50% freight)"
                )
                channel_notes_hi.append(
                    f"ऑपरेशन ग्रीन्स सुझाव: {quantity_quintals:.1f} क्विंटल मात्रा 90 क्विंटल (9 टन) सड़क सीमा से कम है; "
                    f"किसान रेल या समूह बुकिंग (≥9 टन) से ₹{potential:,.2f} (50% भाड़ा) बचाया जा सकता है"
                )

        # ── Scheme 2: State APMC FPO / Direct Marketing / Horticulture Waiver ─
        fpo_rule = self._match_state_fpo_rule(destination_state)
        is_fv = c_clean in self.op_greens_crops

        if gross_apmc_tax > 0:
            if is_fpo_member and fpo_rule.get("fpo_mandi_fee_waiver_pct", 0.0) > 0:
                waiver_pct = float(fpo_rule["fpo_mandi_fee_waiver_pct"])
                tax_waiver = round(gross_apmc_tax * waiver_pct, 2)
                applied_schemes.append(
                    f"State APMC FPO Direct-Marketing Waiver "
                    f"({waiver_pct * 100:.0f}% of ₹{gross_apmc_tax:,.2f})"
                )
                applied_schemes_hi.append(
                    f"राज्य APMC एफपीओ (FPO) प्रत्यक्ष बिक्री शुल्क छूट ({waiver_pct * 100:.0f}% छूट)"
                )
                citations.append(fpo_rule["citation"])
            elif not is_fpo_member and fpo_rule.get("fruits_veg_exempt_inside_apmc") and is_fv:
                tax_waiver = round(gross_apmc_tax, 2)
                applied_schemes.append(
                    "State Horticulture APMC Market Fee Exemption (100% F&V waiver inside mandi)"
                )
                applied_schemes_hi.append(
                    "राज्य बागवानी मंडी शुल्क छूट (फल-सब्जी पर मंडी में 100% छूट)"
                )
                citations.append(fpo_rule["citation"])
            elif not is_fpo_member and fpo_rule.get("fruits_veg_delisted_outside_apmc") and is_fv:
                channel_notes.append(
                    f"APMC Reform Tip: Selling F&V via FPO or outside APMC yard waives ₹{gross_apmc_tax:,.2f} fee ({fpo_rule['citation']})"
                )
                channel_notes_hi.append(
                    f"मंडी सुधार सुझाव: FPO या मंडी यार्ड के बाहर सीधी बिक्री से ₹{gross_apmc_tax:,.2f} मंडी शुल्क बच सकता है"
                )
        elif fpo_rule.get("fruits_veg_exempt_inside_apmc"):
            channel_notes.append(f"Zero-APMC state already active ({fpo_rule['citation']})")
            channel_notes_hi.append(f"शून्य-मंडी शुल्क राज्य लागू ({_state_to_hi(destination_state)})")

        # ── Scheme 3: PM-AASHA PDPS (Oilseeds DBT) & PSS / MSP Floor ────────
        msp_entry = self.msp_benchmarks.get(c_clean)
        if msp_entry:
            msp_val = float(msp_entry["msp"])
            msp_benchmark = msp_val
            intervention = msp_entry.get("intervention_type", "MSP_PROCUREMENT")
            season = msp_entry.get("season", "")
            season_hi = "खरीफ" if season.lower() == "kharif" else ("रबी" if season.lower() == "rabi" else season)

            if modal_price < msp_val:
                raw_gap = msp_val - modal_price
                if intervention == "PDPS":
                    capped_gap = min(raw_gap, self.pdps_cap_pct * msp_val)
                    if capped_gap > 0 and quantity_quintals > 0:
                        price_support = round(capped_gap * quantity_quintals, 2)
                        applied_schemes.append(
                            f"PM-AASHA PDPS (₹{capped_gap:,.2f}/qtl DBT vs MSP ₹{msp_val:,.0f}/qtl "
                            f"[{season}], {self.pdps_cap_pct * 100:.0f}% cap)"
                        )
                        applied_schemes_hi.append(
                            f"PM-AASHA भावांतर योजना (PDPS: {season_hi} MSP ₹{msp_val:,.0f}/क्विंटल पर ₹{capped_gap:,.2f}/क्विंटल सहायता, {self.pdps_cap_pct * 100:.0f}% सीमा)"
                        )
                        citations.append(
                            f"DA&FW PM-AASHA PDPS Guidelines & {msp_entry['citation']}"
                        )
                else:
                    agency = "NAFED/NCCF (PM-AASHA PSS)" if intervention == "PSS" else "FCI/CCI/State Agency"
                    agency_hi = "NAFED/NCCF (PM-AASHA PSS)" if intervention == "PSS" else "FCI/CCI या सरकारी खरीद केंद्र"
                    potential_gain = round(raw_gap * quantity_quintals, 2)
                    msp_advisory = (
                        f"MSP Procurement Alert: Mandi price ₹{modal_price:,.2f}/qtl is ₹{raw_gap:,.2f}/qtl "
                        f"below MSP ₹{msp_val:,.0f}/qtl ({msp_entry['citation']}). "
                        f"Selling at {agency} MSP center yields +₹{potential_gain:,.2f} vs open mandi."
                    )
                    msp_advisory_hi = (
                        f"MSP खरीद चेतावनी: मंडी भाव ₹{modal_price:,.2f}/क्विंटल, सरकारी MSP ₹{msp_val:,.0f}/क्विंटल से ₹{raw_gap:,.2f}/क्विंटल कम है — "
                        f"{agency_hi} पर बेचने से +₹{potential_gain:,.2f} अतिरिक्त लाभ मिल सकता है।"
                    )
                    channel_notes.append(msp_advisory)
                    channel_notes_hi.append(msp_advisory_hi)

        # ── Aggregate ───────────────────────────────────────────────────────
        total_benefit = round(transport_sub + tax_waiver + price_support, 2)
        is_eligible = total_benefit > 0

        if is_eligible:
            parts = []
            parts_hi = []
            if transport_sub > 0:
                parts.append(f"Freight Relief ₹{transport_sub:,.2f}")
                parts_hi.append(f"भाड़ा राहत ₹{transport_sub:,.2f}")
            if tax_waiver > 0:
                parts.append(f"Mandi Fee Waiver ₹{tax_waiver:,.2f}")
                parts_hi.append(f"मंडी शुल्क छूट ₹{tax_waiver:,.2f}")
            if price_support > 0:
                parts.append(f"Price Support ₹{price_support:,.2f}")
                parts_hi.append(f"मूल्य सहायता ₹{price_support:,.2f}")

            note = (
                f"Government Subsidy: ₹{total_benefit:,.2f} total "
                f"({' + '.join(parts)}) via {'; '.join(applied_schemes)} "
                f"[{'; '.join(citations)}]"
            )
            note_hi = (
                f"सरकारी सब्सिडी लाभ: कुल ₹{total_benefit:,.2f} "
                f"({' + '.join(parts_hi)}) — {'; '.join(applied_schemes_hi)}"
            )
        else:
            extra = f" | {'; '.join(channel_notes)}" if channel_notes else ""
            extra_hi = f" | {'; '.join(channel_notes_hi)}" if channel_notes_hi else ""
            note = (
                "Government Subsidy: ₹0.00 "
                "(no direct DBT/freight offset triggered for this route/crop)"
                f"{extra}"
            )
            note_hi = (
                "सरकारी सब्सिडी: ₹0.00 "
                "(इस फसल/मार्ग पर कोई प्रत्यक्ष सब्सिडी लागू नहीं)"
                f"{extra_hi}"
            )

        return DynamicSubsidyResult(
            scheme_name=" + ".join(applied_schemes) if applied_schemes else "None Applicable",
            transport_subsidy_amount=transport_sub,
            mandi_fee_waiver_amount=tax_waiver,
            price_support_amount=price_support,
            total_subsidy_benefit=total_benefit,
            is_eligible=is_eligible,
            citation="; ".join(citations) if citations else "N/A",
            provenance_note=note,
            msp_benchmark_per_qtl=msp_benchmark,
            msp_advisory_note=msp_advisory,
            channel_details="; ".join(channel_notes),
            provenance_note_hi=note_hi,
            msp_advisory_note_hi=msp_advisory_hi,
            channel_details_hi="; ".join(channel_notes_hi),
        )

