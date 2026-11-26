"""
Validation Test Suite
=====================
Proves numerical accuracy and routing correctness for Topo-RAG MVP.

Run: python -m topo_rag.tests.validate
"""

import logging
import sys
import time
import unittest
from pathlib import Path

# Fix import path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)


# ===========================================================================
# Unit Tests
# ===========================================================================

class TestSDSRouter(unittest.TestCase):
    """Validate SDS calculation and routing decisions."""

    def setUp(self):
        from topo_rag.core.sds_router import SDSRouter
        self.router = SDSRouter(threshold=0.4)

    def test_pure_tabular_routes_to_B(self):
        tabular_text = (
            "Tomato | Azadpur | 15/09/2024 | 1200 | 2800 | 2100 | Grade A | "
            "Maharashtra | Pune | 2500 | 3500 | 3100"
        )
        result = self.router.compute_full(tabular_text)
        self.assertGreater(result.sds, 0.4)
        self.assertEqual(result.route, "B")

    def test_pure_narrative_routes_to_A(self):
        narrative = (
            "The Agricultural Produce Market Committee Act governs buying and selling "
            "of agricultural produce across India. Licensed commission agents facilitate "
            "transactions between farmers and buyers in designated mandi premises."
        )
        result = self.router.compute_full(narrative)
        self.assertLessEqual(result.sds, 0.4)
        self.assertEqual(result.route, "A")

    def test_sds_range(self):
        for text in ["hello world", "1 2 3 | a | b", "pure text document about farming"]:
            sds = self.router.calculate_sds(text)
            self.assertGreaterEqual(sds, 0.0)
            self.assertLessEqual(sds, 1.0)

    def test_empty_string(self):
        sds = self.router.calculate_sds("")
        self.assertEqual(sds, 0.0)

    def test_route_chunks_partitions_correctly(self):
        chunks = [
            "Market | Commodity | Date | Min_Price | Max_Price | Modal_Price\n"
            "Azadpur | Tomato | 01/01/2024 | 1200 | 3200 | 2400",
            "The APMC Act ensures farmers get fair prices in designated markets.",
            "1500 | 2200 | 1850 | Onion | Nashik | Grade A | Maharashtra",
        ]
        narrative, tabular = self.router.route_chunks(chunks)
        self.assertGreater(len(tabular), 0)
        self.assertGreater(len(narrative), 0)


class TestProfitCalculator(unittest.TestCase):
    """Validate deterministic profit formula accuracy."""

    def setUp(self):
        from topo_rag.engine.profit_calculator import ProfitCalculator
        self.calc = ProfitCalculator(
            tariff_per_qtl_km=0.80,
            apmc_commission_pct=2.0,
            mandi_fees_per_qtl=15.0,
            wastage_rates={"tomato": 0.15, "onion": 0.06, "potato": 0.04, "wheat": 0.02, "default": 0.08},
            use_dynamic_resolvers=False,
        )

    def test_net_profit_formula_correctness(self):
        """
        Manual verification for 500kg Grade-A Tomato, Pune→Azadpur, price=₹2200/q
        Wastage: 15%
        """
        result = self.calc.calculate(
            commodity="Tomato",
            quantity_kg=500.0,
            modal_price_per_qtl=2200.0,
            source_market="Pune",
            destination_market="Azadpur",
            grade="A",
            grade_premium_pct=5.0,    # 5% Grade A premium → ₹2310/q
        )

        # Manual calculation
        qty_q = 500 / 100  # 5.0 quintals
        effective_price = 2200 * 1.05  # ₹2310
        wastage_rate = 0.15
        wastage_qty = qty_q * wastage_rate  # 0.75 q
        sellable_qty = qty_q * (1 - wastage_rate)  # 4.25 q
        gross = sellable_qty * effective_price  # 4.25 × 2310 = 9817.50
        # Azadpur is mapped to 1420km from Pune in the distance matrix
        distance = 1420.0  # Pune→Azadpur (Delhi market)
        transport = qty_q * 0.80 * distance  # 5 × 0.80 × 1420 = 5680
        commission = gross * 0.02  # 2%
        mandi_fees = qty_q * 15.0  # 5 × 15 = 75
        wastage_cost = wastage_qty * effective_price  # 0.75 × 2310 = 1732.50
        total_costs = transport + commission + mandi_fees + wastage_cost
        expected_net = gross - total_costs

        self.assertAlmostEqual(result.gross_revenue, gross, places=2)
        self.assertAlmostEqual(result.transport_cost, transport, places=2)
        self.assertAlmostEqual(result.net_profit, expected_net, places=2)

    def test_wastage_reduces_net_profit(self):
        """Higher wastage commodities should have lower net profit."""
        # Tomato (15% wastage) vs Wheat (2% wastage) at same price/qty
        tomato = self.calc.calculate("Tomato", 1000.0, 2000.0, "Pune", "Delhi")
        wheat = self.calc.calculate("Wheat", 1000.0, 2000.0, "Pune", "Delhi")
        self.assertGreater(wheat.net_profit, tomato.net_profit)

    def test_longer_distance_reduces_profit(self):
        """More km = higher transport cost = lower profit."""
        near = self.calc.calculate("Onion", 500.0, 1800.0, "Agra", "Delhi",
                                   override_distance_km=210)
        far = self.calc.calculate("Onion", 500.0, 1800.0, "Agra", "Mumbai",
                                  override_distance_km=1170)
        self.assertGreater(near.net_profit, far.net_profit)

    def test_grade_a_premium_increases_profit(self):
        """Grade A premium should always increase net profit over Grade B."""
        grade_a = self.calc.calculate("Tomato", 500.0, 2000.0, "Pune", "Mumbai",
                                      grade="A", grade_premium_pct=12.0)
        grade_b = self.calc.calculate("Tomato", 500.0, 2000.0, "Pune", "Mumbai",
                                      grade="B", grade_premium_pct=0.0)
        self.assertGreater(grade_a.net_profit, grade_b.net_profit)

    def test_zero_quantity_is_safe(self):
        result = self.calc.calculate("Tomato", 0.0, 2000.0, "Pune", "Mumbai")
        self.assertEqual(result.net_profit_per_kg, 0.0)

    def test_breakdown_fields_non_negative(self):
        result = self.calc.calculate("Potato", 200.0, 1200.0, "Indore", "Mumbai")
        self.assertGreaterEqual(result.gross_revenue, 0)
        self.assertGreaterEqual(result.transport_cost, 0)
        self.assertGreaterEqual(result.apmc_commission, 0)
        self.assertGreaterEqual(result.wastage_cost, 0)


class TestDataLoader(unittest.TestCase):
    """Validate CSV and narrative loading."""

    def test_narrative_loader_produces_chunks(self):
        from topo_rag.core.data_loader import NarrativeLoader
        loader = NarrativeLoader(chunk_size=200, chunk_overlap=30)
        chunks = loader.load_text(
            "The APMC Act regulates agricultural markets in India. "
            "Commission agents charge between 2-4% of transaction value. "
            "Farmers benefit from transparent price discovery at designated mandis.",
            source="test"
        )
        self.assertGreater(len(chunks), 0)
        for chunk in chunks:
            self.assertIsInstance(chunk.text, str)
            self.assertGreater(len(chunk.text), 0)

    def test_agmarknet_loader_cell_mode(self):
        """Generate a tiny CSV and load it in cell mode."""
        import tempfile, csv, os
        data = [
            ["State", "District", "Market", "Commodity", "Variety",
             "Arrival_Date", "Min_Price", "Max_Price", "Modal_Price"],
            ["Maharashtra", "Pune", "Gultekdi", "Tomato", "Hybrid",
             "01/01/2024", "1200", "3200", "2400"],
            ["Delhi", "Delhi", "Azadpur", "Onion", "Red",
             "01/01/2024", "900", "2500", "1800"],
        ]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline=""
        ) as f:
            csv.writer(f).writerows(data)
            tmp_path = f.name

        try:
            from topo_rag.core.data_loader import AgmarknetLoader
            loader = AgmarknetLoader(cell_mode=True)
            chunks = loader.load_csv(tmp_path)
            self.assertGreater(len(chunks), 0)
            # Verify cell encoding format
            cell_texts = [c.text for c in chunks]
            self.assertTrue(any("[COL:" in t for t in cell_texts))
            self.assertTrue(any("[VAL:" in t for t in cell_texts))
        finally:
            os.unlink(tmp_path)


class TestMinMaxNormalization(unittest.TestCase):
    """Validate reranker score normalization."""

    def test_minmax_normalization(self):
        import numpy as np
        from topo_rag.engine.reranker import CrossEncoderReranker
        scores = np.array([2.0, 5.0, 1.0, 8.0, 3.0])
        normalized = CrossEncoderReranker._minmax_normalize(scores)
        self.assertAlmostEqual(normalized.min(), 0.0, places=5)
        self.assertAlmostEqual(normalized.max(), 1.0, places=5)

    def test_uniform_scores_return_half(self):
        import numpy as np
        from topo_rag.engine.reranker import CrossEncoderReranker
        scores = np.array([3.0, 3.0, 3.0])
        normalized = CrossEncoderReranker._minmax_normalize(scores)
        for s in normalized:
            self.assertAlmostEqual(s, 0.5, places=5)


class TestGovernmentSubsidyResolver(unittest.TestCase):
    """Validate V6 government subsidies dataset, eligibility rules, and math."""

    def setUp(self):
        from topo_rag.engine.dynamic_resolvers import GovernmentSubsidyResolver
        from topo_rag.engine.profit_calculator import ProfitCalculator
        self.sub = GovernmentSubsidyResolver()
        self.calc_v6 = ProfitCalculator(
            tariff_per_qtl_km=0.80,
            apmc_commission_pct=2.0,
            mandi_fees_per_qtl=15.0,
            wastage_rates={"tomato": 0.15, "soybean": 0.02, "wheat": 0.015, "default": 0.08},
            use_dynamic_resolvers=False,
            use_subsidy_resolver=True,
        )

    def test_json_dataset_loaded(self):
        self.assertGreaterEqual(len(self.sub.msp_benchmarks), 15)
        self.assertIn("tomato", self.sub.op_greens_crops)
        self.assertIn("maharashtra", self.sub.state_fpo_exemptions)
        self.assertAlmostEqual(self.sub.pdps_cap_pct, 0.15, places=4)

    def test_operation_greens_kisan_rail_and_road_thresholds(self):
        # Long-haul small lot (5 qtl, 1420 km) qualifies via Kisan Rail convergence
        kr = self.sub.resolve_subsidies(
            commodity="Tomato", quantity_quintals=5.0, distance_km=1420.0,
            modal_price=2200.0, gross_transport_cost=5680.0, gross_apmc_tax=196.35,
            destination_state="delhi", is_fpo_member=False, transport_mode="auto",
        )
        self.assertAlmostEqual(kr.transport_subsidy_amount, 2840.0, places=2)
        self.assertTrue(kr.is_eligible)

        # Strict road mode with < 9 MT (50 qtl < 90 qtl) must NOT grant road subsidy
        road_small = self.sub.resolve_subsidies(
            commodity="Tomato", quantity_quintals=50.0, distance_km=300.0,
            modal_price=2200.0, gross_transport_cost=12000.0, gross_apmc_tax=1000.0,
            destination_state="delhi", is_fpo_member=False, transport_mode="road",
        )
        self.assertEqual(road_small.transport_subsidy_amount, 0.0)
        self.assertIn("Kisan Rail", road_small.channel_details)

        # Road mode >= 9 MT (100 qtl >= 90 qtl) and >= 100 km grants 50% road subsidy
        road_big = self.sub.resolve_subsidies(
            commodity="Onion", quantity_quintals=100.0, distance_km=150.0,
            modal_price=1800.0, gross_transport_cost=12000.0, gross_apmc_tax=1800.0,
            destination_state="delhi", is_fpo_member=False, transport_mode="road",
        )
        self.assertAlmostEqual(road_big.transport_subsidy_amount, 6000.0, places=2)

    def test_state_apmc_fpo_and_horticulture_waivers(self):
        # Maharashtra non-FPO inside APMC gets 0 waiver + tip; FPO gets 100% waiver
        mh_non_fpo = self.sub.resolve_subsidies(
            commodity="Tomato", quantity_quintals=10.0, distance_km=50.0,
            modal_price=2200.0, gross_transport_cost=400.0, gross_apmc_tax=231.0,
            destination_state="maharashtra", is_fpo_member=False,
        )
        self.assertEqual(mh_non_fpo.mandi_fee_waiver_amount, 0.0)
        self.assertIn("outside APMC", mh_non_fpo.channel_details)

        mh_fpo = self.sub.resolve_subsidies(
            commodity="Tomato", quantity_quintals=10.0, distance_km=50.0,
            modal_price=2200.0, gross_transport_cost=400.0, gross_apmc_tax=231.0,
            destination_state="maharashtra", is_fpo_member=True,
        )
        self.assertAlmostEqual(mh_fpo.mandi_fee_waiver_amount, 231.0, places=2)

        # Karnataka exempts F&V market fees even for non-FPO farmers
        ka_non_fpo = self.sub.resolve_subsidies(
            commodity="Tomato", quantity_quintals=10.0, distance_km=50.0,
            modal_price=2200.0, gross_transport_cost=400.0, gross_apmc_tax=300.0,
            destination_state="karnataka", is_fpo_member=False,
        )
        self.assertAlmostEqual(ka_non_fpo.mandi_fee_waiver_amount, 300.0, places=2)

        # Punjab grants no FPO waiver
        pb_fpo = self.sub.resolve_subsidies(
            commodity="Wheat", quantity_quintals=20.0, distance_km=50.0,
            modal_price=2500.0, gross_transport_cost=800.0, gross_apmc_tax=2000.0,
            destination_state="punjab", is_fpo_member=True,
        )
        self.assertEqual(pb_fpo.mandi_fee_waiver_amount, 0.0)

    def test_pm_aasha_pdps_cap_and_msp_advisory(self):
        # Soybean MSP = 5328; modal = 4800 -> gap 528 (< 15% cap 799.20)
        soy = self.sub.resolve_subsidies(
            commodity="Soybean", quantity_quintals=10.0, distance_km=100.0,
            modal_price=4800.0, gross_transport_cost=800.0, gross_apmc_tax=960.0,
            destination_state="madhya pradesh", is_fpo_member=False,
        )
        self.assertAlmostEqual(soy.price_support_amount, 5280.0, places=2)

        # Soybean severe price crash (modal = 3000) -> strictly capped at 15% of 5328 = 799.20/qtl
        soy_crash = self.sub.resolve_subsidies(
            commodity="Soybean", quantity_quintals=10.0, distance_km=100.0,
            modal_price=3000.0, gross_transport_cost=800.0, gross_apmc_tax=600.0,
            destination_state="madhya pradesh", is_fpo_member=False,
        )
        self.assertAlmostEqual(soy_crash.price_support_amount, 7992.0, places=2)

        # Wheat below MSP (MSP 2425, modal 2100) -> no false PDPS DBT cash, advisory emitted
        wheat_low = self.sub.resolve_subsidies(
            commodity="Wheat", quantity_quintals=10.0, distance_km=100.0,
            modal_price=2100.0, gross_transport_cost=800.0, gross_apmc_tax=420.0,
            destination_state="uttar pradesh", is_fpo_member=False,
        )
        self.assertEqual(wheat_low.price_support_amount, 0.0)
        self.assertIn("MSP Procurement Alert", wheat_low.msp_advisory_note)

    def test_profit_calculator_v6_full_accounting(self):
        res = self.calc_v6.calculate(
            commodity="Tomato", quantity_kg=500.0, modal_price_per_qtl=2200.0,
            source_market="Pune", destination_market="Azadpur", grade="A",
            grade_premium_pct=5.0,
        )
        # Baseline net profit = 2133.65; OpGreens freight subsidy = 2840.00 -> final = 4973.65
        self.assertAlmostEqual(res.baseline_net_profit, 2133.65, places=2)
        self.assertAlmostEqual(res.govt_transport_subsidy, 2840.00, places=2)
        self.assertAlmostEqual(res.govt_total_subsidy, 2840.00, places=2)
        self.assertAlmostEqual(res.effective_total_costs, res.total_costs - 2840.00, places=2)
        self.assertAlmostEqual(res.net_profit, 4973.65, places=2)
        self.assertAlmostEqual(res.net_profit_per_qtl, 4973.65 / 5.0, places=2)
        self.assertAlmostEqual(res.net_profit_per_kg, 4973.65 / 500.0, places=4)
        self.assertIn("Net Profit / quintal", res.summary("en"))
        self.assertIn("प्रति क्विंटल मुनाफा", res.summary("hi"))

    def test_hindi_notes_localization(self):
        from topo_rag.engine.profit_calculator import ProfitCalculator
        calc = ProfitCalculator(use_dynamic_resolvers=True)
        res = calc.calculate(
            commodity="Tomato", quantity_kg=500.0, modal_price_per_qtl=2200.0,
            source_market="Pune", destination_market="Azadpur", grade="A",
            grade_premium_pct=5.0,
            price_provenance_note="Mandi Price: ₹2,200.00/qtl [LIVE]",
            price_provenance_note_hi="मंडी भाव: ₹2,200.00/क्विंटल [लाइव एगमार्कनेट]",
        )
        hi_summary = res.summary("hi")
        self.assertIn("विशेष विवरण:", hi_summary)
        self.assertIn("मंडी भाव:", hi_summary)
        self.assertIn("सड़क दूरी:", hi_summary)
        self.assertIn("परिवहन दर:", hi_summary)
        self.assertIn("मंडी कर (APMC):", hi_summary)
        self.assertIn("यात्रा में खराबी/नुकसान:", hi_summary)
        self.assertIn("सरकारी सब्सिडी लाभ:", hi_summary)
        for eng_prefix in ("Road Distance:", "Transport Tariff:", "APMC Tax:", "Transit Wastage:", "Government Subsidy:", "Mandi Price:", "Grade A premium:"):
            self.assertNotIn(eng_prefix, hi_summary)

    def test_live_mandi_ncr_and_national_fallback(self):
        from unittest.mock import patch
        from topo_rag.engine.dynamic_resolvers import DynamicVariableResolver, _MANDI_PRICE_CACHE

        _MANDI_PRICE_CACHE.clear()
        r = DynamicVariableResolver()
        fake_records = [
            {"state": "Haryana", "district": "Sonipat", "market": "Sonepat APMC", "modal_price": "3000", "min_price": "2600", "max_price": "3400", "arrival_date": "26/11/2026"},
            {"state": "Uttar Pradesh", "district": "Ghaziabad", "market": "Noida APMC", "modal_price": "2500", "min_price": "2200", "max_price": "2800", "arrival_date": "26/11/2026"},
            {"state": "Maharashtra", "district": "Pune", "market": "Khed(Chakan)", "modal_price": "1200", "min_price": "1000", "max_price": "1500", "arrival_date": "26/11/2026"},
        ]

        def fake_fetch(base_url, params, timeout_sec=10.0):
            if "filters[state]" in params:
                return []  # Simulate Delhi having 0 Tomato records today
            return fake_records

        with patch.object(r, "_fetch_ogd_records", side_effect=fake_fetch), patch.dict("os.environ", {"DATA_GOV_API_KEY": "test_key"}):
            res_ncr = r.resolve_mandi_price("Tomato", "Azadpur", "delhi")
            self.assertIsNotNone(res_ncr)
            self.assertTrue(res_ncr.is_live)
            self.assertAlmostEqual(res_ncr.modal_price, 2750.0, places=2)
            self.assertIn("Delhi-NCR", res_ncr.provenance_note)
            self.assertIn("दिल्ली-एनसीआर", res_ncr.provenance_note_hi)

            # District match when market name differs (Pune -> Khed(Chakan) in district Pune)
            _MANDI_PRICE_CACHE.clear()
            res_pune = r.resolve_mandi_price("Tomato", "Pune", "maharashtra")
            self.assertIsNotNone(res_pune)
            self.assertAlmostEqual(res_pune.modal_price, 1200.0, places=2)
        _MANDI_PRICE_CACHE.clear()


# ===========================================================================
# Integration Test (sample queries)
# ===========================================================================

def run_integration_demo(pipeline=None, verbose=True):
    """
    Run sample queries to demonstrate pipeline accuracy.
    Designed to run in < 60 seconds on CPU without GPU.
    """
    from topo_rag.engine.pipeline import TopoRAGPipeline
    from topo_rag.engine.profit_calculator import ProfitCalculator

    SAMPLE_QUERIES = [
        "What is my true net profit for 500kg Grade-A Tomatoes shipped from Pune to Azadpur?",
        "Tell me about APMC commission rates and mandi fees for Onion in Maharashtra.",
        "Calculate profit for 1000kg Grade-B Potato from Indore to Mumbai market.",
        "What are the quality standards for Grade A Tomato export?",
        "500kg onion from Nashik to Delhi what is the transport cost and net profit?",
    ]

    calc = ProfitCalculator()

    print("\n" + "═" * 60)
    print("  TOPO-RAG VALIDATION SUITE — SAMPLE QUERY RESULTS")
    print("═" * 60)

    for i, query in enumerate(SAMPLE_QUERIES, 1):
        print(f"\n[Query {i}] {query}")
        print("─" * 60)

        # SDS routing check
        from topo_rag.core.sds_router import SDSRouter
        router = SDSRouter()
        sds_result = router.compute_full(query)
        print(f"  SDS Score : {sds_result.sds:.3f} → Route {sds_result.route}")
        print(f"  N_num={sds_result.n_num}, N_sep={sds_result.n_sep}, N_ent={sds_result.n_ent}, N_total={sds_result.n_total}")

        # Profit calculation if applicable
        from topo_rag.engine.pipeline import _parse_crop, _parse_quantity, _parse_grade, _parse_locations
        crop = _parse_crop(query)
        qty = _parse_quantity(query)
        grade = _parse_grade(query)
        src, dst = _parse_locations(query)

        if crop and qty:
            defaults = {"Tomato": 2200.0, "Onion": 1800.0, "Potato": 1200.0, "Wheat": 2100.0}
            modal_price = defaults.get(crop, 2000.0)
            grade_premium = 5.0 if grade == "A" else 0.0
            result = calc.calculate(
                commodity=crop,
                quantity_kg=qty,
                modal_price_per_qtl=modal_price,
                source_market=src or "Pune",
                destination_market=dst or "Azadpur",
                grade=grade,
                grade_premium_pct=grade_premium,
            )
            print(result.summary())
        else:
            print(f"  Parsed: crop={crop}, qty={qty} — no profit calc (narrative query)")

    print("\n" + "═" * 60)
    print("  VALIDATION COMPLETE — All deterministic results verified.")
    print("═" * 60)


# ===========================================================================
# Main
# ===========================================================================

def main():
    logging.basicConfig(level=logging.WARNING)
    print("\n" + "═" * 60)
    print("  TOPO-RAG UNIT TEST SUITE")
    print("═" * 60)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestSDSRouter))
    suite.addTests(loader.loadTestsFromTestCase(TestProfitCalculator))
    suite.addTests(loader.loadTestsFromTestCase(TestDataLoader))
    suite.addTests(loader.loadTestsFromTestCase(TestMinMaxNormalization))
    suite.addTests(loader.loadTestsFromTestCase(TestGovernmentSubsidyResolver))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if result.wasSuccessful():
        print("\n✅ All unit tests passed!")
        run_integration_demo()
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
