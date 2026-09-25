"""
Mock Dataset Generator
======================
Generates synthetic Agmarknet-schema CSV files for MVP testing.
Produces data matching the real Agmarknet column schema.

Run: python -m topo_rag.utils.mock_data_generator
"""

import csv
import logging
import os
import random
from datetime import date, timedelta
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

MARKETS = [
    ("Maharashtra", "Pune", "Gultekdi"),
    ("Maharashtra", "Mumbai", "Vashi"),
    ("Maharashtra", "Nashik", "Nashik"),
    ("Delhi", "Delhi", "Azadpur"),
    ("Karnataka", "Bengaluru", "Yeshwanthpur"),
    ("Telangana", "Hyderabad", "Bowenpally"),
    ("Tamil Nadu", "Chennai", "Koyambedu"),
    ("Madhya Pradesh", "Indore", "Indore Krishi Upaj Mandi"),
    ("Uttar Pradesh", "Agra", "Agra Mandi"),
    ("Punjab", "Amritsar", "Amritsar Mandi"),
    ("Haryana", "Karnal", "Karnal Mandi"),
    ("Gujarat", "Ahmedabad", "Jamalpur Mandi"),
]

COMMODITIES = {
    "Tomato": {
        "varieties": ["Hybrid", "Local", "Cherry", "Roma"],
        "base_min": 800,  "base_max": 4500,  "grade_a_premium": 1.12,
    },
    "Onion": {
        "varieties": ["Red", "White", "Yellow", "Nashik"],
        "base_min": 500,  "base_max": 3200,  "grade_a_premium": 1.10,
    },
    "Potato": {
        "varieties": ["Jyoti", "Kufri Bahar", "Local", "Pukhraj"],
        "base_min": 600,  "base_max": 2000,  "grade_a_premium": 1.08,
    },
    "Wheat": {
        "varieties": ["Sharbati", "Lokwan", "HD-2967", "PBW-343"],
        "base_min": 1800, "base_max": 2800,  "grade_a_premium": 1.05,
    },
    "Rice": {
        "varieties": ["Basmati", "Sona Masoori", "IR-64", "Ponni"],
        "base_min": 2000, "base_max": 6000,  "grade_a_premium": 1.15,
    },
    "Maize": {
        "varieties": ["Yellow", "White", "Hybrid"],
        "base_min": 1400, "base_max": 2200,  "grade_a_premium": 1.06,
    },
    "Soybean": {
        "varieties": ["JS-335", "NRC-7", "MACS-58"],
        "base_min": 3800, "base_max": 5500,  "grade_a_premium": 1.07,
    },
    "Garlic": {
        "varieties": ["Desi", "G-50", "Yamuna Safed"],
        "base_min": 4000, "base_max": 15000, "grade_a_premium": 1.20,
    },
}

GRADES = ["Grade A", "Grade B", "Grade C"]


def _simulate_price(
    base_min: int,
    base_max: int,
    day_offset: int,
    grade_premium: float,
    grade: str,
) -> tuple:
    """Simulate min/max/modal price with seasonal variation."""
    # Seasonal sinusoidal variation
    import math
    seasonal = 1.0 + 0.3 * math.sin(day_offset / 180.0 * math.pi)
    market_noise = random.uniform(0.85, 1.15)
    
    effective_min = int(base_min * seasonal * market_noise)
    effective_max = int(base_max * seasonal * market_noise)
    
    if grade == "Grade A":
        effective_min = int(effective_min * grade_premium)
        effective_max = int(effective_max * grade_premium)
    elif grade == "Grade C":
        effective_min = int(effective_min * 0.80)
        effective_max = int(effective_max * 0.80)
    
    # Ensure sensible range
    effective_min = max(100, effective_min)
    effective_max = max(effective_min + 50, effective_max)
    effective_modal = int(effective_min + (effective_max - effective_min) * 0.60)
    
    return effective_min, effective_max, effective_modal


def generate_agmarknet_csv(
    output_dir: str,
    num_days: int = 90,
    start_date: date = None,
    filename: str = "agmarknet_prices.csv",
) -> str:
    """
    Generate a synthetic Agmarknet-schema CSV.

    Parameters
    ----------
    output_dir : Directory to write the CSV.
    num_days   : Number of calendar days of data.
    start_date : Start date (defaults to 90 days ago).
    filename   : Output filename.

    Returns
    -------
    str: Absolute path to the generated CSV.
    """
    if start_date is None:
        start_date = date.today() - timedelta(days=num_days)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(output_dir) / filename

    headers = [
        "State", "District", "Market", "Commodity", "Variety",
        "Grade", "Arrival_Date", "Min_Price", "Max_Price", "Modal_Price",
        "Arrivals_Tonnes",
    ]

    rows = []
    for day_offset in range(num_days):
        current_date = start_date + timedelta(days=day_offset)
        # Skip Sundays (most mandis closed)
        if current_date.weekday() == 6:
            continue
        date_str = current_date.strftime("%d/%m/%Y")

        for state, district, market in MARKETS:
            for commodity, specs in COMMODITIES.items():
                # Not every commodity at every market every day
                if random.random() < 0.35:
                    continue
                variety = random.choice(specs["varieties"])
                grade = random.choice(GRADES)
                min_p, max_p, modal_p = _simulate_price(
                    specs["base_min"],
                    specs["base_max"],
                    day_offset,
                    specs["grade_a_premium"],
                    grade,
                )
                arrivals = random.randint(5, 500)
                rows.append([
                    state, district, market, commodity, variety,
                    grade, date_str, min_p, max_p, modal_p, arrivals,
                ])

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    logger.info("Generated %d rows → %s", len(rows), out_path)
    print(f"✓ Generated {len(rows):,} price records → {out_path}")
    return str(out_path)


def generate_transport_matrix_csv(output_dir: str) -> str:
    """Generate a transport distance + tariff matrix CSV."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(output_dir) / "transport_matrix.csv"

    city_pairs = [
        ("Pune", "Mumbai", 145, 0.85),
        ("Pune", "Delhi", 1420, 0.72),
        ("Pune", "Bengaluru", 840, 0.78),
        ("Nashik", "Mumbai", 170, 0.82),
        ("Nashik", "Delhi", 1380, 0.70),
        ("Nashik", "Hyderabad", 525, 0.75),
        ("Indore", "Delhi", 780, 0.73),
        ("Indore", "Mumbai", 590, 0.76),
        ("Indore", "Bengaluru", 1180, 0.69),
        ("Agra", "Delhi", 210, 0.88),
        ("Agra", "Mumbai", 1170, 0.71),
        ("Lucknow", "Delhi", 555, 0.78),
        ("Lucknow", "Kolkata", 775, 0.74),
        ("Amritsar", "Delhi", 450, 0.80),
        ("Karnal", "Delhi", 130, 0.90),
        ("Ahmedabad", "Mumbai", 530, 0.77),
        ("Ahmedabad", "Delhi", 900, 0.72),
        ("Chennai", "Hyderabad", 625, 0.76),
        ("Chennai", "Bengaluru", 350, 0.82),
        ("Kolkata", "Delhi", 1480, 0.68),
    ]

    headers = ["Source_City", "Destination_City", "Distance_KM", "Tariff_Per_Quintal_Per_KM"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for src, dst, dist, tariff in city_pairs:
            writer.writerow([src, dst, dist, tariff])
            writer.writerow([dst, src, dist, tariff])   # bidirectional

    print(f"✓ Generated {len(city_pairs)*2} transport routes → {out_path}")
    return str(out_path)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    base = Path(__file__).parent.parent / "data" / "raw"
    generate_agmarknet_csv(str(base), num_days=90)
    generate_transport_matrix_csv(str(base))
    print("\nMock data generation complete.")
