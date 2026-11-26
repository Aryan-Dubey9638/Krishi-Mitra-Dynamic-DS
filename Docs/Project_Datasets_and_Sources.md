# Datasets & Official Data Sources — Agricultural Price Discovery & Profit Advisory Platform

This document summarizes all primary datasets, government reports, statutory schedules, and live public APIs used across the platform for mandi price discovery, logistics estimation, wastage modeling, and government subsidy evaluation.

---

## 1. Agricultural Market Prices (Mandi Rates)

| Dataset / Source | Authority / Provider | Scope & Coverage | Official Portal / API Link |
| :--- | :--- | :--- | :--- |
| **Live Daily Mandi Prices (Agmarknet OGD API)** | Ministry of Agriculture & Farmers Welfare (via Open Government Data Platform India) | Real-time daily commodity-wise minimum, maximum, and modal prices across Indian wholesale mandis (`Resource ID: 9ef84268-d588-465a-a308-a864a43d0070`). | [OGD Platform India (data.gov.in)](https://data.gov.in) · [AGMARKNET Portal](https://agmarknet.gov.in) |
| **Historical Agricultural Commodity Price Dataset** | AGMARKNET / Directorate of Marketing & Inspection (DMI) | ~737,000+ historical wholesale mandi price records spanning multiple states, districts, markets, crop varieties, quality grades, and arrival dates. | [AGMARKNET Portal](https://agmarknet.gov.in) · [data.gov.in](https://data.gov.in) |

---

## 2. Fuel Prices & Road Logistics Tariffs

| Dataset / Source | Authority / Provider | Scope & Coverage | Official Portal / PDF Link |
| :--- | :--- | :--- | :--- |
| **PPAC Ready Reckoner (Table 8.14) & Daily Diesel Price Reports** | Petroleum Planning & Analysis Cell (PPAC), Ministry of Petroleum & Natural Gas / IOCL | State-wise Retail Selling Prices (RSP) of High-Speed Diesel (HSD), petrol, and LPG across Indian states and union territories used to index commercial truck freight rates. | [PPAC Retail Selling Price Portal & PDFs](https://ppac.gov.in/retail-selling-price-rsp-of-petrol-diesel-and-domestic-lpg) · [PPAC Home](https://ppac.gov.in) |
| **Indian Highway Routing & Geospatial Network** | Open Source Routing Machine (OSRM) & OpenStreetMap (Nominatim) | Live highway driving distance (km), transit duration, and latitude/longitude geocoding across Indian cities and wholesale mandis. | [OSRM API Documentation](https://project-osrm.org/docs/v5.24.0/api/) · [Nominatim Geocoding API](https://nominatim.org/release-docs/latest/api/Search/) |

---

## 3. Statutory Mandi Taxes & Agricultural Market Regulations

| Dataset / Source | Authority / Provider | Scope & Coverage | Official Portal / Reference Link |
| :--- | :--- | :--- | :--- |
| **State APMC Market Fee & Development Cess Schedules** | State Agricultural Marketing Boards (MSAMB, DAMB, PSAMB, UP Mandi Parishad, KSAMB, etc.) & DMI | State-wise statutory market fees, rural development cesses, commission caps, and loading/handling charges notified under respective State APMC Acts. | [Directorate of Marketing & Inspection (DMI)](https://dmi.gov.in) · [MSAMB](https://www.msamb.com) |
| **e-NAM Trading & Market Fee Norms** | Small Farmers' Agribusiness Consortium (SFAC), Ministry of Agriculture | Electronic wholesale mandi integration, assay-based grading, and inter-state market linkage norms across 1,000+ mandis. | [e-NAM National Agriculture Market](https://enam.gov.in) |

---

## 4. Post-Harvest Wastage & Live Ambient Weather

| Dataset / Source | Authority / Provider | Scope & Coverage | Official Portal / Report Link |
| :--- | :--- | :--- | :--- |
| **Harvest & Post-Harvest Losses Study** | ICAR – Central Institute of Post-Harvest Engineering & Technology (CIPHET) | Empirical crop-wise farm-gate, transit, and mandi handling loss benchmarks for perishable fruits, vegetables, cereals, pulses, and oilseeds. | [ICAR–CIPHET Report (ResearchGate Archive)](https://www.researchgate.net/publication/280780214_Harvest_and_post_harvest_losses_of_major_crops_and_livestock_produce_in_India) · [MoFPI Portal](https://www.mofpi.gov.in) |
| **Live Transit Corridor Weather Feed** | Open-Meteo Weather API | Real-time ambient temperature (°C) and relative humidity (%) along source and destination coordinates to model thermal spoilage risk. | [Open-Meteo Forecast API](https://open-meteo.com/en/docs) |

---

## 5. Government Subsidies, Freight Concessions & Price Support Schemes

| Dataset / Scheme | Authority / Provider | Scope & Coverage | Official Portal / Guidelines Link |
| :--- | :--- | :--- | :--- |
| **Operation Greens (TOP to TOTAL) & Kisan Rail Guidelines** | Ministry of Food Processing Industries (MoFPI) & Ministry of Railways | Operational guidelines providing a 50% freight and cold-storage subsidy for 22 notified fruit/vegetable categories (with zero minimum quantity via Kisan Rail). | [MoFPI Operation Greens](https://www.mofpi.gov.in) · [PMKSY SAMPADA Portal](https://www.sampada-mofpi.gov.in/) |
| **PM-AASHA (PDPS / PSS) & CACP Minimum Support Prices (MSP)** | Department of Agriculture & Farmers Welfare (DA&FW), CACP & CCEA | Official 2024–25 and 2025–26 Kharif/Rabi MSP rates, Price Deficiency Payment Scheme (15% MSP compensation cap for oilseeds), and Price Support Scheme (PSS) procurement norms. | [DA&FW Portal](https://agriwelfare.gov.in) · [Directorate of Economics & Statistics (MSP)](https://desagri.gov.in) · [PIB Releases](https://pib.gov.in) |
| **State APMC Direct Marketing & FPO Exemptions** | Directorate of Marketing & Inspection (DMI) & State Marketing Boards | State-level APMC amendments granting market fee waivers for Farmer Producer Organizations (FPOs), direct marketing, and horticulture produce (e.g., Maharashtra 2016 Ordinance, Karnataka F&V exemption, Bihar APMC repeal). | [DMI Agricultural Marketing Reforms](https://dmi.gov.in) |

---

## 6. Commodity Quality, Grading & Export Standards

| Dataset / Source | Authority / Provider | Scope & Coverage | Official Portal / Reference Link |
| :--- | :--- | :--- | :--- |
| **AGMARK & BIS Agricultural Grading Standards** | Directorate of Marketing & Inspection (DMI) & Bureau of Indian Standards (BIS) | Quality grading specifications (Grade A/B/C), moisture limits, export sizing/MRL requirements, certification fees, and model contract-farming clauses. | [DMI AGMARK Standards](https://dmi.gov.in) · [AGMARKNET](https://agmarknet.gov.in) |
