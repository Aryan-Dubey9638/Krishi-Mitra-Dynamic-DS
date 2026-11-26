# Government Schemes & Subsidies for Indian Farmers
## Compiled Reference Document for Topo-RAG Narrative Store (Variable V6)

---

## 1. Overview
The Government of India and State Agricultural Marketing Boards operate targeted schemes that reduce supply-chain costs for farmers and protect market realization against price crashes. In Topo-RAG, these interventions form Variable **V6**, evaluated deterministically alongside Mandi Price (V1), Route Distance (V2), APMC Taxes (V3), Fuel-Indexed Tariff (V4), and Weather-Aware Wastage (V5).

---

## 2. Scheme 1: MoFPI Operation Greens (TOP to TOTAL) & Kisan Rail

### Authority & Provenance
- **Ministry:** Ministry of Food Processing Industries (MoFPI) in convergence with Indian Railways.
- **Legal Basis:** Operational Guidelines for Operation Greens under Pradhan Mantri Kisan SAMPADA Yojana (`mofpi.gov.in`, `sampada-mofpi.gov.in`, `pib.gov.in`).

### Pattern of Assistance
- **50% Freight Subsidy:** Covers 50% of transportation cost from surplus clusters to consumption centres.
- **50% Storage Subsidy:** Covers 50% of hired cold storage/warehouse cost for up to 3 months.

### Eligible Crops (22 Notified Categories — TOP to TOTAL)
- **Fruits:** Mango, Banana, Apple, Pineapple, Orange/Citrus (Kinnow, Mosambi, Lime, Lemon), Grapes, Aonla/Amla, Pomegranate, Guava, Litchi (plus Kiwi, Papaya, Jackfruit).
- **Vegetables:** Tomato, Onion, Potato (TOP), Green Peas, Carrot, Cauliflower, Beans, Gourd family (Bottle, Bitter, Ridge/Sponge, Pointed, Ash Gourd), Okra (Bhindi), Garlic, Ginger, Green Chillies, Cabbage, Capsicum, Cucumber, Turmeric.

### Channel & Quantity Thresholds
- **Standard Road Transport:** Minimum distance **100 km** (250 km for state federations/retailers) and minimum lot size **9 MT (90 quintals)** for individual farmers/groups (**100 MT** for FPOs/co-ops).
- **Kisan Rail Convergence:** **Zero minimum quantity limit** and no price-fall trigger; Indian Railways deducts 50% of the freight charge upfront at booking and claims reimbursement from MoFPI.

---

## 3. Scheme 2: State APMC FPO & Direct Marketing Fee Waivers

### Authority & Provenance
- **Sources:** Directorate of Marketing & Inspection (`dmi.gov.in`), State APMC Amendment Acts, and State Agricultural Marketing Boards (MSAMB, KSAMB, UP Mandi Parishad).

### State-Specific Provisions
- **Maharashtra:** Under the 2016 APMC Amendment Ordinance, fruits and vegetables are delisted outside APMC yards; trading via **FPOs / Direct Marketing / Shetkari Bazars** outside notified yards is exempt from the 1.05% APMC market fee.
- **Karnataka:** Under KSAMB rules, **0% market fee** is levied on flowers, fruits, and vegetables inside APMCs; FPOs and direct purchase centers also receive exemptions under the APMC Amendment Act.
- **Bihar & Kerala:** Bihar repealed its APMC Act in 2006 and Kerala uses a non-APMC cooperative system (VFPCK/Horticorp), keeping statutory APMC market fees at **0%** statewide.
- **Uttar Pradesh, Gujarat, Rajasthan, Tamil Nadu, Andhra Pradesh, Telangana:** FPOs operating direct purchase or Kisan Bazaar channels outside principal mandi yards qualify for full market fee exemption.
- **Madhya Pradesh:** FPOs qualify for a 50% concession on statutory mandi charges under promotional guidelines.
- **Punjab & Haryana:** Full Market Fee (2%) and Rural Development Fund cess (2%) apply across notified agricultural trade.

---

## 4. Scheme 3: PM-AASHA — Price Deficiency Payment Scheme (PDPS) & MSP Floor

### Authority & Provenance
- **Ministry:** Department of Agriculture & Farmers Welfare (DA&FW), based on CACP recommendations approved by the Cabinet Committee on Economic Affairs (CCEA) (`agriwelfare.gov.in`, `desagri.gov.in`, `pib.gov.in`).

### PDPS Mechanics (Oilseeds)
- For notified oilseeds (**Soybean, Mustard, Groundnut, Sunflower, Sesamum, Nigerseed**), registered farmers selling in notified mandis receive a **Direct Benefit Transfer (DBT)** covering the difference between MSP and market modal price, **capped at 15% of MSP** (with 40% state production coverage per the September 2024 Cabinet revamp).

### PSS & MSP Procurement Floor (Pulses, Cereals, Cotton)
- For pulses (**Tur/Arhar, Moong, Urad, Gram**) under PM-AASHA **Price Support Scheme (PSS)** and cereals/cotton (**Paddy, Wheat, Maize, Cotton**), protection operates via physical procurement at MSP by NAFED, NCCF, FCI, or CCI rather than open-market DBT top-ups. When mandi prices drop below MSP, farmers should route produce to an official procurement center.

### Official MSP Benchmarks (₹/quintal, 2024–25 & 2025–26)
- **Soybean (Yellow, KMS 2025-26):** ₹5,328
- **Mustard (RMS 2025-26):** ₹6,200
- **Groundnut (KMS 2024-25):** ₹6,783
- **Gram / Chana (RMS 2025-26):** ₹5,650
- **Tur / Arhar (KMS 2024-25):** ₹7,550
- **Moong (KMS 2024-25):** ₹8,682
- **Wheat (RMS 2025-26):** ₹2,425
- **Paddy Common / Grade A (KMS 2025-26):** ₹2,369 / ₹2,389
- **Cotton Medium / Long Staple (KMS 2025-26):** ₹7,710 / ₹8,110

---

## 5. Deterministic Integration into Topo-RAG Net Profit Equation

```
Gross Revenue             = Sellable Quantity × Effective Modal Price (V1, V5)
Effective Gross Revenue   = Gross Revenue + Price Deficiency Support (V6 PDPS)
Gross Total Costs         = Transport (V4, V2) + APMC Commission (V3) + Mandi Fees (V3) + Wastage (V5)
Effective Total Costs     = Gross Total Costs - Freight Subsidy (V6) - Mandi Fee Waiver (V6)

Baseline Net Profit       = Gross Revenue - Gross Total Costs
Subsidized Net Profit     = Effective Gross Revenue - Effective Total Costs
                          = Baseline Net Profit + Total Subsidy Benefit
```
