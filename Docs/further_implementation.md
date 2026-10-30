# Future Implementation Roadmap: AI-Powered Agri-Advisory Platform

This document outlines the comprehensive list of features, frameworks, and technologies that need to be implemented to evolve the current MVP into the final production system as defined in the **Architecture Proposal** and **Project Synopsis**.

*(Note: LLM synthesis layers and dataset integrations are excluded from this list per requirements).*

---

## 1. Core Architecture & Infrastructure Upgrades
The current shared RAG engine needs to be upgraded to handle scale and real-time execution.

* **Vector Storage Upgrade (WARP/Qdrant)**: Migrate the current FAISS-based tabular store approximation to a true multi-vector engine (Qdrant or a WARP/CRISP engine) to support efficient late-interaction (ColBERT v2) at scale.
* **Backend Framework**: Port the current local Python scripts to an asynchronous **FastAPI** backend (Python 3.11+).
* **Database & Geospatial Systems**: Implement **PostgreSQL with PostGIS** for spatial location mapping and dynamic distance matrix calculations.
* **Caching & Message Queues**: Integrate **Redis** (for fast mandi price caching) and **Celery** (for asynchronous scraping and background tasks).

---

## 2. Feature 1: Smart Market & Mandi Intelligence Advisor
Expanding the current deterministic math engine into a robust, real-time decision system.

* **Live Data ETL Pipelines**: Build automated scrapers using **Scrapy/Playwright** (and Pandas/Polars) for live Agmarknet/e-NAM mandi rates, transport tariffs, and road distance matrices.
* **Weather API Integration**: Implement live tool calls to **Open-Meteo / OpenWeatherMap** to dynamically calculate post-harvest crop wastage multipliers based on current weather forecasts.
* **Cold Storage Integration**: Incorporate cold storage cost and availability data into the retrieval pipeline.
* **Expanded Decision Engine**: Upgrade the `ProfitCalculator` to evaluate three parallel options—**Sell vs. Store vs. Transport**—factoring in weather-adjusted wastage and cold storage holding costs.
* **Historical Trend Retrieval**: Enable the system to retrieve and factor in historical price trends (both tabular and narrative).

---

## 3. Feature 2: Scheme & Subsidy News Advisor
A new vertical designed to route farmers to relevant, active government subsidies.

* **Ministerial Web Scraping**: Implement scrapers for PM-KISAN, state agricultural tech portals, and other government sites.
* **Metadata-Enriched Ingestion**: Chunk and vectorize scheme documents into the `NarrativeStore`, classified by metadata tags: `state`, `category`, and `status` (active/expired).
* **Staleness Tracking**: Implement a system to track the `last_verified_timestamp` of scraped schemes.
* **Dynamic Query Filtering**: Implement retrieval logic that filters out expired schemes *during the query phase* (not at ingestion) and strictly filters schemes based on the farmer's state/region.

---

## 4. Delivery & Interfaces
Transitioning from a CLI tool to accessible farmer-facing platforms.

* **Multilingual Support (Hindi)**: Implement **Hindi text and Hindi conversation** capabilities so farmers can interact with the advisory platform natively in their local language.
* **Web Dashboard**: Build the frontend interface using **Streamlit** (or similar web framework).
* **Conversational Bot Integration**: Integrate the backend with the **Twilio API** to deploy the platform as a WhatsApp conversational bot.

---

## 5. Advanced Evaluation & Future Roadmap
Long-term validation and phase-two architectural shifts.

* **Retrieval Quality Evaluation**: Implement automated benchmarking using **nDCG@10** for Type A (narrative), Type B (cell-precise), and Type C (hybrid) queries to compare Topo-RAG against naive linearization baselines.
* **GraphRAG Integration**: Build a Knowledge Graph to connect retrieved table cells (e.g., supplier IDs) directly to narrative contract nodes for 2nd-order multi-hop reasoning.
* **Shared Truck Pooling Engine**: Develop a logistics optimizer that groups neighboring farmers delivering to the same wholesale market, aggregating crop loads to cut transport costs by 30-50%.
