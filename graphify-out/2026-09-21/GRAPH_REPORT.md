# Graph Report - airloop-aqi  (2026-09-21)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 267 nodes · 607 edges · 15 communities (14 shown, 1 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 23 edges (avg confidence: 0.95)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `b95c19e0`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- advection.py
- correction-model/main.py
- AQIResidualTrainer
- CoupledAtmosphericChemistryEngine
- SmokeAdvectionSimulator
- run_coupled_system
- smoke-tracker/main.py
- smoke_tracker/fetch_data.py
- test_smoke_tracker.py
- smoke_tracker/__init__.py
- .simulate
- DomainBounds
- index.py
- classify_state
- vercel.json

## God Nodes (most connected - your core abstractions)
1. `AQIResidualTrainer` - 20 edges
2. `SmokeAdvectionSimulator` - 17 edges
3. `DomainBounds` - 15 edges
4. `fetch_firms_data()` - 15 edges
5. `CoupledAtmosphericChemistryEngine` - 14 edges
6. `SimulationConfig` - 14 edges
7. `fetch_wind_forecast()` - 14 edges
8. `TestSmokeTracker` - 13 edges
9. `CoupledStationAQIPredictor` - 12 edges
10. `run_pipeline()` - 12 edges

## Surprising Connections (you probably didn't know these)
- `build_coupled_dashboard()` --uses--> `CoupledSimulationResult`  [INFERRED]
  coupled_dashboard.py → coupled_feedback.py
- `run_simulation_cached()` --uses--> `CoupledStationAQIPredictor`  [INFERRED]
  streamlit_app.py → coupled_predictor.py
- `TestCoupledFeedbackPhysics` --uses--> `CoupledStationAQIPredictor`  [INFERRED]
  tests/test_coupled_system.py → coupled_predictor.py
- `build_coupled_dashboard()` --uses--> `Station72hForecast`  [INFERRED]
  coupled_dashboard.py → coupled_predictor.py
- `run_simulation_cached()` --uses--> `CoupledAtmosphericChemistryEngine`  [INFERRED]
  streamlit_app.py → coupled_feedback.py

## Import Cycles
- None detected.

## Communities (15 total, 1 thin omitted)

### Community 0 - "advection.py"
Cohesion: 0.10
Nodes (41): altair, Configuration constants, CPCB station metadata, and parameter definitions for…, Evaluation and Visualization Layer for AQI Forecast Error Correction Pipeline.…, Feature Engineering Layer for AQI Forecast Error Correction Pipeline.…, Data Ingestion Layer for Delhi NCR AQI Forecast Error Correction Pipeline.…, Model Training Layer for AQI Forecast Error Correction Pipeline. Trains…, Coupled Operational Web Dashboard Generator for Delhi NCR 72-Hour AQI…, CoupledSimulationResult (+33 more)

### Community 1 - "correction-model/main.py"
Cohesion: 0.07
Nodes (40): argparse, compute_metrics(), evaluate_forecasts(), plot_evaluation_comparison(), Any, DataFrame, ndarray, Path (+32 more)

### Community 2 - "AQIResidualTrainer"
Cohesion: 0.09
Nodes (19): AQIResidualTrainer, Any, DataFrame, ndarray, Path, Series, Fit the final regression model on all provided training data., Predict the expected AQI residual (actual - baseline). (+11 more)

### Community 3 - "CoupledAtmosphericChemistryEngine"
Cohesion: 0.09
Nodes (15): CoupledAtmosphericChemistryEngine, CoupledFeedbackStep, Estimate clear-sky surface solar irradiance (W/m²) in Delhi winter., Solve the coupled feedback equations at a single forecast time-step., Execute full 72-hour coupled atmospheric-chemical simulation across Delhi NCR., Diagnostic state of the coupled atmospheric-chemical column at hour t., Two-way coupled numerical solver modeling aerosol-radiative-boundary layer…, Estimate typical unperturbed diurnal Planetary Boundary Layer (PBL) height (m)… (+7 more)

### Community 4 - "SmokeAdvectionSimulator"
Cohesion: 0.18
Nodes (13): Simulates Lagrangian puff transport and 2D Gaussian dispersion over time., SmokeAdvectionSimulator, Atmospheric physics, grid resolution, and simulation parameters., SimulationConfig, generate_synthetic_wind_field(), Any, Generate realistic North-Westerly wind field typical of post-monsoon farm fire…, Test export to NumPy .npz and GeoJSON time series. (+5 more)

### Community 5 - "run_coupled_system"
Cohesion: 0.14
Nodes (13): cache_data, build_coupled_dashboard(), Any, Path, Recursively convert NumPy objects to standard Python types for JSON…, Generate the standalone interactive Leaflet application with coupled feedback…, to_json_safe(), main() (+5 more)

### Community 6 - "smoke-tracker/main.py"
Cohesion: 0.19
Nodes (14): main(), Path, Launch a lightweight local HTTP preview server., Main entry point and CLI pipeline for the Delhi NCR Farm-Fire Smoke Advection…, Execute the full end-to-end smoke advection simulation pipeline., run_pipeline(), serve_directory(), __init__() (+6 more)

### Community 7 - "smoke_tracker/fetch_data.py"
Cohesion: 0.23
Nodes (11): io, requests, ensure_cache_dir(), fetch_wind_forecast(), is_cache_valid(), Path, Data fetching module for NASA FIRMS Active Fire Data and Open-Meteo Wind…, Fetch wind speed and direction forecasts from Open-Meteo API for the next 72… (+3 more)

### Community 8 - "test_smoke_tracker.py"
Cohesion: 0.22
Nodes (9): os, Holds the complete multi-hour advection simulation results and grids., SimulationResult, create_leaflet_map(), Any, DataFrame, Path, Generate a standalone, high-performance, interactive Leaflet web application.… (+1 more)

### Community 9 - "smoke_tracker/__init__.py"
Cohesion: 0.28
Nodes (7): fetch_firms_data(), generate_synthetic_fires(), DataFrame, Fetch active fire data from NASA FIRMS API (or open near-real-time CSV feeds).…, Generate realistic synthetic farm-fire hotspots across Punjab and Haryana as a…, Delhi NCR Farm-Fire Smoke Advection & Dispersion Simulation Package., Verify synthetic fire dataset structure and columns.

### Community 10 - ".simulate"
Cohesion: 0.29
Nodes (6): Any, DataFrame, Execute the 72-hour smoke advection and Gaussian dispersion simulation.…, Represents a discrete Lagrangian smoke puff emitted from a fire hotspot., Build regular grid interpolators for u (Eastward) and v (Northward) wind…, SmokePuff

### Community 11 - "DomainBounds"
Cohesion: 0.25
Nodes (5): DomainBounds, Geographical bounding box definition., Returns min_lon,min_lat,max_lon,max_lat (FIRMS API format)., Check if a coordinate is within the bounding box., Test bounding box containment and string representation.

### Community 12 - "index.py"
Cohesion: 0.40
Nodes (3): handler, BaseHTTPRequestHandler, http_server

### Community 13 - "classify_state"
Cohesion: 0.50
Nodes (3): classify_state(), Assign coarse state identification based on coordinates for Punjab/Haryana/NCR., Test coordinate state classifier for Punjab, Haryana, and Delhi.

## Knowledge Gaps
- **2 isolated node(s):** `cleanUrls`, `$schema`
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 125 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `AQIResidualTrainer` connect `AQIResidualTrainer` to `advection.py`, `correction-model/main.py`?**
  _High betweenness centrality (0.176) - this node is a cross-community bridge._
- **Why does `CoupledAtmosphericChemistryEngine` connect `CoupledAtmosphericChemistryEngine` to `advection.py`, `run_coupled_system`?**
  _High betweenness centrality (0.085) - this node is a cross-community bridge._
- **Why does `TestCoupledFeedbackPhysics` connect `CoupledAtmosphericChemistryEngine` to `advection.py`?**
  _High betweenness centrality (0.075) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `AQIResidualTrainer` (e.g. with `evaluate_forecasts()` and `plot_evaluation_comparison()`) actually correct?**
  _`AQIResidualTrainer` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `SmokeAdvectionSimulator` (e.g. with `run_coupled_system()` and `DomainBounds`) actually correct?**
  _`SmokeAdvectionSimulator` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `DomainBounds` (e.g. with `SmokeAdvectionSimulator` and `fetch_firms_data()`) actually correct?**
  _`DomainBounds` has 5 INFERRED edges - model-reasoned connections that need verification._
- **What connects `cleanUrls`, `$schema` to the rest of the system?**
  _2 weakly-connected nodes found - possible documentation gaps or missing edges._