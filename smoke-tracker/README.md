# Delhi NCR Farm-Fire Smoke Tracker & Dispersion Simulator

A high-performance Python package and interactive simulation system for modeling agricultural stubble-burning (farm-fire) smoke transport from Punjab and Haryana into Delhi NCR over a 72-hour forecast horizon.

---

## 🌟 Overview

Every autumn (October–November), seasonal paddy crop residue burning in Punjab and Haryana injects massive quantities of particulate matter ($PM_{2.5}$) into the planetary boundary layer. Prevailing north-westerly winds advect these dense smoke plumes towards Delhi National Capital Region (NCR), causing severe air quality crises.

This package provides an end-to-end pipeline:
1. **Live Data Ingestion**:
   - **NASA FIRMS**: Fetches near-real-time thermal anomalies / active fire hotspots (VIIRS NOAA-20, Suomi-NPP, NOAA-21, and MODIS) for Punjab and Haryana.
   - **Open-Meteo**: Queries 72-hour multi-level wind speed and direction forecasts across a spatial grid covering NW India and Delhi NCR.
2. **Atmospheric Physics Simulation**:
   - Lagrangian puff tracking with continuous hourly puff release.
   - Spatial wind vector interpolation ($u, v$).
   - 2D expanding Gaussian dispersion kernel: $\sigma(t) = \sigma_0 + \alpha \sqrt{t}$.
   - Scavenging and atmospheric exponential decay ($e^{-\lambda t}$).
   - Fire Radiative Power (FRP) emission scaling.
3. **Data Serialization**:
   - Hourly 3D NumPy array `(72, n_lat, n_lon)` saved to compressed `.npz`.
   - Hourly contour/grid polygon time series serialized to GeoJSON.
4. **Interactive Leaflet Web Dashboard**:
   - Interactive 72-hour timeline scrubber and playback animation controls.
   - Real-time glassmorphic HUD displaying mean/peak smoke index, active fire counts, and predicted peak arrival hour.
   - Dynamic 72-hour sparkline chart with synchronized scrub cursor.
   - Interactive fire hotspot markers with FRP, satellite, and detection timestamps.
   - Vector wind field arrows and toggles for smoke, fire, puff, and station layers.

---

## 📁 Repository Structure

```
smoke-tracker/
├── smoke_tracker/
│   ├── __init__.py          # Package exports and version
│   ├── config.py            # Spatial domains, grid definitions, and physics parameters
│   ├── fetch_data.py        # NASA FIRMS & Open-Meteo API clients with local disk caching
│   ├── advection.py         # Wind interpolation, Lagrangian puff advection, Gaussian kernel
│   └── visualize.py         # Leaflet/HTML5 standalone web map generator
├── data/
│   ├── cache/               # Local cache for NASA FIRMS CSVs and Open-Meteo JSONs
│   └── outputs/             # Generated simulation outputs (.npz, .geojson, .html)
├── tests/
│   └── test_smoke_tracker.py# Comprehensive unit and integration test suite
├── main.py                  # CLI entrypoint for running simulations
├── requirements.txt         # Package dependencies
└── README.md                # Documentation
```

---

## 🚀 Installation & Quickstart

### 1. Requirements
Python 3.10+ is recommended. Install dependencies via:

```bash
pip install -r requirements.txt
```

### 2. Run Default Simulation (Live or Cached Data)
To run an end-to-end 72-hour simulation:

```bash
python main.py
```

This will:
- Query NASA FIRMS (or fallback to synthetic scenario if offline).
- Query Open-Meteo 72h wind forecast grid.
- Compute Lagrangian advection and dispersion across Punjab, Haryana, and Delhi NCR.
- Save outputs to `data/outputs/`:
  - `delhi_smoke_simulation_72h.npz`
  - `delhi_smoke_timeseries.geojson`
  - `smoke_simulation_map.html`

### 3. Open the Interactive Web Map
Open `data/outputs/smoke_simulation_map.html` in any modern web browser or serve it locally:

```bash
python main.py --serve
```

---

## ⚙️ CLI Options

```text
usage: main.py [-h] [--firms-key FIRMS_KEY] [--lookback LOOKBACK] [--no-cache]
               [--output-dir OUTPUT_DIR] [--resolution RESOLUTION]
               [--synthetic] [--serve] [--port PORT]

Delhi NCR Farm-Fire Smoke Advection Simulation

options:
  -h, --help            show this help message and exit
  --firms-key FIRMS_KEY NASA FIRMS MAP_KEY (optional)
  --lookback LOOKBACK   Lookback days for active fire detection (default: 1)
  --no-cache            Bypass cache and force fresh API requests
  --output-dir OUTPUT_DIR
                        Directory to save simulation outputs
  --resolution RESOLUTION
                        Delhi grid resolution: 'low' (25x25), 'standard' (50x50), or 'high' (75x75)
  --synthetic           Force generation of historical synthetic fire scenario
  --serve               Launch local HTTP server to preview map in browser
  --port PORT           Port for HTTP server (default: 8080)
```

---

## 🐍 Python API Usage

You can use `smoke_tracker` as a Python library in your own data pipelines:

```python
from smoke_tracker.fetch_data import fetch_firms_data, fetch_wind_forecast
from smoke_tracker.advection import SmokeAdvectionSimulator
from smoke_tracker.visualize import create_leaflet_map

# 1. Fetch fire hotspots and wind forecast
fires_df = fetch_firms_data(lookback_days=1, use_cache=True)
wind_data = fetch_wind_forecast(forecast_days=3, use_cache=True)

# 2. Initialize and run 72-hour dispersion simulation
simulator = SmokeAdvectionSimulator(delhi_grid_size=50)
result = simulator.simulate(fires_df, wind_data, forecast_hours=72)

print(f"Peak Delhi smoke impact: Hour {result.peak_hour} ({result.peak_time})")

# 3. Export data products
simulator.save_to_numpy(result, "data/outputs/delhi_smoke_simulation_72h.npz")
simulator.save_to_geojson(result, "data/outputs/delhi_smoke_timeseries.geojson")

# 4. Generate standalone interactive Leaflet web map
map_path = create_leaflet_map(result, fires_df, wind_data)
print(f"Interactive web map written to: {map_path}")
```

---

## 🔬 Mathematical Modeling Details

### 1. Emission Scaling
Each fire hotspot detected by VIIRS/MODIS provides Fire Radiative Power ($FRP$, in MW). Emission strength $Q$ is proportional to FRP:
$$Q_i = Q_0 \cdot \sqrt{FRP_i}$$
where $Q_0 = 100$ is the base emission factor.

### 2. Lagrangian Puff Advection
Continuous puffs are released hourly from each fire coordinate $(x_0, y_0)$. At each time step $\Delta t$, puff coordinates are updated using bi-linearly interpolated wind vector components $u(x, y, t)$ and $v(x, y, t)$:
$$x(t + \Delta t) = x(t) + u(x, y, t) \cdot \Delta t$$
$$y(t + \Delta t) = y(t) + v(x, y, t) \cdot \Delta t$$

### 3. Expanding Gaussian Kernel & Atmospheric Decay
The spatial smoke concentration contribution $C$ at grid point $(x_g, y_g)$ from a puff at $(x_p, y_p)$ with age $t_{\text{age}}$ is modeled by:
$$C(x_g, y_g) = \frac{Q_i}{2\pi \sigma^2(t_{\text{age}})} \exp\left(-\frac{(x_g - x_p)^2 + (y_g - y_p)^2}{2\sigma^2(t_{\text{age}})}\right) \cdot \exp(-\lambda t_{\text{age}})$$
where:
- $\sigma(t) = \sigma_0 + \alpha \sqrt{t}$ represents turbulent eddy diffusion ($\alpha = 0.045$).
- $\lambda = 0.025\ \text{hr}^{-1}$ accounts for dry deposition and atmospheric scavenging.

---

## 🧪 Testing

Run the automated test suite with Python's built-in `unittest`:

```bash
python -m unittest tests/test_smoke_tracker.py
```
All 7 test cases validate data fetching fallbacks, wind coordinate interpolations, physical mass conservation, GeoJSON serialization, and HTML generation.
