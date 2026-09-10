#!/usr/bin/env python3
"""
Main entry point and CLI pipeline for the Delhi NCR Farm-Fire Smoke Advection Simulator.
Fetches NASA FIRMS fire data, Open-Meteo wind forecasts, simulates 72h smoke dispersion,
and outputs GeoJSON, NumPy arrays, and an interactive Leaflet web map.
"""

import argparse
import logging
import sys
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Optional

# Ensure package root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from smoke_tracker.config import (
    SimulationConfig,
    DEFAULT_CONFIG,
    OUTPUT_DIR,
    REGIONAL_DOMAIN,
    DELHI_NCR_DOMAIN
)
from smoke_tracker.fetch_data import fetch_firms_data, fetch_wind_forecast
from smoke_tracker.advection import (
    SmokeAdvectionSimulator,
    save_to_numpy,
    save_to_geojson
)
from smoke_tracker.visualize import create_leaflet_map

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

logger = logging.getLogger("smoke_tracker.main")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)


def run_pipeline(
    firms_key: Optional[str] = None,
    use_cache: bool = True,
    force_synthetic: bool = False,
    forecast_hours: int = 72,
    output_dir: Optional[Path] = None,
    open_browser: bool = False,
    serve_port: Optional[int] = None
):
    """Execute the full end-to-end smoke advection simulation pipeline."""
    target_output_dir = output_dir or OUTPUT_DIR
    target_output_dir.mkdir(parents=True, exist_ok=True)
    
    config = SimulationConfig(
        forecast_hours=forecast_hours,
        use_cache=use_cache
    )
    
    print("\n" + "="*70)
    print(" [SMOKE-TRACKER] DELHI NCR FARM-FIRE SMOKE DISPERSION (72-HOUR FORECAST)")
    print("="*70 + "\n")
    
    # 1. Fetch NASA FIRMS Fire Data
    print("[1/5] Fetching Active Fire Data for Punjab & Haryana (NASA FIRMS)...")
    fires_df = fetch_firms_data(
        api_key=firms_key,
        use_cache=use_cache,
        force_synthetic=force_synthetic
    )
    print(f"   ✓ Loaded {len(fires_df)} active fire hotspots in NW India.")
    if "state" in fires_df.columns:
        counts = fires_df["state"].value_counts().to_dict()
        for state, count in counts.items():
            print(f"     - {state}: {count} fires")
    
    # 2. Fetch Open-Meteo Wind Forecast
    print("\n[2/5] Fetching 72-Hour Atmospheric Wind Forecast (Open-Meteo)...")
    wind_data = fetch_wind_forecast(
        grid_step=config.meteo_grid_step,
        forecast_days=max(1, forecast_hours // 24),
        use_cache=use_cache
    )
    print(f"   [+] Retrieved wind field for {len(wind_data['times'])} hourly timesteps.")
    
    # 3. Execute Lagrangian Advection & Dispersion
    print("\n[3/5] Simulating Smoke Advection & Gaussian Dispersion...")
    simulator = SmokeAdvectionSimulator(config=config)
    result = simulator.simulate(fires_df, wind_data)
    print("   [+] Simulation complete!")
    print(f"     * Peak Delhi NCR Smog Arrival: Hour +{result.peak_delhi_hour} ({result.peak_delhi_time})")
    print(f"     * Peak Smog Concentration Index: {result.peak_delhi_density:.1f}")
    print(f"     * Contributing Fires reaching NCR: {result.contributing_fire_count} / {result.total_active_fires}")
    
    # 4. Save Outputs (NumPy array and GeoJSON)
    print("\n[4/5] Exporting Data Formats...")
    npz_path = save_to_numpy(result, target_output_dir / "delhi_smoke_simulation_72h.npz")
    geojson_path = save_to_geojson(result, target_output_dir / "delhi_smoke_timeseries.geojson")
    print(f"   [+] NumPy Array (.npz):  {npz_path}")
    print(f"   [+] GeoJSON Time Series: {geojson_path}")
    
    # 5. Generate Web Map
    print("\n[5/5] Generating Interactive Leaflet Simulation Web Map...")
    html_path = create_leaflet_map(
        result=result,
        fires_df=fires_df,
        wind_data=wind_data,
        output_html_path=target_output_dir / "smoke_simulation_map.html"
    )
    print(f"   [+] Standalone Web Map:  {html_path}")
    
    print("\n" + "="*70)
    print(" [OK] PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70 + "\n")
    
    # Optional Serve / Open
    if open_browser:
        webbrowser.open(f"file:///{html_path.resolve()}")
        
    if serve_port:
        serve_directory(target_output_dir, serve_port)


def serve_directory(directory: Path, port: int = 8000):
    """Launch a lightweight local HTTP preview server."""
    class CustomHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(directory), **kwargs)
            
    print(f"📡 Serving web map at http://localhost:{port}/smoke_simulation_map.html (Press Ctrl+C to stop)...")
    server = HTTPServer(("0.0.0.0", port), CustomHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
        server.server_close()


def main():
    parser = argparse.ArgumentParser(
        description="Simulate farm-fire smoke advection from Punjab/Haryana over Delhi NCR."
    )
    parser.add_argument("--firms-key", type=str, default=None, help="NASA FIRMS MAP_KEY (optional)")
    parser.add_argument("--no-cache", action="store_true", help="Force re-fetching live API data without cache")
    parser.add_argument("--synthetic", action="store_true", help="Use realistic synthetic fires scenario")
    parser.add_argument("--hours", type=int, default=72, help="Forecast hours (default: 72)")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom directory for generated outputs")
    parser.add_argument("--open", action="store_true", help="Open generated HTML map in default browser")
    parser.add_argument("--serve", type=int, nargs="?", const=8080, default=None, help="Start HTTP server on given port (default: 8080)")

    args = parser.parse_args()
    
    out_dir = Path(args.output_dir) if args.output_dir else None
    run_pipeline(
        firms_key=args.firms_key,
        use_cache=not args.no_cache,
        force_synthetic=args.synthetic,
        forecast_hours=args.hours,
        output_dir=out_dir,
        open_browser=args.open,
        serve_port=args.serve
    )


if __name__ == "__main__":
    main()
