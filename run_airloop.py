"""
AirLoop Delhi NCR: Master CLI Orchestrator for Coupled Atmospheric-Chemical 72-Hour AQI Forecasting.
Runs end-to-end:
1. NASA FIRMS Active Fire Ingestion (Punjab/Haryana stubble burning).
2. Open-Meteo 72h Wind Grid Forecasting.
3. Lagrangian Puff Advection Simulation.
4. Two-Way Coupled Feedback Modeling (Solar Dimming, PBL Collapse, Induced Stillness).
5. 72-Hour Station-by-Station CPCB AQI Predictions (Decoupled vs Coupled).
6. Generates the Operational Leaflet Web Dashboard.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

# Add root and submodules to Python path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

smoke_tracker_dir = ROOT_DIR / "smoke-tracker"
correction_model_dir = ROOT_DIR / "correction-model"
if str(smoke_tracker_dir) not in sys.path:
    sys.path.insert(0, str(smoke_tracker_dir))
if str(correction_model_dir) not in sys.path:
    sys.path.insert(0, str(correction_model_dir))

import pandas as pd
import numpy as np

from smoke_tracker.fetch_data import fetch_firms_data, fetch_wind_forecast
from smoke_tracker.advection import SmokeAdvectionSimulator
from coupled_feedback import CoupledAtmosphericChemistryEngine
from coupled_predictor import CoupledStationAQIPredictor
from coupled_dashboard import build_coupled_dashboard
from correction_model.config import CPCB_STATIONS

logger = logging.getLogger("airloop.master")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)


def run_coupled_system(
    forecast_hours: int = 72,
    use_cache: bool = True,
    serve: bool = False,
    port: int = 8080,
    output_dir: Optional[Path] = None
):
    """
    Execute the entire coupled atmospheric-chemical forecasting system end-to-end.
    """
    out_dir = output_dir or (ROOT_DIR / "data" / "outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 78)
    print("  AIRLOOP DELHI NCR: COUPLED ATMOSPHERIC PHYSICS & CHEMICAL TRANSPORT SYSTEM")
    print("  72-Hour Operational AQI Prediction with Weather-Pollution Feedback Loops")
    print("=" * 78 + "\n")

    # -------------------------------------------------------------
    # Step 1: Ingest Live / Cached Data
    # -------------------------------------------------------------
    logger.info("STEP 1: Ingesting active fire hotspots and 72h wind forecast grid...")
    fires_df = fetch_firms_data(lookback_days=1, use_cache=use_cache)
    wind_data = fetch_wind_forecast(forecast_days=3, use_cache=use_cache)
    logger.info(f"Loaded {len(fires_df)} active fire hotspots across Punjab & Haryana.")
    logger.info(f"Loaded 72h wind field grid with {len(wind_data.get('lats', []))}x{len(wind_data.get('lons', []))} points.")

    # -------------------------------------------------------------
    # Step 2: Physical Lagrangian Dispersion Simulation
    # -------------------------------------------------------------
    logger.info("\nSTEP 2: Simulating Lagrangian puff advection over Punjab, Haryana, and Delhi NCR...")
    simulator = SmokeAdvectionSimulator()
    sim_result = simulator.simulate(fires_df, wind_data)
    logger.info(f"Dispersion completed. Peak Delhi plume arrival predicted at Hour +{sim_result.peak_delhi_hour} ({sim_result.peak_delhi_time}).")

    # -------------------------------------------------------------
    # Step 3: Two-Way Bidirectional Coupled Feedback Modeling
    # -------------------------------------------------------------
    logger.info("\nSTEP 3: Solving two-way coupled atmospheric-chemical feedback equations...")
    logger.info("  • Radiative Extinction & Solar Dimming (Beer-Lambert Law)")
    logger.info("  • Planetary Boundary Layer (PBL) Height Thermal Collapse")
    logger.info("  • Ground-Level Volumetric Compression & Smog Entrapment")
    logger.info("  • Aerosol-Induced Surface Stagnation & Wind Deceleration")
    
    coupled_engine = CoupledAtmosphericChemistryEngine()
    
    # Ambient winter temperature and wind traces
    ambient_temps = [24.0 - (h % 24) * 0.45 for h in range(forecast_hours)]
    ambient_winds = [max(0.6, float(np.mean(wind_data["speed"][h]))) for h in range(forecast_hours)]
    
    coupled_result = coupled_engine.simulate_72h_coupled_trajectory(
        decoupled_delhi_mean=sim_result.delhi_mean_series,
        decoupled_delhi_max=sim_result.delhi_max_series,
        ambient_temps=ambient_temps,
        ambient_winds=ambient_winds,
        timestamps=wind_data.get("times", [f"+{h}h" for h in range(forecast_hours)])
    )

    # -------------------------------------------------------------
    # Step 4: 72-Hour Station-by-Station CPCB AQI Predictions
    # -------------------------------------------------------------
    logger.info("\nSTEP 4: Generating 72-hour AQI forecasts for all 10 CPCB monitoring stations...")
    predictor = CoupledStationAQIPredictor(CPCB_STATIONS)
    station_forecasts = predictor.predict_72h_station_forecasts(
        coupled_result=coupled_result,
        station_smoke_traces=sim_result.station_series,
    )

    # Print Summary Table
    print("\n" + "-" * 78)
    print(f"{'STATION':<18} | {'CURRENT':<8} | {'DECOUPLED PEAK':<15} | {'COUPLED PEAK':<15} | {'UNDERPRED. GAP':<14}")
    print("-" * 78)
    for st_name, st_fc in station_forecasts.items():
        gap = st_fc.peak_coupled_aqi - st_fc.peak_decoupled_aqi
        print(f"{st_name:<18} | {st_fc.current_measured_aqi:<8.1f} | {st_fc.peak_decoupled_aqi:<15.1f} | {st_fc.peak_coupled_aqi:<15.1f} | +{gap:<13.1f}")
    print("-" * 78)
    print(f"Regional Average Underprediction Gap: +{coupled_result.underprediction_max_gap:.1f} AQI points")
    print("-" * 78 + "\n")

    # -------------------------------------------------------------
    # Step 5: Render Standalone Web Dashboard & Save Artifacts
    # -------------------------------------------------------------
    logger.info("STEP 5: Generating interactive operational Leaflet dashboard...")
    fires_list = [
        {"lat": float(r["latitude"]), "lon": float(r["longitude"]), "frp": float(r.get("frp", 30)), "state": r.get("state", "Punjab")}
        for _, r in fires_df.iterrows()
    ]
    dashboard_path = build_coupled_dashboard(
        coupled_result=coupled_result,
        station_forecasts=station_forecasts,
        fires_data=fires_list,
        wind_data=wind_data,
        output_html_path=out_dir / "airloop_coupled_dashboard.html"
    )

    # Save data outputs
    forecast_json_path = out_dir / "coupled_72h_forecast.json"
    with open(forecast_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "peak_hour": coupled_result.peak_smog_hour,
            "peak_time": coupled_result.peak_smog_time,
            "underprediction_max_gap": coupled_result.underprediction_max_gap,
            "decoupled_delhi_mean": coupled_result.decoupled_delhi_mean,
            "coupled_delhi_mean": coupled_result.coupled_delhi_mean,
            "stations": {
                st_name: {
                    "peak_coupled_aqi": st.peak_coupled_aqi,
                    "peak_decoupled_aqi": st.peak_decoupled_aqi,
                    "severity": st.peak_severity_category,
                    "attribution": st.feedback_attribution
                }
                for st_name, st in station_forecasts.items()
            }
        }, f, indent=2)
    logger.info(f"Saved coupled forecast summary to {forecast_json_path}")

    # -------------------------------------------------------------
    # Step 6: Launch Local Server Preview if Requested
    # -------------------------------------------------------------
    if serve:
        logger.info(f"\nLaunching local HTTP server to preview dashboard at http://localhost:{port} ...")
        import http.server
        import socketserver
        import webbrowser
        
        class QuietHandler(http.server.SimpleHTTPRequestHandler):
            def log_message(self, format, *args):
                pass
                
        def start_server():
            # Change directory to outputs
            import os
            os.chdir(str(out_dir))
            with socketserver.TCPServer(("", port), QuietHandler) as httpd:
                print(f"Server active at http://localhost:{port}/airloop_coupled_dashboard.html")
                webbrowser.open(f"http://localhost:{port}/airloop_coupled_dashboard.html")
                try:
                    httpd.serve_forever()
                except KeyboardInterrupt:
                    print("\nServer stopped.")

        start_server()

    return dashboard_path, coupled_result, station_forecasts


def main():
    parser = argparse.ArgumentParser(
        description="AirLoop Delhi NCR: Coupled Atmospheric-Chemical 72-Hour AQI Forecasting System"
    )
    parser.add_argument("--hours", type=int, default=72, help="Forecast horizon in hours (default: 72)")
    parser.add_argument("--no-cache", action="store_true", help="Force fresh live data fetch from APIs")
    parser.add_argument("--serve", action="store_true", help="Launch local HTTP server and open browser preview")
    parser.add_argument("--port", type=int, default=8080, help="HTTP server preview port (default: 8080)")

    args = parser.parse_args()
    run_coupled_system(
        forecast_hours=args.hours,
        use_cache=not args.no_cache,
        serve=args.serve,
        port=args.port
    )


if __name__ == "__main__":
    main()
