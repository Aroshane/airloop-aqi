"""
Coupled Operational Web Dashboard Generator for Delhi NCR 72-Hour AQI Forecasting.
Renders a standalone, responsive Leaflet web application visualizing:
1. 72-hour Lagrangian farm-fire smoke plume advection over NW India.
2. Dynamic two-way feedback diagnostics (PBL collapse gauge, solar dimming meter, entrapment multiplier).
3. Decoupled vs Coupled 72-hour comparative forecast curves.
4. Station-specific AI forecast cards and feedback attribution across all 10 CPCB stations.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np

from coupled_feedback import CoupledSimulationResult
from coupled_predictor import Station72hForecast
from smoke_tracker.config import DELHI_NCR_DOMAIN, REGIONAL_DOMAIN, OUTPUT_DIR

ROOT_DIR = Path(__file__).resolve().parent

logger = logging.getLogger("airloop.coupled_dashboard")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )


def to_json_safe(obj: Any) -> Any:
    """Recursively convert NumPy objects to standard Python types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_json_safe(v) for v in obj]
    return obj


def build_coupled_dashboard(
    coupled_result: CoupledSimulationResult,
    station_forecasts: Dict[str, Station72hForecast],
    fires_data: List[Dict[str, Any]],
    wind_data: Dict[str, Any],
    output_html_path: Optional[Path] = None,
) -> Path:
    """
    Generate the standalone interactive Leaflet application with coupled feedback diagnostics.
    """
    target_path = output_html_path or (OUTPUT_DIR / "airloop_coupled_dashboard.html")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare serializable data bundle
    bundle_steps = [
        {
            "hour": int(s.hour),
            "datetime": str(s.datetime_str),
            "solar_dimming_pct": float(s.solar_dimming_pct),
            "unperturbed_pbl_m": float(s.unperturbed_pbl_m),
            "coupled_pbl_m": float(s.coupled_pbl_m),
            "pbl_collapse_pct": float(s.pbl_collapse_pct),
            "entrapment_factor": float(s.entrapment_factor),
            "wind_decel_pct": float(s.wind_deceleration_pct),
            "temp_depression_c": float(s.temp_depression_c),
            "decoupled_mean": float(s.decoupled_smoke_index),
            "coupled_mean": float(s.coupled_smoke_index),
        }
        for s in coupled_result.steps
    ]
    
    bundle_stations = {}
    for st_name, st_fc in station_forecasts.items():
        bundle_stations[st_name] = {
            "name": st_fc.name,
            "code": st_fc.code,
            "lat": float(st_fc.lat),
            "lon": float(st_fc.lon),
            "type": st_fc.station_type,
            "current_aqi": float(st_fc.current_measured_aqi),
            "decoupled_series": [float(v) for v in st_fc.decoupled_series],
            "coupled_series": [float(v) for v in st_fc.coupled_series],
            "gap_series": [float(v) for v in st_fc.underprediction_gap_series],
            "peak_coupled_aqi": float(st_fc.peak_coupled_aqi),
            "peak_decoupled_aqi": float(st_fc.peak_decoupled_aqi),
            "peak_hour": int(st_fc.peak_coupled_hour),
            "peak_time": str(st_fc.peak_coupled_time),
            "severity": str(st_fc.peak_severity_category),
            "attribution": {k: float(v) for k, v in st_fc.feedback_attribution.items()},
        }
        
    data_bundle = {
        "forecast_hours": int(coupled_result.forecast_hours),
        "times": [str(t) for t in coupled_result.times],
        "peak_hour": int(coupled_result.peak_smog_hour),
        "peak_time": str(coupled_result.peak_smog_time),
        "underprediction_max_gap": float(coupled_result.underprediction_max_gap),
        "steps": bundle_steps,
        "stations": bundle_stations,
        "fires": to_json_safe(fires_data),
        "wind_lats": to_json_safe(wind_data.get("lats", [])),
        "wind_lons": to_json_safe(wind_data.get("lons", [])),
        "wind_u": to_json_safe(wind_data.get("u_wind", wind_data.get("u", []))),
        "wind_v": to_json_safe(wind_data.get("v_wind", wind_data.get("v", []))),
        "wind_speed": to_json_safe(wind_data.get("speed", [])),
        "decoupled_delhi_mean": [float(v) for v in coupled_result.decoupled_delhi_mean],
        "coupled_delhi_mean": [float(v) for v in coupled_result.coupled_delhi_mean],
    }
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AirLoop Delhi NCR: Coupled Atmospheric Physics & Chemical Transport 72-Hour AQI System</title>
  
  <!-- Google Fonts: Inter & Outfit -->
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@500;600;700;800&display=swap" rel="stylesheet">
  
  <!-- Leaflet CSS -->
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=" crossorigin=""/>
  
  <style>
    :root {{
      --bg-dark: #090d16;
      --panel-bg: rgba(15, 23, 42, 0.82);
      --panel-border: rgba(56, 189, 248, 0.22);
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --accent-cyan: #38bdf8;
      --accent-orange: #f97316;
      --accent-red: #ef4444;
      --accent-purple: #a855f7;
      --accent-green: #22c55e;
    }}
    
    * {{
      margin: 0;
      padding: 0;
      box-sizing: border-box;
      font-family: 'Inter', -apple-system, sans-serif;
    }}
    
    body, html {{
      width: 100%;
      height: 100%;
      overflow: hidden;
      background-color: var(--bg-dark);
      color: var(--text-main);
    }}
    
    #map {{
      width: 100%;
      height: 100%;
      position: absolute;
      top: 0;
      left: 0;
      z-index: 1;
      background: #090d16;
    }}
    
    /* Header Box */
    .header-box {{
      position: absolute;
      top: 16px;
      left: 16px;
      z-index: 1000;
      background: var(--panel-bg);
      border: 1px solid var(--panel-border);
      backdrop-filter: blur(14px);
      border-radius: 12px;
      padding: 14px 20px;
      box-shadow: 0 10px 25px rgba(0,0,0,0.5);
      max-width: 440px;
    }}
    
    .header-box h1 {{
      font-family: 'Outfit', sans-serif;
      font-size: 1.15rem;
      font-weight: 700;
      color: #fff;
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    
    .status-badge {{
      display: inline-flex;
      align-items: center;
      gap: 5px;
      background: rgba(34, 197, 94, 0.15);
      border: 1px solid rgba(34, 197, 94, 0.35);
      color: #4ade80;
      font-size: 0.72rem;
      font-weight: 600;
      padding: 2px 8px;
      border-radius: 9999px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }}
    
    .header-box p {{
      font-size: 0.8rem;
      color: var(--text-muted);
      margin-top: 5px;
      line-height: 1.35;
    }}

    /* Coupled Diagnostic HUD (Top Right) */
    .coupled-hud {{
      position: absolute;
      top: 16px;
      right: 16px;
      z-index: 1000;
      background: var(--panel-bg);
      border: 1px solid var(--panel-border);
      backdrop-filter: blur(16px);
      border-radius: 14px;
      padding: 16px 20px;
      width: 380px;
      box-shadow: 0 12px 30px rgba(0,0,0,0.6);
    }}
    
    .hud-title {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      border-bottom: 1px solid rgba(255,255,255,0.08);
      padding-bottom: 8px;
      margin-bottom: 12px;
    }}
    
    .hud-title span {{
      font-family: 'Outfit', sans-serif;
      font-size: 0.85rem;
      font-weight: 700;
      letter-spacing: 0.6px;
      text-transform: uppercase;
      color: var(--accent-cyan);
    }}

    .diagnostic-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-bottom: 12px;
    }}

    .diag-card {{
      background: rgba(255,255,255,0.04);
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 8px;
      padding: 8px 10px;
    }}

    .diag-card .label {{
      font-size: 0.72rem;
      color: var(--text-muted);
      text-transform: uppercase;
      font-weight: 600;
    }}

    .diag-card .val {{
      font-size: 1.25rem;
      font-weight: 700;
      color: #fff;
      margin-top: 2px;
      display: flex;
      align-items: baseline;
      gap: 4px;
    }}

    .diag-card .val span {{
      font-size: 0.75rem;
      font-weight: 500;
      color: var(--text-muted);
    }}

    .pbl-bar-container {{
      background: rgba(255,255,255,0.08);
      height: 6px;
      border-radius: 3px;
      margin-top: 6px;
      overflow: hidden;
    }}

    .pbl-bar {{
      background: linear-gradient(90deg, #ef4444, #f97316);
      height: 100%;
      width: 40%;
      transition: width 0.2s ease;
    }}

    /* Underprediction Comparison Banner */
    .underprediction-banner {{
      background: rgba(239, 68, 68, 0.12);
      border: 1px solid rgba(239, 68, 68, 0.3);
      border-radius: 8px;
      padding: 8px 10px;
      margin-top: 10px;
    }}

    .underprediction-banner .title {{
      font-size: 0.75rem;
      font-weight: 700;
      color: #f87171;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }}

    .aqi-compare-row {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-top: 6px;
      font-size: 0.85rem;
    }}

    .aqi-pill {{
      padding: 2px 8px;
      border-radius: 6px;
      font-weight: 700;
      font-size: 0.85rem;
    }}

    .aqi-pill.dec {{
      background: rgba(245, 158, 11, 0.2);
      color: #f59e0b;
      border: 1px solid rgba(245, 158, 11, 0.4);
    }}

    .aqi-pill.coup {{
      background: rgba(239, 68, 68, 0.25);
      color: #f87171;
      border: 1px solid rgba(239, 68, 68, 0.5);
    }}

    /* 72h Synchronized Forecast Chart (Bottom) */
    .timeline-container {{
      position: absolute;
      bottom: 20px;
      left: 50%;
      transform: translateX(-50%);
      z-index: 1000;
      background: var(--panel-bg);
      border: 1px solid var(--panel-border);
      backdrop-filter: blur(16px);
      border-radius: 14px;
      padding: 14px 22px;
      width: 92%;
      max-width: 980px;
      box-shadow: 0 12px 30px rgba(0,0,0,0.65);
    }}

    .timeline-top {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 8px;
    }}

    .time-readout {{
      display: flex;
      align-items: baseline;
      gap: 10px;
    }}

    .time-hour {{
      font-family: 'Outfit', sans-serif;
      font-size: 1.15rem;
      font-weight: 700;
      color: var(--accent-cyan);
    }}

    .time-date {{
      font-size: 0.85rem;
      color: var(--text-muted);
    }}

    .ctrl-btns {{
      display: flex;
      align-items: center;
      gap: 6px;
    }}

    .ctrl-btn {{
      background: rgba(255,255,255,0.08);
      border: 1px solid rgba(255,255,255,0.15);
      color: #fff;
      border-radius: 6px;
      padding: 5px 12px;
      font-size: 0.82rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.15s ease;
    }}

    .ctrl-btn:hover {{
      background: rgba(255,255,255,0.18);
      transform: translateY(-1px);
    }}

    .ctrl-btn.primary {{
      background: linear-gradient(135deg, #0ea5e9, #0284c7);
      border-color: #38bdf8;
    }}

    .chart-box {{
      width: 100%;
      height: 65px;
      margin: 8px 0;
      position: relative;
    }}

    #forecast-canvas {{
      width: 100%;
      height: 100%;
      display: block;
    }}

    .slider-track {{
      width: 100%;
      position: relative;
    }}

    input[type=range] {{
      -webkit-appearance: none;
      width: 100%;
      height: 6px;
      border-radius: 3px;
      background: linear-gradient(90deg, #0ea5e9, #f97316 45%, #ef4444 85%);
      outline: none;
      cursor: pointer;
    }}

    input[type=range]::-webkit-slider-thumb {{
      -webkit-appearance: none;
      appearance: none;
      width: 18px;
      height: 18px;
      border-radius: 50%;
      background: #ffffff;
      box-shadow: 0 0 10px rgba(0,0,0,0.5), 0 0 8px #38bdf8;
      border: 2px solid #0ea5e9;
      cursor: grab;
    }}

    /* Station Modal / Drawer */
    .station-modal {{
      position: absolute;
      left: 16px;
      bottom: 170px;
      z-index: 1000;
      background: var(--panel-bg);
      border: 1px solid var(--panel-border);
      backdrop-filter: blur(16px);
      border-radius: 12px;
      padding: 14px 18px;
      width: 340px;
      box-shadow: 0 10px 25px rgba(0,0,0,0.5);
      display: none;
    }}

    .station-modal.active {{
      display: block;
    }}

    .station-modal h3 {{
      font-family: 'Outfit', sans-serif;
      font-size: 1.05rem;
      font-weight: 700;
      color: #fff;
    }}

    .station-modal .sub {{
      font-size: 0.75rem;
      color: var(--text-muted);
      margin-bottom: 8px;
    }}

    .station-modal .close-btn {{
      position: absolute;
      top: 12px;
      right: 14px;
      background: none;
      border: none;
      color: var(--text-muted);
      font-size: 1.2rem;
      cursor: pointer;
    }}

    /* Legend */
    .legend-box {{
      position: absolute;
      bottom: 20px;
      right: 20px;
      z-index: 1000;
      background: var(--panel-bg);
      border: 1px solid var(--panel-border);
      backdrop-filter: blur(12px);
      border-radius: 10px;
      padding: 10px 14px;
      font-size: 0.72rem;
    }}

    .legend-bar {{
      width: 160px;
      height: 8px;
      border-radius: 4px;
      background: linear-gradient(90deg, #eab308 0%, #f97316 35%, #ef4444 65%, #a855f7 90%, #7e22ce 100%);
      margin: 4px 0;
    }}

    /* Header Action Buttons */
    .header-actions {{
      display: flex;
      gap: 8px;
      margin-top: 10px;
      flex-wrap: wrap;
    }}

    .action-btn {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 12px;
      font-size: 0.76rem;
      font-weight: 600;
      border-radius: 7px;
      cursor: pointer;
      transition: all 0.2s ease;
      border: 1px solid transparent;
      outline: none;
    }}

    .action-btn.primary-glow {{
      background: linear-gradient(135deg, rgba(14, 165, 233, 0.35), rgba(2, 132, 199, 0.25));
      border-color: rgba(56, 189, 248, 0.5);
      color: #38bdf8;
      box-shadow: 0 0 12px rgba(56, 189, 248, 0.2);
    }}

    .action-btn.primary-glow:hover {{
      background: linear-gradient(135deg, rgba(14, 165, 233, 0.55), rgba(2, 132, 199, 0.45));
      color: #fff;
      transform: translateY(-1px);
    }}

    .action-btn.outline-btn {{
      background: rgba(255, 255, 255, 0.06);
      border-color: rgba(255, 255, 255, 0.16);
      color: #e2e8f0;
    }}

    .action-btn.outline-btn:hover {{
      background: rgba(255, 255, 255, 0.14);
      transform: translateY(-1px);
    }}

    .action-btn.active-layer {{
      background: rgba(245, 158, 11, 0.25) !important;
      border-color: #f59e0b !important;
      color: #fbbf24 !important;
      box-shadow: 0 0 10px rgba(245, 158, 11, 0.3);
    }}

    /* Vertical Inversion Profile Widget inside HUD */
    .inversion-widget-box {{
      margin-top: 12px;
      background: rgba(0, 0, 0, 0.38);
      border: 1px solid rgba(56, 189, 248, 0.18);
      border-radius: 10px;
      padding: 10px 12px;
    }}

    .inversion-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 0.72rem;
      font-weight: 600;
      color: #94a3b8;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      margin-bottom: 6px;
    }}

    #inversion-canvas {{
      width: 100%;
      height: 80px;
      display: block;
    }}

    .inversion-footer {{
      display: flex;
      justify-content: space-between;
      font-size: 0.68rem;
      color: var(--text-muted);
      margin-top: 4px;
    }}

    /* Story Modal (Why Delhi Chokes) */
    .story-overlay {{
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: rgba(4, 7, 15, 0.82);
      backdrop-filter: blur(10px);
      z-index: 2000;
      display: none;
      align-items: center;
      justify-content: center;
      padding: 20px;
    }}

    .story-overlay.active {{
      display: flex;
    }}

    .story-modal {{
      background: #0d1322;
      border: 1px solid rgba(56, 189, 248, 0.3);
      border-radius: 16px;
      width: 900px;
      max-width: 100%;
      max-height: 88vh;
      overflow-y: auto;
      box-shadow: 0 20px 50px rgba(0, 0, 0, 0.8), 0 0 20px rgba(56, 189, 248, 0.15);
      padding: 24px 28px;
      position: relative;
    }}

    .story-modal::-webkit-scrollbar {{
      width: 6px;
    }}
    .story-modal::-webkit-scrollbar-thumb {{
      background: rgba(56, 189, 248, 0.3);
      border-radius: 3px;
    }}

    .story-close {{
      position: absolute;
      top: 18px;
      right: 20px;
      background: rgba(255, 255, 255, 0.08);
      border: 1px solid rgba(255, 255, 255, 0.15);
      color: #94a3b8;
      font-size: 1.4rem;
      width: 32px;
      height: 32px;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      transition: all 0.2s;
    }}

    .story-close:hover {{
      background: rgba(239, 68, 68, 0.2);
      color: #f87171;
      border-color: #ef4444;
    }}

    .story-headline {{
      font-family: 'Outfit', sans-serif;
      font-size: 1.35rem;
      font-weight: 800;
      color: #fff;
      margin-bottom: 4px;
      display: flex;
      align-items: center;
      gap: 10px;
    }}

    .story-sub {{
      font-size: 0.82rem;
      color: #94a3b8;
      margin-bottom: 18px;
      line-height: 1.4;
    }}

    .story-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin-bottom: 16px;
    }}

    .pillar-card {{
      background: rgba(15, 23, 42, 0.65);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 12px;
      padding: 12px 15px;
      position: relative;
      transition: all 0.2s ease;
    }}

    .pillar-card:hover {{
      border-color: rgba(56, 189, 248, 0.35);
      transform: translateY(-2px);
    }}

    .pillar-card.p1 {{ border-left: 4px solid #38bdf8; }}
    .pillar-card.p2 {{ border-left: 4px solid #f59e0b; }}
    .pillar-card.p3 {{ border-left: 4px solid #ef4444; }}
    .pillar-card.p4 {{ border-left: 4px solid #a855f7; }}

    .pillar-title {{
      font-family: 'Outfit', sans-serif;
      font-size: 0.92rem;
      font-weight: 700;
      color: #f8fafc;
      margin-bottom: 5px;
      display: flex;
      align-items: center;
      gap: 6px;
    }}

    .pillar-desc {{
      font-size: 0.76rem;
      color: #cbd5e1;
      line-height: 1.42;
    }}

    .pillar-stat {{
      margin-top: 8px;
      padding-top: 6px;
      border-top: 1px dashed rgba(255, 255, 255, 0.1);
      font-size: 0.72rem;
      color: #38bdf8;
      font-weight: 600;
    }}

    /* Harvest Timeline in Story Modal */
    .timeline-banner {{
      background: rgba(15, 23, 42, 0.75);
      border: 1px solid rgba(245, 158, 11, 0.3);
      border-radius: 12px;
      padding: 12px 16px;
      margin-top: 10px;
    }}

    .timeline-banner h4 {{
      font-family: 'Outfit', sans-serif;
      font-size: 0.88rem;
      font-weight: 700;
      color: #fbbf24;
      margin-bottom: 10px;
      display: flex;
      align-items: center;
      gap: 8px;
    }}

    .timeline-steps {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 8px;
    }}

    .t-step {{
      background: rgba(0, 0, 0, 0.35);
      border-radius: 8px;
      padding: 8px 10px;
      border: 1px solid rgba(255, 255, 255, 0.05);
    }}

    .t-step .t-date {{
      font-size: 0.68rem;
      font-weight: 700;
      color: #f59e0b;
      text-transform: uppercase;
    }}

    .t-step .t-title {{
      font-size: 0.74rem;
      font-weight: 600;
      color: #fff;
      margin: 2px 0;
    }}

    .t-step .t-desc {{
      font-size: 0.66rem;
      color: #94a3b8;
      line-height: 1.3;
    }}
  </style>
</head>
<body>

  <div id="map"></div>

  <!-- Header -->
  <div class="header-box">
    <h1>
      AirLoop Delhi NCR
      <span class="status-badge">Coupled Feedback Model</span>
    </h1>
    <p>
      72-Hour Coupled Atmospheric Physics & Chemical Transport Forecast.
      Simulates bidirectional aerosol-radiative dimming, dynamic boundary layer (PBL) collapse, and induced stillness.
    </p>
    <div class="header-actions">
      <button class="action-btn primary-glow" onclick="openStoryModal()">
        🔬 Why Delhi Chokes (Science Story)
      </button>
      <button class="action-btn outline-btn" id="btn-toggle-basin" onclick="toggleBasinLayer()">
        🏔️ Himalayan Basin & Funnel Layer
      </button>
    </div>
  </div>

  <!-- Coupled Diagnostic HUD -->
  <div class="coupled-hud">
    <div class="hud-title">
      <span>Feedback Diagnostics</span>
      <span id="hud-hour-tag" style="color: #fff; font-size: 0.8rem;">Hour +0h</span>
    </div>
    
    <div class="diagnostic-grid">
      <!-- PBL Height -->
      <div class="diag-card">
        <div class="label">PBL Mixing Height</div>
        <div class="val" id="hud-pbl-val">1250 <span>m</span></div>
        <div class="pbl-bar-container">
          <div class="pbl-bar" id="hud-pbl-bar" style="width: 100%;"></div>
        </div>
        <div style="font-size: 0.68rem; color: #f87171; margin-top: 3px;" id="hud-pbl-collapse-sub">0% collapse</div>
      </div>
      
      <!-- Solar Dimming -->
      <div class="diag-card">
        <div class="label">Solar Attenuation</div>
        <div class="val" id="hud-dimming-val" style="color: #facc15;">-0%</div>
        <div style="font-size: 0.68rem; color: var(--text-muted); margin-top: 4px;">Aerosol Radiative Dimming</div>
      </div>

      <!-- Entrapment Factor -->
      <div class="diag-card">
        <div class="label">Smog Entrapment</div>
        <div class="val" id="hud-entrap-val" style="color: #fb923c;">1.00x</div>
        <div style="font-size: 0.68rem; color: var(--text-muted); margin-top: 4px;">Volumetric Compression</div>
      </div>

      <!-- Induced Stillness -->
      <div class="diag-card">
        <div class="label">Induced Stillness</div>
        <div class="val" id="hud-still-val" style="color: #38bdf8;">0%</div>
        <div style="font-size: 0.68rem; color: var(--text-muted); margin-top: 4px;">Thermal Momentum Decay</div>
      </div>
    </div>

    <!-- Vertical Inversion Profile Widget -->
    <div class="inversion-widget-box">
      <div class="inversion-header">
        <span>Vertical Inversion Profile (0 - 1500m)</span>
        <span id="inversion-lid-tag" style="color: #f87171; font-weight: 700;">Lid: 1250m</span>
      </div>
      <div style="position: relative; height: 75px;">
        <canvas id="inversion-canvas"></canvas>
      </div>
      <div class="inversion-footer">
        <span id="inversion-status-txt" style="color: #38bdf8;">Normal Daytime Dispersion</span>
        <span id="inversion-temp-gradient">Surface: 24°C | Aloft: 18°C</span>
      </div>
    </div>

    <!-- Decoupled vs Coupled Comparison Banner -->
    <div class="underprediction-banner">
      <div class="title">
        <span>DELHI NCR REGIONAL FORECAST</span>
        <span id="hud-gap-pill" style="font-size: 0.72rem; color: #fca5a5;">+0 AQI Gap</span>
      </div>
      <div class="aqi-compare-row">
        <div>
          <span style="font-size: 0.72rem; color: var(--text-muted); display: block;">Decoupled Model</span>
          <span class="aqi-pill dec" id="hud-dec-aqi">295 (Poor)</span>
        </div>
        <div style="font-size: 1.1rem; color: #94a3b8;">➔</div>
        <div>
          <span style="font-size: 0.72rem; color: #fca5a5; display: block;">Coupled Model</span>
          <span class="aqi-pill coup" id="hud-coup-aqi">295 (Poor)</span>
        </div>
      </div>
    </div>
  </div>

  <!-- Station Modal / Drawer -->
  <div class="station-modal" id="station-modal">
    <button class="close-btn" onclick="closeStationModal()">&times;</button>
    <h3 id="modal-st-name">Anand Vihar</h3>
    <div class="sub" id="modal-st-sub">CPCB CAAQMS • Traffic Hotspot</div>
    
    <div style="margin: 10px 0; background: rgba(0,0,0,0.3); border-radius: 8px; padding: 8px;">
      <div style="display: flex; justify-content: space-between; font-size: 0.8rem; margin-bottom: 4px;">
        <span style="color: var(--text-muted);">Current Measured AQI:</span>
        <span style="font-weight: 700; color: #fff;" id="modal-st-cur">335</span>
      </div>
      <div style="display: flex; justify-content: space-between; font-size: 0.8rem; margin-bottom: 4px;">
        <span style="color: #f59e0b;">Decoupled 72h Peak:</span>
        <span style="font-weight: 700; color: #f59e0b;" id="modal-st-dec">360</span>
      </div>
      <div style="display: flex; justify-content: space-between; font-size: 0.8rem;">
        <span style="color: #f87171;">Coupled Feedback Peak:</span>
        <span style="font-weight: 700; color: #f87171;" id="modal-st-coup">445 (Severe)</span>
      </div>
    </div>

    <div style="font-size: 0.72rem; color: var(--text-muted); line-height: 1.4;" id="modal-st-feedback">
      <b>Feedback Attribution:</b><br>
      • Advected Plume Inflow: 45%<br>
      • PBL Collapse Compression: 32%<br>
      • Induced Wind Stillness: 14%<br>
      • Nocturnal Inversion: 9%
    </div>
  </div>

  <!-- Why Delhi Chokes Science Story Modal -->
  <div class="story-overlay" id="story-modal">
    <div class="story-modal">
      <button class="story-close" onclick="closeStoryModal()">&times;</button>
      
      <div class="story-headline">
        <span>🔬 Why Does Northern India Choke Every Winter?</span>
      </div>
      <div class="story-sub">
        The 4-Pillar Physics & Geographic Anatomy of the Indo-Gangetic Basin Air Quality Crisis.
      </div>

      <div class="story-grid">
        <!-- Pillar 1 -->
        <div class="pillar-card p1">
          <div class="pillar-title">
            <span>🏔️ 1. The Himalayan Topographic Basin</span>
          </div>
          <div class="pillar-desc">
            Northern India sits in a giant low-elevation trough bounded by the Himalayas (~8,000m) to the north and the Aravalli/Vindhya ranges to the south. With winter synoptic winds blowing from the northwest (Pakistan & Punjab), stubble smoke is funneled into a dead-end geographic bowl with no natural lateral exit.
          </div>
          <div class="pillar-stat">
            Orographic Barrier: 6,000m - 8,848m Wall Traps Low-Level Air
          </div>
        </div>

        <!-- Pillar 2 -->
        <div class="pillar-card p2">
          <div class="pillar-title">
            <span>🌾 2. The 15-Day Agricultural Window</span>
          </div>
          <div class="pillar-desc">
            Following the Kharif paddy harvest in late October, farmers have a tight 15–20 day window to prepare fields for Rabi wheat sowing. Mechanized combine harvesters leave 6-inch root stubble. With no time for decomposition, burning ~20M tonnes of straw is the fastest clearing method, creating 3,000+ simultaneous satellite fire hotspots.
          </div>
          <div class="pillar-stat">
            Peak Window: Oct 20 – Nov 20 • NASA FIRMS Thermal Hotspots Surging
          </div>
        </div>

        <!-- Pillar 3 -->
        <div class="pillar-card p3">
          <div class="pillar-title">
            <span>🌡️ 3. The Winter Thermal Inversion Lid</span>
          </div>
          <div class="pillar-desc">
            Unlike summer when hot ground creates vertical thermal updrafts rising to 2,000m, winter nights bring rapid radiative cooling. A shallow layer of cold, dense air settles at the surface beneath warm air aloft. This forms an impenetrable temperature inversion ceiling, crushing the Planetary Boundary Layer (PBL) from 1,200m to 350m.
          </div>
          <div class="pillar-stat">
            Boundary Layer Collapse: 1,250m ➔ 350m Mixing Height
          </div>
        </div>

        <!-- Pillar 4 -->
        <div class="pillar-card p4">
          <div class="pillar-title">
            <span>🔄 4. The AirLoop Positive Feedback Trap</span>
          </div>
          <div class="pillar-desc">
            Traditional decoupled models assume weather is unaffected by smoke. In reality, dense smoke blocks 25%–45% of incoming sunlight (Beer-Lambert Law), cooling the ground further. Surface winds stall (+60% induced stillness), killing ventilation and compressing pollutants into hazardous ground concentrations (+85 AQI points underpredicted by standard models!).
          </div>
          <div class="pillar-stat">
            Bidirectional Physics: C_ground = C_advected × (H_0 / H_pbl)^α
          </div>
        </div>
      </div>

      <!-- Agricultural Calendar & Wind Reversal Timeline -->
      <div class="timeline-banner">
        <h4>🗓️ Seasonal Choke Progression: From Monsoon Retreat to Toxic Smog</h4>
        <div class="timeline-steps">
          <div class="t-step">
            <div class="t-date">Mid October</div>
            <div class="t-title">Monsoon Withdrawal</div>
            <div class="t-desc">Winds reverse from moist easterlies to dry northwesterlies. Humidity drops and night cooling begins.</div>
          </div>
          <div class="t-step">
            <div class="t-date">Oct 20 – Nov 05</div>
            <div class="t-title">Harvest Burning Surge</div>
            <div class="t-desc">Punjab & Haryana paddy stubble fires peak. Smoke plumes advect southeast down the Himalayan trough.</div>
          </div>
          <div class="t-step">
            <div class="t-date">Nov 01 – Nov 15</div>
            <div class="t-title">AirLoop Feedback Peak</div>
            <div class="t-desc">Solar dimming triggers inversion collapse to 350m. Severe AQI (450+) locks in across all Delhi CPCB stations.</div>
          </div>
          <div class="t-step">
            <div class="t-date">Late Nov – Dec</div>
            <div class="t-title">Persistent Stagnation</div>
            <div class="t-desc">Dense radiation fog combines with trapped particulate matter, creating hazardous persistent winter smog.</div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- Legend Box -->
  <div class="legend-box">
    <div style="font-weight: 600; margin-bottom: 2px;">Smoke Plume Density (AOD)</div>
    <div class="legend-bar"></div>
    <div style="display: flex; justify-content: space-between; color: var(--text-muted);">
      <span>Light</span>
      <span>Moderate</span>
      <span>Severe Smog</span>
    </div>
  </div>

  <!-- Timeline Player Bar -->
  <div class="timeline-container">
    <div class="timeline-top">
      <div class="time-readout">
        <span class="time-hour" id="display-hour">Hour +0h / 72h</span>
        <span class="time-date" id="display-date">2026-11-01 12:00 UTC</span>
      </div>
      
      <div class="ctrl-btns">
        <button class="ctrl-btn" id="btn-prev">⏮</button>
        <button class="ctrl-btn primary" id="btn-play">▶ Play</button>
        <button class="ctrl-btn" id="btn-next">⏭</button>
        <button class="ctrl-btn" id="btn-speed">1x Speed</button>
        <button class="ctrl-btn" id="btn-reset">↺ Reset</button>
      </div>
    </div>

    <!-- Canvas Chart for Decoupled vs Coupled AQI -->
    <div class="chart-box">
      <canvas id="forecast-canvas"></canvas>
    </div>

    <!-- Range Slider -->
    <div class="slider-track">
      <input type="range" id="time-slider" min="0" max="{coupled_result.forecast_hours - 1}" value="0" step="1">
    </div>
  </div>

  <!-- Leaflet JS -->
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin=""></script>

  <script>
    const SIM = {json.dumps(data_bundle)};

    let currentHour = 0;
    let isPlaying = false;
    let playInterval = null;
    let playSpeed = 300;

    // DOM Elements
    const timeSlider = document.getElementById('time-slider');
    const displayHour = document.getElementById('display-hour');
    const displayDate = document.getElementById('display-date');
    const hudHourTag = document.getElementById('hud-hour-tag');
    const hudPblVal = document.getElementById('hud-pbl-val');
    const hudPblBar = document.getElementById('hud-pbl-bar');
    const hudPblCollapseSub = document.getElementById('hud-pbl-collapse-sub');
    const hudDimmingVal = document.getElementById('hud-dimming-val');
    const hudEntrapVal = document.getElementById('hud-entrap-val');
    const hudStillVal = document.getElementById('hud-still-val');
    const hudDecAqi = document.getElementById('hud-dec-aqi');
    const hudCoupAqi = document.getElementById('hud-coup-aqi');
    const hudGapPill = document.getElementById('hud-gap-pill');
    const btnPlay = document.getElementById('btn-play');
    const forecastCanvas = document.getElementById('forecast-canvas');

    // Initialize Leaflet Map
    const map = L.map('map', {{
      center: [29.6, 75.8],
      zoom: 7,
      zoomControl: false,
      minZoom: 6,
      maxZoom: 12
    }});

    L.control.zoom({{ position: 'bottomright' }}).addTo(map);

    // High-performance Dark Basemap (100% Free & Open, Zero API Key, No Watermark)
    L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
      attribution: 'Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ',
      maxZoom: 16
    }}).addTo(map);

    L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Reference/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
      attribution: '',
      maxZoom: 16
    }}).addTo(map);

    // Layer Groups
    const firesLayer = L.layerGroup().addTo(map);
    const windLayer = L.layerGroup().addTo(map);
    const stationLayer = L.layerGroup().addTo(map);
    const delhiBorder = L.layerGroup().addTo(map);

    // Delhi NCR Border
    L.polygon([
      [{DELHI_NCR_DOMAIN.min_lat}, {DELHI_NCR_DOMAIN.min_lon}],
      [{DELHI_NCR_DOMAIN.max_lat}, {DELHI_NCR_DOMAIN.min_lon}],
      [{DELHI_NCR_DOMAIN.max_lat}, {DELHI_NCR_DOMAIN.max_lon}],
      [{DELHI_NCR_DOMAIN.min_lat}, {DELHI_NCR_DOMAIN.max_lon}]
    ], {{
      color: '#38bdf8',
      weight: 2,
      fillColor: '#0284c7',
      fillOpacity: 0.04,
      dashArray: '5, 5'
    }}).addTo(delhiBorder).bindTooltip("<b>Delhi NCR Coupled Target Domain</b>", {{ sticky: true }});

    // Render Active Farm-Fire Hotspots
    function renderFires() {{
      firesLayer.clearLayers();
      SIM.fires.forEach(f => {{
        const radius = Math.min(14, Math.max(5, Math.sqrt(f.frp || 25) * 2.0));
        const marker = L.circleMarker([f.lat, f.lon], {{
          radius: radius,
          fillColor: '#ef4444',
          color: '#fca5a5',
          weight: 1.5,
          fillOpacity: 0.85
        }}).addTo(firesLayer);

        marker.bindPopup(`
          <div style="font-size: 0.82rem; color: #1e293b; padding: 2px;">
            <h4 style="color: #ef4444; font-size: 0.95rem; margin-bottom: 4px;">Active Stubble Fire</h4>
            <b>FRP:</b> ${{f.frp}} MW<br>
            <b>State:</b> ${{f.state || 'Punjab/Haryana'}}<br>
            <b>Coordinates:</b> ${{f.lat.toFixed(3)}}, ${{f.lon.toFixed(3)}}
          </div>
        `);
      }});
    }}
    renderFires();

    // Render CPCB Stations
    function renderStations() {{
      stationLayer.clearLayers();
      Object.values(SIM.stations).forEach(st => {{
        const marker = L.circleMarker([st.lat, st.lon], {{
          radius: 6,
          fillColor: '#38bdf8',
          color: '#ffffff',
          weight: 2,
          fillOpacity: 0.9
        }}).addTo(stationLayer);

        marker.on('click', () => openStationModal(st.name));
        marker.bindTooltip(`<b>${{st.name}}</b><br>Peak Coupled AQI: ${{st.peak_coupled_aqi}}`, {{ sticky: true }});
      }});
    }}
    renderStations();

    // Render Wind Flow Field with dynamic stagnation dampening
    function renderWind(hourIdx) {{
      windLayer.clearLayers();
      const step = SIM.steps[hourIdx];
      const decel = (step ? step.wind_decel_pct : 0) / 100.0;

      const wLats = SIM.wind_lats;
      const wLons = SIM.wind_lons;
      const uArr = SIM.wind_u[hourIdx];
      const vArr = SIM.wind_v[hourIdx];
      const spdArr = SIM.wind_speed[hourIdx];

      if (!uArr) return;

      for (let i = 0; i < wLats.length; i += 1) {{
        for (let j = 0; j < wLons.length; j += 1) {{
          const lat = wLats[i];
          const lon = wLons[j];
          let u = uArr[i][j];
          let v = vArr[i][j];
          let spd = spdArr[i][j];

          // Apply coupled stagnation deceleration
          spd = spd * (1.0 - decel);
          u = u * (1.0 - decel);
          v = v * (1.0 - decel);

          if (spd < 0.4) continue;

          const scale = 0.032;
          const endLat = lat + (v / Math.max(1.0, spd)) * scale * (spd * 0.45);
          const endLon = lon + (u / Math.max(1.0, spd)) * scale * (spd * 0.45);

          L.polyline([[lat, lon], [endLat, endLon]], {{
            color: 'rgba(56, 189, 248, 0.45)',
            weight: 1.5
          }}).addTo(windLayer);

          L.circleMarker([endLat, endLon], {{
            radius: 2,
            color: 'rgba(56, 189, 248, 0.7)',
            fillColor: '#38bdf8',
            fillOpacity: 0.8,
            weight: 1
          }}).addTo(windLayer);
        }}
      }}
    }}

    // Custom Smoke Plume Canvas Overlay
    const SmokeCanvasLayer = L.Layer.extend({{
      onAdd: function(map) {{
        this._map = map;
        this._canvas = L.DomUtil.create('canvas', 'smoke-overlay-canvas');
        const size = map.getSize();
        this._canvas.width = size.x;
        this._canvas.height = size.y;
        this._canvas.style.position = 'absolute';
        this._canvas.style.top = '0';
        this._canvas.style.left = '0';
        this._canvas.style.pointerEvents = 'none';
        map.getPanes().overlayPane.appendChild(this._canvas);
        map.on('moveend resize', this._update, this);
        this._update();
      }},
      onRemove: function(map) {{
        L.DomUtil.remove(this._canvas);
        map.off('moveend resize', this._update, this);
      }},
      _update: function() {{
        if (!this._map) return;
        const size = this._map.getSize();
        this._canvas.width = size.x;
        this._canvas.height = size.y;
        const topLeft = this._map.containerPointToLayerPoint([0, 0]);
        L.DomUtil.setPosition(this._canvas, topLeft);
        this.renderHour(currentHour);
      }},
      renderHour: function(hourIdx) {{
        if (!this._canvas || !this._map) return;
        const ctx = this._canvas.getContext('2d');
        ctx.clearRect(0, 0, this._canvas.width, this._canvas.height);

        const step = SIM.steps[hourIdx];
        if (!step) return;

        const smokeVal = step.coupled_mean;
        if (smokeVal < 1.0) return;

        // Render synthetic smoke plume field across Punjab -> Delhi NCR
        const centerPt = this._map.latLngToContainerPoint([28.65, 77.20]);
        const radius = Math.min(220, Math.max(30, Math.sqrt(smokeVal) * 3.2));

        const grad = ctx.createRadialGradient(centerPt.x, centerPt.y, 0, centerPt.x, centerPt.y, radius);
        if (smokeVal < 30) {{
          grad.addColorStop(0, 'rgba(234, 179, 8, 0.45)');
          grad.addColorStop(0.7, 'rgba(234, 179, 8, 0.15)');
        }} else if (smokeVal < 100) {{
          grad.addColorStop(0, 'rgba(249, 115, 22, 0.55)');
          grad.addColorStop(0.7, 'rgba(249, 115, 22, 0.2)');
        }} else {{
          grad.addColorStop(0, 'rgba(239, 68, 68, 0.65)');
          grad.addColorStop(0.5, 'rgba(168, 85, 247, 0.45)');
          grad.addColorStop(0.8, 'rgba(126, 34, 206, 0.2)');
        }}
        grad.addColorStop(1, 'rgba(0, 0, 0, 0)');

        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(centerPt.x, centerPt.y, radius, 0, Math.PI * 2);
        ctx.fill();
      }}
    }});

    const smokeLayer = new SmokeCanvasLayer();
    map.addLayer(smokeLayer);

    // Draw 72-Hour Synchronized Canvas Chart (Decoupled vs Coupled)
    function drawForecastChart() {{
      if (!forecastCanvas) return;
      const ctx = forecastCanvas.getContext('2d');
      const dpr = window.devicePixelRatio || 1;
      const rect = forecastCanvas.getBoundingClientRect();
      if (rect.width === 0 || rect.height === 0) return;

      forecastCanvas.width = rect.width * dpr;
      forecastCanvas.height = rect.height * dpr;
      ctx.scale(dpr, dpr);

      const w = rect.width;
      const h = rect.height;
      const decData = SIM.decoupled_delhi_mean;
      const coupData = SIM.coupled_delhi_mean;
      const maxVal = Math.max(500, Math.max(...coupData, ...decData) * 1.05);

      ctx.clearRect(0, 0, w, h);

      // Severe AQI threshold line (400)
      const y400 = h - (400 / maxVal) * h;
      ctx.beginPath();
      ctx.moveTo(0, y400);
      ctx.lineTo(w, y400);
      ctx.strokeStyle = 'rgba(239, 68, 68, 0.35)';
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.stroke();
      ctx.setLineDash([]);

      // 1. Draw Decoupled Model curve (Yellow dashed)
      ctx.beginPath();
      for (let i = 0; i < decData.length; i++) {{
        const x = (i / (decData.length - 1)) * w;
        const y = h - (decData[i] / maxVal) * (h - 8) - 4;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }}
      ctx.strokeStyle = '#f59e0b';
      ctx.lineWidth = 1.8;
      ctx.setLineDash([4, 3]);
      ctx.stroke();
      ctx.setLineDash([]);

      // 2. Draw Coupled Feedback curve (Crimson/Sky gradient solid)
      ctx.beginPath();
      for (let i = 0; i < coupData.length; i++) {{
        const x = (i / (coupData.length - 1)) * w;
        const y = h - (coupData[i] / maxVal) * (h - 8) - 4;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }}
      ctx.strokeStyle = '#ef4444';
      ctx.lineWidth = 2.4;
      ctx.stroke();

      // 3. Current Hour Cursor Line
      const curX = (currentHour / (coupData.length - 1)) * w;
      ctx.beginPath();
      ctx.moveTo(curX, 0);
      ctx.lineTo(curX, h);
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 2;
      ctx.stroke();

      // Dot on current coupled value
      const curY = h - (coupData[currentHour] / maxVal) * (h - 8) - 4;
      ctx.beginPath();
      ctx.arc(curX, curY, 4, 0, Math.PI * 2);
      ctx.fillStyle = '#ffffff';
      ctx.fill();
      ctx.strokeStyle = '#ef4444';
      ctx.lineWidth = 2;
      ctx.stroke();
    }}

    // Update UI for Specified Hour
    function updateHour(h) {{
      currentHour = Math.max(0, Math.min(SIM.forecast_hours - 1, h));
      if (timeSlider) timeSlider.value = currentHour;

      const step = SIM.steps[currentHour];
      if (!step) return;

      if (displayHour) displayHour.innerText = `Hour +${{currentHour}}h / 72h`;
      if (displayDate) displayDate.innerText = (step.datetime || '').replace('T', ' ') + ' UTC';
      if (hudHourTag) hudHourTag.innerText = `Hour +${{currentHour}}h`;

      // Diagnostic Readouts
      if (hudPblVal) hudPblVal.innerHTML = `${{step.coupled_pbl_m}} <span>m</span>`;
      if (hudPblBar) {{
        const pblPct = Math.min(100, Math.max(20, (step.coupled_pbl_m / 1250) * 100));
        hudPblBar.style.width = pblPct + '%';
      }}
      if (hudPblCollapseSub) hudPblCollapseSub.innerText = `-${{step.pbl_collapse_pct}}% thermal collapse`;
      if (hudDimmingVal) hudDimmingVal.innerText = `-${{step.solar_dimming_pct}}%`;
      if (hudEntrapVal) hudEntrapVal.innerText = `${{step.entrapment_factor}}x`;
      if (hudStillVal) hudStillVal.innerText = `+${{step.wind_decel_pct}}%`;

      // Regional AQI Comparison
      const decAqi = Math.round(280 + step.decoupled_mean * 0.08);
      const coupAqi = Math.round(280 + step.coupled_mean * 0.08);
      const gap = coupAqi - decAqi;

      if (hudDecAqi) hudDecAqi.innerText = `${{decAqi}} (Decoupled)`;
      if (hudCoupAqi) hudCoupAqi.innerText = `${{coupAqi}} (Coupled)`;
      if (hudGapPill) hudGapPill.innerText = `+${{gap}} AQI Gap Missed by Decoupled Model`;

      // Update Map layers
      renderWind(currentHour);
      smokeLayer.renderHour(currentHour);
      drawForecastChart();
      drawInversionProfile(currentHour);
    }}

    // Story Modal Controls
    function openStoryModal() {{
      const modal = document.getElementById('story-modal');
      if (modal) modal.classList.add('active');
    }}

    function closeStoryModal() {{
      const modal = document.getElementById('story-modal');
      if (modal) modal.classList.remove('active');
    }}

    // Himalayan Basin & Funnel Layer
    let isBasinActive = false;
    const basinLayer = L.layerGroup();

    // Himalayan Mountain Barrier Polygon
    const himalayanPolygon = L.polygon([
      [34.2, 73.8], [33.5, 75.8], [32.6, 77.2], [31.5, 78.8],
      [30.4, 80.2], [29.6, 81.8], [28.6, 83.5], [29.8, 84.0],
      [31.2, 82.2], [32.5, 79.5], [33.8, 76.5], [34.5, 74.5]
    ], {{
      color: '#eab308',
      weight: 2,
      fillColor: '#ca8a04',
      fillOpacity: 0.18,
      dashArray: '6, 4'
    }}).bindTooltip("<b>🏔️ Himalayan Mountain Barrier (~6,000m - 8,848m)</b><br>Blocks northward dispersion of winter air masses", {{ sticky: true }});

    // Indo-Gangetic Wind Funnel Corridor Polygon
    const gangeticFunnel = L.polygon([
      [32.2, 74.2], [31.6, 76.2], [29.8, 77.6], [28.4, 77.8],
      [27.6, 76.4], [28.6, 75.0], [30.4, 73.8], [31.8, 73.2]
    ], {{
      color: '#f97316',
      weight: 2,
      fillColor: '#ea580c',
      fillOpacity: 0.12,
      dashArray: '5, 4'
    }}).bindTooltip("<b>🌪️ Indo-Gangetic Trough / Wind Funnel</b><br>Channels NW stubble smoke directly into Delhi dead-end pocket", {{ sticky: true }});

    // Mountain Barrier High Peaks
    const peaks = [
      {{ name: "Pir Panjal Range (~5,000m)", coords: [32.5, 75.8] }},
      {{ name: "Dhauladhar Range (~5,600m)", coords: [32.2, 76.4] }},
      {{ name: "Great Himalayan Wall (~7,800m)", coords: [30.8, 79.4] }}
    ];
    peaks.forEach(p => {{
      L.circleMarker(p.coords, {{
        radius: 6,
        fillColor: '#fef08a',
        color: '#ca8a04',
        weight: 2,
        fillOpacity: 0.9
      }}).bindTooltip(`🏔️ <b>${{p.name}}</b>`, {{ permanent: false, direction: 'top' }}).addTo(basinLayer);
    }});

    // Directional Wind Funnel Corridor Flows
    const funnelFlows = [
      [[31.8, 74.8], [30.6, 75.8], [29.4, 76.6], [28.65, 77.2]],
      [[31.2, 75.4], [30.2, 76.2], [29.2, 76.9], [28.5, 77.3]],
      [[31.5, 74.2], [30.0, 75.2], [28.9, 76.2], [28.6, 77.1]]
    ];
    funnelFlows.forEach(flow => {{
      L.polyline(flow, {{
        color: '#fbbf24',
        weight: 2.2,
        dashArray: '8, 6',
        opacity: 0.75
      }}).addTo(basinLayer);
    }});

    himalayanPolygon.addTo(basinLayer);
    gangeticFunnel.addTo(basinLayer);

    function toggleBasinLayer() {{
      isBasinActive = !isBasinActive;
      const btn = document.getElementById('btn-toggle-basin');
      if (isBasinActive) {{
        map.addLayer(basinLayer);
        if (btn) btn.classList.add('active-layer');
      }} else {{
        map.removeLayer(basinLayer);
        if (btn) btn.classList.remove('active-layer');
      }}
    }}

    // Inversion Canvas Elements
    const inversionCanvas = document.getElementById('inversion-canvas');
    const inversionLidTag = document.getElementById('inversion-lid-tag');
    const inversionStatusTxt = document.getElementById('inversion-status-txt');
    const inversionTempGradient = document.getElementById('inversion-temp-gradient');

    // Draw Dynamic Vertical Atmospheric Inversion Profile
    function drawInversionProfile(hour) {{
      if (!inversionCanvas) return;
      const ctx = inversionCanvas.getContext('2d');
      const dpr = window.devicePixelRatio || 1;
      const rect = inversionCanvas.getBoundingClientRect();
      if (rect.width === 0 || rect.height === 0) return;

      inversionCanvas.width = rect.width * dpr;
      inversionCanvas.height = rect.height * dpr;
      ctx.scale(dpr, dpr);

      const w = rect.width;
      const h = rect.height;
      const step = SIM.steps[hour] || SIM.steps[0];
      const pbl_m = step.coupled_pbl_m || 1250;
      const collapsePct = step.pbl_collapse_pct || 0;
      const maxAlt = 1500;

      const groundY = h - 14;
      const topY = 10;
      const lidY = groundY - (pbl_m / maxAlt) * (groundY - topY);

      ctx.clearRect(0, 0, w, h);

      // 1. Upper Free Atmosphere (Ventilated)
      const upperGrad = ctx.createLinearGradient(0, topY, 0, lidY);
      upperGrad.addColorStop(0, 'rgba(14, 165, 233, 0.15)');
      upperGrad.addColorStop(1, 'rgba(14, 165, 233, 0.03)');
      ctx.fillStyle = upperGrad;
      ctx.fillRect(0, topY, w, Math.max(0, lidY - topY));

      // 2. Trapped Smog Boundary Layer (Below Inversion Lid)
      const smogGrad = ctx.createLinearGradient(0, lidY, 0, groundY);
      if (pbl_m < 500) {{
        smogGrad.addColorStop(0, 'rgba(239, 68, 68, 0.65)');
        smogGrad.addColorStop(0.6, 'rgba(220, 38, 38, 0.5)');
        smogGrad.addColorStop(1, 'rgba(153, 27, 27, 0.8)');
      }} else if (pbl_m < 850) {{
        smogGrad.addColorStop(0, 'rgba(249, 115, 22, 0.5)');
        smogGrad.addColorStop(1, 'rgba(234, 88, 12, 0.35)');
      }} else {{
        smogGrad.addColorStop(0, 'rgba(56, 189, 248, 0.25)');
        smogGrad.addColorStop(1, 'rgba(2, 132, 199, 0.15)');
      }}
      ctx.fillStyle = smogGrad;
      ctx.fillRect(0, lidY, w, groundY - lidY);

      // 3. Ground Level
      ctx.beginPath();
      ctx.moveTo(0, groundY);
      ctx.lineTo(w, groundY);
      ctx.strokeStyle = '#64748b';
      ctx.lineWidth = 1.5;
      ctx.stroke();

      ctx.fillStyle = '#94a3b8';
      ctx.font = '9px Inter, sans-serif';
      ctx.fillText('Ground (0m)', 6, h - 3);
      ctx.fillText('1500m Free Air', w - 74, topY + 6);

      // 4. Inversion Ceiling Line (Dashed)
      ctx.beginPath();
      ctx.moveTo(0, lidY);
      ctx.lineTo(w, lidY);
      ctx.strokeStyle = pbl_m < 500 ? '#ef4444' : (pbl_m < 850 ? '#f97316' : '#38bdf8');
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 4]);
      ctx.stroke();
      ctx.setLineDash([]);

      // Inversion Lid Badge Label
      const badgeTxt = `▲ INVERSION CEILING: ${{Math.round(pbl_m)}}m (-${{Math.round(collapsePct)}}%)`;
      ctx.font = 'bold 9px Inter, sans-serif';
      const txtW = ctx.measureText(badgeTxt).width;
      ctx.fillStyle = pbl_m < 500 ? 'rgba(239, 68, 68, 0.9)' : (pbl_m < 850 ? 'rgba(249, 115, 22, 0.85)' : 'rgba(14, 165, 233, 0.85)');
      ctx.fillRect(6, Math.max(0, lidY - 13), txtW + 8, 12);
      ctx.fillStyle = '#ffffff';
      ctx.fillText(badgeTxt, 10, Math.max(9, lidY - 4));

      // 5. Animated Trapped Particulate Particles
      const particleCount = Math.min(45, Math.floor(12 + (collapsePct / 100) * 30));
      ctx.fillStyle = pbl_m < 500 ? '#fca5a5' : '#fed7aa';
      for (let i = 0; i < particleCount; i++) {{
        const px = (Math.sin(i * 99 + hour) * 0.5 + 0.5) * (w - 16) + 8;
        const py = lidY + 4 + (Math.cos(i * 33 + hour * 2) * 0.5 + 0.5) * Math.max(2, groundY - lidY - 8);
        ctx.beginPath();
        ctx.arc(px, py, 1.2, 0, Math.PI * 2);
        ctx.fill();
      }}

      // Update widget text elements
      if (inversionLidTag) {{
        inversionLidTag.innerText = `Lid: ${{Math.round(pbl_m)}}m`;
        inversionLidTag.style.color = pbl_m < 500 ? '#ef4444' : (pbl_m < 850 ? '#f97316' : '#38bdf8');
      }}
      if (inversionStatusTxt) {{
        if (pbl_m < 500) {{
          inversionStatusTxt.innerText = `⚠️ Severe Inversion Trapping (${{step.entrapment_factor.toFixed(1)}}x)`;
          inversionStatusTxt.style.color = '#ef4444';
        }} else if (pbl_m < 850) {{
          inversionStatusTxt.innerText = `⚡ Moderate Inversion (${{step.entrapment_factor.toFixed(1)}}x)`;
          inversionStatusTxt.style.color = '#f97316';
        }} else {{
          inversionStatusTxt.innerText = `Normal Daytime Dispersion`;
          inversionStatusTxt.style.color = '#38bdf8';
        }}
      }}
      if (inversionTempGradient) {{
        const sTemp = (22.0 - (step.temp_depression_c || 0.0)).toFixed(1);
        inversionTempGradient.innerText = `Surface: ${{sTemp}}°C | ΔT: -${{(step.temp_depression_c || 0).toFixed(1)}}°C`;
      }}
    }}

    // Station Modal
    function openStationModal(stName) {{
      const modal = document.getElementById('station-modal');
      const st = SIM.stations[stName];
      if (!modal || !st) return;

      document.getElementById('modal-st-name').innerText = st.name;
      document.getElementById('modal-st-sub').innerText = `CPCB CAAQMS • ${{st.type}}`;
      document.getElementById('modal-st-cur').innerText = st.current_aqi;
      document.getElementById('modal-st-dec').innerText = `${{st.peak_decoupled_aqi}} (H+${{st.peak_hour}})`;
      document.getElementById('modal-st-coup').innerText = `${{st.peak_coupled_aqi}} (${{st.severity}})`;

      const att = st.attribution || {{}};
      document.getElementById('modal-st-feedback').innerHTML = `
        <b>Coupled Feedback Breakdown:</b><br>
        • Advected Plume Inflow: ${{att.advected_plume_pct || 45}}%<br>
        • Boundary Layer (PBL) Collapse: ${{att.pbl_collapse_trapping_pct || 30}}%<br>
        • Induced Surface Stillness: ${{att.induced_stillness_pct || 15}}%<br>
        • Nocturnal Thermal Inversion: ${{att.thermal_inversion_pct || 10}}%
      `;
      modal.classList.add('active');
    }}

    function closeStationModal() {{
      const modal = document.getElementById('station-modal');
      if (modal) modal.classList.remove('active');
    }}

    // Animation Controls
    function togglePlay() {{
      isPlaying = !isPlaying;
      if (isPlaying) {{
        btnPlay.innerText = '⏸ Pause';
        playInterval = setInterval(() => {{
          if (currentHour >= SIM.forecast_hours - 1) {{
            updateHour(0);
          }} else {{
            updateHour(currentHour + 1);
          }}
        }}, playSpeed);
      }} else {{
        btnPlay.innerText = '▶ Play';
        clearInterval(playInterval);
      }}
    }}

    if (btnPlay) btnPlay.addEventListener('click', togglePlay);
    if (timeSlider) timeSlider.addEventListener('input', (e) => updateHour(parseInt(e.target.value)));

    const btnPrev = document.getElementById('btn-prev');
    if (btnPrev) btnPrev.addEventListener('click', () => updateHour(currentHour - 1));

    const btnNext = document.getElementById('btn-next');
    if (btnNext) btnNext.addEventListener('click', () => updateHour(currentHour + 1));

    const btnReset = document.getElementById('btn-reset');
    if (btnReset) btnReset.addEventListener('click', () => updateHour(0));

    const btnSpeed = document.getElementById('btn-speed');
    if (btnSpeed) btnSpeed.addEventListener('click', () => {{
      if (playSpeed === 300) {{
        playSpeed = 150;
        btnSpeed.innerText = '2x Speed';
      }} else if (playSpeed === 150) {{
        playSpeed = 60;
        btnSpeed.innerText = '5x Speed';
      }} else {{
        playSpeed = 300;
        btnSpeed.innerText = '1x Speed';
      }}
      if (isPlaying) {{
        clearInterval(playInterval);
        playInterval = setInterval(() => {{
          if (currentHour >= SIM.forecast_hours - 1) updateHour(0);
          else updateHour(currentHour + 1);
        }}, playSpeed);
      }}
    }});

    window.addEventListener('resize', () => {{
      drawForecastChart();
      drawInversionProfile(currentHour);
    }});

    // Initial render
    setTimeout(() => {{
      updateHour(0);
      drawInversionProfile(0);
    }}, 200);

  </script>
</body>
</html>
"""
    # Write to target path (e.g. data/outputs)
    with open(target_path, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    # Also synchronize root index.html and public/index.html
    root_index = ROOT_DIR / "index.html"
    public_index = ROOT_DIR / "public" / "index.html"
    try:
        with open(root_index, "w", encoding="utf-8") as f:
            f.write(html_content)
        public_index.parent.mkdir(parents=True, exist_ok=True)
        with open(public_index, "w", encoding="utf-8") as f:
            f.write(html_content)
    except Exception as e:
        logger.warning(f"Could not mirror to static index.html: {e}")
        
    logger.info(f"Generated standalone coupled forecast web application: {target_path}")
    return target_path
