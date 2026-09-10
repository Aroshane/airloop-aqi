"""
Interactive Web Map Visualization Module for Farm-Fire Smoke Advection.
Generates a standalone, responsive Leaflet web application with a 72-hour time slider,
animated smoke density overlays, wind flow vectors, fire source markers, and a glassmorphic HUD dashboard.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd

from smoke_tracker.config import (
    DELHI_STATIONS,
    DELHI_NCR_DOMAIN,
    REGIONAL_DOMAIN,
    OUTPUT_DIR
)
from smoke_tracker.advection import SimulationResult

logger = logging.getLogger("smoke_tracker.visualize")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )


def create_leaflet_map(
    result: SimulationResult,
    fires_df: pd.DataFrame,
    wind_data: Dict[str, Any],
    output_html_path: Optional[Path] = None,
    title: str = "Delhi NCR Farm-Fire Smoke Dispersion (72-Hour Forecast)"
) -> Path:
    """
    Generate a standalone, high-performance, interactive Leaflet web application.
    
    Parameters:
        result: SimulationResult containing computed smoke grids.
        fires_df: DataFrame of active fire hotspots from NASA FIRMS.
        wind_data: Open-Meteo wind field dictionary.
        output_html_path: Destination path for HTML file.
        title: Page and header title.
        
    Returns:
        Path to the generated HTML file.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    target_path = output_html_path or (OUTPUT_DIR / "smoke_simulation_map.html")
    
    # Prepare JSON serializable payloads
    # 1. Fire hotspots list
    fires_json = []
    for _, row in fires_df.iterrows():
        conf_raw = row.get("confidence", "80")
        if pd.isna(conf_raw):
            conf_str = "80%"
        elif str(conf_raw).strip().lower() in ("l", "low"):
            conf_str = "Low"
        elif str(conf_raw).strip().lower() in ("n", "nominal"):
            conf_str = "Nominal"
        elif str(conf_raw).strip().lower() in ("h", "high"):
            conf_str = "High"
        else:
            conf_str = f"{conf_raw}%" if "%" not in str(conf_raw) else str(conf_raw)

        fires_json.append({
            "lat": round(float(row["latitude"]), 4),
            "lon": round(float(row["longitude"]), 4),
            "frp": round(float(row.get("frp", 15.0)), 1),
            "state": str(row.get("state", "Punjab")),
            "satellite": str(row.get("satellite", "VIIRS")),
            "confidence": conf_str,
            "date": str(row.get("acq_date", "")),
            "time": str(row.get("acq_time", ""))
        })

    # 2. Downsample regional and Delhi grids for efficient in-browser rendering
    # Regional smoke density: shape (T, n_reg_lat, n_reg_lon)
    # We round float values to 1 decimal place to minimize payload
    t_count = len(result.times)
    
    # Regional smoke matrices per hour
    reg_smoke_payload = []
    for t in range(t_count):
        reg_smoke_payload.append(np.round(result.regional_smoke_density[t], 1).tolist())
        
    # Delhi high-res smoke matrices per hour
    delhi_smoke_payload = []
    for t in range(t_count):
        delhi_smoke_payload.append(np.round(result.delhi_smoke_density[t], 1).tolist())
        
    # Wind vectors at downsampled grid for streamlines/arrows
    wind_u_payload = []
    wind_v_payload = []
    wind_spd_payload = []
    
    # Downsample wind spatial grid by step of 2 for clean vector visualization
    step = 2
    sub_w_lats = wind_data["lats"][::step].tolist()
    sub_w_lons = wind_data["lons"][::step].tolist()
    
    for t in range(t_count):
        sub_u = np.round(wind_data["u_wind"][t, ::step, ::step], 2).tolist()
        sub_v = np.round(wind_data["v_wind"][t, ::step, ::step], 2).tolist()
        sub_spd = np.round(wind_data["speed"][t, ::step, ::step], 2).tolist()
        wind_u_payload.append(sub_u)
        wind_v_payload.append(sub_v)
        wind_spd_payload.append(sub_spd)

    # Compile data bundle
    simulation_bundle = {
        "title": title,
        "times": result.times,
        "forecast_hours": result.forecast_hours,
        "delhi_lats": [round(float(x), 4) for x in result.delhi_lats],
        "delhi_lons": [round(float(x), 4) for x in result.delhi_lons],
        "delhi_smoke": delhi_smoke_payload,
        "regional_lats": [round(float(x), 4) for x in result.regional_lats],
        "regional_lons": [round(float(x), 4) for x in result.regional_lons],
        "regional_smoke": reg_smoke_payload,
        "wind_lats": sub_w_lats,
        "wind_lons": sub_w_lons,
        "wind_u": wind_u_payload,
        "wind_v": wind_v_payload,
        "wind_speed": wind_spd_payload,
        "stations": DELHI_STATIONS,
        "station_series": result.station_series,
        "delhi_mean_series": result.delhi_mean_series,
        "delhi_max_series": result.delhi_max_series,
        "peak_hour": result.peak_delhi_hour,
        "peak_time": result.peak_delhi_time,
        "peak_density": result.peak_delhi_density,
        "fires": fires_json,
        "total_fires": result.total_active_fires,
        "contributing_fires": result.contributing_fire_count,
        "puffs_history": result.hourly_puff_positions
    }

    # Generate HTML content with Leaflet and custom UI
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  
  <!-- Google Fonts: Inter & Outfit -->
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@500;600;700;800&display=swap" rel="stylesheet">
  
  <!-- Leaflet CSS -->
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=" crossorigin=""/>
  
  <style>
    :root {{
      --bg-dark: #0b0f19;
      --panel-bg: rgba(15, 23, 42, 0.85);
      --panel-border: rgba(255, 255, 255, 0.12);
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --accent-cyan: #06b6d4;
      --accent-orange: #f97316;
      --accent-fire: #ef4444;
      --accent-smoke: #a855f7;
      --font-display: 'Outfit', sans-serif;
      --font-body: 'Inter', sans-serif;
    }}
    
    * {{
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }}
    
    body, html {{
      width: 100%;
      height: 100%;
      font-family: var(--font-body);
      background-color: var(--bg-dark);
      color: var(--text-main);
      overflow: hidden;
    }}
    
    #map {{
      width: 100%;
      height: 100%;
      z-index: 1;
      background: #090d16;
    }}
    
    /* Top Header Bar */
    .header-bar {{
      position: absolute;
      top: 16px;
      left: 16px;
      z-index: 1000;
      background: var(--panel-bg);
      backdrop-filter: blur(12px);
      -webkit-backdrop-filter: blur(12px);
      border: 1px solid var(--panel-border);
      border-radius: 12px;
      padding: 12px 20px;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
      max-width: 480px;
    }}
    
    .header-bar h1 {{
      font-family: var(--font-display);
      font-size: 1.25rem;
      font-weight: 700;
      letter-spacing: -0.02em;
      background: linear-gradient(135deg, #38bdf8, #fb923c, #f43f5e);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    
    .header-bar p {{
      font-size: 0.8rem;
      color: var(--text-muted);
      margin-top: 4px;
      line-height: 1.3;
    }}

    .badge-live {{
      display: inline-block;
      width: 8px;
      height: 8px;
      background: #ef4444;
      border-radius: 50%;
      box-shadow: 0 0 10px #ef4444;
      animation: pulse 1.8s infinite;
    }}
    
    @keyframes pulse {{
      0%, 100% {{ transform: scale(1); opacity: 1; }}
      50% {{ transform: scale(1.4); opacity: 0.6; }}
    }}
    
    /* Right HUD Analytics Panel */
    .hud-panel {{
      position: absolute;
      top: 16px;
      right: 16px;
      z-index: 1000;
      background: var(--panel-bg);
      backdrop-filter: blur(14px);
      -webkit-backdrop-filter: blur(14px);
      border: 1px solid var(--panel-border);
      border-radius: 14px;
      padding: 18px;
      width: 320px;
      box-shadow: 0 12px 40px rgba(0, 0, 0, 0.6);
      display: flex;
      flex-direction: column;
      gap: 14px;
    }}
    
    .hud-section-title {{
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--text-muted);
      font-weight: 600;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    
    .metric-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }}
    
    .metric-card {{
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid rgba(255, 255, 255, 0.06);
      border-radius: 10px;
      padding: 10px;
    }}
    
    .metric-card.accent {{
      border-color: rgba(249, 115, 22, 0.3);
      background: rgba(249, 115, 22, 0.08);
    }}
    
    .metric-label {{
      font-size: 0.7rem;
      color: var(--text-muted);
      margin-bottom: 2px;
    }}
    
    .metric-value {{
      font-family: var(--font-display);
      font-size: 1.35rem;
      font-weight: 700;
      color: #fff;
    }}
    
    .metric-sub {{
      font-size: 0.68rem;
      color: #38bdf8;
      margin-top: 2px;
    }}
    
    /* Air Quality Severity Pill */
    .severity-badge {{
      display: inline-block;
      padding: 4px 10px;
      border-radius: 20px;
      font-size: 0.75rem;
      font-weight: 600;
      text-align: center;
      width: 100%;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    
    /* Sparkline Chart Container */
    .sparkline-box {{
      background: rgba(0, 0, 0, 0.25);
      border-radius: 8px;
      padding: 8px;
      position: relative;
    }}
    
    .sparkline-canvas {{
      width: 100%;
      height: 60px;
      display: block;
    }}

    /* Bottom Control Bar & Timeline */
    .timeline-bar {{
      position: absolute;
      bottom: 24px;
      left: 50%;
      transform: translateX(-50%);
      z-index: 1000;
      background: var(--panel-bg);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      border: 1px solid var(--panel-border);
      border-radius: 16px;
      padding: 14px 24px;
      width: min(880px, 92vw);
      box-shadow: 0 16px 48px rgba(0, 0, 0, 0.7);
      display: flex;
      flex-direction: column;
      gap: 10px;
    }}
    
    .timeline-top {{
      display: flex;
      align-items: center;
      justify-content: space-between;
    }}
    
    .time-info {{
      display: flex;
      align-items: baseline;
      gap: 12px;
    }}
    
    .time-hour {{
      font-family: var(--font-display);
      font-size: 1.4rem;
      font-weight: 800;
      color: #38bdf8;
    }}
    
    .time-date {{
      font-size: 0.85rem;
      color: var(--text-muted);
    }}
    
    .player-controls {{
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    
    .ctrl-btn {{
      background: rgba(255, 255, 255, 0.08);
      border: 1px solid rgba(255, 255, 255, 0.15);
      color: #fff;
      border-radius: 8px;
      padding: 6px 12px;
      font-size: 0.85rem;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.15s ease;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 4px;
    }}
    
    .ctrl-btn:hover {{
      background: rgba(255, 255, 255, 0.18);
      border-color: rgba(255, 255, 255, 0.3);
      transform: translateY(-1px);
    }}
    
    .ctrl-btn.primary {{
      background: linear-gradient(135deg, #0ea5e9, #0284c7);
      border-color: #38bdf8;
      font-weight: 600;
    }}
    
    .ctrl-btn.active {{
      background: #f97316;
      border-color: #fdba74;
    }}
    
    /* Range Slider Styling */
    .slider-track {{
      position: relative;
      width: 100%;
    }}
    
    input[type=range] {{
      -webkit-appearance: none;
      width: 100%;
      height: 8px;
      border-radius: 4px;
      background: linear-gradient(90deg, #0284c7 0%, #f97316 50%, #ef4444 100%);
      outline: none;
      cursor: pointer;
    }}
    
    input[type=range]::-webkit-slider-thumb {{
      -webkit-appearance: none;
      appearance: none;
      width: 22px;
      height: 22px;
      border-radius: 50%;
      background: #ffffff;
      box-shadow: 0 0 10px rgba(0,0,0,0.5), 0 0 12px #38bdf8;
      cursor: grab;
      border: 2px solid #0ea5e9;
      transition: transform 0.1s ease;
    }}
    
    input[type=range]::-webkit-slider-thumb:active {{
      cursor: grabbing;
      transform: scale(1.2);
    }}
    
    /* Map Legend Overlay */
    .legend-overlay {{
      position: absolute;
      bottom: 120px;
      right: 16px;
      z-index: 1000;
      background: var(--panel-bg);
      backdrop-filter: blur(12px);
      -webkit-backdrop-filter: blur(12px);
      border: 1px solid var(--panel-border);
      border-radius: 10px;
      padding: 10px 14px;
      font-size: 0.75rem;
    }}
    
    .legend-title {{
      font-weight: 600;
      margin-bottom: 6px;
      color: var(--text-muted);
    }}
    
    .legend-scale {{
      display: flex;
      height: 10px;
      width: 160px;
      border-radius: 4px;
      background: linear-gradient(to right, 
        rgba(30, 41, 59, 0.4), 
        rgba(234, 179, 8, 0.7), 
        rgba(249, 115, 22, 0.85), 
        rgba(239, 68, 68, 0.9), 
        rgba(168, 85, 247, 0.95),
        rgba(126, 34, 206, 1.0)
      );
    }}
    
    .legend-labels {{
      display: flex;
      justify-content: space-between;
      margin-top: 4px;
      font-size: 0.65rem;
      color: var(--text-muted);
    }}

    /* Custom Leaflet Popups */
    .leaflet-popup-content-wrapper {{
      background: var(--panel-bg) !important;
      color: var(--text-main) !important;
      border: 1px solid var(--panel-border);
      border-radius: 10px;
      backdrop-filter: blur(12px);
    }}
    
    .leaflet-popup-tip {{
      background: var(--panel-bg) !important;
    }}
    
    .popup-box h4 {{
      font-family: var(--font-display);
      font-size: 0.9rem;
      margin-bottom: 4px;
      color: #38bdf8;
    }}
    
    .popup-box p {{
      font-size: 0.75rem;
      color: var(--text-muted);
      line-height: 1.4;
    }}
    
    /* Layer Switcher Checkboxes */
    .layer-toggles {{
      display: flex;
      gap: 12px;
      font-size: 0.75rem;
    }}
    
    .toggle-label {{
      display: flex;
      align-items: center;
      gap: 4px;
      cursor: pointer;
      color: var(--text-muted);
    }}
    
    .toggle-label input {{
      accent-color: #0ea5e9;
      cursor: pointer;
    }}
  </style>
</head>
<body>

  <!-- Map Container -->
  <div id="map"></div>

  <!-- Header -->
  <div class="header-bar">
    <h1>
      <span class="badge-live"></span>
      Delhi NCR Smoke Tracker
    </h1>
    <p>Lagrangian Puff & Gaussian Dispersion Simulation of Farm-Fire Emissions (Punjab & Haryana) advecting into Delhi NCR.</p>
  </div>

  <!-- Right HUD Panel -->
  <div class="hud-panel">
    <div class="hud-section-title">
      <span>Delhi NCR Impact</span>
      <span id="current-hour-tag">Hour +0h</span>
    </div>

    <!-- Severity Status -->
    <div id="severity-pill" class="severity-badge" style="background: rgba(34, 197, 94, 0.2); color: #4ade80; border: 1px solid rgba(74, 222, 128, 0.4);">
      Moderate Air Impact
    </div>

    <div class="metric-grid">
      <div class="metric-card">
        <div class="metric-label">Mean Smoke Index</div>
        <div class="metric-value" id="hud-mean-smoke">0.0</div>
        <div class="metric-sub" id="hud-pm25">~0 µg/m³ farm contrib.</div>
      </div>
      <div class="metric-card accent">
        <div class="metric-label">Peak Concentration</div>
        <div class="metric-value" id="hud-peak-smoke">0.0</div>
        <div class="metric-sub">Hotspot grid cell</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Total Active Fires</div>
        <div class="metric-value" id="hud-total-fires">{result.total_active_fires}</div>
        <div class="metric-sub">Punjab & Haryana (24h)</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Peak Arrival Time</div>
        <div class="metric-value" style="font-size: 1.05rem;" id="hud-peak-hour">H+{result.peak_delhi_hour}</div>
        <div class="metric-sub" id="hud-peak-time">{result.peak_delhi_time.split('T')[1] if 'T' in result.peak_delhi_time else result.peak_delhi_time}</div>
      </div>
    </div>

    <!-- 72h Sparkline Chart -->
    <div class="hud-section-title">
      <span>72-Hour Smoke Trend (Delhi)</span>
    </div>
    <div class="sparkline-box">
      <canvas id="sparkline" class="sparkline-canvas"></canvas>
    </div>

    <!-- Layer Toggles -->
    <div class="layer-toggles">
      <label class="toggle-label"><input type="checkbox" id="chk-smoke" checked> Smoke Plume</label>
      <label class="toggle-label"><input type="checkbox" id="chk-fires" checked> Fire Hotspots</label>
      <label class="toggle-label"><input type="checkbox" id="chk-wind" checked> Wind Vectors</label>
      <label class="toggle-label"><input type="checkbox" id="chk-puffs" checked> Puffs</label>
    </div>
  </div>

  <!-- Legend Overlay -->
  <div class="legend-overlay">
    <div class="legend-title">Smoke Density (Arbitrary Intensity Units)</div>
    <div class="legend-scale"></div>
    <div class="legend-labels">
      <span>Low (5)</span>
      <span>Moderate (50)</span>
      <span>Severe (200+)</span>
    </div>
  </div>

  <!-- Timeline Player Bar -->
  <div class="timeline-bar">
    <div class="timeline-top">
      <div class="time-info">
        <span class="time-hour" id="display-hour">Hour 0 / 72</span>
        <span class="time-date" id="display-datetime">--:--</span>
      </div>
      <div class="player-controls">
        <button class="ctrl-btn" id="btn-prev" title="Step Back 1 Hour">⏮</button>
        <button class="ctrl-btn primary" id="btn-play" title="Play / Pause Animation">▶ Play</button>
        <button class="ctrl-btn" id="btn-next" title="Step Forward 1 Hour">⏭</button>
        <button class="ctrl-btn" id="btn-speed">1x Speed</button>
        <button class="ctrl-btn" id="btn-reset" title="Reset to Start">↺ Reset</button>
      </div>
    </div>
    <div class="slider-track">
      <input type="range" id="time-slider" min="0" max="{result.forecast_hours - 1}" value="0" step="1">
    </div>
  </div>

  <!-- Leaflet JS -->
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin=""></script>
  
  <script>
    // Injected Data Bundle from Python
    const SIM_DATA = {json.dumps(simulation_bundle)};
    
    // State & UI variables (initialized at top to avoid Temporal Dead Zone)
    let currentHour = 0;
    let isPlaying = false;
    let playInterval = null;
    let playSpeed = 350; // ms per frame

    // DOM Elements
    const timeSlider = document.getElementById('time-slider');
    const displayHour = document.getElementById('display-hour');
    const displayDatetime = document.getElementById('display-datetime');
    const currentHourTag = document.getElementById('current-hour-tag');
    const hudMeanSmoke = document.getElementById('hud-mean-smoke');
    const hudPeakSmoke = document.getElementById('hud-peak-smoke');
    const hudPm25 = document.getElementById('hud-pm25');
    const severityPill = document.getElementById('severity-pill');
    const btnPlay = document.getElementById('btn-play');
    const sparkCanvas = document.getElementById('sparkline');

    // Color mapper for smoke concentration
    function getSmokeColor(val) {{
      if (val < 1) return [0, 0, 0, 0];
      if (val < 15) return [234, 179, 8, Math.min(0.5, val / 15 * 0.5)];    // Yellow
      if (val < 50) return [249, 115, 22, 0.65];                            // Orange
      if (val < 120) return [239, 68, 68, 0.78];                            // Red
      if (val < 250) return [168, 85, 247, 0.85];                           // Purple
      return [126, 34, 206, 0.95];                                          // Hazardous Deep Purple
    }}

    // Initialize Map with dark tiles
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

    // Layer groups
    const firesLayerGroup = L.layerGroup().addTo(map);
    const windLayerGroup = L.layerGroup().addTo(map);
    const stationLayerGroup = L.layerGroup().addTo(map);
    const puffLayerGroup = L.layerGroup().addTo(map);
    const delhiBorderGroup = L.layerGroup().addTo(map);

    // Delhi NCR Bounding Polygon
    const delhiPoly = L.polygon([
      [{DELHI_NCR_DOMAIN.min_lat}, {DELHI_NCR_DOMAIN.min_lon}],
      [{DELHI_NCR_DOMAIN.max_lat}, {DELHI_NCR_DOMAIN.min_lon}],
      [{DELHI_NCR_DOMAIN.max_lat}, {DELHI_NCR_DOMAIN.max_lon}],
      [{DELHI_NCR_DOMAIN.min_lat}, {DELHI_NCR_DOMAIN.max_lon}]
    ], {{
      color: '#38bdf8',
      weight: 2,
      fillColor: '#0284c7',
      fillOpacity: 0.05,
      dashArray: '6, 6'
    }}).addTo(delhiBorderGroup);
    
    delhiPoly.bindTooltip("<b>Delhi NCR Target Domain</b>", {{ sticky: true }});

    // Monitoring Stations
    SIM_DATA.stations.forEach(st => {{
      const marker = L.circleMarker([st.lat, st.lon], {{
        radius: 6,
        fillColor: '#38bdf8',
        color: '#ffffff',
        weight: 1.5,
        fillOpacity: 0.9
      }}).addTo(stationLayerGroup);
      
      marker.bindPopup(`
        <div class="popup-box">
          <h4>${{st.name}}</h4>
          <p><b>Type:</b> ${{st.type}}<br>
          <b>Coordinates:</b> ${{st.lat.toFixed(4)}}, ${{st.lon.toFixed(4)}}<br>
          <span id="popup-st-${{st.name.replace(/\\s+/g, '-')}}">Current Smoke: 0.0</span></p>
        </div>
      `);
    }});

    // Render Fire Hotspots
    function renderFires() {{
      firesLayerGroup.clearLayers();
      SIM_DATA.fires.forEach(f => {{
        const radius = Math.min(16, Math.max(5, Math.sqrt(f.frp) * 2.2));
        const marker = L.circleMarker([f.lat, f.lon], {{
          radius: radius,
          fillColor: '#ef4444',
          color: '#fca5a5',
          weight: 1.5,
          fillOpacity: 0.85
        }}).addTo(firesLayerGroup);
        
        marker.bindPopup(`
          <div class="popup-box">
            <h4 style="color: #f87171;">Active Fire Hotspot</h4>
            <p><b>State:</b> ${{f.state}}<br>
            <b>FRP:</b> ${{f.frp}} MW<br>
            <b>Satellite:</b> ${{f.satellite}}<br>
            <b>Confidence:</b> ${{f.confidence}}<br>
            <b>Detection:</b> ${{f.date}} ${{f.time}} UTC</p>
          </div>
        `);
      }});
    }}
    renderFires();

    // Wind Vector rendering
    function renderWind(hourIdx) {{
      windLayerGroup.clearLayers();
      if (!document.getElementById('chk-wind').checked) return;

      const wLats = SIM_DATA.wind_lats;
      const wLons = SIM_DATA.wind_lons;
      const uArr = SIM_DATA.wind_u[hourIdx];
      const vArr = SIM_DATA.wind_v[hourIdx];
      const spdArr = SIM_DATA.wind_speed[hourIdx];

      if (!uArr) return;

      for (let i = 0; i < wLats.length; i++) {{
        for (let j = 0; j < wLons.length; j++) {{
          const lat = wLats[i];
          const lon = wLons[j];
          const u = uArr[i][j];
          const v = vArr[i][j];
          const spd = spdArr[i][j];

          if (spd < 0.5) continue;

          const scale = 0.035;
          const endLat = lat + (v / Math.max(1.0, spd)) * scale * (spd * 0.5);
          const endLon = lon + (u / Math.max(1.0, spd)) * scale * (spd * 0.5);

          L.polyline([[lat, lon], [endLat, endLon]], {{
            color: 'rgba(56, 189, 248, 0.45)',
            weight: 1.5
          }}).addTo(windLayerGroup);

          L.circleMarker([endLat, endLon], {{
            radius: 2,
            color: 'rgba(56, 189, 248, 0.7)',
            fillColor: '#38bdf8',
            fillOpacity: 0.8,
            weight: 1
          }}).addTo(windLayerGroup);
        }}
      }}
    }}

    // Lagrangian Puffs rendering
    function renderPuffs(hourIdx) {{
      puffLayerGroup.clearLayers();
      if (!document.getElementById('chk-puffs').checked) return;

      const puffs = SIM_DATA.puffs_history[hourIdx] || [];
      puffs.forEach(p => {{
        if (p.intensity > 1.0) {{
          L.circleMarker([p.lat, p.lon], {{
            radius: Math.min(6, Math.max(2, Math.sqrt(p.intensity) * 0.4)),
            color: 'rgba(251, 146, 60, 0.4)',
            fillColor: '#fb923c',
            fillOpacity: 0.5,
            weight: 1
          }}).addTo(puffLayerGroup);
        }}
      }});
    }}

    // Draw 72-Hour Sparkline Chart on Canvas
    function drawSparkline() {{
      if (!sparkCanvas) return;
      const ctx = sparkCanvas.getContext('2d');
      const dpr = window.devicePixelRatio || 1;
      const rect = sparkCanvas.getBoundingClientRect();
      if (rect.width === 0 || rect.height === 0) return;
      
      sparkCanvas.width = rect.width * dpr;
      sparkCanvas.height = rect.height * dpr;
      ctx.scale(dpr, dpr);
      
      const w = rect.width;
      const h = rect.height;
      const data = SIM_DATA.delhi_mean_series;
      const maxVal = Math.max(10, Math.max(...data));

      ctx.clearRect(0, 0, w, h);

      // Area gradient
      const grad = ctx.createLinearGradient(0, 0, 0, h);
      grad.addColorStop(0, 'rgba(249, 115, 22, 0.4)');
      grad.addColorStop(1, 'rgba(249, 115, 22, 0.0)');

      ctx.beginPath();
      for (let i = 0; i < data.length; i++) {{
        const x = (i / (data.length - 1)) * w;
        const y = h - (data[i] / maxVal) * (h - 8) - 4;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }}
      ctx.lineTo(w, h);
      ctx.lineTo(0, h);
      ctx.closePath();
      ctx.fillStyle = grad;
      ctx.fill();

      // Stroke line
      ctx.beginPath();
      for (let i = 0; i < data.length; i++) {{
        const x = (i / (data.length - 1)) * w;
        const y = h - (data[i] / maxVal) * (h - 8) - 4;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }}
      ctx.strokeStyle = '#f97316';
      ctx.lineWidth = 2;
      ctx.stroke();

      // Current hour cursor line
      const curX = (currentHour / (data.length - 1)) * w;
      ctx.beginPath();
      ctx.moveTo(curX, 0);
      ctx.lineTo(curX, h);
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 2;
      ctx.setLineDash([3, 3]);
      ctx.stroke();
      ctx.setLineDash([]);

      // Current hour circle dot
      const curY = h - (data[currentHour] / maxVal) * (h - 8) - 4;
      ctx.beginPath();
      ctx.arc(curX, curY, 4, 0, Math.PI * 2);
      ctx.fillStyle = '#ffffff';
      ctx.fill();
      ctx.strokeStyle = '#0284c7';
      ctx.lineWidth = 2;
      ctx.stroke();
    }}

    // Smoke Layer using HTML5 Canvas
    const CustomSmokeOverlay = L.Layer.extend({{
      onAdd: function(map) {{
        this._map = map;
        this._canvas = L.DomUtil.create('canvas', 'leaflet-smoke-canvas');
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

        if (!document.getElementById('chk-smoke').checked) return;

        // Render Regional Plumes
        const regSmoke = SIM_DATA.regional_smoke[hourIdx];
        const regLats = SIM_DATA.regional_lats;
        const regLons = SIM_DATA.regional_lons;
        
        if (!regSmoke) return;

        const nLat = regLats.length;
        const nLon = regLons.length;
        
        for (let i = 0; i < nLat; i++) {{
          for (let j = 0; j < nLon; j++) {{
            const val = regSmoke[i][j];
            if (val >= 2.0) {{
              const pt = this._map.latLngToContainerPoint([regLats[i], regLons[j]]);
              const [r, g, b, a] = getSmokeColor(val);
              
              const rad = Math.max(12, Math.min(38, Math.sqrt(val) * 3.0));
              const grad = ctx.createRadialGradient(pt.x, pt.y, 0, pt.x, pt.y, rad);
              grad.addColorStop(0, `rgba(${{r}}, ${{g}}, ${{b}}, ${{a}})`);
              grad.addColorStop(0.6, `rgba(${{r}}, ${{g}}, ${{b}}, ${{a * 0.4}})`);
              grad.addColorStop(1, `rgba(${{r}}, ${{g}}, ${{b}}, 0)`);
              
              ctx.fillStyle = grad;
              ctx.beginPath();
              ctx.arc(pt.x, pt.y, rad, 0, Math.PI * 2);
              ctx.fill();
            }}
          }}
        }}

        // Render Delhi high-res plume overlay
        const delhiSmoke = SIM_DATA.delhi_smoke[hourIdx];
        const dLats = SIM_DATA.delhi_lats;
        const dLons = SIM_DATA.delhi_lons;
        if (delhiSmoke) {{
          for (let i = 0; i < dLats.length; i += 2) {{
            for (let j = 0; j < dLons.length; j += 2) {{
              const val = delhiSmoke[i][j];
              if (val >= 3.0) {{
                const pt = this._map.latLngToContainerPoint([dLats[i], dLons[j]]);
                const [r, g, b, a] = getSmokeColor(val);
                const rad = Math.max(14, Math.min(32, Math.sqrt(val) * 2.8));
                
                const grad = ctx.createRadialGradient(pt.x, pt.y, 0, pt.x, pt.y, rad);
                grad.addColorStop(0, `rgba(${{r}}, ${{g}}, ${{b}}, ${{a * 1.1}})`);
                grad.addColorStop(0.7, `rgba(${{r}}, ${{g}}, ${{b}}, ${{a * 0.3}})`);
                grad.addColorStop(1, `rgba(${{r}}, ${{g}}, ${{b}}, 0)`);
                
                ctx.fillStyle = grad;
                ctx.beginPath();
                ctx.arc(pt.x, pt.y, rad, 0, Math.PI * 2);
                ctx.fill();
              }}
            }}
          }}
        }}
      }}
    }});

    const customSmoke = new CustomSmokeOverlay();
    map.addLayer(customSmoke);

    // Update UI for specified hour
    function updateHour(hourIdx) {{
      currentHour = Math.max(0, Math.min(SIM_DATA.forecast_hours - 1, hourIdx));
      if (timeSlider) timeSlider.value = currentHour;
      
      const timeStr = SIM_DATA.times[currentHour] || `+${{currentHour}}h`;
      if (displayHour) displayHour.innerText = `Hour +${{currentHour}}h / 72h`;
      if (displayDatetime) displayDatetime.innerText = timeStr.replace('T', ' ') + ' UTC';
      if (currentHourTag) currentHourTag.innerText = `Hour +${{currentHour}}h`;

      const meanVal = SIM_DATA.delhi_mean_series[currentHour] || 0;
      const peakVal = SIM_DATA.delhi_max_series[currentHour] || 0;
      
      if (hudMeanSmoke) hudMeanSmoke.innerText = meanVal.toFixed(1);
      if (hudPeakSmoke) hudPeakSmoke.innerText = peakVal.toFixed(1);
      if (hudPm25) hudPm25.innerText = `~${{(meanVal * 0.85).toFixed(0)}} µg/m³ farm contrib.`;

      // Update severity badge
      if (severityPill) {{
        if (meanVal < 10) {{
          severityPill.style.background = 'rgba(34, 197, 94, 0.2)';
          severityPill.style.color = '#4ade80';
          severityPill.style.borderColor = 'rgba(74, 222, 128, 0.4)';
          severityPill.innerText = 'Minimal Smoke Impact';
        }} else if (meanVal < 45) {{
          severityPill.style.background = 'rgba(234, 179, 8, 0.2)';
          severityPill.style.color = '#facc15';
          severityPill.style.borderColor = 'rgba(250, 204, 21, 0.4)';
          severityPill.innerText = 'Moderate Smoke Inflow';
        }} else if (meanVal < 100) {{
          severityPill.style.background = 'rgba(249, 115, 22, 0.25)';
          severityPill.style.color = '#fb923c';
          severityPill.style.borderColor = 'rgba(251, 146, 60, 0.4)';
          severityPill.innerText = 'Heavy Smoke Plume';
        }} else {{
          severityPill.style.background = 'rgba(239, 68, 68, 0.3)';
          severityPill.style.color = '#f87171';
          severityPill.style.borderColor = 'rgba(248, 113, 113, 0.5)';
          severityPill.innerText = 'Severe Smog Intrusion';
        }}
      }}

      // Update Station Popups
      SIM_DATA.stations.forEach(st => {{
        const el = document.getElementById(`popup-st-${{st.name.replace(/\\s+/g, '-')}}`);
        if (el && SIM_DATA.station_series[st.name]) {{
          el.innerText = `Current Smoke: ${{SIM_DATA.station_series[st.name][currentHour]}}`;
        }}
      }});

      // Redraw dynamic map layers
      customSmoke.renderHour(currentHour);
      renderWind(currentHour);
      renderPuffs(currentHour);
      drawSparkline();
    }}

    // Controls Event Listeners
    if (timeSlider) {{
      timeSlider.addEventListener('input', (e) => {{
        updateHour(parseInt(e.target.value));
      }});
    }}

    function togglePlay() {{
      isPlaying = !isPlaying;
      if (isPlaying) {{
        btnPlay.innerText = '⏸ Pause';
        btnPlay.classList.add('active');
        playInterval = setInterval(() => {{
          if (currentHour >= SIM_DATA.forecast_hours - 1) {{
            updateHour(0);
          }} else {{
            updateHour(currentHour + 1);
          }}
        }}, playSpeed);
      }} else {{
        btnPlay.innerText = '▶ Play';
        btnPlay.classList.remove('active');
        clearInterval(playInterval);
      }}
    }}

    if (btnPlay) btnPlay.addEventListener('click', togglePlay);

    const btnPrev = document.getElementById('btn-prev');
    if (btnPrev) btnPrev.addEventListener('click', () => {{
      if (isPlaying) togglePlay();
      updateHour(currentHour - 1);
    }});

    const btnNext = document.getElementById('btn-next');
    if (btnNext) btnNext.addEventListener('click', () => {{
      if (isPlaying) togglePlay();
      updateHour(currentHour + 1);
    }});

    const btnReset = document.getElementById('btn-reset');
    if (btnReset) btnReset.addEventListener('click', () => {{
      if (isPlaying) togglePlay();
      updateHour(0);
    }});

    const speedBtn = document.getElementById('btn-speed');
    const speeds = [{{ label: '1x Speed', ms: 350 }}, {{ label: '2x Speed', ms: 180 }}, {{ label: '4x Speed', ms: 80 }}];
    let speedIdx = 0;
    if (speedBtn) speedBtn.addEventListener('click', () => {{
      speedIdx = (speedIdx + 1) % speeds.length;
      speedBtn.innerText = speeds[speedIdx].label;
      playSpeed = speeds[speedIdx].ms;
      if (isPlaying) {{
        clearInterval(playInterval);
        playInterval = setInterval(() => {{
          if (currentHour >= SIM_DATA.forecast_hours - 1) {{
            updateHour(0);
          }} else {{
            updateHour(currentHour + 1);
          }}
        }}, playSpeed);
      }}
    }});

    // Layer checkboxes
    const chkSmoke = document.getElementById('chk-smoke');
    if (chkSmoke) chkSmoke.addEventListener('change', () => customSmoke.renderHour(currentHour));
    
    const chkFires = document.getElementById('chk-fires');
    if (chkFires) chkFires.addEventListener('change', (e) => {{
      if (e.target.checked) firesLayerGroup.addTo(map);
      else map.removeLayer(firesLayerGroup);
    }});
    
    const chkWind = document.getElementById('chk-wind');
    if (chkWind) chkWind.addEventListener('change', () => renderWind(currentHour));
    
    const chkPuffs = document.getElementById('chk-puffs');
    if (chkPuffs) chkPuffs.addEventListener('change', () => renderPuffs(currentHour));

    // Keyboard shortcuts (Space = Play/Pause, ArrowLeft = -1, ArrowRight = +1)
    window.addEventListener('keydown', (e) => {{
      if (e.code === 'Space') {{
        e.preventDefault();
        togglePlay();
      }} else if (e.code === 'ArrowLeft') {{
        if (isPlaying) togglePlay();
        updateHour(currentHour - 1);
      }} else if (e.code === 'ArrowRight') {{
        if (isPlaying) togglePlay();
        updateHour(currentHour + 1);
      }}
    }});

    // Initial render
    window.addEventListener('resize', drawSparkline);
    setTimeout(() => {{
      updateHour(0);
      drawSparkline();
    }}, 100);
  </script>
</body>
</html>
"""

    with open(target_path, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    logger.info(f"Generated standalone interactive Leaflet simulation map: {target_path}")
    return target_path
