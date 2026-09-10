"""
Configuration constants, CPCB station metadata, and parameter definitions for AQI Forecast Error Correction.
"""

from pathlib import Path
from typing import Dict, List, Any, Tuple

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
MODEL_DIR = DATA_DIR / "models"
OUTPUT_DIR = DATA_DIR / "outputs"

for d in [DATA_DIR, CACHE_DIR, MODEL_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Key CPCB Monitoring Stations across Delhi NCR
CPCB_STATIONS: List[Dict[str, Any]] = [
    {"name": "Anand Vihar", "code": "DL001", "lat": 28.6469, "lon": 77.3160, "type": "Traffic / Hotspot"},
    {"name": "ITO", "code": "DL002", "lat": 28.6289, "lon": 77.2405, "type": "Commercial / Central"},
    {"name": "RK Puram", "code": "DL003", "lat": 28.5633, "lon": 77.1863, "type": "Residential / South"},
    {"name": "Punjabi Bagh", "code": "DL004", "lat": 28.6740, "lon": 77.1310, "type": "Residential / West"},
    {"name": "Rohini", "code": "DL005", "lat": 28.7325, "lon": 77.1199, "type": "Residential / North-West"},
    {"name": "IGI Airport T3", "code": "DL006", "lat": 28.5629, "lon": 77.0878, "type": "Industrial / Airport"},
    {"name": "Bawana", "code": "DL007", "lat": 28.7997, "lon": 77.0326, "type": "Industrial / North"},
    {"name": "Okhla Phase-2", "code": "DL008", "lat": 28.5308, "lon": 77.2713, "type": "Industrial / South-East"},
    {"name": "Jahangirpuri", "code": "DL009", "lat": 28.7328, "lon": 77.1706, "type": "Residential / North"},
    {"name": "Dwarka Sector 8", "code": "DL010", "lat": 28.5710, "lon": 77.0667, "type": "Residential / South-West"},
]

# Seasonal date configurations focused on autumn-winter stubble burning & thermal inversion
SEASONS: Dict[str, Dict[str, str]] = {
    "2022-2023": {"start": "2022-10-01", "end": "2023-01-31"},
    "2023-2024": {"start": "2023-10-01", "end": "2024-01-31"},
    "2024-2025": {"start": "2024-10-01", "end": "2025-01-31"},
}

# Critical severe smog window for focused evaluation (Peak Stubble Burning & Post-Diwali Inversion)
SEVERE_SMOG_EVAL_PERIOD = {
    "start_month": 11,
    "start_day": 1,
    "end_month": 11,
    "end_day": 15,
}

# Official CPCB Breakpoint Table for PM2.5 (24-hour average in µg/m³) -> Sub-index (AQI)
# Format: (C_low, C_high, I_low, I_high, Category)
CPCB_PM25_BREAKPOINTS: List[Tuple[float, float, float, float, str]] = [
    (0.0, 30.0, 0.0, 50.0, "Good"),
    (30.1, 60.0, 51.0, 100.0, "Satisfactory"),
    (60.1, 90.0, 101.0, 200.0, "Moderate"),
    (90.1, 120.0, 201.0, 300.0, "Poor"),
    (120.1, 250.0, 301.0, 400.0, "Very Poor"),
    (250.1, 380.0, 401.0, 500.0, "Severe"),
]

# Physical bounds for Indian AQI
MIN_AQI = 0.0
MAX_AQI = 500.0

# Northwest wind direction range (Punjab/Haryana towards Delhi): 270° (W) to 345° (NNW)
UPWIND_DIR_MIN = 270.0
UPWIND_DIR_MAX = 345.0

# Threshold for wind stillness (m/s)
STILL_WIND_THRESHOLD = 1.5

# Engineered feature names used in regression modeling
FEATURE_COLS: List[str] = [
    # Baseline AQI state
    "baseline_aqi",
    # Wind stillness & ventilation
    "wind_speed_mean",
    "still_hours_count",
    "wind_speed_max",
    # Wind direction & NW alignment
    "nw_wind_fraction",
    "wind_u_mean",
    "wind_v_mean",
    # Temperature & thermal inversion
    "temp_mean",
    "temp_min",
    "diurnal_temp_range",
    "temp_change_24h",
    # Precipitation & dry spell
    "precip_sum",
    "days_since_rain",
    # Upwind fire emissions (past 48h)
    "fire_count_48h",
    "frp_sum_48h",
    "upwind_fire_index",
    # AQI trends and persistence dynamics
    "aqi_trend_24h",
    "aqi_roll_mean_3d",
    "aqi_roll_std_3d",
    # Station location coordinates
    "station_lat",
    "station_lon",
    # Calendar features
    "day_of_year",
    "day_of_week",
    "is_weekend",
    "month",
]

TARGET_COL = "residual"  # actual_aqi_tomorrow - baseline_aqi_today
