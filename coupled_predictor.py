"""
72-Hour Coupled AQI Prediction Engine for Delhi NCR Monitoring Stations.
Generates hourly station-specific AQI forecasts comparing:
1. Traditional Decoupled Forecast (standard dispersion without feedback).
2. Coupled Feedback Forecast (modeling PBL collapse, solar dimming, and induced stillness).
Uses official Central Pollution Control Board (CPCB) NAQI breakpoint standards.
"""

import logging
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Ensure submodules can be imported
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
correction_model_dir = ROOT_DIR / "correction-model"
smoke_tracker_dir = ROOT_DIR / "smoke-tracker"
if str(correction_model_dir) not in sys.path:
    sys.path.insert(0, str(correction_model_dir))
if str(smoke_tracker_dir) not in sys.path:
    sys.path.insert(0, str(smoke_tracker_dir))

import numpy as np
import pandas as pd

from coupled_feedback import CoupledSimulationResult
from correction_model.config import (
    CPCB_STATIONS,
    CPCB_PM25_BREAKPOINTS,
    MIN_AQI,
    MAX_AQI,
)
from correction_model.fetch_data import calculate_cpcb_aqi

logger = logging.getLogger("airloop.coupled_predictor")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )


@dataclass
class Station72hForecast:
    """72-hour coupled vs decoupled forecast profile for a specific monitoring station."""
    name: str
    code: str
    lat: float
    lon: float
    station_type: str
    current_measured_aqi: float
    decoupled_series: List[float]
    coupled_series: List[float]
    underprediction_gap_series: List[float]
    peak_coupled_aqi: float
    peak_decoupled_aqi: float
    peak_coupled_hour: int
    peak_coupled_time: str
    peak_severity_category: str
    feedback_attribution: Dict[str, float]


def get_aqi_category(aqi_val: float) -> str:
    """Classify AQI according to official Government of India CPCB NAQI standard."""
    if aqi_val <= 50:
        return "Good"
    elif aqi_val <= 100:
        return "Satisfactory"
    elif aqi_val <= 200:
        return "Moderate"
    elif aqi_val <= 300:
        return "Poor"
    elif aqi_val <= 400:
        return "Very Poor"
    else:
        return "Severe"


class CoupledStationAQIPredictor:
    """
    Computes 72-hour coupled station AQI trajectories across Delhi NCR.
    """

    def __init__(self, stations: Optional[List[Dict[str, Any]]] = None):
        self.stations = stations or CPCB_STATIONS

    def predict_72h_station_forecasts(
        self,
        coupled_result: CoupledSimulationResult,
        station_smoke_traces: Optional[Dict[str, List[float]]] = None,
        base_measured_aqis: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Station72hForecast]:
        """
        Generate continuous 72-hour AQI forecasts for all CPCB stations.
        """
        n_hours = coupled_result.forecast_hours
        times = coupled_result.times
        station_forecasts: Dict[str, Station72hForecast] = {}
        
        # Default baseline measured AQIs typical of late October / early November in Delhi
        default_station_aqis = {
            "Anand Vihar": 335.0,    # Traffic hotspot
            "ITO": 310.0,            # Commercial / Central
            "RK Puram": 285.0,        # Residential South
            "Punjabi Bagh": 315.0,    # West Delhi corridor
            "Rohini": 325.0,          # North-West downwind
            "IGI Airport T3": 265.0,  # Open airport area
            "Bawana": 345.0,          # North Industrial hotspot
            "Okhla Phase-2": 310.0,   # South-East Industrial
            "Jahangirpuri": 340.0,    # North residential hotspot
            "Dwarka Sector 8": 280.0, # South-West
        }
        
        logger.info(f"Computing 72-hour coupled vs decoupled forecasts for {len(self.stations)} CPCB stations...")
        
        for st in self.stations:
            st_name = st["name"]
            st_code = st.get("code", "DL000")
            st_lat = st["lat"]
            st_lon = st["lon"]
            st_type = st.get("type", "Urban")
            
            base_aqi = (
                base_measured_aqis.get(st_name, default_station_aqis.get(st_name, 300.0))
                if base_measured_aqis else default_station_aqis.get(st_name, 300.0)
            )
            
            # Obtain physical smoke trace for this station from advection simulation
            if station_smoke_traces and st_name in station_smoke_traces:
                st_trace = station_smoke_traces[st_name][:n_hours]
            else:
                # Regional fallback with geographical spatial modifier
                geo_factor = 1.0 + (st_lat - 28.6) * 0.4 - (st_lon - 77.2) * 0.2
                st_trace = [max(0.0, v * geo_factor) for v in coupled_result.decoupled_delhi_mean]
                
            decoupled_aqi_series = []
            coupled_aqi_series = []
            gap_series = []
            
            for h in range(n_hours):
                step = coupled_result.steps[h]
                h_of_day = (h + 12) % 24
                
                # Diurnal baseline variation (traffic rush hours and nighttime shallow mixing)
                # Morning peak ~8-10 AM, evening rush ~6-9 PM
                diurnal_rush = 15.0 * math.sin((h_of_day - 8) / 12.0 * math.pi)
                background_aqi = base_aqi + diurnal_rush * 0.5
                
                # 1. Decoupled Forecast:
                # Standard advection without radiative feedback
                decoupled_smoke_contrib = st_trace[h] * 0.012
                # Convert to AQI scaling
                dec_aqi = np.clip(background_aqi + decoupled_smoke_contrib, MIN_AQI, MAX_AQI)
                decoupled_aqi_series.append(round(float(dec_aqi), 1))
                
                # 2. Coupled Feedback Forecast:
                # Includes PBL collapse compression, solar dimming, and induced stillness
                entrapment = step.entrapment_factor
                wind_stillness_boost = 1.0 + (step.wind_deceleration_pct / 100.0) * 0.7
                
                coupled_smoke_contrib = decoupled_smoke_contrib * entrapment * wind_stillness_boost
                
                # Additional thermal inversion penalty during nighttime
                inversion_penalty = 12.0 * (1.0 - (step.coupled_pbl_m / max(300.0, step.unperturbed_pbl_m)))
                
                coup_aqi = np.clip(background_aqi + coupled_smoke_contrib + inversion_penalty, MIN_AQI, MAX_AQI)
                coupled_aqi_series.append(round(float(coup_aqi), 1))
                
                # Underprediction gap
                gap = coup_aqi - dec_aqi
                gap_series.append(round(float(gap), 1))
                
            peak_coup_idx = int(np.argmax(coupled_aqi_series))
            peak_coup_val = float(coupled_aqi_series[peak_coup_idx])
            peak_dec_val = float(decoupled_aqi_series[peak_coup_idx])
            peak_time_str = times[peak_coup_idx] if peak_coup_idx < len(times) else f"+{peak_coup_idx}h"
            
            # Feedback factor attribution at peak hour
            peak_step = coupled_result.steps[peak_coup_idx]
            attribution = {
                "advected_plume_pct": 45.0,
                "pbl_collapse_trapping_pct": round(min(35.0, peak_step.pbl_collapse_pct * 0.45), 1),
                "induced_stillness_pct": round(min(20.0, peak_step.wind_deceleration_pct * 0.5), 1),
                "thermal_inversion_pct": 12.0,
            }
            
            station_forecasts[st_name] = Station72hForecast(
                name=st_name,
                code=st_code,
                lat=st_lat,
                lon=st_lon,
                station_type=st_type,
                current_measured_aqi=round(base_aqi, 1),
                decoupled_series=decoupled_aqi_series,
                coupled_series=coupled_aqi_series,
                underprediction_gap_series=gap_series,
                peak_coupled_aqi=round(peak_coup_val, 1),
                peak_decoupled_aqi=round(peak_dec_val, 1),
                peak_coupled_hour=peak_coup_idx,
                peak_coupled_time=peak_time_str,
                peak_severity_category=get_aqi_category(peak_coup_val),
                feedback_attribution=attribution,
            )
            
        return station_forecasts
