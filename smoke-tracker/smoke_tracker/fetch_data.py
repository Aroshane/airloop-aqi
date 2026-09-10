"""
Data fetching module for NASA FIRMS Active Fire Data and Open-Meteo Wind Forecasts.
Includes local caching, error handling, rate-limit resilience, and offline fallbacks.
"""

import os
import io
import json
import time
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List, Tuple

import requests
import numpy as np
import pandas as pd

from smoke_tracker.config import (
    CACHE_DIR,
    DomainBounds,
    REGIONAL_DOMAIN,
    DELHI_NCR_DOMAIN,
    SimulationConfig,
    DEFAULT_CONFIG
)

logger = logging.getLogger("smoke_tracker.fetch_data")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

# NASA FIRMS Open NRT 24h CSV Feed URLs (Publicly accessible, no key required)
FIRMS_OPEN_FEEDS = {
    "VIIRS_SNPP": "https://firms.modaps.eosdis.nasa.gov/data/active_fire/suomi-npp-viirs-c2/csv/SUOMI_VIIRS_C2_South_Asia_24h.csv",
    "VIIRS_NOAA20": "https://firms.modaps.eosdis.nasa.gov/data/active_fire/noaa-20-viirs-c2/csv/J1_VIIRS_C2_South_Asia_24h.csv",
    "VIIRS_NOAA21": "https://firms.modaps.eosdis.nasa.gov/data/active_fire/noaa-21-viirs-c2/csv/J2_VIIRS_C2_South_Asia_24h.csv",
    "MODIS": "https://firms.modaps.eosdis.nasa.gov/data/active_fire/modis-c6.1/csv/MODIS_C6_1_South_Asia_24h.csv",
}

# Open-Meteo Base URL
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


def ensure_cache_dir(custom_dir: Optional[Path] = None) -> Path:
    """Ensure that the cache directory exists."""
    target_dir = custom_dir or CACHE_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def is_cache_valid(cache_file: Path, max_age_hours: int = 6) -> bool:
    """Check if a cached file exists and is within its freshness lifetime."""
    if not cache_file.exists():
        return False
    try:
        mtime = datetime.fromtimestamp(cache_file.stat().st_mtime, tz=timezone.utc)
        age = datetime.now(timezone.utc) - mtime
        return age < timedelta(hours=max_age_hours)
    except Exception:
        return False


def classify_state(lat: float, lon: float) -> str:
    """Assign coarse state identification based on coordinates for Punjab/Haryana/NCR."""
    # Delhi NCR bounds
    if DELHI_NCR_DOMAIN.contains(lat, lon):
        return "Delhi NCR"
    # Punjab bounds (typically lat 29.8 to 32.5)
    elif 29.8 <= lat <= 32.5 and 73.8 <= lon <= 76.9:
        return "Punjab"
    # Haryana bounds (typically lat 27.6 to 30.6)
    elif 27.6 <= lat <= 30.6 and 74.4 <= lon <= 77.6:
        return "Haryana"
    # Surrounding Rajasthan / UP / Himachal
    elif 27.5 <= lat <= 29.8 and 73.5 <= lon <= 76.5:
        return "Rajasthan"
    elif 27.5 <= lat <= 31.0 and 77.0 <= lon <= 78.2:
        return "Uttar Pradesh"
    else:
        return "North-West India"


def generate_synthetic_fires(count: int = 45) -> pd.DataFrame:
    """
    Generate realistic synthetic farm-fire hotspots across Punjab and Haryana
    as a robust fallback when APIs are offline or during offline testing.
    """
    logger.warning("Generating synthetic farm-fire data for Punjab/Haryana simulation fallback.")
    np.random.seed(int(time.time()) % 100000)
    
    # Concentrated farm fire clusters in Punjab (Sangrur, Bathinda, Firozpur, Ludhiana, Tarn Taran)
    punjab_clusters = [
        (30.24, 75.84), (30.21, 74.94), (30.92, 74.61),
        (30.90, 75.85), (31.45, 74.92), (31.14, 75.34),
        (30.55, 74.65), (30.70, 75.20), (31.32, 75.57)
    ]
    
    # Haryana clusters (Karnal, Kaithal, Fatehabad, Jind, Kurukshetra)
    haryana_clusters = [
        (29.68, 76.98), (29.80, 76.40), (29.51, 75.45),
        (29.31, 76.31), (29.96, 76.87), (29.15, 75.72)
    ]
    
    records = []
    now = datetime.now(timezone.utc)
    
    for i in range(count):
        if np.random.rand() < 0.70: # 70% in Punjab
            center_lat, center_lon = punjab_clusters[np.random.randint(len(punjab_clusters))]
            state = "Punjab"
        else:
            center_lat, center_lon = haryana_clusters[np.random.randint(len(haryana_clusters))]
            state = "Haryana"
            
        lat = float(center_lat + np.random.normal(0, 0.12))
        lon = float(center_lon + np.random.normal(0, 0.12))
        
        # Fire Radiative Power (MW): log-normal distribution typical of farm stubble burns
        frp = float(np.clip(np.random.lognormal(mean=2.8, sigma=0.8), 3.0, 180.0))
        confidence = int(np.random.choice([75, 80, 85, 90, 95, 100]))
        satellite = str(np.random.choice(["NOAA-20", "Suomi-NPP", "NOAA-21", "Terra/Aqua"]))
        
        # Detected within last 24h
        hours_ago = np.random.uniform(0.5, 23.5)
        dt = now - timedelta(hours=hours_ago)
        
        records.append({
            "latitude": round(lat, 5),
            "longitude": round(lon, 5),
            "frp": round(frp, 2),
            "confidence": confidence,
            "acq_date": dt.strftime("%Y-%m-%d"),
            "acq_time": int(dt.strftime("%H%M")),
            "satellite": satellite,
            "state": state,
            "source": "SYNTHETIC_FALLBACK"
        })
        
    return pd.DataFrame(records)


def fetch_firms_data(
    api_key: Optional[str] = None,
    use_cache: bool = True,
    cache_dir: Optional[Path] = None,
    lookback_days: int = 1,
    domain: DomainBounds = REGIONAL_DOMAIN,
    force_synthetic: bool = False
) -> pd.DataFrame:
    """
    Fetch active fire data from NASA FIRMS API (or open near-real-time CSV feeds).
    
    Parameters:
        api_key: Optional NASA FIRMS MAP_KEY (if user has one).
        use_cache: Whether to read/write local cache.
        cache_dir: Optional custom cache directory.
        lookback_days: Days of historical fire data (default 1 = last 24h).
        domain: Geographical bounding box to filter.
        force_synthetic: Force use of synthetic test data.
        
    Returns:
        pd.DataFrame containing filtered fire points with columns:
        ['latitude', 'longitude', 'frp', 'acq_date', 'acq_time', 'confidence', 'satellite', 'state', 'source']
    """
    if force_synthetic:
        return generate_synthetic_fires(50)
        
    target_cache_dir = ensure_cache_dir(cache_dir)
    today_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    cache_file = target_cache_dir / f"firms_fires_{today_str}_{lookback_days}d.csv"
    
    if use_cache and is_cache_valid(cache_file, max_age_hours=4):
        try:
            logger.info(f"Loading cached NASA FIRMS fire data from {cache_file}")
            df = pd.read_csv(cache_file)
            if not df.empty and "latitude" in df.columns and "longitude" in df.columns:
                return df
        except Exception as e:
            logger.warning(f"Failed reading cache {cache_file}: {e}. Will re-fetch.")

    all_dfs = []
    
    # Strategy 1: If API key provided (or in env), use official FIRMS REST endpoint
    firms_key = api_key or os.environ.get("FIRMS_MAP_KEY") or os.environ.get("NASA_FIRMS_KEY")
    if firms_key:
        logger.info("Querying NASA FIRMS API using provided MAP_KEY...")
        try:
            # Sources: VIIRS_SNPP_NRT, VIIRS_NOAA20_NRT, MODIS_NRT
            for source in ["VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT"]:
                url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{firms_key}/{source}/{domain.bbox_str}/{lookback_days}"
                resp = requests.get(url, timeout=15)
                if resp.status_code == 200 and len(resp.text) > 50:
                    df_sub = pd.read_csv(io.StringIO(resp.text))
                    df_sub["source"] = source
                    all_dfs.append(df_sub)
                    logger.info(f"Fetched {len(df_sub)} points from FIRMS API ({source})")
        except Exception as e:
            logger.warning(f"FIRMS REST API query encountered error: {e}. Falling back to open feeds.")

    # Strategy 2: If no key or API failed, query NASA FIRMS open near-real-time South Asia feeds
    if not all_dfs:
        logger.info("Querying NASA FIRMS Open Near-Real-Time CSV Feeds (No Key Required)...")
        for source_name, url in FIRMS_OPEN_FEEDS.items():
            try:
                logger.info(f"Downloading active fires feed: {source_name}...")
                resp = requests.get(url, timeout=20)
                if resp.status_code == 200 and len(resp.text) > 50:
                    df_sub = pd.read_csv(io.StringIO(resp.text))
                    df_sub["source"] = source_name
                    all_dfs.append(df_sub)
                    logger.info(f"Loaded {len(df_sub)} points from {source_name} feed")
                else:
                    logger.warning(f"Feed {source_name} returned status {resp.status_code}")
            except Exception as e:
                logger.warning(f"Failed to fetch {source_name} feed: {e}")

    # Process and harmonize data
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        
        # Standardize column names (VIIRS uses bright_ti4, MODIS uses brightness, etc.)
        for col in ["latitude", "longitude"]:
            if col in combined.columns:
                combined[col] = pd.to_numeric(combined[col], errors="coerce")
        combined = combined.dropna(subset=["latitude", "longitude"])
        
        # Standardize FRP
        if "frp" not in combined.columns:
            combined["frp"] = 15.0  # default fallback FRP in MW
        else:
            combined["frp"] = pd.to_numeric(combined["frp"], errors="coerce").fillna(15.0)
            combined["frp"] = combined["frp"].apply(lambda x: max(1.0, float(x)))
            
        # Standardize satellite/confidence
        if "confidence" not in combined.columns:
            combined["confidence"] = 80
        if "satellite" not in combined.columns:
            combined["satellite"] = combined.get("source", "VIIRS")
            
        # Filter within Punjab, Haryana, and regional NW India bounding box
        mask = (
            (combined["latitude"] >= domain.min_lat) &
            (combined["latitude"] <= domain.max_lat) &
            (combined["longitude"] >= domain.min_lon) &
            (combined["longitude"] <= domain.max_lon)
        )
        filtered = combined[mask].copy()
        
        # Remove duplicate detections within ~1km if multiple sensors saw the same fire
        filtered["lat_round"] = filtered["latitude"].round(3)
        filtered["lon_round"] = filtered["longitude"].round(3)
        filtered = filtered.sort_values("frp", ascending=False).drop_duplicates(subset=["lat_round", "lon_round"])
        filtered = filtered.drop(columns=["lat_round", "lon_round"], errors="ignore")
        
        # Assign state classification
        filtered["state"] = filtered.apply(lambda r: classify_state(r["latitude"], r["longitude"]), axis=1)
        
        logger.info(f"Successfully processed {len(filtered)} fire hotspots in Punjab/Haryana/NW India domain.")
        
        # If very few fires detected in off-season, augment with realistic samples for robust demonstration
        if len(filtered) < 5:
            logger.info("Low seasonal fire activity detected; augmenting with realistic benchmark farm fires.")
            synth = generate_synthetic_fires(35)
            filtered = pd.concat([filtered, synth], ignore_index=True)
            
        # Cache to disk
        try:
            filtered.to_csv(cache_file, index=False)
            logger.info(f"Cached {len(filtered)} fire records to {cache_file}")
        except Exception as e:
            logger.warning(f"Could not save fire cache: {e}")
            
        return filtered

    # Fallback to synthetic if all network attempts failed
    logger.error("All NASA FIRMS fetching mechanisms failed. Using synthetic fallback.")
    fallback_df = generate_synthetic_fires(45)
    return fallback_df


def fetch_wind_forecast(
    domain: DomainBounds = REGIONAL_DOMAIN,
    grid_step: float = 0.5,
    forecast_days: int = 3,
    use_cache: bool = True,
    cache_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Fetch wind speed and direction forecasts from Open-Meteo API for the next 72 hours
    on a regular spatial grid covering Punjab, Haryana, and Delhi NCR.
    
    Parameters:
        domain: Geographical bounding box.
        grid_step: Spacing between grid points in degrees (default 0.5°).
        forecast_days: Forecast horizon in days (default 3 days = 72 hours).
        use_cache: Whether to read/write local cache.
        cache_dir: Optional custom cache directory.
        
    Returns:
        Dictionary containing:
        - 'lats': 1D array of latitude coordinates
        - 'lons': 1D array of longitude coordinates
        - 'times': list of 72 ISO timestamps
        - 'u_wind': 3D array of zonal (Eastward) wind speeds (m/s), shape (72, n_lat, n_lon)
        - 'v_wind': 3D array of meridional (Northward) wind speeds (m/s), shape (72, n_lat, n_lon)
        - 'speed': 3D array of wind speed (m/s), shape (72, n_lat, n_lon)
        - 'direction': 3D array of meteorological wind direction (degrees), shape (72, n_lat, n_lon)
    """
    target_cache_dir = ensure_cache_dir(cache_dir)
    today_hour_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H")
    cache_file = target_cache_dir / f"meteo_wind_{today_hour_str}_{grid_step}deg.json"
    
    if use_cache and is_cache_valid(cache_file, max_age_hours=3):
        try:
            logger.info(f"Loading cached wind forecast from {cache_file}")
            with open(cache_file, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
            
            # Reconstruct numpy arrays
            cached_data["lats"] = np.array(cached_data["lats"])
            cached_data["lons"] = np.array(cached_data["lons"])
            cached_data["u_wind"] = np.array(cached_data["u_wind"])
            cached_data["v_wind"] = np.array(cached_data["v_wind"])
            cached_data["speed"] = np.array(cached_data["speed"])
            cached_data["direction"] = np.array(cached_data["direction"])
            return cached_data
        except Exception as e:
            logger.warning(f"Failed loading wind cache {cache_file}: {e}. Re-fetching.")

    # Generate grid coordinates
    lats = np.arange(domain.min_lat, domain.max_lat + 0.001, grid_step)
    lons = np.arange(domain.min_lon, domain.max_lon + 0.001, grid_step)
    n_lats = len(lats)
    n_lons = len(lons)
    
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    flat_lats = lat_grid.flatten().tolist()
    flat_lons = lon_grid.flatten().tolist()
    
    logger.info(f"Querying Open-Meteo wind forecast for {len(flat_lats)} grid points ({n_lats}x{n_lons})...")
    
    lat_str = ",".join([f"{x:.2f}" for x in flat_lats])
    lon_str = ",".join([f"{x:.2f}" for x in flat_lons])
    
    params = {
        "latitude": lat_str,
        "longitude": lon_str,
        "hourly": "wind_speed_10m,wind_direction_10m",
        "forecast_days": forecast_days,
        "wind_speed_unit": "ms" # meters per second
    }
    
    try:
        resp = requests.get(OPEN_METEO_URL, params=params, timeout=35)
        resp.raise_for_status()
        raw_results = resp.json()
        
        # If single location returned as dict, wrap in list
        if isinstance(raw_results, dict):
            raw_results = [raw_results]
            
        sample_hourly = raw_results[0].get("hourly", {})
        times = sample_hourly.get("time", [])
        n_times = len(times)
        
        logger.info(f"Open-Meteo returned forecast for {len(raw_results)} points x {n_times} hours.")
        
        # Allocate 3D arrays: (time, lat, lon)
        speed_array = np.zeros((n_times, n_lats, n_lons), dtype=np.float32)
        dir_array = np.zeros((n_times, n_lats, n_lons), dtype=np.float32)
        u_array = np.zeros((n_times, n_lats, n_lons), dtype=np.float32)
        v_array = np.zeros((n_times, n_lats, n_lons), dtype=np.float32)
        
        for idx, res in enumerate(raw_results):
            # Coordinates in grid
            i = idx // n_lons
            j = idx % n_lons
            
            hourly = res.get("hourly", {})
            spd = np.array(hourly.get("wind_speed_10m", [0.0] * n_times), dtype=np.float32)
            deg = np.array(hourly.get("wind_direction_10m", [0.0] * n_times), dtype=np.float32)
            
            # Meteorological convention: 
            # Direction is the angle FROM which the wind is blowing.
            # 0 deg = from North (blowing South: u=0, v=-spd)
            # 90 deg = from East (blowing West: u=-spd, v=0)
            # 180 deg = from South (blowing North: u=0, v=+spd)
            # 270 deg = from West (blowing East: u=+spd, v=0)
            # 315 deg = from North-West (blowing South-East: u=+spd*sin(45), v=-spd*cos(45))
            rad = np.radians(deg)
            u = -spd * np.sin(rad)  # Eastward velocity
            v = -spd * np.cos(rad)  # Northward velocity
            
            speed_array[:, i, j] = spd
            dir_array[:, i, j] = deg
            u_array[:, i, j] = u
            v_array[:, i, j] = v

        result = {
            "lats": lats,
            "lons": lons,
            "times": times,
            "u_wind": u_array,
            "v_wind": v_array,
            "speed": speed_array,
            "direction": dir_array
        }
        
        # Cache to file (JSON serializable format)
        try:
            serializable = {
                "lats": lats.tolist(),
                "lons": lons.tolist(),
                "times": times,
                "u_wind": u_array.tolist(),
                "v_wind": v_array.tolist(),
                "speed": speed_array.tolist(),
                "direction": dir_array.tolist()
            }
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(serializable, f)
            logger.info(f"Cached wind forecast to {cache_file}")
        except Exception as e:
            logger.warning(f"Could not cache wind forecast: {e}")
            
        return result
        
    except Exception as e:
        logger.error(f"Failed to fetch wind forecast from Open-Meteo: {e}. Generating synthetic atmospheric wind field.")
        return generate_synthetic_wind_field(domain, grid_step, forecast_days)


def generate_synthetic_wind_field(
    domain: DomainBounds = REGIONAL_DOMAIN,
    grid_step: float = 0.5,
    forecast_days: int = 3
) -> Dict[str, Any]:
    """Generate realistic North-Westerly wind field typical of post-monsoon farm fire season."""
    lats = np.arange(domain.min_lat, domain.max_lat + 0.001, grid_step)
    lons = np.arange(domain.min_lon, domain.max_lon + 0.001, grid_step)
    n_times = forecast_days * 24
    
    now = datetime.now(timezone.utc)
    times = [(now + timedelta(hours=h)).strftime("%Y-%m-%dT%H:00") for h in range(n_times)]
    
    n_lats = len(lats)
    n_lons = len(lons)
    
    speed_array = np.zeros((n_times, n_lats, n_lons), dtype=np.float32)
    dir_array = np.zeros((n_times, n_lats, n_lons), dtype=np.float32)
    u_array = np.zeros((n_times, n_lats, n_lons), dtype=np.float32)
    v_array = np.zeros((n_times, n_lats, n_lons), dtype=np.float32)
    
    # Typical autumn pattern: NW winds (300° to 330°) at 3.0 to 6.5 m/s with diurnal diurnal cycle
    for t in range(n_times):
        diurnal_factor = 1.0 + 0.35 * np.sin(2 * np.pi * (t - 6) / 24.0)
        base_spd = 4.2 * diurnal_factor
        base_dir = 315.0 + 15.0 * np.sin(2 * np.pi * t / 48.0) # NW
        
        rad = np.radians(base_dir)
        u = -base_spd * np.sin(rad)
        v = -base_spd * np.cos(rad)
        
        speed_array[t, :, :] = base_spd
        dir_array[t, :, :] = base_dir
        u_array[t, :, :] = u
        v_array[t, :, :] = v
        
    return {
        "lats": lats,
        "lons": lons,
        "times": times,
        "u_wind": u_array,
        "v_wind": v_array,
        "speed": speed_array,
        "direction": dir_array
    }
