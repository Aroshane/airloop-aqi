"""
Lagrangian Puff Advection and Gaussian Dispersion Simulation Engine.
Simulates hour-by-hour farm-fire smoke transport from Punjab and Haryana across North-West India into Delhi NCR.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from smoke_tracker.config import (
    DomainBounds,
    REGIONAL_DOMAIN,
    DELHI_NCR_DOMAIN,
    DELHI_STATIONS,
    SimulationConfig,
    DEFAULT_CONFIG,
    OUTPUT_DIR
)

logger = logging.getLogger("smoke_tracker.advection")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

# Earth geometry conversion constants
METERS_PER_DEG_LAT = 111139.0
SECONDS_PER_HOUR = 3600.0


@dataclass
class SmokePuff:
    """Represents a discrete Lagrangian smoke puff emitted from a fire hotspot."""
    fire_id: int
    lat: float
    lon: float
    t_emit: int
    q0: float
    state: str
    satellite: str
    frp: float
    is_active: bool = True


@dataclass
class SimulationResult:
    """Holds the complete multi-hour advection simulation results and grids."""
    times: List[str]
    forecast_hours: int
    
    # Delhi NCR Target High-Res Grid
    delhi_lats: np.ndarray
    delhi_lons: np.ndarray
    delhi_smoke_density: np.ndarray  # Shape: (T, n_lat, n_lon)
    
    # Regional Domain Grid (Punjab, Haryana, Delhi NCR)
    regional_lats: np.ndarray
    regional_lons: np.ndarray
    regional_smoke_density: np.ndarray  # Shape: (T, n_reg_lat, n_reg_lon)
    
    # Monitoring Stations Time Series
    station_series: Dict[str, List[float]]
    
    # Summary Metrics
    delhi_mean_series: List[float]
    delhi_max_series: List[float]
    peak_delhi_hour: int
    peak_delhi_time: str
    peak_delhi_density: float
    total_active_fires: int
    contributing_fire_count: int
    
    # Puff Trajectories for Animation
    hourly_puff_positions: List[List[Dict[str, Any]]]


class SmokeAdvectionSimulator:
    """
    Simulates Lagrangian puff transport and 2D Gaussian dispersion over time.
    """
    def __init__(self, config: SimulationConfig = DEFAULT_CONFIG):
        self.config = config
        
    def _create_wind_interpolator(self, wind_data: Dict[str, Any]):
        """Build regular grid interpolators for u (Eastward) and v (Northward) wind components."""
        times_idx = np.arange(len(wind_data["times"]))
        lats = wind_data["lats"]
        lons = wind_data["lons"]
        
        u_array = wind_data["u_wind"]
        v_array = wind_data["v_wind"]
        
        # Grid coordinates for RegularGridInterpolator: (time, lat, lon)
        u_interp = RegularGridInterpolator(
            (times_idx, lats, lons),
            u_array,
            bounds_error=False,
            fill_value=None  # Extrapolates to nearest boundary
        )
        
        v_interp = RegularGridInterpolator(
            (times_idx, lats, lons),
            v_array,
            bounds_error=False,
            fill_value=None
        )
        
        return u_interp, v_interp

    def simulate(
        self,
        fires_df: pd.DataFrame,
        wind_data: Dict[str, Any],
        delhi_domain: DomainBounds = DELHI_NCR_DOMAIN,
        regional_domain: DomainBounds = REGIONAL_DOMAIN
    ) -> SimulationResult:
        """
        Execute the 72-hour smoke advection and Gaussian dispersion simulation.
        
        Parameters:
            fires_df: DataFrame of active fire points from NASA FIRMS.
            wind_data: Dictionary of Open-Meteo wind field forecast.
            delhi_domain: Bounding box for Delhi NCR high-res grid.
            regional_domain: Bounding box for regional domain.
            
        Returns:
            SimulationResult containing 3D density arrays, metrics, and station time series.
        """
        times = wind_data["times"]
        n_hours = min(len(times), self.config.forecast_hours)
        times = times[:n_hours]
        
        logger.info(f"Initializing smoke simulation for {len(fires_df)} fire sources over {n_hours} hours...")
        
        # 1. Build wind vector interpolators
        u_interp, v_interp = self._create_wind_interpolator(wind_data)
        
        # 2. Build spatial grids
        n_delhi_lat, n_delhi_lon = self.config.grid_res_delhi
        delhi_lats = np.linspace(delhi_domain.min_lat, delhi_domain.max_lat, n_delhi_lat)
        delhi_lons = np.linspace(delhi_domain.min_lon, delhi_domain.max_lon, n_delhi_lon)
        DLAT, DLON = np.meshgrid(delhi_lats, delhi_lons, indexing="ij")
        
        n_reg_lat, n_reg_lon = self.config.grid_res_regional
        regional_lats = np.linspace(regional_domain.min_lat, regional_domain.max_lat, n_reg_lat)
        regional_lons = np.linspace(regional_domain.min_lon, regional_domain.max_lon, n_reg_lon)
        RLAT, RLON = np.meshgrid(regional_lats, regional_lons, indexing="ij")
        
        # Allocate output grids: shape (T, n_lat, n_lon)
        delhi_smoke = np.zeros((n_hours, n_delhi_lat, n_delhi_lon), dtype=np.float32)
        regional_smoke = np.zeros((n_hours, n_reg_lat, n_reg_lon), dtype=np.float32)
        
        # 3. Instantiate Lagrangian Puffs
        puffs: List[SmokePuff] = []
        puff_interval = self.config.puff_release_interval_hours
        burn_duration = self.config.active_burn_duration_hours
        
        for idx, row in fires_df.iterrows():
            lat = float(row["latitude"])
            lon = float(row["longitude"])
            frp = float(row.get("frp", 15.0))
            state = str(row.get("state", "Punjab"))
            satellite = str(row.get("satellite", "VIIRS"))
            
            # Initial mass proportional to FRP (MW)
            q_base = frp * self.config.frp_scale_factor
            
            # Release puffs across the active burning window (e.g. t=0 to burn_duration)
            for t_emit in range(0, burn_duration, puff_interval):
                # Diurnal burn cycle: peak burning in early afternoon (13:00 - 16:00)
                diurnal_scale = 1.0 + 0.5 * np.cos(2 * np.pi * ((t_emit + 14) % 24) / 24.0)
                q0 = q_base * diurnal_scale
                
                puffs.append(SmokePuff(
                    fire_id=idx,
                    lat=lat,
                    lon=lon,
                    t_emit=t_emit,
                    q0=q0,
                    state=state,
                    satellite=satellite,
                    frp=frp
                ))
                
        logger.info(f"Created {len(puffs)} Lagrangian smoke puffs. Simulating hour-by-hour advection...")
        
        # 4. Step through each hour
        hourly_puff_positions: List[List[Dict[str, Any]]] = []
        contributing_fire_ids = set()
        
        for t in range(n_hours):
            # Active puffs up to current hour
            active_puffs = [p for p in puffs if p.t_emit <= t and p.is_active]
            puff_snapshot = []
            
            for p in active_puffs:
                age = t - p.t_emit
                
                # Advect puff with local wind vector if age > 0
                if age > 0:
                    query_point = np.array([[t, p.lat, p.lon]])
                    u_val = float(u_interp(query_point)[0])
                    v_val = float(v_interp(query_point)[0])
                    
                    # Convert velocity (m/s) to displacement in degrees
                    # dx = u * dt, dy = v * dt
                    d_lat = (v_val * SECONDS_PER_HOUR) / METERS_PER_DEG_LAT
                    meters_per_deg_lon = METERS_PER_DEG_LAT * np.cos(np.radians(p.lat))
                    d_lon = (u_val * SECONDS_PER_HOUR) / max(meters_per_deg_lon, 1000.0)
                    
                    p.lat += d_lat
                    p.lon += d_lon
                    
                # Deactivate if out of regional bounding box + margin
                if not (regional_domain.min_lat - 1.0 <= p.lat <= regional_domain.max_lat + 1.0 and
                        regional_domain.min_lon - 1.0 <= p.lon <= regional_domain.max_lon + 1.0):
                    p.is_active = False
                    continue
                    
                # Gaussian dispersion standard deviation sigma (in degrees)
                # Grows with puff age: sigma(tau) = sigma_0 + alpha * sqrt(tau + 1)
                sigma = self.config.base_sigma_deg + self.config.dispersion_rate * np.sqrt(age + 1.0)
                sigma2 = 2.0 * (sigma ** 2)
                
                # Atmospheric scavenging / deposition decay: Q(tau) = Q0 * exp(-lambda * tau)
                q_decayed = p.q0 * np.exp(-self.config.atmospheric_decay_rate * age)
                
                # Record snapshot for web animation (downsample snapshot for UI efficiency)
                if p.fire_id % 2 == 0 or age == 0:
                    puff_snapshot.append({
                        "id": p.fire_id,
                        "lat": round(p.lat, 4),
                        "lon": round(p.lon, 4),
                        "age": age,
                        "intensity": round(q_decayed, 1),
                        "state": p.state,
                        "frp": p.frp
                    })
                
                # -----------------------------------------------------------------
                # Gaussian Kernel on Delhi NCR Target Grid
                # -----------------------------------------------------------------
                # Check if puff is close enough to Delhi to compute density
                dlat_delhi = DLAT - p.lat
                # Metric aspect correction for longitude
                cos_lat = np.cos(np.radians(p.lat))
                dlon_delhi = (DLON - p.lon) * cos_lat
                dist2_delhi = (dlat_delhi ** 2) + (dlon_delhi ** 2)
                
                # Cutoff at 3.5 * sigma (~99.9% of mass) to optimize computation
                cutoff_radius2 = (3.5 * sigma) ** 2
                in_cutoff_delhi = dist2_delhi < cutoff_radius2
                
                if np.any(in_cutoff_delhi):
                    contributing_fire_ids.add(p.fire_id)
                    kernel_val = (q_decayed / (2.0 * np.pi * (sigma ** 2))) * np.exp(-dist2_delhi[in_cutoff_delhi] / sigma2)
                    delhi_smoke[t][in_cutoff_delhi] += kernel_val.astype(np.float32)
                    
                # -----------------------------------------------------------------
                # Gaussian Kernel on Regional Map Grid
                # -----------------------------------------------------------------
                dlat_reg = RLAT - p.lat
                dlon_reg = (RLON - p.lon) * cos_lat
                dist2_reg = (dlat_reg ** 2) + (dlon_reg ** 2)
                in_cutoff_reg = dist2_reg < cutoff_radius2
                
                if np.any(in_cutoff_reg):
                    kernel_val_reg = (q_decayed / (2.0 * np.pi * (sigma ** 2))) * np.exp(-dist2_reg[in_cutoff_reg] / sigma2)
                    regional_smoke[t][in_cutoff_reg] += kernel_val_reg.astype(np.float32)

            hourly_puff_positions.append(puff_snapshot)

        # 5. Extract Station Time Series
        station_series: Dict[str, List[float]] = {}
        for st in DELHI_STATIONS:
            s_lat, s_lon = st["lat"], st["lon"]
            # Find closest Delhi grid cell
            lat_idx = int(np.argmin(np.abs(delhi_lats - s_lat)))
            lon_idx = int(np.argmin(np.abs(delhi_lons - s_lon)))
            series = delhi_smoke[:, lat_idx, lon_idx].tolist()
            station_series[st["name"]] = [round(v, 2) for v in series]

        # 6. Compute summary metrics
        delhi_mean_series = [round(float(delhi_smoke[t].mean()), 2) for t in range(n_hours)]
        delhi_max_series = [round(float(delhi_smoke[t].max()), 2) for t in range(n_hours)]
        
        peak_delhi_hour = int(np.argmax(delhi_mean_series))
        peak_delhi_time = times[peak_delhi_hour]
        peak_delhi_density = delhi_max_series[peak_delhi_hour]
        
        logger.info(f"Simulation completed! Peak Delhi smoke impact predicted at Hour {peak_delhi_hour} ({peak_delhi_time})")
        logger.info(f"Contributing farm fires: {len(contributing_fire_ids)} of {len(fires_df)}")
        
        return SimulationResult(
            times=times,
            forecast_hours=n_hours,
            delhi_lats=delhi_lats,
            delhi_lons=delhi_lons,
            delhi_smoke_density=delhi_smoke,
            regional_lats=regional_lats,
            regional_lons=regional_lons,
            regional_smoke_density=regional_smoke,
            station_series=station_series,
            delhi_mean_series=delhi_mean_series,
            delhi_max_series=delhi_max_series,
            peak_delhi_hour=peak_delhi_hour,
            peak_delhi_time=peak_delhi_time,
            peak_delhi_density=peak_delhi_density,
            total_active_fires=len(fires_df),
            contributing_fire_count=len(contributing_fire_ids),
            hourly_puff_positions=hourly_puff_positions
        )


def save_to_numpy(
    result: SimulationResult,
    output_path: Optional[Path] = None
) -> Path:
    """
    Save simulation grids and metadata as compressed NumPy archive (.npz).
    
    Parameters:
        result: SimulationResult instance.
        output_path: Target destination path.
        
    Returns:
        Path to saved .npz file.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    target = output_path or (OUTPUT_DIR / "delhi_smoke_simulation_72h.npz")
    
    np.savez_compressed(
        target,
        times=np.array(result.times),
        delhi_lats=result.delhi_lats,
        delhi_lons=result.delhi_lons,
        delhi_smoke_density=result.delhi_smoke_density,
        regional_lats=result.regional_lats,
        regional_lons=result.regional_lons,
        regional_smoke_density=result.regional_smoke_density,
        delhi_mean_series=np.array(result.delhi_mean_series),
        delhi_max_series=np.array(result.delhi_max_series),
        peak_hour=result.peak_delhi_hour
    )
    logger.info(f"Saved NumPy simulation results to {target}")
    return target


def save_to_geojson(
    result: SimulationResult,
    output_path: Optional[Path] = None,
    min_density_threshold: float = 0.5
) -> Path:
    """
    Export Delhi NCR hourly smoke density time series to GeoJSON format.
    
    Parameters:
        result: SimulationResult instance.
        output_path: Target GeoJSON file path.
        min_density_threshold: Minimum density threshold to filter noise.
        
    Returns:
        Path to saved GeoJSON file.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    target = output_path or (OUTPUT_DIR / "delhi_smoke_timeseries.geojson")
    
    features = []
    d_lats = result.delhi_lats
    d_lons = result.delhi_lons
    d_lat_step = (d_lats[1] - d_lats[0]) / 2.0
    d_lon_step = (d_lons[1] - d_lons[0]) / 2.0
    
    for t_idx, timestamp in enumerate(result.times):
        grid_t = result.delhi_smoke_density[t_idx]
        
        # Find active cells
        active_indices = np.argwhere(grid_t >= min_density_threshold)
        
        for (i, j) in active_indices:
            lat_center = float(d_lats[i])
            lon_center = float(d_lons[j])
            val = float(grid_t[i, j])
            
            # Bounding box polygon for grid cell
            poly_coords = [[
                [round(lon_center - d_lon_step, 5), round(lat_center - d_lat_step, 5)],
                [round(lon_center + d_lon_step, 5), round(lat_center - d_lat_step, 5)],
                [round(lon_center + d_lon_step, 5), round(lat_center + d_lat_step, 5)],
                [round(lon_center - d_lon_step, 5), round(lat_center + d_lat_step, 5)],
                [round(lon_center - d_lon_step, 5), round(lat_center - d_lat_step, 5)]
            ]]
            
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": poly_coords
                },
                "properties": {
                    "hour": t_idx,
                    "timestamp": timestamp,
                    "smoke_density": round(val, 2),
                    "pm25_estimate": round(val * 0.85, 2),
                    "latitude": round(lat_center, 4),
                    "longitude": round(lon_center, 4)
                }
            })
            
    geojson_doc = {
        "type": "FeatureCollection",
        "metadata": {
            "title": "Delhi NCR Farm-Fire Smoke Advection Forecast (72h)",
            "generated_at": result.times[0] if result.times else "",
            "forecast_hours": result.forecast_hours,
            "peak_hour": result.peak_delhi_hour,
            "peak_time": result.peak_delhi_time,
            "total_features": len(features)
        },
        "features": features
    }
    
    with open(target, "w", encoding="utf-8") as f:
        json.dump(geojson_doc, f, indent=2)
        
    logger.info(f"Saved GeoJSON time series ({len(features)} cells) to {target}")
    return target
