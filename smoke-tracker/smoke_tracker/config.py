"""
Configuration settings, geographical domain definitions, and simulation parameters
for the Delhi NCR Farm-Fire Smoke Advection Simulation package.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Tuple, List, Dict


# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
OUTPUT_DIR = DATA_DIR / "outputs"


@dataclass
class DomainBounds:
    """Geographical bounding box definition."""
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float

    @property
    def bbox_str(self) -> str:
        """Returns min_lon,min_lat,max_lon,max_lat (FIRMS API format)."""
        return f"{self.min_lon},{self.min_lat},{self.max_lon},{self.max_lat}"

    def contains(self, lat: float, lon: float) -> bool:
        """Check if a coordinate is within the bounding box."""
        return self.min_lat <= lat <= self.max_lat and self.min_lon <= lon <= self.max_lon


# Domain Definitions
# Regional domain covers Punjab, Haryana, Himachal foothills, Chandigarh, parts of Rajasthan & Delhi NCR
REGIONAL_DOMAIN = DomainBounds(
    min_lat=27.5,
    max_lat=32.6,
    min_lon=73.5,
    max_lon=78.2
)

# Delhi NCR high-resolution focus domain
DELHI_NCR_DOMAIN = DomainBounds(
    min_lat=28.20,
    max_lat=28.95,
    min_lon=76.75,
    max_lon=77.65
)

# Key monitoring stations and reference points in Delhi NCR
DELHI_STATIONS = [
    {"name": "Anand Vihar", "lat": 28.6508, "lon": 77.3152, "type": "Hotspot Station"},
    {"name": "Connaught Place (Central)", "lat": 28.6315, "lon": 77.2167, "type": "Urban Core"},
    {"name": "ITO", "lat": 28.6289, "lon": 77.2410, "type": "Traffic Core"},
    {"name": "IGI Airport (T3)", "lat": 28.5562, "lon": 77.1000, "type": "Western Corridor"},
    {"name": "RK Puram", "lat": 28.5638, "lon": 77.1864, "type": "South Delhi"},
    {"name": "Noida Sector 62", "lat": 28.6258, "lon": 77.3649, "type": "East NCR"},
    {"name": "Gurugram Cyber City", "lat": 28.4950, "lon": 77.0895, "type": "South-West NCR"},
    {"name": "Faridabad", "lat": 28.4089, "lon": 77.3178, "type": "South NCR"},
]


@dataclass
class SimulationConfig:
    """Atmospheric physics, grid resolution, and simulation parameters."""
    forecast_hours: int = 72
    grid_res_delhi: Tuple[int, int] = (50, 50)  # (n_lat, n_lon) for Delhi NCR
    grid_res_regional: Tuple[int, int] = (60, 60)  # (n_lat, n_lon) for regional map
    
    # Wind field grid spacing in degrees for Open-Meteo queries
    meteo_grid_step: float = 0.5
    
    # Physics parameters for Gaussian dispersion and puff dynamics
    puff_release_interval_hours: int = 1  # How often active fires release new smoke puffs
    active_burn_duration_hours: int = 18   # Expected active burning duration per detected fire
    frp_scale_factor: float = 12.0        # Multiplier from FRP (MW) to initial smoke mass Q0
    base_sigma_deg: float = 0.04          # Initial puff horizontal standard deviation (deg ~ 4.4 km)
    dispersion_rate: float = 0.008        # Horizontal diffusion growth rate (deg / sqrt(hour))
    atmospheric_decay_rate: float = 0.025 # Scavenging/deposition loss rate per hour (half-life ~ 28h)
    
    # Cache settings
    use_cache: bool = True
    cache_ttl_hours: int = 6


# Default configuration instance
DEFAULT_CONFIG = SimulationConfig()
