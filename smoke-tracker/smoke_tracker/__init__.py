"""
Delhi NCR Farm-Fire Smoke Advection & Dispersion Simulation Package.
"""

from smoke_tracker.config import (
    DomainBounds,
    REGIONAL_DOMAIN,
    DELHI_NCR_DOMAIN,
    DELHI_STATIONS,
    SimulationConfig,
    DEFAULT_CONFIG,
    CACHE_DIR,
    OUTPUT_DIR
)
from smoke_tracker.fetch_data import (
    fetch_firms_data,
    fetch_wind_forecast,
    generate_synthetic_fires,
    generate_synthetic_wind_field
)
from smoke_tracker.advection import (
    SmokeAdvectionSimulator,
    SimulationResult,
    save_to_numpy,
    save_to_geojson
)
from smoke_tracker.visualize import (
    create_leaflet_map
)

__version__ = "1.0.0"
__all__ = [
    "DomainBounds",
    "REGIONAL_DOMAIN",
    "DELHI_NCR_DOMAIN",
    "DELHI_STATIONS",
    "SimulationConfig",
    "DEFAULT_CONFIG",
    "CACHE_DIR",
    "OUTPUT_DIR",
    "fetch_firms_data",
    "fetch_wind_forecast",
    "generate_synthetic_fires",
    "generate_synthetic_wind_field",
    "SmokeAdvectionSimulator",
    "SimulationResult",
    "save_to_numpy",
    "save_to_geojson",
    "create_leaflet_map"
]
