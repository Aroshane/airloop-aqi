"""
Unit and Integration Test Suite for Delhi NCR Farm-Fire Smoke Tracker.
"""

import sys
import os
import unittest
import numpy as np
import pandas as pd
from pathlib import Path

# Add package root
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE_ROOT))

from smoke_tracker.config import (
    DomainBounds,
    REGIONAL_DOMAIN,
    DELHI_NCR_DOMAIN,
    SimulationConfig
)
from smoke_tracker.fetch_data import (
    generate_synthetic_fires,
    generate_synthetic_wind_field,
    classify_state,
    fetch_firms_data,
    fetch_wind_forecast
)
from smoke_tracker.advection import (
    SmokeAdvectionSimulator,
    SimulationResult,
    save_to_numpy,
    save_to_geojson
)
from smoke_tracker.visualize import create_leaflet_map


class TestSmokeTracker(unittest.TestCase):
    """Comprehensive tests for all components of the smoke tracker module."""

    def test_domain_bounds(self):
        """Test bounding box containment and string representation."""
        bbox = DomainBounds(min_lat=28.0, max_lat=29.0, min_lon=76.0, max_lon=78.0)
        self.assertEqual(bbox.bbox_str, "76.0,28.0,78.0,29.0")
        self.assertTrue(bbox.contains(28.5, 77.0))
        self.assertFalse(bbox.contains(30.0, 77.0))

    def test_state_classification(self):
        """Test coordinate state classifier for Punjab, Haryana, and Delhi."""
        self.assertEqual(classify_state(30.5, 75.0), "Punjab")
        self.assertEqual(classify_state(29.5, 76.5), "Haryana")
        self.assertEqual(classify_state(28.6, 77.2), "Delhi NCR")

    def test_synthetic_fires_generation(self):
        """Verify synthetic fire dataset structure and columns."""
        df = generate_synthetic_fires(count=20)
        self.assertEqual(len(df), 20)
        required_cols = ["latitude", "longitude", "frp", "confidence", "acq_date", "acq_time", "state"]
        for col in required_cols:
            self.assertIn(col, df.columns)
        self.assertTrue((df["frp"] > 0).all())
        self.assertTrue(df["state"].isin(["Punjab", "Haryana"]).all())

    def test_synthetic_wind_field(self):
        """Verify synthetic wind field array dimensions and physical vectors."""
        wind = generate_synthetic_wind_field(REGIONAL_DOMAIN, grid_step=1.0, forecast_days=3)
        self.assertEqual(len(wind["times"]), 72)
        self.assertEqual(wind["speed"].shape[0], 72)
        self.assertEqual(wind["u_wind"].shape[0], 72)
        self.assertEqual(wind["v_wind"].shape[0], 72)
        # Check that speed is approximately sqrt(u^2 + v^2)
        calc_spd = np.sqrt(wind["u_wind"][0]**2 + wind["v_wind"][0]**2)
        np.testing.assert_allclose(calc_spd, wind["speed"][0], rtol=1e-3)

    def test_advection_simulation_execution(self):
        """Test 72-hour Lagrangian advection and Gaussian dispersion engine."""
        fires_df = generate_synthetic_fires(count=15)
        wind_data = generate_synthetic_wind_field(REGIONAL_DOMAIN, grid_step=1.0, forecast_days=3)
        
        config = SimulationConfig(
            forecast_hours=72,
            grid_res_delhi=(20, 20),
            grid_res_regional=(25, 25),
            active_burn_duration_hours=12
        )
        
        sim = SmokeAdvectionSimulator(config=config)
        result = sim.simulate(fires_df, wind_data)
        
        self.assertIsInstance(result, SimulationResult)
        self.assertEqual(result.forecast_hours, 72)
        self.assertEqual(result.delhi_smoke_density.shape, (72, 20, 20))
        self.assertEqual(result.regional_smoke_density.shape, (72, 25, 25))
        self.assertEqual(len(result.delhi_mean_series), 72)
        self.assertEqual(len(result.delhi_max_series), 72)
        self.assertGreaterEqual(result.peak_delhi_density, 0)
        self.assertIn("Anand Vihar", result.station_series)

    def test_serialization_numpy_and_geojson(self, tmp_path=None):
        """Test export to NumPy .npz and GeoJSON time series."""
        test_dir = PACKAGE_ROOT / "data" / "test_outputs"
        test_dir.mkdir(parents=True, exist_ok=True)
        
        fires_df = generate_synthetic_fires(count=10)
        wind_data = generate_synthetic_wind_field(REGIONAL_DOMAIN, grid_step=1.0, forecast_days=2)
        
        config = SimulationConfig(
            forecast_hours=48,
            grid_res_delhi=(15, 15),
            grid_res_regional=(20, 20)
        )
        sim = SmokeAdvectionSimulator(config=config)
        result = sim.simulate(fires_df, wind_data)
        
        # Test NumPy export
        npz_file = test_dir / "test_smoke.npz"
        save_to_numpy(result, npz_file)
        self.assertTrue(npz_file.exists())
        
        loaded = np.load(npz_file)
        self.assertEqual(loaded["delhi_smoke_density"].shape, (48, 15, 15))
        self.assertEqual(len(loaded["times"]), 48)
        
        # Test GeoJSON export
        geojson_file = test_dir / "test_smoke.geojson"
        save_to_geojson(result, geojson_file, min_density_threshold=0.1)
        self.assertTrue(geojson_file.exists())
        self.assertGreater(geojson_file.stat().st_size, 100)

    def test_leaflet_html_generation(self):
        """Test interactive Leaflet HTML generation and bundle injection."""
        test_dir = PACKAGE_ROOT / "data" / "test_outputs"
        test_dir.mkdir(parents=True, exist_ok=True)
        
        fires_df = generate_synthetic_fires(count=10)
        wind_data = generate_synthetic_wind_field(REGIONAL_DOMAIN, grid_step=1.0, forecast_days=2)
        
        config = SimulationConfig(
            forecast_hours=48,
            grid_res_delhi=(15, 15),
            grid_res_regional=(20, 20)
        )
        sim = SmokeAdvectionSimulator(config=config)
        result = sim.simulate(fires_df, wind_data)
        
        html_file = test_dir / "test_map.html"
        create_leaflet_map(result, fires_df, wind_data, output_html_path=html_file)
        
        self.assertTrue(html_file.exists())
        content = html_file.read_text(encoding="utf-8")
        self.assertIn("SIM_DATA", content)
        self.assertIn("leaflet.js", content)
        self.assertIn("time-slider", content)
        self.assertIn("Delhi NCR Smoke Tracker", content)


if __name__ == "__main__":
    unittest.main()
