import sys
import unittest
from pathlib import Path

# Add root and submodules to path
TEST_DIR = Path(__file__).resolve().parent
ROOT_DIR = TEST_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
correction_model_dir = ROOT_DIR / "correction-model"
smoke_tracker_dir = ROOT_DIR / "smoke-tracker"
if str(correction_model_dir) not in sys.path:
    sys.path.insert(0, str(correction_model_dir))
if str(smoke_tracker_dir) not in sys.path:
    sys.path.insert(0, str(smoke_tracker_dir))

import numpy as np

from coupled_feedback import (
    CoupledAtmosphericChemistryEngine,
    CoupledFeedbackStep,
    CoupledSimulationResult,
)
from coupled_predictor import (
    CoupledStationAQIPredictor,
    get_aqi_category,
    Station72hForecast,
)
from correction_model.config import CPCB_STATIONS, MIN_AQI, MAX_AQI


class TestCoupledFeedbackPhysics(unittest.TestCase):

    def setUp(self):
        self.engine = CoupledAtmosphericChemistryEngine()

    def test_solar_dimming_beer_lambert(self):
        """Verify Beer-Lambert extinction: solar flux strictly decreases with aerosol density."""
        step_clean = self.engine.solve_coupled_step(
            hour_idx=2,
            datetime_str="+2h",
            unperturbed_smoke=0.0,
            ambient_temp_c=22.0,
            ambient_wind_speed=2.5
        )
        self.assertEqual(step_clean.solar_dimming_pct, 0.0)
        self.assertEqual(step_clean.unperturbed_solar_flux, step_clean.coupled_solar_flux)

        step_heavy_smoke = self.engine.solve_coupled_step(
            hour_idx=2,
            datetime_str="+2h",
            unperturbed_smoke=250.0,
            ambient_temp_c=22.0,
            ambient_wind_speed=2.5
        )
        self.assertGreater(step_heavy_smoke.solar_dimming_pct, 15.0)
        self.assertLess(step_heavy_smoke.coupled_solar_flux, step_heavy_smoke.unperturbed_solar_flux)

    def test_pbl_height_collapse(self):
        """Verify that aerosol thermal suppression collapses Planetary Boundary Layer depth."""
        step_clean = self.engine.solve_coupled_step(
            hour_idx=2,
            datetime_str="+2h",
            unperturbed_smoke=0.0,
            ambient_temp_c=22.0,
            ambient_wind_speed=2.5
        )
        step_smoke = self.engine.solve_coupled_step(
            hour_idx=2,
            datetime_str="+2h",
            unperturbed_smoke=200.0,
            ambient_temp_c=22.0,
            ambient_wind_speed=2.5
        )
        # Smoke must collapse PBL mixing depth
        self.assertLess(step_smoke.coupled_pbl_m, step_clean.unperturbed_pbl_m)
        self.assertGreater(step_smoke.pbl_collapse_pct, 20.0)
        # Bounded above minimum threshold
        self.assertGreaterEqual(step_smoke.coupled_pbl_m, step_smoke.unperturbed_pbl_m * self.engine.min_pbl_ratio * 0.8)

    def test_ground_concentration_entrapment(self):
        """Verify volume compression: coupled ground concentration exceeds decoupled concentration."""
        step = self.engine.solve_coupled_step(
            hour_idx=5,
            datetime_str="+5h",
            unperturbed_smoke=150.0,
            ambient_temp_c=20.0,
            ambient_wind_speed=2.0
        )
        # Entrapment factor must be > 1.0
        self.assertGreater(step.entrapment_factor, 1.10)
        # Coupled smoke must be amplified compared to uncoupled dispersion
        self.assertGreater(step.coupled_smoke_index, step.decoupled_smoke_index)

    def test_induced_surface_stagnation(self):
        """Verify aerosol-induced wind deceleration / surface stillness."""
        step = self.engine.solve_coupled_step(
            hour_idx=5,
            datetime_str="+5h",
            unperturbed_smoke=180.0,
            ambient_temp_c=20.0,
            ambient_wind_speed=3.0
        )
        self.assertLess(step.coupled_wind_speed, step.unperturbed_wind_speed)
        self.assertGreater(step.wind_deceleration_pct, 5.0)

    def test_72h_simulation_trajectory(self):
        """Verify 72-hour full coupled simulation execution and output integrity."""
        hours = 72
        decoupled_mean = [float(5.0 + 80.0 * np.exp(-((h - 20) ** 2) / 40.0)) for h in range(hours)]
        decoupled_max = [v * 2.5 for v in decoupled_mean]
        ambient_temps = [24.0 - (h % 24) * 0.4 for h in range(hours)]
        ambient_winds = [2.2 for _ in range(hours)]
        timestamps = [f"2026-11-01T{h:02d}:00" for h in range(hours)]

        result = self.engine.simulate_72h_coupled_trajectory(
            decoupled_delhi_mean=decoupled_mean,
            decoupled_delhi_max=decoupled_max,
            ambient_temps=ambient_temps,
            ambient_winds=ambient_winds,
            timestamps=timestamps,
        )

        self.assertEqual(result.forecast_hours, 72)
        self.assertEqual(len(result.steps), 72)
        self.assertEqual(len(result.coupled_delhi_mean), 72)
        self.assertGreater(result.underprediction_max_gap, 0.0)
        self.assertTrue(0 <= result.peak_smog_hour < 72)

    def test_station_forecast_predictions(self):
        """Verify 72-hour station AQI calculations and CPCB categories."""
        hours = 72
        decoupled_mean = [float(10.0 + 60.0 * np.exp(-((h - 18) ** 2) / 50.0)) for h in range(hours)]
        decoupled_max = [v * 2.0 for v in decoupled_mean]
        ambient_temps = [22.0 for _ in range(hours)]
        ambient_winds = [2.0 for _ in range(hours)]
        timestamps = [f"2026-11-01T{h:02d}:00" for h in range(hours)]

        result = self.engine.simulate_72h_coupled_trajectory(
            decoupled_delhi_mean=decoupled_mean,
            decoupled_delhi_max=decoupled_max,
            ambient_temps=ambient_temps,
            ambient_winds=ambient_winds,
            timestamps=timestamps,
        )

        predictor = CoupledStationAQIPredictor(CPCB_STATIONS[:5])
        forecasts = predictor.predict_72h_station_forecasts(result)

        self.assertEqual(len(forecasts), 5)
        for st_name, fc in forecasts.items():
            self.assertEqual(len(fc.decoupled_series), 72)
            self.assertEqual(len(fc.coupled_series), 72)
            # Physical bounds check
            self.assertTrue(all(MIN_AQI <= v <= MAX_AQI for v in fc.coupled_series))
            # Coupled peak must capture feedback amplification
            self.assertGreaterEqual(fc.peak_coupled_aqi, fc.peak_decoupled_aqi)
            # Valid category
            self.assertIn(fc.peak_severity_category, ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"])


if __name__ == "__main__":
    unittest.main()
