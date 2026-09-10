"""
Unit and Integration Tests for Delhi NCR AQI Forecast Error Correction Pipeline.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from correction_model.config import (
    FEATURE_COLS,
    TARGET_COL,
    MIN_AQI,
    MAX_AQI,
)
from correction_model.fetch_data import calculate_cpcb_aqi
from correction_model.features import (
    compute_days_since_rain,
    engineer_feedback_features,
    build_feature_dataset,
)
from correction_model.train import AQIResidualTrainer
from correction_model.evaluate import compute_metrics, evaluate_forecasts


class TestAQICorrectionModel(unittest.TestCase):
    
    def test_calculate_cpcb_aqi(self):
        """Test official CPCB NAQI breakpoint calculations for PM2.5."""
        # Good range (0 - 30 µg/m³ -> 0 - 50 AQI)
        self.assertAlmostEqual(calculate_cpcb_aqi(0.0), 0.0, places=1)
        self.assertAlmostEqual(calculate_cpcb_aqi(15.0), 25.0, places=1)
        self.assertAlmostEqual(calculate_cpcb_aqi(30.0), 50.0, places=1)
        
        # Satisfactory range (30.1 - 60 µg/m³ -> 51 - 100 AQI)
        aqi_sat = calculate_cpcb_aqi(45.0)
        self.assertTrue(51.0 <= aqi_sat <= 100.0)
        
        # Moderate range (60.1 - 90 µg/m³ -> 101 - 200 AQI)
        aqi_mod = calculate_cpcb_aqi(75.0)
        self.assertTrue(101.0 <= aqi_mod <= 200.0)
        
        # Poor range (90.1 - 120 µg/m³ -> 201 - 300 AQI)
        aqi_poor = calculate_cpcb_aqi(105.0)
        self.assertTrue(201.0 <= aqi_poor <= 300.0)
        
        # Very Poor range (120.1 - 250 µg/m³ -> 301 - 400 AQI)
        aqi_vp = calculate_cpcb_aqi(185.0)
        self.assertTrue(301.0 <= aqi_vp <= 400.0)
        
        # Severe range (250.1 - 380 µg/m³ -> 401 - 500 AQI)
        aqi_sev = calculate_cpcb_aqi(315.0)
        self.assertTrue(401.0 <= aqi_sev <= 500.0)
        
        # Extreme capping above 380 µg/m³
        self.assertEqual(calculate_cpcb_aqi(450.0), MAX_AQI)
        self.assertEqual(calculate_cpcb_aqi(-10.0), MIN_AQI)

    def test_compute_days_since_rain(self):
        """Test consecutive dry days counter logic."""
        # 0.0 (dry), 0.2 (dry), 15.0 (rain!), 0.0 (dry), 0.0 (dry)
        precip = pd.Series([0.0, 0.2, 15.0, 0.0, 0.0])
        dry_days = compute_days_since_rain(precip, threshold=0.5)
        self.assertEqual(list(dry_days), [1, 2, 0, 1, 2])

    def test_feature_engineering_pipeline(self):
        """Test merging and feature engineering transformations."""
        dates = pd.date_range("2023-10-01", periods=10, freq="D")
        
        df_aqi = pd.DataFrame({
            "date": dates,
            "station_name": ["Anand Vihar"] * 10,
            "station_code": ["DL001"] * 10,
            "station_lat": [28.6469] * 10,
            "station_lon": [77.3160] * 10,
            "pm25": [45, 60, 85, 110, 150, 180, 210, 260, 310, 350],
            "aqi": [75, 100, 180, 260, 320, 345, 370, 410, 450, 480],
        })
        
        df_weather = pd.DataFrame({
            "date": dates,
            "station_name": ["Anand Vihar"] * 10,
            "temp_mean": [26.0 - i * 0.5 for i in range(10)],
            "temp_min": [18.0 - i * 0.5 for i in range(10)],
            "temp_max": [32.0 - i * 0.5 for i in range(10)],
            "wind_speed_mean": [2.5, 1.8, 1.2, 1.0, 0.9, 1.1, 1.4, 2.0, 2.2, 2.5],
            "wind_speed_max": [4.0, 3.2, 2.0, 1.8, 1.5, 1.9, 2.2, 3.5, 4.0, 4.2],
            "wind_u_mean": [-1.0] * 10,
            "wind_v_mean": [-0.5] * 10,
            "still_hours_count": [4, 8, 14, 18, 20, 16, 12, 6, 4, 3],
            "nw_wind_fraction": [0.2, 0.3, 0.6, 0.8, 0.85, 0.7, 0.5, 0.4, 0.3, 0.2],
            "precip_sum": [0.0] * 10,
            "diurnal_temp_range": [14.0] * 10,
        })
        
        df_fires = pd.DataFrame({
            "date": dates,
            "fire_count": [50, 80, 200, 500, 1200, 2500, 3000, 1800, 800, 300],
            "frp_sum": [1000, 1800, 5000, 15000, 38000, 85000, 95000, 55000, 22000, 7000],
            "mean_frp": [20.0, 22.5, 25.0, 30.0, 31.6, 34.0, 31.6, 30.5, 27.5, 23.3],
        })
        
        clean_df = engineer_feedback_features(df_aqi, df_weather, df_fires)
        
        # Verify columns exist
        for col in FEATURE_COLS:
            self.assertIn(col, clean_df.columns, f"Missing feature column {col}")
        self.assertIn(TARGET_COL, clean_df.columns)
        
        # Verify persistence residual identity: residual = actual_next_day_aqi - baseline_aqi
        residuals = clean_df["actual_next_day_aqi"] - clean_df["baseline_aqi"]
        np.testing.assert_allclose(clean_df["residual"].values, residuals.values)
        
        # Verify dataset split extraction
        X, y, meta = build_feature_dataset(clean_df)
        self.assertEqual(len(X), len(y))
        self.assertEqual(X.shape[1], len(FEATURE_COLS))

    def test_model_training_and_saving(self):
        """Test training LightGBM regressor and saving/loading round-trip."""
        np.random.seed(42)
        n_samples = 60
        X = pd.DataFrame({
            col: np.random.uniform(0, 100, n_samples) for col in FEATURE_COLS
        })
        # Synthetic residual with some signal from wind stillness and fire index
        y = pd.Series(
            0.3 * X["upwind_fire_index"] + 0.2 * X["still_hours_count"] - 0.1 * X["wind_speed_mean"] + np.random.normal(0, 5, n_samples),
            name="residual"
        )
        
        trainer = AQIResidualTrainer(model_type="lightgbm")
        trainer.fit(X, y)
        preds = trainer.predict(X)
        self.assertEqual(len(preds), n_samples)
        
        imp = trainer.get_feature_importances()
        self.assertEqual(len(imp), len(FEATURE_COLS))
        self.assertAlmostEqual(imp["importance_pct"].sum(), 100.0, places=1)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            m_path, meta_path = trainer.save(tmp_path)
            self.assertTrue(m_path.exists())
            self.assertTrue(meta_path.exists())
            
            # Load and verify prediction identity
            loaded_trainer = AQIResidualTrainer.load(tmp_path)
            loaded_preds = loaded_trainer.predict(X)
            np.testing.assert_allclose(preds, loaded_preds, rtol=1e-5)

    def test_evaluation_metrics_and_correction(self):
        """Test evaluation calculations and forecast correction additivity."""
        y_true = np.array([200.0, 300.0, 400.0, 450.0])
        y_base = np.array([150.0, 220.0, 320.0, 380.0])
        
        m_base = compute_metrics(y_true, y_base)
        self.assertGreater(m_base["mae"], 0.0)
        self.assertGreater(m_base["rmse"], 0.0)
        
        # Perfect predictions
        m_perfect = compute_metrics(y_true, y_true)
        self.assertEqual(m_perfect["mae"], 0.0)
        self.assertEqual(m_perfect["rmse"], 0.0)
        self.assertEqual(m_perfect["r2"], 1.0)


if __name__ == "__main__":
    unittest.main()
