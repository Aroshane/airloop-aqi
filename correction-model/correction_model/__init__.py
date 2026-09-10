"""
Delhi NCR AQI Forecast Error Correction Pipeline.
Predicts the residual in naive persistence AQI forecasts using weather-pollution feedback loops.
"""

__version__ = "1.0.0"

from correction_model.config import CPCB_STATIONS, SEASONS
from correction_model.fetch_data import (
    fetch_historical_aqi,
    fetch_historical_weather,
    fetch_historical_fires,
    calculate_cpcb_aqi,
)
from correction_model.features import (
    engineer_feedback_features,
    build_feature_dataset,
)
from correction_model.train import AQIResidualTrainer
from correction_model.evaluate import (
    evaluate_forecasts,
    plot_evaluation_comparison,
)

__all__ = [
    "CPCB_STATIONS",
    "SEASONS",
    "fetch_historical_aqi",
    "fetch_historical_weather",
    "fetch_historical_fires",
    "calculate_cpcb_aqi",
    "engineer_feedback_features",
    "build_feature_dataset",
    "AQIResidualTrainer",
    "evaluate_forecasts",
    "plot_evaluation_comparison",
]
