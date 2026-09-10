"""
Feature Engineering Layer for AQI Forecast Error Correction Pipeline.
Constructs naive persistence baseline, target residuals, meteorological stillness indicators,
upwind agricultural fire proxies, and rolling persistence dynamics per station per day.
"""

import logging
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd

from correction_model.config import (
    FEATURE_COLS,
    TARGET_COL,
    STILL_WIND_THRESHOLD,
    UPWIND_DIR_MIN,
    UPWIND_DIR_MAX,
    MIN_AQI,
    MAX_AQI,
)

logger = logging.getLogger("correction_model.features")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )


def compute_days_since_rain(precip_series: pd.Series, threshold: float = 0.5) -> pd.Series:
    """
    Compute a running counter of consecutive days with precipitation below threshold (dry days).
    Precipitation scavenges suspended PM2.5; prolonged dry spells enable extreme accumulation.
    """
    days = []
    count = 0
    for p in precip_series:
        if pd.isna(p) or p < threshold:
            count += 1
        else:
            count = 0
        days.append(count)
    return pd.Series(days, index=precip_series.index)


def engineer_feedback_features(
    df_aqi: pd.DataFrame,
    df_weather: pd.DataFrame,
    df_fires: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge raw AQI, weather, and fire datasets and engineer all physical feedback features.
    
    Parameters:
        df_aqi: Daily station AQI DataFrame with ['date', 'station_name', 'station_lat', 'station_lon', 'aqi']
        df_weather: Daily station weather DataFrame with ['date', 'station_name', 'temp_mean', 'wind_speed_mean', ...]
        df_fires: Daily farm-fire DataFrame with ['date', 'fire_count', 'frp_sum']
        
    Returns:
        Clean, feature-engineered DataFrame ready for training or evaluation.
    """
    logger.info("Merging AQI, meteorological, and fire datasets...")
    
    # 1. Merge AQI with Station Weather on (date, station_name)
    merged = pd.merge(df_aqi, df_weather, on=["date", "station_name"], how="inner")
    
    # 2. Merge with regional Punjab/Haryana Fire emissions on date
    merged = pd.merge(merged, df_fires, on="date", how="left")
    merged["fire_count"] = merged["fire_count"].fillna(0)
    merged["frp_sum"] = merged["frp_sum"].fillna(0.0)
    
    merged["date"] = pd.to_datetime(merged["date"])
    merged.sort_values(by=["station_name", "date"], inplace=True)
    merged.reset_index(drop=True, inplace=True)
    
    station_dfs = []
    
    # 3. Engineer Station-Level Lagged, Trend, and Feedback Features
    for st_name, group in merged.groupby("station_name", as_index=False):
        g = group.copy().sort_values("date").reset_index(drop=True)
        
        # A. Baseline Forecast & Target Residual Definition
        # Baseline forecast for tomorrow (t+1) is today's AQI: Baseline(t+1) = AQI(t)
        # Target to predict at day t is: Residual(t+1) = Actual_AQI(t+1) - Baseline(t+1)
        g["baseline_aqi"] = g["aqi"]
        g["actual_next_day_aqi"] = g["aqi"].shift(-1)
        g["residual"] = g["actual_next_day_aqi"] - g["baseline_aqi"]
        
        # B. AQI Dynamics & Persistence Lags
        # Recent 24h trend: AQI(t) - AQI(t-1)
        g["aqi_trend_24h"] = g["aqi"] - g["aqi"].shift(1)
        # 3-day rolling mean and volatility (standard deviation)
        g["aqi_roll_mean_3d"] = g["aqi"].rolling(window=3, min_periods=1).mean()
        g["aqi_roll_std_3d"] = g["aqi"].rolling(window=3, min_periods=1).std().fillna(0.0)
        
        # C. Thermal Inversion & Temperature Proxies
        # 24h temperature change: Temp(t) - Temp(t-1)
        g["temp_change_24h"] = g["temp_mean"] - g["temp_mean"].shift(1)
        
        # D. Consecutive Dry Days (Days since rain >= 0.5mm)
        g["days_since_rain"] = compute_days_since_rain(g["precip_sum"])
        
        # E. Upwind Farm-Fire Emissions (Past 48 Hours)
        # Rolling 48h sum of fire count and FRP across today and yesterday (t and t-1)
        g["fire_count_48h"] = g["fire_count"].rolling(window=2, min_periods=1).sum()
        g["frp_sum_48h"] = g["frp_sum"].rolling(window=2, min_periods=1).sum()
        
        # Feedback Interaction: Upwind Fire Index (FRP weighted by NW wind alignment)
        # If wind is blowing directly from NW (towards Delhi), fire impact scales multiplicatively
        g["upwind_fire_index"] = g["frp_sum_48h"] * g["nw_wind_fraction"]
        
        # F. Calendar & Seasonality
        g["day_of_year"] = g["date"].dt.dayofyear
        g["day_of_week"] = g["date"].dt.dayofweek
        g["is_weekend"] = (g["day_of_week"] >= 5).astype(int)
        g["month"] = g["date"].dt.month
        
        station_dfs.append(g)
        
    feat_df = pd.concat(station_dfs, ignore_index=True)
    
    # Fill remaining forward/backward lag edges
    feat_df["aqi_trend_24h"] = feat_df["aqi_trend_24h"].fillna(0.0)
    feat_df["temp_change_24h"] = feat_df["temp_change_24h"].fillna(0.0)
    
    # Drop rows where actual_next_day_aqi is NaN (the very last day of dataset)
    clean_df = feat_df.dropna(subset=["actual_next_day_aqi", "residual"]).copy()
    clean_df.reset_index(drop=True, inplace=True)
    
    logger.info(f"Feature engineering completed: {len(clean_df)} station-day records with {len(FEATURE_COLS)} features.")
    return clean_df


def build_feature_dataset(
    clean_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Extract X (feature matrix), y (residual target vector), and metadata dataframe.
    
    Returns:
        (X, y, meta_df)
    """
    cols = feature_cols or FEATURE_COLS
    
    # Verify all columns exist
    missing = [c for c in cols if c not in clean_df.columns]
    if missing:
        raise KeyError(f"Missing required feature columns: {missing}")
        
    X = clean_df[cols].copy()
    y = clean_df[TARGET_COL].copy()
    meta = clean_df[["date", "station_name", "baseline_aqi", "actual_next_day_aqi"]].copy()
    
    return X, y, meta
