# Delhi NCR AQI Forecast Error Correction Pipeline

A machine learning system that predicts and corrects the error (residual) in naive persistence Air Quality Index (AQI) forecasts for Central Pollution Control Board (CPCB) monitoring stations across Delhi NCR, using atmospheric and weather-pollution feedback loops.

---

## 🌟 Background & Problem Formulation

In air quality forecasting, official multi-day dynamical forecasts are rarely archived or openly accessible with historic daily predictions. Operational forecasters and benchmarks routinely use a **naive persistence baseline**:
$$AQI_{\text{baseline}}(t+1) = AQI(t)$$

While persistence works well during stable conditions, it **consistently fails during rapid smog inflection points**:
1. **Under-predicts sudden severe smog spikes**: In early November, when stubble-burning smoke from Punjab and Haryana surges under calm north-westerly winds, persistence lags behind by 50–150 AQI points.
2. **Over-predicts during rapid dispersion**: When cold fronts or western disturbances introduce strong ventilation or precipitation, persistence predicts continued hazardous smog long after winds clear the basin.

### The Feedback-Loop Solution
Instead of predicting raw AQI directly (which suffers from non-stationarity and seasonal drift), this pipeline trains a gradient boosting model (LightGBM / XGBoost) to predict the **forecast residual**:
$$\text{Residual}(t+1) = AQI(t+1) - AQI_{\text{baseline}}(t+1)$$

The **corrected forecast** is reconstructed as:
$$\widehat{AQI}_{\text{corrected}}(t+1) = \text{clip}\left(AQI(t) + \widehat{\text{Residual}}(t+1), 0, 500\right)$$

---

## 🏗️ Pipeline Architecture

```
correction-model/
├── correction_model/
│   ├── __init__.py          # Package exports and version
│   ├── config.py            # CPCB stations, seasonal date ranges, feature lists, thresholds
│   ├── fetch_data.py        # CPCB AQI (CPCB NAQI formulas), Open-Meteo weather, and NASA FIRMS fetcher
│   ├── features.py          # Persistence baseline, weather stillness, upwind fire metrics, lag trends
│   ├── train.py             # Residual regression (LightGBM/XGBoost), TimeSeriesSplit CV, model persistence
│   └── evaluate.py          # Metrics (MAE, RMSE, R²), severe smog subset analysis, publication charts
├── data/
│   ├── cache/               # Local JSON/CSV disk caches (zero redundant API calls)
│   ├── models/              # Serialized model artifacts (.joblib, metadata JSON)
│   └── outputs/             # Generated charts (.png), evaluation metrics JSON, and prediction CSVs
├── tests/
│   └── test_correction_model.py # Comprehensive unit and integration test suite
├── main.py                  # CLI pipeline runner
├── requirements.txt         # Package dependencies
└── README.md                # Documentation
```

---

## 🧪 Key Engineered Feedback Features

Per station per day, the pipeline engineers 25 physical features capturing atmospheric stagnation and transport:

| Category | Feature Name | Physical Rationale |
| :--- | :--- | :--- |
| **Wind Stillness & Ventilation** | `wind_speed_mean`, `wind_speed_max` | Low ambient wind speed inhibits horizontal dispersion. |
| | `still_hours_count` | Hours per day with wind $< 1.5\ \text{m/s}$ (stagnation proxy). |
| **Upwind Alignment** | `nw_wind_fraction`, `wind_u_mean`, `wind_v_mean` | Winds from NW ($270^\circ - 345^\circ$) directly transport farm-fire plumes from Punjab & Haryana. |
| **Thermal Inversion** | `temp_mean`, `temp_min`, `diurnal_temp_range` | Low nocturnal temperatures trap pollutants in shallow boundary layer. |
| | `temp_change_24h` | Identifies cold-front intrusions vs warming clearing trends. |
| **Precipitation & Dry Spells** | `precip_sum`, `days_since_rain` | Running dry-day counter; rain scavenges suspended particulate matter ($PM_{2.5}$). |
| **Upwind Fire Emissions** | `fire_count_48h`, `frp_sum_48h` | 48-hour cumulative active fire count & Fire Radiative Power (MW). |
| | `upwind_fire_index` | $\text{FRP}_{48h} \times \text{NW Wind Fraction}$ (interaction feedback). |
| **AQI Persistence Dynamics** | `baseline_aqi`, `aqi_trend_24h` | Current pollutant state and 24-hour momentum ($AQI_t - AQI_{t-1}$). |
| | `aqi_roll_mean_3d`, `aqi_roll_std_3d` | 3-day baseline level and atmospheric volatility. |

---

## 📊 Evaluation Results

Trained on Season 2023–2024 and evaluated strictly out-of-sample on Season 2024–2025 across 10 CPCB Delhi NCR monitoring stations (1,220 station-day test observations):

```text
==============================================================================
      DELHI NCR AQI FORECAST ERROR-CORRECTION EVALUATION RESULTS
==============================================================================
EVALUATION SUBSET         | METRIC   | BASELINE   | CORRECTED  | IMPROVEMENT 
------------------------------------------------------------------------------
Full Test Season          | MAE      | 47.33      | 45.24      |     +4.42%
                          | RMSE     | 70.27      | 64.81      |     +7.77%
                          | R²       | 0.481      | 0.559      |     +0.078
------------------------------------------------------------------------------
Severe Smog (Nov 1-15)    | MAE      | 43.41      | 40.51      |     +6.68%
                          | RMSE     | 54.47      | 47.96      |    +11.95%
                          | R²       | -0.272     | 0.014      |     +0.286
==============================================================================
```

### Key Findings:
- **Severe Smog Improvement**: During peak stubble-burning episodes (November 1–15), forecast **RMSE is reduced by 11.95%** and **MAE by 6.68%**.
- **Variance Shrinkage**: Naive persistence exhibits catastrophic negative $R^2$ (-0.272) during smog onset because it lags behind the surge. The feedback-corrected model captures the onset inflection and drives $R^2$ back into positive territory.

---

## 🚀 Quickstart & Usage

### 1. Installation
Install requirements using Python 3.10+:

```bash
pip install -r requirements.txt
```

### 2. Run the Pipeline (CLI)
Run the full training and evaluation pipeline with LightGBM (default):

```bash
python main.py --model lightgbm
```

Or run with XGBoost:

```bash
python main.py --model xgboost
```

### CLI Options:
```text
options:
  --model {lightgbm,xgboost}  Gradient boosting architecture (default: lightgbm)
  --train-season {2022-2023,2023-2024,2024-2025}
                              Season used for training (default: 2023-2024)
  --test-season {2022-2023,2023-2024,2024-2025}
                              Season used for evaluation (default: 2024-2025)
  --stations STATIONS         Number of CPCB monitoring stations (default: 10)
  --no-cache                  Bypass local cache and force fresh API queries
  --no-plot                   Skip matplotlib chart generation
  --output-dir OUTPUT_DIR     Custom output directory for metrics and charts
```

---

## 🐍 Programmatic Python API

```python
from correction_model.fetch_data import fetch_historical_aqi, fetch_historical_weather, fetch_historical_fires
from correction_model.features import engineer_feedback_features, build_feature_dataset
from correction_model.train import AQIResidualTrainer
from correction_model.evaluate import evaluate_forecasts, plot_evaluation_comparison
from correction_model.config import CPCB_STATIONS, SEASONS

# 1. Ingest data
df_aqi = fetch_historical_aqi(CPCB_STATIONS, "2023-10-01", "2024-01-31")
df_weather = fetch_historical_weather(CPCB_STATIONS, "2023-10-01", "2024-01-31")
df_fires = fetch_historical_fires("2023-10-01", "2024-01-31")

# 2. Engineer physical feedback features
features_df = engineer_feedback_features(df_aqi, df_weather, df_fires)
X, y, meta = build_feature_dataset(features_df)

# 3. Train residual model
trainer = AQIResidualTrainer(model_type="lightgbm")
trainer.cross_validate(X, y, n_splits=4)
trainer.fit(X, y)
trainer.save()

# 4. Evaluate out-of-sample
test_aqi = fetch_historical_aqi(CPCB_STATIONS, "2024-10-01", "2025-01-31")
test_weather = fetch_historical_weather(CPCB_STATIONS, "2024-10-01", "2025-01-31")
test_fires = fetch_historical_fires("2024-10-01", "2025-01-31")
test_features_df = engineer_feedback_features(test_aqi, test_weather, test_fires)

metrics, eval_df = evaluate_forecasts(trainer, test_features_df)
plots = plot_evaluation_comparison(eval_df, trainer)
```

---

## 🧪 Testing

Run automated unit and integration tests:

```bash
python -m unittest tests/test_correction_model.py
```
All 5 unit tests validate CPCB breakpoint interpolation, dry-spell counting, lag-trend feature alignment, model serialization round-trip, and evaluation metrics.
