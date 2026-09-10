"""
Main CLI Pipeline Orchestrator for Delhi NCR AQI Forecast Error Correction.
Runs end-to-end data ingestion, feature engineering, LightGBM/XGBoost residual training,
model evaluation, and publication chart generation.
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Add current directory to path if run directly
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from correction_model.config import (
    CPCB_STATIONS,
    SEASONS,
    MODEL_DIR,
    OUTPUT_DIR,
)
from correction_model.fetch_data import (
    fetch_historical_aqi,
    fetch_historical_weather,
    fetch_historical_fires,
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

logger = logging.getLogger("correction_model.main")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)


def run_pipeline(
    model_type: str = "lightgbm",
    train_season: str = "2023-2024",
    test_season: str = "2024-2025",
    num_stations: int = 10,
    use_cache: bool = True,
    save_model: bool = True,
    plot_charts: bool = True,
    output_dir: Path = OUTPUT_DIR
):
    """
    Execute the entire AQI forecast error-correction pipeline end-to-end.
    """
    selected_stations = CPCB_STATIONS[:num_stations]
    station_names = [s["name"] for s in selected_stations]
    
    logger.info("=" * 70)
    logger.info("  STARTING DELHI NCR AQI FORECAST ERROR-CORRECTION PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Model Architecture : {model_type.upper()}")
    logger.info(f"Training Season    : {train_season} ({SEASONS[train_season]['start']} to {SEASONS[train_season]['end']})")
    logger.info(f"Evaluation Season  : {test_season} ({SEASONS[test_season]['start']} to {SEASONS[test_season]['end']})")
    logger.info(f"Monitoring Stations: {len(selected_stations)} stations: {', '.join(station_names[:4])}...")
    logger.info(f"Local Disk Cache   : {'ENABLED' if use_cache else 'DISABLED'}")
    
    # -------------------------------------------------------------
    # Step 1: Ingest Training Data & Engineer Features
    # -------------------------------------------------------------
    logger.info("\n--- STEP 1: INGESTING TRAINING SEASON DATA ---")
    tr_start = SEASONS[train_season]["start"]
    tr_end = SEASONS[train_season]["end"]
    
    tr_aqi = fetch_historical_aqi(selected_stations, tr_start, tr_end, use_cache=use_cache)
    tr_weather = fetch_historical_weather(selected_stations, tr_start, tr_end, use_cache=use_cache)
    tr_fires = fetch_historical_fires(tr_start, tr_end, use_cache=use_cache)
    
    train_features_df = engineer_feedback_features(tr_aqi, tr_weather, tr_fires)
    X_train, y_train, _ = build_feature_dataset(train_features_df)
    logger.info(f"Training dataset ready: {X_train.shape[0]} samples x {X_train.shape[1]} features.")
    
    # -------------------------------------------------------------
    # Step 2: Ingest Evaluation Data & Engineer Features
    # -------------------------------------------------------------
    logger.info("\n--- STEP 2: INGESTING EVALUATION SEASON DATA ---")
    te_start = SEASONS[test_season]["start"]
    te_end = SEASONS[test_season]["end"]
    
    te_aqi = fetch_historical_aqi(selected_stations, te_start, te_end, use_cache=use_cache)
    te_weather = fetch_historical_weather(selected_stations, te_start, te_end, use_cache=use_cache)
    te_fires = fetch_historical_fires(te_start, te_end, use_cache=use_cache)
    
    test_features_df = engineer_feedback_features(te_aqi, te_weather, te_fires)
    X_test, y_test, _ = build_feature_dataset(test_features_df)
    logger.info(f"Evaluation dataset ready: {X_test.shape[0]} samples x {X_test.shape[1]} features.")
    
    # -------------------------------------------------------------
    # Step 3: Train Regression Model to Predict Residuals
    # -------------------------------------------------------------
    logger.info(f"\n--- STEP 3: TRAINING {model_type.upper()} RESIDUAL REGRESSION MODEL ---")
    trainer = AQIResidualTrainer(model_type=model_type)
    
    # Cross-validate on training set
    trainer.cross_validate(X_train, y_train, n_splits=4)
    
    # Fit final model on all training data
    trainer.fit(X_train, y_train)
    
    if save_model:
        trainer.save(MODEL_DIR)
        
    # Print top 6 driving features
    imp_df = trainer.get_feature_importances()
    logger.info("\nTop 6 Weather-Pollution Feedback Features:")
    for _, row in imp_df.head(6).iterrows():
        logger.info(f"  • {row['feature']:<22}: {row['importance_pct']:>5.2f}%")
        
    # -------------------------------------------------------------
    # Step 4: Evaluate Baseline vs Corrected Forecasts
    # -------------------------------------------------------------
    logger.info("\n--- STEP 4: EVALUATING BASELINE VS CORRECTED FORECASTS ---")
    metrics_summary, eval_df = evaluate_forecasts(trainer, test_features_df, output_dir=output_dir)
    
    # -------------------------------------------------------------
    # Step 5: Generate Comparison Visualization Charts
    # -------------------------------------------------------------
    if plot_charts:
        logger.info("\n--- STEP 5: GENERATING PUBLICATION CHARTS ---")
        plots = plot_evaluation_comparison(eval_df, trainer, output_dir=output_dir, station_focus="Anand Vihar")
        for p_name, p_path in plots.items():
            logger.info(f"  Saved chart [{p_name}]: {p_path.name}")
            
    logger.info("\n============================================================")
    logger.info("  PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    logger.info("============================================================")
    return metrics_summary, eval_df, trainer


def main():
    parser = argparse.ArgumentParser(
        description="Delhi NCR AQI Forecast Error Correction Pipeline (Weather-Pollution Feedback Modeling)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="lightgbm",
        choices=["lightgbm", "xgboost"],
        help="Gradient boosting model architecture (default: lightgbm)"
    )
    parser.add_argument(
        "--train-season",
        type=str,
        default="2023-2024",
        choices=list(SEASONS.keys()),
        help="Season used for training (default: 2023-2024)"
    )
    parser.add_argument(
        "--test-season",
        type=str,
        default="2024-2025",
        choices=list(SEASONS.keys()),
        help="Season used for evaluation (default: 2024-2025)"
    )
    parser.add_argument(
        "--stations",
        type=int,
        default=10,
        help="Number of Delhi NCR CPCB stations to include (1-10, default: 10)"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass local disk cache and force fresh API requests"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save the trained model artifact to disk"
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not generate matplotlib visualization charts"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help="Custom output directory for evaluation reports and charts"
    )

    args = parser.parse_args()
    
    run_pipeline(
        model_type=args.model,
        train_season=args.train_season,
        test_season=args.test_season,
        num_stations=args.stations,
        use_cache=not args.no_cache,
        save_model=not args.no_save,
        plot_charts=not args.no_plot,
        output_dir=Path(args.output_dir)
    )


if __name__ == "__main__":
    main()
