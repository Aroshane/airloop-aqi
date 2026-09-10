"""
Evaluation and Visualization Layer for AQI Forecast Error Correction Pipeline.
Evaluates baseline persistence vs error-corrected forecasts on full test seasons
and critical severe smog episodes, producing metrics tables and visualization charts.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from correction_model.config import (
    OUTPUT_DIR,
    MIN_AQI,
    MAX_AQI,
    SEVERE_SMOG_EVAL_PERIOD,
)
from correction_model.train import AQIResidualTrainer

logger = logging.getLogger("correction_model.evaluate")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute standard regression performance metrics."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    
    # Safe MAPE calculation avoiding zero division
    denom = np.where(y_true <= 0, 1.0, y_true)
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)
    
    r2 = float(r2_score(y_true, y_pred))
    return {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "mape": round(mape, 2),
        "r2": round(r2, 3),
    }


def evaluate_forecasts(
    trainer: AQIResidualTrainer,
    test_df: pd.DataFrame,
    output_dir: Optional[Path] = None,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Evaluate persistence baseline forecast vs error-corrected forecast.
    
    Corrected Forecast = clip(Baseline + Predicted_Residual, 0, 500)
    
    Returns:
        (metrics_summary_dict, test_df_with_predictions)
    """
    out_dir = output_dir or OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    
    eval_df = test_df.copy()
    
    # 1. Generate predicted residuals
    X_test = eval_df[trainer.feature_names]
    eval_df["predicted_residual"] = trainer.predict(X_test)
    
    # 2. Compute corrected forecast
    # Corrected Forecast = Baseline + Residual
    eval_df["corrected_aqi"] = np.clip(
        eval_df["baseline_aqi"] + eval_df["predicted_residual"],
        MIN_AQI,
        MAX_AQI
    )
    
    # Forecast errors
    eval_df["baseline_error"] = eval_df["actual_next_day_aqi"] - eval_df["baseline_aqi"]
    eval_df["corrected_error"] = eval_df["actual_next_day_aqi"] - eval_df["corrected_aqi"]
    
    # 3. Compute Metrics on Full Test Set
    y_true_all = eval_df["actual_next_day_aqi"].values
    y_base_all = eval_df["baseline_aqi"].values
    y_corr_all = eval_df["corrected_aqi"].values
    
    base_metrics = compute_metrics(y_true_all, y_base_all)
    corr_metrics = compute_metrics(y_true_all, y_corr_all)
    
    mae_improvement = ((base_metrics["mae"] - corr_metrics["mae"]) / base_metrics["mae"]) * 100.0
    rmse_improvement = ((base_metrics["rmse"] - corr_metrics["rmse"]) / base_metrics["rmse"]) * 100.0
    
    # 4. Compute Metrics Specifically on Severe Smog Dates (Early November 1-15)
    eval_df["date"] = pd.to_datetime(eval_df["date"])
    smog_mask = (
        (eval_df["date"].dt.month == SEVERE_SMOG_EVAL_PERIOD["start_month"]) &
        (eval_df["date"].dt.day >= SEVERE_SMOG_EVAL_PERIOD["start_day"]) &
        (eval_df["date"].dt.day <= SEVERE_SMOG_EVAL_PERIOD["end_day"])
    )
    smog_df = eval_df[smog_mask]
    
    if not smog_df.empty:
        y_true_smog = smog_df["actual_next_day_aqi"].values
        y_base_smog = smog_df["baseline_aqi"].values
        y_corr_smog = smog_df["corrected_aqi"].values
        
        smog_base_metrics = compute_metrics(y_true_smog, y_base_smog)
        smog_corr_metrics = compute_metrics(y_true_smog, y_corr_smog)
        smog_mae_imp = ((smog_base_metrics["mae"] - smog_corr_metrics["mae"]) / smog_base_metrics["mae"]) * 100.0
        smog_rmse_imp = ((smog_base_metrics["rmse"] - smog_corr_metrics["rmse"]) / smog_base_metrics["rmse"]) * 100.0
    else:
        smog_base_metrics = {}
        smog_corr_metrics = {}
        smog_mae_imp = 0.0
        smog_rmse_imp = 0.0

    # 5. Assemble and Print Comparative Summary Table
    metrics_summary = {
        "sample_count_total": len(eval_df),
        "sample_count_severe_smog": len(smog_df),
        "overall": {
            "baseline": base_metrics,
            "corrected": corr_metrics,
            "mae_improvement_pct": round(mae_improvement, 2),
            "rmse_improvement_pct": round(rmse_improvement, 2),
        },
        "severe_smog_nov_1_15": {
            "baseline": smog_base_metrics,
            "corrected": smog_corr_metrics,
            "mae_improvement_pct": round(smog_mae_imp, 2),
            "rmse_improvement_pct": round(smog_rmse_imp, 2),
        }
    }
    
    print("\n" + "=" * 78)
    print("      DELHI NCR AQI FORECAST ERROR-CORRECTION EVALUATION RESULTS")
    print("=" * 78)
    print(f"{'EVALUATION SUBSET':<25} | {'METRIC':<8} | {'BASELINE':<10} | {'CORRECTED':<10} | {'IMPROVEMENT':<12}")
    print("-" * 78)
    print(f"{'Full Test Season':<25} | {'MAE':<8} | {base_metrics['mae']:<10.2f} | {corr_metrics['mae']:<10.2f} | {mae_improvement:>+9.2f}%")
    print(f"{'':<25} | {'RMSE':<8} | {base_metrics['rmse']:<10.2f} | {corr_metrics['rmse']:<10.2f} | {rmse_improvement:>+9.2f}%")
    print(f"{'':<25} | {'R²':<8} | {base_metrics['r2']:<10.3f} | {corr_metrics['r2']:<10.3f} | {'--':>10}")
    print("-" * 78)
    if not smog_df.empty:
        print(f"{'Severe Smog (Nov 1-15)':<25} | {'MAE':<8} | {smog_base_metrics['mae']:<10.2f} | {smog_corr_metrics['mae']:<10.2f} | {smog_mae_imp:>+9.2f}%")
        print(f"{'':<25} | {'RMSE':<8} | {smog_base_metrics['rmse']:<10.2f} | {smog_corr_metrics['rmse']:<10.2f} | {smog_rmse_imp:>+9.2f}%")
        print(f"{'':<25} | {'R²':<8} | {smog_base_metrics['r2']:<10.3f} | {smog_corr_metrics['r2']:<10.3f} | {'--':>10}")
    print("=" * 78 + "\n")
    
    # Save outputs
    metrics_path = out_dir / "evaluation_metrics.json"
    preds_path = out_dir / "test_predictions.csv"
    
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
        
    eval_df.to_csv(preds_path, index=False)
    logger.info(f"Evaluation metrics written to {metrics_path}")
    logger.info(f"Detailed test predictions saved to {preds_path}")
    
    return metrics_summary, eval_df


def plot_evaluation_comparison(
    eval_df: pd.DataFrame,
    trainer: AQIResidualTrainer,
    output_dir: Optional[Path] = None,
    station_focus: str = "Anand Vihar"
) -> Dict[str, Path]:
    """
    Generate clean, professional evaluation charts:
    1. Time-series comparison of Baseline vs Corrected vs Actual AQI across severe smog period.
    2. Residual error distribution histogram comparing variance shrinkage.
    3. Feature importance ranking bar chart.
    """
    out_dir = output_dir or OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    plots = {}
    
    # Style settings
    plt.rcParams.update({
        "font.family": "sans-serif",
        "figure.facecolor": "#0f172a",
        "axes.facecolor": "#1e293b",
        "axes.edgecolor": "#475569",
        "axes.labelcolor": "#f8fafc",
        "xtick.color": "#94a3b8",
        "ytick.color": "#94a3b8",
        "grid.color": "#334155",
        "text.color": "#f8fafc",
    })
    
    # -------------------------------------------------------------
    # Plot 1: Severe Smog Time Series (Baseline vs Corrected vs Actual)
    # -------------------------------------------------------------
    eval_df["date"] = pd.to_datetime(eval_df["date"])
    st_df = eval_df[eval_df["station_name"] == station_focus].copy()
    if st_df.empty:
        st_df = eval_df.groupby("date").agg({
            "actual_next_day_aqi": "mean",
            "baseline_aqi": "mean",
            "corrected_aqi": "mean"
        }).reset_index()
        station_title = "Delhi NCR Average"
    else:
        station_title = station_focus
        
    # Filter to November peak window
    nov_df = st_df[(st_df["date"].dt.month == 11)].sort_values("date").reset_index(drop=True)
    if nov_df.empty:
        nov_df = st_df.sort_values("date").iloc[:35].reset_index(drop=True)
        
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    
    # Severity background bands
    ax.axhspan(401, 500, color="#7e22ce", alpha=0.15, label="Severe (401-500)")
    ax.axhspan(301, 400, color="#ef4444", alpha=0.12, label="Very Poor (301-400)")
    ax.axhspan(201, 300, color="#f97316", alpha=0.10, label="Poor (201-300)")
    
    dates_str = nov_df["date"].dt.strftime("%b %d")
    x = np.arange(len(nov_df))
    
    # Curves
    ax.plot(x, nov_df["actual_next_day_aqi"], color="#22c55e", linewidth=2.8, marker="o", markersize=5, label="Actual Tomorrow AQI (Ground Truth)", zorder=4)
    ax.plot(x, nov_df["baseline_aqi"], color="#f59e0b", linewidth=2.0, linestyle="--", marker="s", markersize=4, label="Naive Persistence Baseline (Today's AQI)", zorder=3)
    ax.plot(x, nov_df["corrected_aqi"], color="#38bdf8", linewidth=2.4, marker="^", markersize=5, label="Feedback-Corrected Forecast", zorder=5)
    
    # Annotate peak stubble smog inflection
    peak_idx = nov_df["actual_next_day_aqi"].idxmax()
    peak_val = nov_df.loc[peak_idx, "actual_next_day_aqi"]
    ax.annotate(
        f"Peak Smog AQI: {peak_val:.0f}\n(High Stillness + Upwind Fire Surge)",
        xy=(peak_idx, peak_val),
        xytext=(max(0, peak_idx - 5), min(480, peak_val + 25)),
        arrowprops=dict(facecolor="#38bdf8", shrink=0.08, width=1.5, headwidth=6),
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#1e293b", edgecolor="#38bdf8", alpha=0.9),
        fontsize=9,
        color="#ffffff"
    )
    
    ax.set_xticks(x[::2])
    ax.set_xticklabels(dates_str[::2], rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Air Quality Index (CPCB NAQI)", fontsize=11, fontweight="600")
    ax.set_title(f"Delhi AQI Forecast Error Correction: {station_title} (November Stubble Smog Episode)", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylim(150, 510)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", framealpha=0.85, facecolor="#0f172a", edgecolor="#475569", fontsize=9)
    
    plt.tight_layout()
    chart1_path = out_dir / "baseline_vs_corrected_timeseries.png"
    fig.savefig(chart1_path)
    plt.close(fig)
    plots["timeseries"] = chart1_path
    
    # -------------------------------------------------------------
    # Plot 2: Forecast Error Distribution (Variance Shrinkage)
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)
    
    base_err = eval_df["baseline_error"].values
    corr_err = eval_df["corrected_error"].values
    
    # Histograms
    bins = np.linspace(-120, 120, 41)
    ax1.hist(base_err, bins=bins, alpha=0.5, color="#f59e0b", label=f"Baseline Error (Std: {np.std(base_err):.1f})", density=True)
    ax1.hist(corr_err, bins=bins, alpha=0.65, color="#38bdf8", label=f"Corrected Error (Std: {np.std(corr_err):.1f})", density=True)
    ax1.axvline(0, color="#ffffff", linestyle="--", linewidth=1.2, alpha=0.7)
    ax1.set_xlabel("Forecast Error: Actual - Forecast (AQI points)", fontsize=10)
    ax1.set_ylabel("Probability Density", fontsize=10)
    ax1.set_title("Error Distribution (Residual Shrinkage)", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(loc="upper right", framealpha=0.85, facecolor="#0f172a", edgecolor="#475569", fontsize=9)
    
    # Scatter vs Ground Truth
    ax2.scatter(eval_df["actual_next_day_aqi"], eval_df["baseline_aqi"], color="#f59e0b", alpha=0.35, s=20, label="Baseline")
    ax2.scatter(eval_df["actual_next_day_aqi"], eval_df["corrected_aqi"], color="#38bdf8", alpha=0.45, s=20, label="Corrected")
    # Ideal 1:1 diagonal line
    diag = np.linspace(MIN_AQI, MAX_AQI, 100)
    ax2.plot(diag, diag, color="#22c55e", linestyle="--", linewidth=1.5, label="Perfect 1:1 Forecast")
    ax2.set_xlabel("Actual Ground Truth AQI", fontsize=10)
    ax2.set_ylabel("Predicted AQI", fontsize=10)
    ax2.set_title("Forecast vs Actual Alignment", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(loc="lower right", framealpha=0.85, facecolor="#0f172a", edgecolor="#475569", fontsize=9)
    
    plt.tight_layout()
    chart2_path = out_dir / "residual_error_distribution.png"
    fig.savefig(chart2_path)
    plt.close(fig)
    plots["error_distribution"] = chart2_path
    
    # -------------------------------------------------------------
    # Plot 3: Feature Importance Bar Chart
    # -------------------------------------------------------------
    imp_df = trainer.get_feature_importances().head(12)
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
    
    y_pos = np.arange(len(imp_df))[::-1]
    bars = ax.barh(y_pos, imp_df["importance_pct"], color="#0ea5e9", edgecolor="#38bdf8", height=0.65)
    
    # Highlight physical feedback features vs baseline
    for idx, (bar, feat) in enumerate(zip(bars, imp_df["feature"])):
        if "fire" in feat or "frp" in feat:
            bar.set_color("#f97316")  # Orange for fire
        elif "wind" in feat or "still" in feat:
            bar.set_color("#38bdf8")  # Sky blue for wind
        elif "temp" in feat:
            bar.set_color("#ec4899")  # Pink for temperature
            
    ax.set_yticks(y_pos)
    ax.set_yticklabels(imp_df["feature"], fontsize=9.5)
    ax.set_xlabel("Relative Feature Importance (% Gain / Split Share)", fontsize=10, fontweight="600")
    ax.set_title("Top Feedback Loop Features Driving AQI Forecast Error Correction", fontsize=12, fontweight="bold", pad=10)
    ax.grid(True, axis="x", linestyle=":", alpha=0.5)
    
    # Add value labels
    for bar, val in zip(bars, imp_df["importance_pct"]):
        ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%", va="center", ha="left", color="#f8fafc", fontsize=8.5)
        
    plt.tight_layout()
    chart3_path = out_dir / "feature_importance.png"
    fig.savefig(chart3_path)
    plt.close(fig)
    plots["feature_importance"] = chart3_path
    
    logger.info(f"Generated 3 publication-grade evaluation charts in {out_dir}")
    return plots
