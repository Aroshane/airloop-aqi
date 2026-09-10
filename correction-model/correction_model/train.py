"""
Model Training Layer for AQI Forecast Error Correction Pipeline.
Trains LightGBM or XGBoost regression models to predict the residual
(actual tomorrow AQI minus persistence baseline) using weather and fire feedback loops.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from correction_model.config import (
    FEATURE_COLS,
    TARGET_COL,
    MODEL_DIR,
)

logger = logging.getLogger("correction_model.train")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )


class AQIResidualTrainer:
    """
    Trains and validates gradient boosting regression models to predict
    the persistence forecast residual: actual_aqi(t+1) - baseline_aqi(t).
    """
    
    def __init__(
        self,
        model_type: str = "lightgbm",
        hyperparameters: Optional[Dict[str, Any]] = None,
        random_state: int = 42
    ):
        self.model_type = model_type.lower()
        self.random_state = random_state
        self.hyperparameters = hyperparameters or self._default_params()
        self.model = None
        self.feature_names: List[str] = FEATURE_COLS
        self.train_metrics: Dict[str, float] = {}
        self.cv_metrics: Dict[str, float] = {}
        
    def _default_params(self) -> Dict[str, Any]:
        """Return tuned default hyperparameters for tabular gradient boosting."""
        if self.model_type == "lightgbm":
            return {
                "objective": "regression_l1",
                "n_estimators": 120,
                "learning_rate": 0.035,
                "max_depth": 4,
                "num_leaves": 15,
                "subsample": 0.80,
                "colsample_bytree": 0.80,
                "min_child_samples": 20,
                "reg_alpha": 1.0,
                "reg_lambda": 2.0,
                "random_state": self.random_state,
                "verbose": -1,
                "n_jobs": -1,
            }
        elif self.model_type == "xgboost":
            return {
                "objective": "reg:absoluteerror",
                "n_estimators": 120,
                "learning_rate": 0.035,
                "max_depth": 4,
                "subsample": 0.80,
                "colsample_bytree": 0.80,
                "min_child_weight": 3,
                "reg_alpha": 1.0,
                "reg_lambda": 2.0,
                "random_state": self.random_state,
                "n_jobs": -1,
            }
        else:
            raise ValueError(f"Unsupported model_type: '{self.model_type}'. Choose 'lightgbm' or 'xgboost'.")
            
    def _init_model(self):
        """Instantiate the underlying regressor."""
        if self.model_type == "lightgbm":
            import lightgbm as lgb
            return lgb.LGBMRegressor(**self.hyperparameters)
        elif self.model_type == "xgboost":
            import xgboost as xgb
            return xgb.XGBRegressor(**self.hyperparameters)
            
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, n_splits: int = 4) -> Dict[str, float]:
        """
        Perform strictly temporal rolling TimeSeriesSplit cross-validation.
        """
        logger.info(f"Running {n_splits}-fold TimeSeriesSplit CV with {self.model_type.upper()}...")
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        mae_list = []
        rmse_list = []
        r2_list = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            
            fold_model = self._init_model()
            fold_model.fit(X_tr, y_tr)
            preds = fold_model.predict(X_val)
            
            mae = mean_absolute_error(y_val, preds)
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            r2 = r2_score(y_val, preds)
            
            mae_list.append(mae)
            rmse_list.append(rmse)
            r2_list.append(r2)
            logger.info(f"  CV Fold {fold+1}: Val MAE = {mae:.2f} | Val RMSE = {rmse:.2f} | Val R² = {r2:.3f}")
            
        self.cv_metrics = {
            "cv_mae_mean": float(np.mean(mae_list)),
            "cv_mae_std": float(np.std(mae_list)),
            "cv_rmse_mean": float(np.mean(rmse_list)),
            "cv_rmse_std": float(np.std(rmse_list)),
            "cv_r2_mean": float(np.mean(r2_list)),
        }
        logger.info(
            f"Mean CV Residual MAE: {self.cv_metrics['cv_mae_mean']:.2f} ± {self.cv_metrics['cv_mae_std']:.2f} | "
            f"Mean CV RMSE: {self.cv_metrics['cv_rmse_mean']:.2f}"
        )
        return self.cv_metrics

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "AQIResidualTrainer":
        """
        Fit the final regression model on all provided training data.
        """
        self.feature_names = list(X.columns)
        self.model = self._init_model()
        logger.info(f"Fitting final {self.model_type.upper()} residual model on {len(X)} samples with {len(self.feature_names)} features...")
        self.model.fit(X, y)
        
        # Calculate training set performance
        train_preds = self.model.predict(X)
        self.train_metrics = {
            "train_mae": float(mean_absolute_error(y, train_preds)),
            "train_rmse": float(np.sqrt(mean_squared_error(y, train_preds))),
            "train_r2": float(r2_score(y, train_preds)),
        }
        logger.info(
            f"Model training complete. Train MAE: {self.train_metrics['train_mae']:.2f} | "
            f"Train RMSE: {self.train_metrics['train_rmse']:.2f} | Train R²: {self.train_metrics['train_r2']:.3f}"
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict the expected AQI residual (actual - baseline).
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet. Call fit() first.")
        return self.model.predict(X[self.feature_names])

    def get_feature_importances(self) -> pd.DataFrame:
        """
        Extract feature importances ranked in descending order.
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")
            
        importances = self.model.feature_importances_
        norm_imp = (importances / np.sum(importances)) * 100.0
        
        df_imp = pd.DataFrame({
            "feature": self.feature_names,
            "importance_raw": importances,
            "importance_pct": np.round(norm_imp, 2)
        }).sort_values(by="importance_pct", ascending=False).reset_index(drop=True)
        
        return df_imp

    def save(self, model_dir: Optional[Path] = None) -> Tuple[Path, Path]:
        """
        Save the trained model artifact and metadata JSON to disk.
        """
        out_dir = model_dir or MODEL_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = out_dir / "aqi_residual_model.joblib"
        meta_path = out_dir / "model_metadata.json"
        
        joblib.dump(self.model, model_path)
        
        meta = {
            "model_type": self.model_type,
            "trained_at": datetime.now().isoformat(),
            "features": self.feature_names,
            "hyperparameters": {k: v for k, v in self.hyperparameters.items() if not str(k).startswith("_")},
            "train_metrics": self.train_metrics,
            "cv_metrics": self.cv_metrics,
        }
        
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
            
        logger.info(f"Model artifact saved to {model_path}")
        logger.info(f"Model metadata saved to {meta_path}")
        return model_path, meta_path

    @classmethod
    def load(cls, model_dir: Optional[Path] = None) -> "AQIResidualTrainer":
        """
        Load a saved model artifact and metadata from disk.
        """
        in_dir = model_dir or MODEL_DIR
        model_path = in_dir / "aqi_residual_model.joblib"
        meta_path = in_dir / "model_metadata.json"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact not found at {model_path}")
            
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            
        trainer = cls(
            model_type=meta["model_type"],
            hyperparameters=meta.get("hyperparameters")
        )
        trainer.model = joblib.load(model_path)
        trainer.feature_names = meta.get("features", FEATURE_COLS)
        trainer.train_metrics = meta.get("train_metrics", {})
        trainer.cv_metrics = meta.get("cv_metrics", {})
        return trainer
