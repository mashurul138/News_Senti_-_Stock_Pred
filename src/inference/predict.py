#!/usr/bin/env python3
"""
NEXUS Stock AI — Standalone Inference Engine (Phases 23–28)
Provides production-ready inference and local factor explainability for next-day direction.
Seamlessly supports both CalibratedClassifierCV joblib pipelines and native XGBoost JSON models.
"""

import os
import json
from typing import List, Dict, Any, Union, Optional
import numpy as np
import pandas as pd
import xgboost as xgb


class NexusInferenceEngine:
    """
    Production inference engine for NEXUS Stock AI.
    Loads CalibratedClassifierCV joblib models or native XGBoost JSON artifacts,
    and runs validated, explainable predictions.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        schema_path: str = "./models/feature_schema.json"
    ):
        # 1. Resolve model path with prioritized fallback candidates
        resolved_model_path = None
        candidates = []
        if model_path:
            candidates.extend([
                model_path,
                os.path.join("/content/drive/MyDrive/NEXUS_Stock_AI/models", os.path.basename(model_path))
            ])
        candidates.extend([
            "./models/xgb_direction_calibrated.joblib",
            "/content/drive/MyDrive/NEXUS_Stock_AI/models/xgb_direction_calibrated.joblib",
            "./models/xgb_direction.json",
            "/content/drive/MyDrive/NEXUS_Stock_AI/models/xgb_direction.json"
        ])

        for p in candidates:
            if p and os.path.exists(p):
                resolved_model_path = p
                break

        if resolved_model_path is None:
            raise FileNotFoundError(f"Model file not found. Checked candidate locations: {candidates}")

        # 2. Resolve feature schema path
        resolved_schema_path = None
        schema_candidates = [
            schema_path,
            os.path.join("/content/drive/MyDrive/NEXUS_Stock_AI/models", os.path.basename(schema_path)),
            "./models/feature_schema.json"
        ]
        for sp in schema_candidates:
            if sp and os.path.exists(sp):
                resolved_schema_path = sp
                break

        if resolved_schema_path is None:
            raise FileNotFoundError(f"Schema file not found. Checked candidate locations: {schema_candidates}")

        with open(resolved_schema_path, "r", encoding="utf-8") as f:
            schema_data = json.load(f)
            if isinstance(schema_data, dict) and "features" in schema_data:
                self.feature_names = schema_data["features"]
            elif isinstance(schema_data, list):
                self.feature_names = schema_data
            else:
                raise ValueError("Unexpected schema format. Expected list or dict with 'features' key.")

        self.model_path = resolved_model_path
        self.schema_path = resolved_schema_path

        # 3. Load model based on file format
        if self.model_path.endswith(".joblib"):
            import joblib
            self.model = joblib.load(self.model_path)
            self.booster = None

            # Extract underlying XGBoost booster from CalibratedClassifierCV
            if hasattr(self.model, "calibrated_classifiers_") and len(self.model.calibrated_classifiers_) > 0:
                first_cal = self.model.calibrated_classifiers_[0]
                base_est = getattr(first_cal, "estimator", getattr(first_cal, "base_estimator", None))
                if hasattr(base_est, "get_booster"):
                    self.booster = base_est.get_booster()
                elif hasattr(base_est, "booster"):
                    self.booster = base_est.booster
            elif hasattr(self.model, "get_booster"):
                self.booster = self.model.get_booster()
        else:
            self.model = xgb.XGBClassifier()
            self.model.load_model(self.model_path)
            self.booster = self.model.get_booster()

    def predict(
        self,
        input_features: pd.DataFrame,
        top_k_factors: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Runs batch directional prediction and SHAP factor attribution.
        """
        if not isinstance(input_features, pd.DataFrame):
            raise TypeError("input_features must be a pandas DataFrame.")

        missing_cols = [c for c in self.feature_names if c not in input_features.columns]
        if missing_cols:
            raise ValueError(f"Input features missing required schema columns ({len(missing_cols)}): {missing_cols}")

        X = input_features[self.feature_names].copy()

        if X.isna().any().any():
            nan_cols = X.columns[X.isna().any()].tolist()
            raise ValueError(f"Input features contain NaN values in columns: {nan_cols}")

        probs = self.model.predict_proba(X)
        preds = (probs[:, 1] >= 0.5).astype(int)

        if self.booster is not None:
            dmatrix = xgb.DMatrix(X, feature_names=self.feature_names)
            contribs = self.booster.predict(dmatrix, pred_contribs=True)
        else:
            contribs = np.zeros((len(X), len(self.feature_names) + 1))

        results = []
        for i in range(len(input_features)):
            row_raw = input_features.iloc[i]
            ticker = str(row_raw.get("stock_symbol", row_raw.get("ticker", "UNKNOWN")))
            
            trade_date_val = row_raw.get("date", row_raw.get("trade_date", "UNKNOWN"))
            if isinstance(trade_date_val, (pd.Timestamp, np.datetime64)):
                trade_date = pd.to_datetime(trade_date_val).strftime("%Y-%m-%d")
            else:
                trade_date = str(trade_date_val)

            up_prob = float(round(probs[i, 1], 4))
            down_prob = float(round(probs[i, 0], 4))
            prediction_label = "UP" if preds[i] == 1 else "DOWN"

            row_contribs = contribs[i, :-1]
            factor_list = []
            for feat_name, impact_val in zip(self.feature_names, row_contribs):
                factor_list.append({
                    "feature": feat_name,
                    "direction": "UP" if impact_val > 0 else "DOWN",
                    "impact": float(round(abs(impact_val), 4))
                })

            factor_list.sort(key=lambda x: x["impact"], reverse=True)
            top_factors = factor_list[:top_k_factors]

            results.append({
                "ticker": ticker,
                "trade_date": trade_date,
                "prediction": prediction_label,
                "up_probability": up_prob,
                "down_probability": down_prob,
                "top_factors": top_factors
            })

        return results
