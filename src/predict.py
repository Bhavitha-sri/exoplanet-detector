"""
Prediction Inference Module for AI-Powered Exoplanet Detector.
Loads trained model artifacts and provides inference endpoints for web API.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from src.feature_extraction import extract_features_from_flux_array, FEATURE_NAMES


MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "lightcurve_model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "lightcurve_scaler.joblib")
FEATURE_NAMES_PATH = os.path.join(MODEL_DIR, "feature_names.json")


class ExoplanetPredictor:
    """
    Production predictor engine.
    """

    def __init__(self, model_path=MODEL_PATH, scaler_path=SCALER_PATH):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self.feature_names = FEATURE_NAMES
        self.load_artifacts()

    def load_artifacts(self):
        """Loads serialized model binary, scaler, and feature names."""
        if not os.path.exists(self.model_path) or not os.path.exists(self.scaler_path):
            raise FileNotFoundError(
                f"Model artifact not found at '{self.model_path}'. "
                "Please run 'python train.py' to train and save the model."
            )
        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)

        if os.path.exists(FEATURE_NAMES_PATH):
            with open(FEATURE_NAMES_PATH, "r") as f:
                self.feature_names = json.load(f)

    def predict_flux_sequence(self, flux_array, time_array=None):
        """
        Executes exoplanet classification on a raw time-series flux array F(t).

        Parameters:
        -----------
        flux_array : np.ndarray or list
            Sequence of stellar flux observations.
        time_array : np.ndarray or list, optional

        Returns:
        --------
        result : dict
        """
        # 1. Feature Extraction
        feat_dict = extract_features_from_flux_array(flux_array, time=time_array)
        feat_df = pd.DataFrame([feat_dict], columns=self.feature_names)

        # 2. Scale features with pre-fitted scaler
        scaled_array = self.scaler.transform(feat_df)
        scaled_df = pd.DataFrame(scaled_array, columns=self.feature_names)

        # 3. Model Inference
        pred_class = int(self.model.predict(scaled_df)[0])
        prob_scores = self.model.predict_proba(scaled_df)[0] if hasattr(self.model, "predict_proba") else [1.0 - pred_class, float(pred_class)]
        exoplanet_prob = float(prob_scores[1])

        # 4. Feature Importance / Contribution
        top_contributions = self._get_feature_contributions(feat_dict)

        # 5. Explainability Synthesis
        explanation = self._synthesize_explanation(pred_class, exoplanet_prob, feat_dict)

        return {
            "prediction": "EXOPLANET CANDIDATE" if pred_class == 1 else "NON-EXOPLANET",
            "is_exoplanet_candidate": bool(pred_class == 1),
            "confidence_score": float(np.max(prob_scores)),
            "exoplanet_probability": exoplanet_prob,
            "extracted_features": feat_dict,
            "top_features": top_contributions,
            "explanation": explanation
        }

    def _get_feature_contributions(self, feat_dict):
        """Returns top feature importance weights if model supports feature_importances_."""
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            feat_imp = [
                {
                    "feature": name,
                    "importance": float(imp),
                    "value": float(feat_dict.get(name, 0.0))
                }
                for name, imp in zip(self.feature_names, importances)
            ]
            feat_imp.sort(key=lambda x: x["importance"], reverse=True)
            return feat_imp[:6]
        return []

    def _synthesize_explanation(self, pred_class, prob, feat_dict):
        """Generates domain-informed astronomical explanation based on physical transit features."""
        snr = feat_dict.get("dip_snr", 0.0)
        depth_ppm = feat_dict.get("max_dip_depth_ppm", 0.0)
        skew = feat_dict.get("flux_skew", 0.0)
        pgram_power = feat_dict.get("lomb_scargle_max_power", 0.0)

        if pred_class == 1:
            return (
                f"Classified as EXOPLANET CANDIDATE with {prob:.1%} confidence. "
                f"The stellar light curve exhibits a characteristic transit dip profile with a depth of "
                f"{depth_ppm:.1f} ppm and a signal-to-noise ratio (SNR) of {snr:.2f}. "
                f"Negative skewness ({skew:.2f}) and Lomb-Scargle periodic power ({pgram_power:.3f}) "
                f"strongly indicate periodic starlight obstruction consistent with a planetary transit."
            )
        else:
            return (
                f"Classified as NON-EXOPLANET with {1.0 - prob:.1%} confidence. "
                f"The light curve lacks a significant U-shaped periodic transit dip (SNR: {snr:.2f}, "
                f"max drop: {depth_ppm:.1f} ppm). Observed flux variation is consistent with background noise, "
                f"stellar rotation/variability, or non-planetary stellar activity."
            )
