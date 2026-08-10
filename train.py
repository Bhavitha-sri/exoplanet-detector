"""
Standalone Training CLI Script for AI-Powered Exoplanet Detector.
Runs a strict, leakage-free machine learning benchmark on Kepler time-series data,
selects the best model, evaluates it on unseen test data, and saves empirical metrics.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from src.data_loader import load_kepler_raw_dataset
from src.feature_extraction import extract_features_dataframe, FEATURE_NAMES
from src.preprocessing import (
    inspect_class_distribution,
    partition_dataset,
    StrictPreprocessingPipeline,
    apply_imbalance_strategy
)
from src.models import (
    get_candidate_models,
    run_cross_validation_benchmark,
    evaluate_predictions
)


MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")


def run_training_pipeline(seed=42):
    """
    Executes end-to-end training and evaluation pipeline.
    """
    print("==========================================================================")
    print("         AI-POWERED EXOPLANET DETECTOR - MODEL TRAINING ENGINE           ")
    print("==========================================================================\n")

    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. Load Raw Time-Series Dataset
    train_raw, test_raw = load_kepler_raw_dataset()

    # Combine raw data to perform a single clean 70/15/15 stratified partition
    full_raw = pd.concat([train_raw, test_raw], ignore_index=True)
    
    print("\n--- Extracting Domain Light-Curve Features from Raw Flux Sequences ---")
    X_features, y_all = extract_features_dataframe(full_raw, label_col="LABEL")
    print(f"Extracted {X_features.shape[1]} domain features for {len(X_features)} light curves.")

    # 2. Partition into 70% Train, 15% Validation, 15% Held-Out Test
    X_train, y_train, X_val, y_val, X_test, y_test = partition_dataset(
        X_features, y_all, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=seed
    )

    print("\n--- Inspecting Class Distribution Across Splits ---")
    train_dist = inspect_class_distribution(y_train, "Training Set (70%)")
    val_dist = inspect_class_distribution(y_val, "Validation Set (15%)")
    test_dist = inspect_class_distribution(y_test, "Held-Out Test Set (15%)")

    # 3. Fit Leakage-Free Preprocessing Pipeline STRICTLY on Training Split
    print("--- Fitting Preprocessing & Scaler strictly on Training Set ---")
    pipeline = StrictPreprocessingPipeline()
    X_train_scaled = pipeline.fit_transform(X_train)
    X_val_scaled = pipeline.transform(X_val)
    X_test_scaled = pipeline.transform(X_test)

    # 4. Benchmark Imbalance Strategies & Candidate Classifiers via 3-Fold Stratified CV on Train Set
    imbalance_strategies = ["class_weight", "undersample", "smote"]
    best_strategy = "class_weight"
    best_strategy_score = -1.0
    best_strategy_cv_table = {}

    print("\n--- Benchmarking Class Imbalance Strategies via 3-Fold Stratified CV ---")
    for strat in imbalance_strategies:
        cv_res = run_cross_validation_benchmark(X_train_scaled, y_train, imbalance_strategy=strat, n_splits=3, seed=seed)
        top_f1_in_strat = max([res["val_f1_score"] for res in cv_res.values()])
        top_prauc_in_strat = max([res["val_pr_auc"] for res in cv_res.values()])
        combined_score = 0.6 * top_prauc_in_strat + 0.4 * top_f1_in_strat

        print(f"-> Strategy '{strat}' Combined Score: {combined_score:.4f} (Top PR-AUC: {top_prauc_in_strat:.4f})")
        if combined_score > best_strategy_score:
            best_strategy_score = combined_score
            best_strategy = strat
            best_strategy_cv_table = cv_res

    print(f"\n[Selected Optimal Imbalance Strategy]: '{best_strategy}'")

    # 5. Select Best Model Based on Cross-Validation F1 / PR-AUC
    models_dict = get_candidate_models(seed=seed)
    best_model_name = None
    best_val_score = -1.0

    for model_name in models_dict.keys():
        f1_sc = best_strategy_cv_table[model_name]["val_f1_score"]
        pr_sc = best_strategy_cv_table[model_name]["val_pr_auc"]
        score = 0.5 * f1_sc + 0.5 * pr_sc
        if score > best_val_score:
            best_val_score = score
            best_model_name = model_name

    print(f"[Selected Best Candidate Classifier]: {best_model_name}")

    # 6. Fit Selected Best Model on Full Training Split (with selected imbalance strategy)
    X_tr_final, y_tr_final = apply_imbalance_strategy(X_train_scaled, y_train, strategy=best_strategy, seed=seed)
    final_model = models_dict[best_model_name]
    final_model.fit(X_tr_final, y_tr_final)

    # 7. Evaluate Selected Model ONLY ONCE on Unseen Held-Out Test Set
    print("\n==========================================================================")
    print("       FINAL UNBIASED EVALUATION ON UNSEEN HELD-OUT TEST SET (15%)       ")
    print("==========================================================================")

    y_test_pred = final_model.predict(X_test_scaled)
    y_test_prob = final_model.predict_proba(X_test_scaled)[:, 1] if hasattr(final_model, "predict_proba") else y_test_pred

    test_metrics = evaluate_predictions(y_test, y_test_pred, y_test_prob)

    print(f"\nModel: {best_model_name}")
    print(f"Test Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall:    {test_metrics['recall']:.4f}")
    print(f"Test F1-Score:  {test_metrics['f1_score']:.4f}")
    print(f"Test PR-AUC:    {test_metrics['pr_auc']:.4f}")
    print(f"Test ROC-AUC:   {test_metrics['roc_auc']:.4f}")
    print(f"Confusion Matrix: {test_metrics['confusion_matrix']}")

    # Collect full comparison results across all candidate models on Test set
    test_comparison = {}
    for name, m_obj in models_dict.items():
        if name != best_model_name:
            m_obj.fit(X_tr_final, y_tr_final)
        preds = m_obj.predict(X_test_scaled)
        probs = m_obj.predict_proba(X_test_scaled)[:, 1] if hasattr(m_obj, "predict_proba") else preds
        test_comparison[name] = evaluate_predictions(y_test, preds, probs)

    # 8. Save Artifacts
    model_save_path = os.path.join(MODEL_DIR, "lightcurve_model.joblib")
    scaler_save_path = os.path.join(MODEL_DIR, "lightcurve_scaler.joblib")
    feat_save_path = os.path.join(MODEL_DIR, "feature_names.json")
    metrics_save_path = os.path.join(MODEL_DIR, "metrics.json")

    joblib.dump(final_model, model_save_path)
    joblib.dump(pipeline.scaler, scaler_save_path)

    with open(feat_save_path, "w") as f:
        json.dump(FEATURE_NAMES, f, indent=2)

    export_metrics = {
        "best_model_name": best_model_name,
        "imbalance_strategy": best_strategy,
        "test_metrics": test_metrics,
        "model_comparison": test_comparison,
        "cv_benchmark_results": best_strategy_cv_table,
        "dataset_split": {
            "train_size": len(X_train),
            "val_size": len(X_val),
            "test_size": len(X_test)
        }
    }

    with open(metrics_save_path, "w") as f:
        json.dump(export_metrics, f, indent=2)

    print(f"\n[Success] Artifacts saved to '{MODEL_DIR}':")
    print(f"  - Model Binary: {model_save_path}")
    print(f"  - Scaler:       {scaler_save_path}")
    print(f"  - Feature Names:{feat_save_path}")
    print(f"  - Test Metrics: {metrics_save_path}")

    return export_metrics


if __name__ == "__main__":
    run_training_pipeline()
