"""
Models Module for AI-Powered Exoplanet Detector.
Defines candidate classification models, 5-Fold Stratified Cross-Validation,
model selection, and empirical evaluation routines.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)
from src.preprocessing import apply_imbalance_strategy


def get_candidate_models(seed=42):
    """
    Returns a dictionary of candidate classification models.
    """
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=4,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1
        ),
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.08,
            max_depth=5,
            class_weight="balanced",
            random_state=seed
        ),
        "Support Vector Machine": SVC(
            C=1.5,
            kernel="rbf",
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=seed
        ),
        "Logistic Regression": LogisticRegression(
            C=1.0,
            solver="liblinear",
            class_weight="balanced",
            random_state=seed
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=5,
            weights="distance",
            n_jobs=-1
        )
    }
    return models


def evaluate_predictions(y_true, y_pred, y_prob=None):
    """
    Calculates empirical evaluation metrics.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        try:
            roc_auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            roc_auc = 0.5
        try:
            pr_auc = float(average_precision_score(y_true, y_prob))
        except Exception:
            pr_auc = prec
    else:
        roc_auc = 0.5
        pr_auc = prec

    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm
    }


def run_cross_validation_benchmark(X_train, y_train, imbalance_strategy="class_weight", n_splits=3, seed=42):
    """
    Runs Stratified Cross-Validation on training set ONLY.
    Benchmarking all candidate models and returning empirical validation metrics.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    models = get_candidate_models(seed=seed)
    cv_results = {}

    print(f"-> Benchmark strategy '{imbalance_strategy}' across models...")

    for model_name, model in models.items():
        val_accs, val_precs, val_recs, val_f1s, val_pr_aucs = [], [], [], [], []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            if isinstance(X_train, pd.DataFrame):
                X_tr_f = X_train.iloc[train_idx]
                X_va_f = X_train.iloc[val_idx]
            else:
                X_tr_f = X_train[train_idx]
                X_va_f = X_train[val_idx]

            if isinstance(y_train, (pd.Series, pd.DataFrame)):
                y_tr_f = y_train.iloc[train_idx]
                y_va_f = y_train.iloc[val_idx]
            else:
                y_tr_f = y_train[train_idx]
                y_va_f = y_train[val_idx]

            # Apply imbalance strategy to fold training set ONLY
            X_tr_res, y_tr_res = apply_imbalance_strategy(X_tr_f, y_tr_f, strategy=imbalance_strategy, seed=seed + fold)

            # Fit model
            model.fit(X_tr_res, y_tr_res)

            # Evaluate on fold validation set (unmodified)
            y_va_pred = model.predict(X_va_f)
            y_va_prob = model.predict_proba(X_va_f)[:, 1] if hasattr(model, "predict_proba") else y_va_pred

            metrics = evaluate_predictions(y_va_f, y_va_pred, y_va_prob)
            val_accs.append(metrics["accuracy"])
            val_precs.append(metrics["precision"])
            val_recs.append(metrics["recall"])
            val_f1s.append(metrics["f1_score"])
            val_pr_aucs.append(metrics["pr_auc"])

        mean_metrics = {
            "val_accuracy": float(np.mean(val_accs)),
            "val_precision": float(np.mean(val_precs)),
            "val_recall": float(np.mean(val_recs)),
            "val_f1_score": float(np.mean(val_f1s)),
            "val_pr_auc": float(np.mean(val_pr_aucs))
        }

        cv_results[model_name] = mean_metrics

    return cv_results
