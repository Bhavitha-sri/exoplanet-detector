"""
Preprocessing Module for AI-Powered Exoplanet Detector.
Enforces strict data leakage prevention and data-driven class imbalance handling.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

try:
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False


def inspect_class_distribution(y, dataset_name="Dataset"):
    """
    Inspects and prints the class distribution of binary target vector y.
    """
    y = np.asarray(y)
    n_total = len(y)
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    pos_ratio = n_pos / n_total if n_total > 0 else 0.0
    imbalance_ratio = n_neg / n_pos if n_pos > 0 else 0.0

    print(f"[{dataset_name} Distribution]")
    print(f"  Total Samples: {n_total}")
    print(f"  Exoplanets (Class 1): {n_pos} ({pos_ratio:.2%})")
    print(f"  Non-Planets (Class 0): {n_neg} ({1.0 - pos_ratio:.2%})")
    print(f"  Imbalance Ratio (0:1): {imbalance_ratio:.2f}:1\n")

    return {
        "n_total": n_total,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "positive_ratio": pos_ratio,
        "imbalance_ratio": imbalance_ratio
    }


def partition_dataset(X, y, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Partitions dataset X, y into Train (70%), Validation (15%), and Held-Out Test (15%) splits
    using stratified sampling to maintain class proportions across all splits.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Split ratios must sum to 1.0"

    val_test_ratio = val_ratio + test_ratio
    test_relative_ratio = test_ratio / val_test_ratio

    # First split: Train vs (Val + Test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=val_test_ratio,
        stratify=y,
        random_state=seed
    )

    # Second split: Val vs Test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=test_relative_ratio,
        stratify=y_temp,
        random_state=seed + 1
    )

    return X_train, y_train, X_val, y_val, X_test, y_test


class StrictPreprocessingPipeline:
    """
    Leakage-Free Preprocessing Pipeline wrapper.
    Ensures StandardScaler is fitted STRICTLY on X_train.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X_train):
        """Fits StandardScaler on training feature matrix ONLY."""
        self.scaler.fit(X_train)
        self.is_fitted = True
        return self

    def transform(self, X):
        """Transforms feature matrix X using pre-fitted scaler parameters."""
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted on training data before calling transform.")
        
        if isinstance(X, pd.DataFrame):
            cols = X.columns
            scaled_array = self.scaler.transform(X)
            return pd.DataFrame(scaled_array, columns=cols, index=X.index)
        return self.scaler.transform(X)

    def fit_transform(self, X_train):
        """Fits on X_train and returns transformed X_train."""
        self.fit(X_train)
        return self.transform(X_train)


def apply_imbalance_strategy(X_train, y_train, strategy="class_weight", seed=42):
    """
    Applies the selected class imbalance strategy ONLY to training data.

    Supported strategies:
    - "none": No resampling
    - "class_weight": Returns original data (weighting handled in model fit)
    - "undersample": Random Under-Sampling of majority class
    - "oversample": Random Over-Sampling of minority class
    - "smote": Synthetic Minority Over-sampling Technique
    """
    if strategy in ["none", "class_weight"]:
        return X_train, y_train

    np.random.seed(seed)

    if isinstance(X_train, pd.DataFrame):
        X_arr = X_train.values
        cols = X_train.columns
        is_df = True
    else:
        X_arr = np.asarray(X_train)
        cols = None
        is_df = False

    y_arr = np.asarray(y_train)

    if HAS_IMBLEARN:
        if strategy == "undersample":
            rus = RandomUnderSampler(random_state=seed)
            X_res, y_res = rus.fit_resample(X_arr, y_arr)
        elif strategy == "oversample":
            ros = RandomOverSampler(random_state=seed)
            X_res, y_res = ros.fit_resample(X_arr, y_arr)
        elif strategy == "smote":
            n_pos = sum(y_arr == 1)
            k_n = max(1, min(3, n_pos - 1))
            smote = SMOTE(random_state=seed, k_neighbors=k_n)
            X_res, y_res = smote.fit_resample(X_arr, y_arr)
        else:
            X_res, y_res = X_arr, y_arr
    else:
        # Fallback pure NumPy rebalancers if imblearn is not installed
        pos_idx = np.where(y_arr == 1)[0]
        neg_idx = np.where(y_arr == 0)[0]
        n_pos = len(pos_idx)
        n_neg = len(neg_idx)

        if strategy == "undersample":
            selected_neg_idx = np.random.choice(neg_idx, size=n_pos, replace=False)
            keep_idx = np.concatenate([pos_idx, selected_neg_idx])
            np.random.shuffle(keep_idx)
            X_res, y_res = X_arr[keep_idx], y_arr[keep_idx]
        elif strategy in ["oversample", "smote"]:
            selected_pos_idx = np.random.choice(pos_idx, size=n_neg, replace=True)
            keep_idx = np.concatenate([neg_idx, selected_pos_idx])
            np.random.shuffle(keep_idx)
            X_res, y_res = X_arr[keep_idx], y_arr[keep_idx]
        else:
            X_res, y_res = X_arr, y_arr

    if is_df:
        return pd.DataFrame(X_res, columns=cols), y_res
    return X_res, y_res
