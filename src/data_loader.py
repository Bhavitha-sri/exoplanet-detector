"""
Data Loader Module for AI-Powered Exoplanet Detector.
Loads raw time-series stellar flux datasets (exoTrain.csv / exoTest.csv),
ensures benchmark data availability, and handles pre-packaged sample light curves.
"""

import os
import pandas as pd
import numpy as np
from src.utils import generate_synthetic_lightcurve, create_sample_lightcurves_dataset


RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
SAMPLE_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sample_lightcurves")


def generate_benchmark_kepler_dataset(
    output_train_path,
    output_test_path,
    num_train=800,
    num_test=200,
    exoplanet_ratio=0.08,
    num_points=3197,
    seed=42
):
    """
    Generates a realistic Kepler-style time-series flux benchmark dataset
    (formatted identically to exoTrain.csv / exoTest.csv with LABEL column and FLUX.1..FLUX.3197)
    when external datasets are not present locally.

    Label encoding:
    2 = Exoplanet Candidate
    1 = Non-Exoplanet Star
    """
    np.random.seed(seed)
    os.makedirs(os.path.dirname(output_train_path), exist_ok=True)

    def _generate_dataset_split(n_samples, split_seed):
        np.random.seed(split_seed)
        rows = []
        labels = []
        n_exo = int(round(n_samples * exoplanet_ratio))
        n_non_exo = n_samples - n_exo

        # Generate Exoplanet Candidates (Label 2)
        for i in range(n_exo):
            period = np.random.uniform(12.0, 48.0)
            duration = np.random.uniform(2.0, 5.0)
            depth = np.random.uniform(800.0, 5000.0) # ppm
            noise = np.random.uniform(0.0003, 0.0009)
            var_amp = np.random.uniform(0.0001, 0.0006)
            
            _, flux = generate_synthetic_lightcurve(
                num_points=num_points,
                time_span_hours=80.0,
                has_transit=True,
                period_hours=period,
                transit_duration_hours=duration,
                depth_ppm=depth,
                noise_std=noise,
                stellar_variability_amp=var_amp,
                is_eclipsing_binary=False,
                seed=split_seed + i
            )
            rows.append(flux)
            labels.append(2)

        # Generate Non-Exoplanet Stars (Label 1)
        for i in range(n_non_exo):
            # Mix of quiet stars, active variable stars, and eclipsing binaries
            rand_type = np.random.rand()
            if rand_type < 0.15:
                # Eclipsing Binary false positive
                is_eb = True
                has_t = False
                depth = np.random.uniform(12000.0, 35000.0)
                duration = np.random.uniform(4.0, 7.0)
                period = np.random.uniform(14.0, 36.0)
                var_amp = np.random.uniform(0.0005, 0.002)
            elif rand_type < 0.40:
                # High variability star
                is_eb = False
                has_t = False
                depth = 0.0
                duration = 0.0
                period = np.random.uniform(10.0, 50.0)
                var_amp = np.random.uniform(0.002, 0.006)
            else:
                # Quiet star with noise
                is_eb = False
                has_t = False
                depth = 0.0
                duration = 0.0
                period = 24.0
                var_amp = np.random.uniform(0.0001, 0.0005)

            noise = np.random.uniform(0.0004, 0.0012)
            _, flux = generate_synthetic_lightcurve(
                num_points=num_points,
                time_span_hours=80.0,
                has_transit=has_t,
                period_hours=period,
                transit_duration_hours=duration,
                depth_ppm=depth,
                noise_std=noise,
                stellar_variability_amp=var_amp,
                is_eclipsing_binary=is_eb,
                seed=split_seed + n_exo + i
            )
            rows.append(flux)
            labels.append(1)

        # Shuffle dataset rows
        shuffle_idx = np.random.permutation(n_samples)
        rows = np.array(rows)[shuffle_idx]
        labels = np.array(labels)[shuffle_idx]

        flux_cols = [f"FLUX.{i+1}" for i in range(num_points)]
        df = pd.DataFrame(rows, columns=flux_cols)
        df.insert(0, "LABEL", labels)
        return df

    train_df = _generate_dataset_split(num_train, seed)
    test_df = _generate_dataset_split(num_test, seed + 1000)

    train_df.to_csv(output_train_path, index=False)
    test_df.to_csv(output_test_path, index=False)
    
    return train_df, test_df


def load_kepler_raw_dataset(raw_dir=RAW_DATA_DIR):
    """
    Loads raw Kepler time-series dataset files (exoTrain.csv / exoTest.csv).
    If they do not exist locally, automatically generates benchmark Kepler dataset splits.

    Returns:
    --------
    train_df : pd.DataFrame
    test_df : pd.DataFrame
    """
    os.makedirs(raw_dir, exist_ok=True)
    train_path = os.path.join(raw_dir, "exoTrain.csv")
    test_path = os.path.join(raw_dir, "exoTest.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"[DataLoader] Raw Kepler dataset files not found in '{raw_dir}'.")
        print("[DataLoader] Generating realistic benchmark Kepler flux dataset (exoTrain.csv / exoTest.csv)...")
        train_df, test_df = generate_benchmark_kepler_dataset(train_path, test_path)
        print(f"[DataLoader] Successfully created benchmark datasets: Train shape {train_df.shape}, Test shape {test_df.shape}")
    else:
        print(f"[DataLoader] Loading existing raw dataset from '{train_path}'...")
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        print(f"[DataLoader] Successfully loaded: Train shape {train_df.shape}, Test shape {test_df.shape}")

    # Ensure pre-packaged sample light curves exist for web uploads
    create_sample_lightcurves_dataset(SAMPLE_DATA_DIR)

    return train_df, test_df
