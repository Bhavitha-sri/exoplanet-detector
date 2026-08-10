"""
Feature Extraction Module for AI-Powered Exoplanet Detector.
Extracts domain-informed statistical, transit-dip, volatility, and periodic features
from raw time-series stellar brightness flux data F(t).
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import lombscargle


FEATURE_NAMES = [
    "flux_mean",
    "flux_std",
    "flux_skew",
    "flux_kurtosis",
    "flux_median",
    "flux_mad",
    "flux_min",
    "flux_max",
    "flux_p1",
    "flux_p5",
    "flux_p95",
    "flux_p99",
    "flux_p1_p99_span",
    "asymmetry_ratio",
    "max_dip_depth_ppm",
    "dip_snr",
    "local_std_mean",
    "local_std_max",
    "max_local_drop",
    "dip_duration_fraction",
    "lomb_scargle_max_power",
    "lomb_scargle_dominant_period",
    "secondary_power_ratio"
]


def extract_features_from_flux_array(flux, time=None, window_size=31):
    """
    Extracts structured astrophysical and statistical features from a single time-series flux array F(t).

    Parameters:
    -----------
    flux : np.ndarray or list
        1D array of stellar brightness flux observations over time.
    time : np.ndarray or list, optional
        1D array of timestamps (in hours/days). If None, uniform spacing is assumed.
    window_size : int
        Rolling window size for local volatility analysis.

    Returns:
    --------
    features_dict : dict
        Dictionary of extracted numerical feature names and values.
    """
    flux = np.asarray(flux, dtype=np.float64)
    # Filter out NaNs if any
    flux = flux[~np.isnan(flux)]

    if len(flux) < 10:
        raise ValueError("Flux time series array must contain at least 10 observations.")

    if time is None:
        time = np.arange(len(flux), dtype=np.float64)
    else:
        time = np.asarray(time, dtype=np.float64)

    # Normalize flux baseline around 1.0 if not already normalized
    median_flux = np.median(flux)
    if median_flux != 0 and abs(median_flux - 1.0) > 0.1:
        flux = flux / median_flux

    median_val = np.median(flux)
    mean_val = np.mean(flux)
    std_val = np.std(flux, ddof=1) if len(flux) > 1 else 1e-6
    if std_val < 1e-9:
        std_val = 1e-9

    skew_val = float(stats.skew(flux))
    kurt_val = float(stats.kurtosis(flux))
    mad_val = float(np.median(np.abs(flux - median_val)))

    min_val = float(np.min(flux))
    max_val = float(np.max(flux))

    p1, p5, p95, p99 = np.percentile(flux, [1, 5, 95, 99])
    p1_p99_span = float(p99 - p1)

    # Asymmetry ratio: ratio of variance below median vs above median
    below_median = flux[flux < median_val]
    above_median = flux[flux > median_val]
    var_below = np.var(below_median) if len(below_median) > 0 else 0.0
    var_above = np.var(above_median) if len(above_median) > 0 else 1e-9
    asymmetry_ratio = float(var_below / (var_above + 1e-9))

    # Transit dip parameters
    max_dip_depth_ppm = float((median_val - min_val) * 1e6)
    dip_snr = float((median_val - min_val) / std_val)

    # Rolling window local volatility
    flux_series = pd.Series(flux)
    rolling_std = flux_series.rolling(window=window_size, min_periods=5, center=True).std().dropna()
    local_std_mean = float(rolling_std.mean()) if len(rolling_std) > 0 else std_val
    local_std_max = float(rolling_std.max()) if len(rolling_std) > 0 else std_val

    rolling_min = flux_series.rolling(window=window_size, min_periods=5, center=True).min().dropna()
    rolling_med = flux_series.rolling(window=window_size, min_periods=5, center=True).median().dropna()
    local_drops = rolling_med - rolling_min
    max_local_drop = float(local_drops.max()) if len(local_drops) > 0 else float(median_val - min_val)

    # Dip duration fraction: fraction of points > 3 std below median
    dip_points = np.sum(flux < (median_val - 3.0 * std_val))
    dip_duration_fraction = float(dip_points / len(flux))

    # Spectral / Periodic features using Lomb-Scargle Periodogram
    try:
        # Define trial angular frequencies (excluding extreme DC component)
        time_span = float(time[-1] - time[0]) if len(time) > 1 else 1.0
        if time_span <= 0:
            time_span = 1.0
        
        freqs = np.linspace(2 * np.pi / (time_span * 0.9), 2 * np.pi / (2.0 * (time_span / len(time))), 200)
        pgram = lombscargle(time, flux - mean_val, freqs, normalize=True)
        
        sorted_indices = np.argsort(pgram)[::-1]
        max_power_idx = sorted_indices[0]
        lomb_scargle_max_power = float(pgram[max_power_idx])
        
        dominant_freq = freqs[max_power_idx]
        lomb_scargle_dominant_period = float((2 * np.pi) / dominant_freq) if dominant_freq > 0 else 0.0
        
        # Second peak ratio
        sec_power = float(pgram[sorted_indices[1]]) if len(sorted_indices) > 1 else 0.0
        secondary_power_ratio = float(sec_power / (lomb_scargle_max_power + 1e-9))
    except Exception:
        lomb_scargle_max_power = 0.0
        lomb_scargle_dominant_period = 0.0
        secondary_power_ratio = 0.0

    return {
        "flux_mean": mean_val,
        "flux_std": std_val,
        "flux_skew": skew_val,
        "flux_kurtosis": kurt_val,
        "flux_median": median_val,
        "flux_mad": mad_val,
        "flux_min": min_val,
        "flux_max": max_val,
        "flux_p1": p1,
        "flux_p5": p5,
        "flux_p95": p95,
        "flux_p99": p99,
        "flux_p1_p99_span": p1_p99_span,
        "asymmetry_ratio": asymmetry_ratio,
        "max_dip_depth_ppm": max_dip_depth_ppm,
        "dip_snr": dip_snr,
        "local_std_mean": local_std_mean,
        "local_std_max": local_std_max,
        "max_local_drop": max_local_drop,
        "dip_duration_fraction": dip_duration_fraction,
        "lomb_scargle_max_power": lomb_scargle_max_power,
        "lomb_scargle_dominant_period": lomb_scargle_dominant_period,
        "secondary_power_ratio": secondary_power_ratio
    }


def extract_features_dataframe(flux_df, label_col=None):
    """
    Extracts features for an entire dataset DataFrame where each row is a time-series light curve.

    Parameters:
    -----------
    flux_df : pd.DataFrame
        DataFrame containing flux time series columns.
    label_col : str, optional
        Name of target column (e.g. 'LABEL' or 'target') if present.

    Returns:
    --------
    X_features : pd.DataFrame
        Extracted feature matrix DataFrame.
    y : pd.Series or None
        Target binary labels (1 = Exoplanet Candidate, 0 = Non-Exoplanet) if label_col provided.
    """
    y = None
    if label_col and label_col in flux_df.columns:
        y_raw = flux_df[label_col].values
        # Map labels (e.g. Kaggle exoTrain label 2 -> 1 (exoplanet), label 1 -> 0 (non-planet))
        if set(np.unique(y_raw)).issubset({1, 2}):
            y = np.where(y_raw == 2, 1, 0)
        else:
            y = np.where(y_raw > 0, 1, 0)
        flux_cols_df = flux_df.drop(columns=[label_col])
    else:
        flux_cols_df = flux_df.copy()

    feature_rows = []
    for idx, row in flux_cols_df.iterrows():
        flux_vals = row.values.astype(np.float64)
        feat_dict = extract_features_from_flux_array(flux_vals)
        feature_rows.append(feat_dict)

    X_features = pd.DataFrame(feature_rows, columns=FEATURE_NAMES)
    return X_features, y
