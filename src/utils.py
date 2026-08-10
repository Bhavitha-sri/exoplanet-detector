"""
Utilities Module for AI-Powered Exoplanet Detector.
Provides transit physics modeling, light-curve synthesis, plotting helpers,
and sample dataset generator.
"""

import os
import numpy as np
import pandas as pd


def generate_synthetic_lightcurve(
    num_points=3197,
    time_span_hours=80.0,
    has_transit=False,
    period_hours=24.0,
    transit_duration_hours=3.5,
    depth_ppm=2500.0,
    noise_std=0.0008,
    stellar_variability_amp=0.0005,
    is_eclipsing_binary=False,
    seed=None
):
    """
    Generates a realistic normalized flux light curve F(t) over time.

    Parameters:
    -----------
    num_points : int
        Number of time-series flux sample points.
    time_span_hours : float
        Total time duration of the light curve in hours.
    has_transit : bool
        Whether a planetary transit dip is present.
    period_hours : float
        Orbital period of transit in hours.
    transit_duration_hours : float
        Duration of the transit dip in hours.
    depth_ppm : float
        Transit dip depth in parts-per-million (ppm).
    noise_std : float
        Gaussian white noise standard deviation.
    stellar_variability_amp : float
        Low-frequency sinusoidal stellar variability amplitude.
    is_eclipsing_binary : bool
        If True, generates a V-shaped deep primary and secondary eclipse.
    seed : int, optional
        Random seed for reproducibility.

    Returns:
    --------
    time_array : np.ndarray
        Array of timestamps in hours.
    flux_array : np.ndarray
        Array of normalized relative flux measurements F(t).
    """
    if seed is not None:
        np.random.seed(seed)

    time_array = np.linspace(0, time_span_hours, num_points)
    
    # Baseline normalized flux = 1.0
    flux = np.ones(num_points, dtype=np.float64)

    # 1. Low-frequency stellar variability (rotation / starspots)
    var_period = period_hours * 1.73 + 12.0
    flux += stellar_variability_amp * np.sin(2 * np.pi * time_array / var_period)
    flux += 0.5 * stellar_variability_amp * np.cos(4 * np.pi * time_array / var_period)

    # 2. Add Transit or Eclipsing Binary Signal
    if has_transit or is_eclipsing_binary:
        depth_fraction = depth_ppm / 1e6
        
        # Calculate phase
        phase = np.mod(time_array, period_hours)
        center_phase = period_hours / 2.0
        
        if is_eclipsing_binary:
            # Primary V-shaped deep transit
            dist_primary = np.abs(phase - center_phase)
            in_primary = dist_primary < (transit_duration_hours / 2.0)
            flux[in_primary] -= depth_fraction * (1.0 - (dist_primary[in_primary] / (transit_duration_hours / 2.0)))
            
            # Secondary shallower eclipse
            sec_phase = np.mod(time_array + period_hours / 2.0, period_hours)
            dist_sec = np.abs(sec_phase - center_phase)
            in_sec = dist_sec < (transit_duration_hours * 0.8 / 2.0)
            flux[in_sec] -= (depth_fraction * 0.45) * (1.0 - (dist_sec[in_sec] / (transit_duration_hours * 0.8 / 2.0)))
        else:
            # Planetary Transit (U-shaped with ingress/egress limb darkening)
            dist = np.abs(phase - center_phase)
            half_dur = transit_duration_hours / 2.0
            in_transit = dist < half_dur
            
            if np.any(in_transit):
                # Smooth ingress/egress trapezoidal profile
                ingress_fraction = 0.25
                ingress_dur = half_dur * ingress_fraction
                
                transit_shape = np.ones_like(dist[in_transit])
                edge_dist = half_dur - dist[in_transit]
                
                # Soften edges for ingress/egress
                ingress_mask = edge_dist < ingress_dur
                transit_shape[ingress_mask] = edge_dist[ingress_mask] / ingress_dur
                
                flux[in_transit] -= depth_fraction * transit_shape

    # 3. Add Gaussian Measurement Noise
    flux += np.random.normal(0.0, noise_std, num_points)

    return time_array, flux


def create_sample_lightcurves_dataset(output_dir):
    """
    Creates and saves a pre-packaged suite of sample light-curve CSV files for testing upload
    and web dashboard interactive analysis.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    samples = [
        {
            "filename": "kepler_22b_candidate.csv",
            "name": "Kepler-22b Earth-like Candidate",
            "has_transit": True,
            "period": 24.0,
            "duration": 3.8,
            "depth": 3200.0,
            "noise": 0.0006,
            "is_eb": False,
            "seed": 42
        },
        {
            "filename": "kepler_186f_candidate.csv",
            "name": "Kepler-186f Earth-sized Candidate",
            "has_transit": True,
            "period": 18.5,
            "duration": 2.5,
            "depth": 1400.0,
            "noise": 0.0004,
            "is_eb": False,
            "seed": 101
        },
        {
            "filename": "eclipsing_binary_false_positive.csv",
            "name": "Eclipsing Binary (False Positive)",
            "has_transit": False,
            "period": 16.0,
            "duration": 4.5,
            "depth": 18000.0,
            "noise": 0.0009,
            "is_eb": True,
            "seed": 202
        },
        {
            "filename": "stellar_variability_non_planet.csv",
            "name": "Active Variable Star (Non-Planet)",
            "has_transit": False,
            "period": 30.0,
            "duration": 0.0,
            "depth": 0.0,
            "noise": 0.0012,
            "stellar_var": 0.0025,
            "is_eb": False,
            "seed": 303
        },
        {
            "filename": "quiet_star_baseline.csv",
            "name": "Quiet Solar-type Star (Non-Planet)",
            "has_transit": False,
            "period": 20.0,
            "duration": 0.0,
            "depth": 0.0,
            "noise": 0.0005,
            "stellar_var": 0.0002,
            "is_eb": False,
            "seed": 404
        }
    ]

    saved_files = []
    for s in samples:
        t, f = generate_synthetic_lightcurve(
            num_points=3197,
            time_span_hours=80.0,
            has_transit=s["has_transit"],
            period_hours=s["period"],
            transit_duration_hours=s["duration"],
            depth_ppm=s["depth"],
            noise_std=s["noise"],
            stellar_variability_amp=s.get("stellar_var", 0.0004),
            is_eclipsing_binary=s["is_eb"],
            seed=s["seed"]
        )
        
        filepath = os.path.join(output_dir, s["filename"])
        df = pd.DataFrame({"time_hours": np.round(t, 4), "flux": np.round(f, 6)})
        df.to_csv(filepath, index=False)
        saved_files.append({"name": s["name"], "filename": s["filename"], "path": filepath})

    return saved_files
