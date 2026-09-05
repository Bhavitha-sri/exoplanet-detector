# 🚀 AI-Powered Exoplanet Detector

An end-to-end full-stack machine learning application designed to identify potential exoplanet candidates from NASA Kepler space telescope time-series light-curve data.

---

## 🌐 Live Demo
<p align="center">
  <a href="https://exoplanet-detector-iota.vercel.app/">
    <img src="https://img.shields.io/badge/🚀%20LIVE%20DEMO-Visit%20Website-blue?style=for-the-badge" alt="Live Demo">
  </a>
  <a href="https://github.com/Bhavitha-sri/exoplanet-detector">
    <img src="https://img.shields.io/badge/💻%20GITHUB-View%20Repository-black?style=for-the-badge&logo=github" alt="GitHub">
  </a>
</p>

🚀 **[Open Exoplanet AI Observatory](https://exoplanet-detector-iota.vercel.app/)**

---

## 🧠 1. Problem Statement

When an exoplanet passes directly between its host star and an observer (such as the Kepler space telescope), it obstructs a small fraction of the star's emitted light.

This astronomical event is known as a **planetary transit**, creating a characteristic dip in the star's observed brightness over time—known as a **transit light curve**.

However, identifying true exoplanet transits from stellar light curves presents significant challenges:

### 🔬 Key Challenges

1. **Low Signal-to-Noise Ratio (SNR)**
   Planetary transits often cause tiny brightness drops (often less than 1% or parts-per-million), easily obscured by stellar variability, starspots, or instrument noise.

2. **False Positives**
   Non-planetary phenomena such as eclipsing binary star systems, background eclipsing binaries, and instrumental systematic errors mimic exoplanet transit signatures.

3. **Data Volume**
   Space telescopes record continuous light curves for hundreds of thousands of stars, rendering manual visual inspection unfeasible.

---

## 💡 2. Solution Overview

This project implements a **leakage-free machine learning pipeline and full-stack web application** to automate exoplanet candidate classification:

* 🪐 **Primary Time-Series Light-Curve Pipeline:** Analyzes raw time-series flux measurements $F(t) = [f_1, f_2, \dots, f_N]$ from Kepler stellar observations.

* 🔬 **Astrophysical Feature Extraction:** Extracts 23 domain-informed features per light curve, including statistical moments (skewness, kurtosis, MAD), transit dip depth (ppm), transit SNR, localized rolling volatility, and Lomb-Scargle periodic spectral power.

* 🧪 **Leakage-Free Benchmark:** Trains and compares 5 classification models using a strict 70% Train / 15% Validation / 15% Held-Out Test split. Preprocessing scalers are fitted **strictly on the training split**.

* ⚖️ **Data-Driven Imbalance Strategy:** Evaluates class weighting, random undersampling, oversampling, and SMOTE strictly inside training cross-validation folds.

* 🖥️ **Interactive Full-Stack Web Dashboard:** Built with Flask, HTML5, CSS Glassmorphism, JavaScript, and Chart.js. Features interactive light-curve zooming, pre-packaged Kepler sample targets, custom CSV file uploads, parameter sliders, and physical explainability breakdown.

---

## 🏗️ 3. Project Architecture

```text
exoplanet-detector/
│
├── data/                               # Dataset storage
│   ├── raw/                            # Time-series flux dataset (exoTrain.csv / exoTest.csv)
│   ├── processed/                      # Cleaned features & train/val/test splits
│   └── sample_lightcurves/             # Sample light-curve CSVs for web uploader
│
├── src/                                # Core Python package
│   ├── __init__.py                     # Package metadata
│   ├── data_loader.py                  # Downloads & loads raw time-series flux data & KOI catalog
│   ├── feature_extraction.py           # Extracts statistical, transit-dip & periodic features from F(t)
│   ├── preprocessing.py                # Data-driven class imbalance handling, scaling, train-only fitting
│   ├── models.py                       # Classifier suite (RF, GBDT, SVM, LR, KNN) & CV model selector
│   ├── predict.py                      # Production inference pipeline (raw flux & parameter inputs)
│   └── utils.py                        # Synthetic transit curve generator & transit physics calculations
│
├── model/                              # Trained artifacts & empirical evaluation results
│   ├── lightcurve_model.joblib         # Serialized top-performing light-curve classifier
│   ├── lightcurve_scaler.joblib        # Feature scaler fitted strictly on training data
│   ├── feature_names.json              # Extracted feature column names
│   └── metrics.json                    # Real empirical test metrics & model comparison table
│
├── templates/                          # Flask HTML UI
│   └── index.html                      # Dark glassmorphic dashboard with plot & controls
│
├── static/                             # Web static assets
│   ├── css/
│   │   └── style.css                   # Responsive dark theme styling
│   └── js/
│       ├── main.js                     # Form controls, sample presets, API fetcher
│       └── chart_builder.js             # Interactive Chart.js light-curve visualizer
│
├── app.py                              # Flask Web Server (REST API endpoints + dashboard routes)
├── train.py                            # Standalone training script (runs leakage-free benchmark)
├── requirements.txt                    # Python dependencies
└── README.md                           # Comprehensive documentation
```

---

## 🤖 4. Machine Learning & Feature Extraction Pipeline

### 4.1 🔬 Extracted Light-Curve Features (`src/feature_extraction.py`)

| Feature Category         | Features Extracted                                                                | Physical Significance                                                                        |
| :----------------------- | :-------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------- |
| **Statistical Moments**  | `flux_mean`, `flux_std`, `flux_skew`, `flux_kurtosis`, `flux_mad`                 | Negative skewness captures downward starlight blockage. Heavy kurtosis indicates sharp dips. |
| **Percentile Spans**     | `flux_p1`, `flux_p5`, `flux_p95`, `flux_p99`, `flux_p1_p99_span`                  | Quantifies extreme drop depth relative to total baseline flux variation.                     |
| **Transit Dip Metrics**  | `max_dip_depth_ppm`, `dip_snr`, `dip_duration_fraction`                           | Measures transit depth in parts-per-million and signal-to-noise ratio over baseline.         |
| **Local Volatility**     | `local_std_mean`, `local_std_max`, `max_local_drop`                               | Rolling window statistics to separate sharp transit drops from global stellar variability.   |
| **Spectral Periodicity** | `lomb_scargle_max_power`, `lomb_scargle_dominant_period`, `secondary_power_ratio` | Lomb-Scargle periodogram power peak identifying periodic transit repeating signals.          |

---

### 4.2 🧪 Leakage-Free Validation Strategy

```text
Raw Kepler Time-Series Data (1,000 Stellar Light Curves)
       │
       ├──► 70% Train Set (700 Light Curves)
       │        ├──► Fit StandardScaler strictly on Train
       │        ├──► Benchmark Imbalance Strategies via CV
       │        └──► Select & Train Best Classifier
       │
       ├──► 15% Val Set (150 Light Curves)
       │        └──► Hyperparameter Tuning
       │
       └──► 15% Held-Out Test (150 Curves)
                └──► Final Unbiased Evaluation (metrics.json)
```

---

## 📊 5. Empirical Results & Model Comparison

All metrics reported below were generated by executing `python train.py` on the held-out test set (150 unseen light curves):

### 5.1 🏆 Final Test Set Evaluation (Unseen Data)

| Classifier Model       | Imbalance Strategy | Test Accuracy | Test Precision | Test Recall | Test F1-Score |   PR-AUC  |  ROC-AUC  | Status      |
| :--------------------- | :----------------- | :-----------: | :------------: | :---------: | :-----------: | :-------: | :-------: | :---------- |
| **Random Forest**      | **Undersample**    |   **100.0%**  |   **100.0%**   |  **100.0%** |   **1.000**   | **1.000** | **1.000** | **Optimal** |
| HistGradientBoosting   | Undersample        |     98.0%     |      84.6%     |    91.7%    |     0.880     |   0.988   |   0.999   | Evaluated   |
| Support Vector Machine | Undersample        |     96.0%     |      71.4%     |    83.3%    |     0.769     |   0.911   |   0.978   | Evaluated   |
| K-Nearest Neighbors    | Undersample        |     92.0%     |      50.0%     |    91.7%    |     0.647     |   0.944   |   0.989   | Evaluated   |
| Logistic Regression    | Undersample        |     92.0%     |      50.0%     |    83.3%    |     0.625     |   0.824   |   0.953   | Evaluated   |

---

## ⚙️ 6. Installation & Quick Start

### 6.1 🛠️ Setup Environment

```bash
# Navigate to project directory
cd exoplanet-detector

# Create and activate virtual environment
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Linux/macOS:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

---

### 6.2 🧪 Train & Evaluate Model

```bash
# Execute standalone model training and benchmark pipeline
python train.py
```

---

### 6.3 🌐 Run Flask Web Application

```bash
# Launch Flask server
python app.py
```

Open your browser at:

```text
http://127.0.0.1:5000
```

to access the visual dashboard.

---

## 🖥️ 7. Web Application Features

1. 🌌 **Kepler Sample Presets**
   Instant one-click selection of confirmed Earth-like planet candidates (Kepler-22b, Kepler-186f), false positive eclipsing binaries, active variable stars, and quiet solar-type stars.

2. 📂 **CSV File Upload**
   Drag-and-drop custom CSV files containing time-series flux columns (`time_hours`, `flux`) for automated feature extraction and inference.

3. 📈 **Interactive Light Curve Visualizer**
   Chart.js canvas with smooth zooming, custom hover tooltips, and transit dip highlighting.

4. 🔍 **Physical Explainability Panel**
   Dynamic synthesis explaining transit depth, signal-to-noise ratio, skewness, and Lomb-Scargle power peak.

5. 🎚️ **Secondary KOI Slider Experiment**
   Interactive parameter sliders for scalar KOI catalog properties.

---

## ⚠️ 8. Limitations & Future Work

* **Dataset Size:** The current pipeline runs on benchmark Kepler light curves. Scaling to full Multi-Quarter Kepler FITS files (~200,000 stars) will require distributed parallel preprocessing.

* **Deep Learning Companion:** Future extensions can integrate 1D Convolutional Neural Networks (1D-CNN) or Transformer models to operate directly on raw un-featurized flux time series alongside domain feature extraction.

---

<div align="center">

### 🌌 AI-Powered Exoplanet Detection

**Machine Learning • Astronomy • Flask • Python • NASA Kepler**

</div>
