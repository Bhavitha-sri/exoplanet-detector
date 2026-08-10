"""
Flask Web Application Server for AI-Powered Exoplanet Detector.
Provides REST API endpoints and web UI routes for light-curve analysis and classification.
"""

import os
import json
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from src.predict import ExoplanetPredictor
from src.data_loader import SAMPLE_DATA_DIR
from src.utils import create_sample_lightcurves_dataset
from src.feature_extraction import extract_features_from_flux_array


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 MB max upload limit

predictor = None


def get_predictor():
    global predictor
    if predictor is None:
        predictor = ExoplanetPredictor()
    return predictor


@app.route("/")
def index():
    """Renders main dashboard UI."""
    return render_template("index.html")


@app.route("/api/samples", methods=["GET"])
def get_sample_list():
    """Returns list of pre-packaged Kepler sample light curves."""
    if not os.path.exists(SAMPLE_DATA_DIR) or len(os.listdir(SAMPLE_DATA_DIR)) == 0:
        create_sample_lightcurves_dataset(SAMPLE_DATA_DIR)

    samples = [
        {
            "id": "kepler_22b_candidate.csv",
            "name": "Kepler-22b Earth-like Candidate",
            "description": "Transit profile of Earth-sized planet candidate in habitable zone.",
            "category": "Exoplanet Candidate"
        },
        {
            "id": "kepler_186f_candidate.csv",
            "name": "Kepler-186f Earth-sized Candidate",
            "description": "Shallow transit drop of an Earth-sized planet orbiting an M-dwarf star.",
            "category": "Exoplanet Candidate"
        },
        {
            "id": "eclipsing_binary_false_positive.csv",
            "name": "Eclipsing Binary (False Positive)",
            "description": "V-shaped deep primary and secondary stellar eclipse mimicking planetary transit.",
            "category": "False Positive"
        },
        {
            "id": "stellar_variability_non_planet.csv",
            "name": "Active Variable Star (Non-Planet)",
            "description": "High-amplitude stellar rotational variability with starspot modulation.",
            "category": "Non-Planet Star"
        },
        {
            "id": "quiet_star_baseline.csv",
            "name": "Quiet Solar-type Star (Non-Planet)",
            "description": "Quiet solar-type star with Gaussian measurement noise.",
            "category": "Non-Planet Star"
        }
    ]
    return jsonify({"status": "success", "samples": samples})


@app.route("/api/sample/<filename>", methods=["GET"])
def get_sample_data(filename):
    """Loads flux time-series and runs classification on selected sample file."""
    filepath = os.path.join(SAMPLE_DATA_DIR, filename)
    if not os.path.exists(filepath):
        return jsonify({"status": "error", "message": f"Sample file '{filename}' not found."}), 404

    df = pd.read_csv(filepath)
    time_arr = df["time_hours"].values.tolist() if "time_hours" in df.columns else list(range(len(df)))
    flux_arr = df["flux"].values.tolist()

    try:
        pred_engine = get_predictor()
        res = pred_engine.predict_flux_sequence(flux_arr, time_array=time_arr)
        res["status"] = "success"
        res["filename"] = filename
        res["time"] = time_arr
        res["flux"] = flux_arr
        return jsonify(res)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/predict_lightcurve", methods=["POST"])
def predict_lightcurve():
    """Endpoint for raw flux array JSON submission."""
    data = request.get_json()
    if not data or "flux" not in data:
        return jsonify({"status": "error", "message": "Missing 'flux' array in JSON payload."}), 400

    flux_arr = data["flux"]
    time_arr = data.get("time", None)

    try:
        pred_engine = get_predictor()
        res = pred_engine.predict_flux_sequence(flux_arr, time_array=time_arr)
        res["status"] = "success"
        res["time"] = time_arr if time_arr else list(range(len(flux_arr)))
        res["flux"] = flux_arr
        return jsonify(res)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/upload", methods=["POST"])
def upload_file():
    """Endpoint for CSV file uploads."""
    if "file" not in request.files:
        return jsonify({"status": "error", "message": "No file attached in request."}), 400

    uploaded_file = request.files["file"]
    if uploaded_file.filename == "":
        return jsonify({"status": "error", "message": "Selected file is empty."}), 400

    try:
        df = pd.read_csv(uploaded_file)
        
        # Determine flux columns
        if "flux" in df.columns:
            flux_arr = df["flux"].values.tolist()
            time_arr = df["time_hours"].values.tolist() if "time_hours" in df.columns else (df["time"].values.tolist() if "time" in df.columns else list(range(len(df))))
        elif any(c.startswith("FLUX") for c in df.columns):
            # Format like exoTrain row
            flux_cols = [c for c in df.columns if c.startswith("FLUX")]
            flux_arr = df[flux_cols].iloc[0].values.astype(float).tolist()
            time_arr = list(range(len(flux_arr)))
        else:
            # First numeric column as flux
            num_cols = df.select_dtypes(include=[np.number]).columns
            if len(num_cols) == 0:
                return jsonify({"status": "error", "message": "No numeric flux column found in CSV file."}), 400
            flux_arr = df[num_cols[0]].values.tolist()
            time_arr = list(range(len(flux_arr)))

        pred_engine = get_predictor()
        res = pred_engine.predict_flux_sequence(flux_arr, time_array=time_arr)
        res["status"] = "success"
        res["filename"] = uploaded_file.filename
        res["time"] = time_arr
        res["flux"] = flux_arr
        return jsonify(res)
    except Exception as e:
        return jsonify({"status": "error", "message": f"Error parsing CSV file: {str(e)}"}), 500


@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    """Returns genuine empirical model metrics from model/metrics.json."""
    metrics_path = os.path.join(os.path.dirname(__file__), "model", "metrics.json")
    if not os.path.exists(metrics_path):
        return jsonify({
            "status": "pending",
            "message": "Model training metrics not generated yet. Please run 'python train.py'."
        })

    with open(metrics_path, "r") as f:
        metrics_data = json.load(f)

    metrics_data["status"] = "success"
    return jsonify(metrics_data)


@app.route("/api/predict_catalog", methods=["POST"])
def predict_catalog():
    """Secondary experiment endpoint for catalog parameter inputs."""
    data = request.get_json() or {}
    period = float(data.get("koi_period", 24.0))
    depth = float(data.get("koi_depth", 2500.0))
    prad = float(data.get("koi_prad", 1.8))
    duration = float(data.get("koi_duration", 3.5))
    snr = float(data.get("koi_model_snr", 15.0))

    score = 0.0
    if 500.0 <= depth <= 12000.0:
        score += 0.35
    if 0.5 <= prad <= 10.0:
        score += 0.35
    if snr >= 8.0:
        score += 0.20
    if 1.0 <= duration <= 8.0:
        score += 0.10

    is_candidate = score >= 0.65

    return jsonify({
        "status": "success",
        "prediction": "EXOPLANET CANDIDATE" if is_candidate else "NON-EXOPLANET",
        "is_exoplanet_candidate": is_candidate,
        "score": score,
        "parameters": {
            "koi_period": period,
            "koi_depth": depth,
            "koi_prad": prad,
            "koi_duration": duration,
            "koi_model_snr": snr
        },
        "note": "Secondary KOI catalog parameter estimation."
    })


if __name__ == "__main__":
    print("[Exoplanet Detector Server] Starting Flask dev server on http://127.0.0.1:5000 ...")
    app.run(host="127.0.0.1", port=5000, debug=True)
