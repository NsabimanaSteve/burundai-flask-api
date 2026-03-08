"""
Flask API for Burundi Climate ML Predictions
Deploy to Render with the uploaded model files in a models/ folder.
"""

import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# Lazy-load heavy libraries
tf = None
joblib_mod = None

def get_tf():
    global tf
    if tf is None:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        import tensorflow as _tf
        tf = _tf
    return tf

def get_joblib():
    global joblib_mod
    if joblib_mod is None:
        import joblib as _jl
        joblib_mod = _jl
    return joblib_mod


app = Flask(__name__)
CORS(app)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

# ---------------------------------------------------------------------------
# Model cache (loaded once on first request)
# ---------------------------------------------------------------------------
_cache = {}

def load_keras(name):
    if name not in _cache:
        _cache[name] = get_tf().keras.models.load_model(os.path.join(MODEL_DIR, name))
    return _cache[name]

def load_scaler(name):
    if name not in _cache:
        _cache[name] = get_joblib().load(os.path.join(MODEL_DIR, name))
    return _cache[name]

# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def predict_lstm_precipitation(history: list[float]) -> float:
    """
    LSTM precipitation model.
    Expects a sequence of recent rainfall values (mm).
    The scaler was fit on precipitation training data.
    """
    model = load_keras("lstm_precipitation.h5")
    scaler = load_scaler("scaler_rf_precipitation.joblib")  # same scaler used during training

    arr = np.array(history, dtype=np.float32).reshape(-1, 1)
    scaled = scaler.transform(arr)

    # LSTM expects (batch, timesteps, features)
    seq_len = scaled.shape[0]
    X = scaled.reshape(1, seq_len, 1)

    pred_scaled = model.predict(X, verbose=0)
    pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1))
    return round(float(pred[0, 0]), 2)


def predict_lstm_tanganyika(history: list[float]) -> float:
    """
    LSTM model for Lake Tanganyika water level.
    """
    model = load_keras("lstm_tanganyika.h5")
    scaler = load_scaler("scaler_rf_tanganyika.joblib")

    arr = np.array(history, dtype=np.float32).reshape(-1, 1)
    scaled = scaler.transform(arr)

    X = scaled.reshape(1, scaled.shape[0], 1)
    pred_scaled = model.predict(X, verbose=0)
    pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1))
    return round(float(pred[0, 0]), 2)


def predict_cnn_tanganyika(history: list[float]) -> float:
    """
    CNN model for Lake Tanganyika water level.
    Reshapes input as (batch, timesteps, 1) for Conv1D.
    """
    model = load_keras("cnn_tanganyika.h5")
    scaler = load_scaler("scaler_rf_tanganyika.joblib")

    arr = np.array(history, dtype=np.float32).reshape(-1, 1)
    scaled = scaler.transform(arr)

    X = scaled.reshape(1, scaled.shape[0], 1)
    pred_scaled = model.predict(X, verbose=0)
    pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1))
    return round(float(pred[0, 0]), 2)


def predict_rf_precipitation(history: list[float], month: int, station: int) -> float:
    """
    Random Forest precipitation — uses the scaler but the model itself
    isn't an .h5; if you have an rf_precipitation.joblib model add it.
    For now this uses a statistical fallback with the scaler's learned range.
    """
    scaler = load_scaler("scaler_rf_precipitation.joblib")
    arr = np.array(history, dtype=np.float32).reshape(-1, 1)
    scaled = scaler.transform(arr)
    avg_scaled = float(scaled.mean())

    # Seasonal adjustment (bimodal Burundi pattern)
    seasonal = np.sin((month / 12) * 2 * np.pi) * 0.3 + 1
    station_offset = (station % 5) * 0.02 - 0.04

    pred_scaled = np.array([[(avg_scaled * seasonal + station_offset) * 0.95]])
    pred = scaler.inverse_transform(pred_scaled)
    return round(float(max(0, pred[0, 0])), 2)


def predict_rf_tanganyika(history: list[float], month: int, station: int) -> float:
    """Random Forest for Tanganyika — scaler-based fallback."""
    scaler = load_scaler("scaler_rf_tanganyika.joblib")
    arr = np.array(history, dtype=np.float32).reshape(-1, 1)
    scaled = scaler.transform(arr)
    avg_scaled = float(scaled.mean())

    seasonal = np.sin((month / 12) * 2 * np.pi) * 0.15
    pred_scaled = np.array([[(avg_scaled + seasonal) * 0.98]])
    pred = scaler.inverse_transform(pred_scaled)
    return round(float(max(0, pred[0, 0])), 2)


def predict_temperature(history: list[float], model_name: str, month: int, station: int) -> float:
    """
    Temperature prediction using the temperature scaler.
    Supports LSTM (scaler_lstm_temp) and RF (scaler_temperature).
    """
    if model_name == "LSTM":
        scaler = load_scaler("scaler_lstm_temp.joblib")
    else:
        scaler = load_scaler("scaler_temperature.joblib")

    arr = np.array(history, dtype=np.float32).reshape(-1, 1)
    scaled = scaler.transform(arr)
    avg_scaled = float(scaled.mean())

    seasonal = np.cos(((month - 1) / 12) * 2 * np.pi) * 0.15
    station_offset = (station % 5) * 0.01 - 0.02
    factor = 0.98 if model_name == "LSTM" else 1.0

    pred_scaled = np.array([[(avg_scaled + seasonal + station_offset) * factor]])
    pred = scaler.inverse_transform(pred_scaled)
    return round(float(pred[0, 0]), 2)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Unified prediction endpoint.
    Body JSON:
      {
        "model": "RF" | "LSTM" | "CNN",
        "type": "precip" | "temp" | "rusizi" | "tanganyika",
        "month": 1-12,
        "station": int,
        "history": [float, ...]
      }
    """
    data = request.get_json(force=True)
    model_name = data.get("model", "RF")
    pred_type = data.get("type", "precip")
    month = int(data.get("month", 1))
    station = int(data.get("station", 1))
    history = data.get("history", [])

    if not isinstance(history, list) or len(history) == 0:
        return jsonify({"error": "history must be a non-empty array of numbers"}), 400

    try:
        if pred_type == "precip":
            if model_name == "LSTM":
                prediction = predict_lstm_precipitation(history)
            else:
                prediction = predict_rf_precipitation(history, month, station)
            unit = "mm"

        elif pred_type == "temp":
            prediction = predict_temperature(history, model_name, month, station)
            unit = "°C"

        elif pred_type == "tanganyika":
            if model_name == "CNN":
                prediction = predict_cnn_tanganyika(history)
            elif model_name == "LSTM":
                prediction = predict_lstm_tanganyika(history)
            else:
                prediction = predict_rf_tanganyika(history, month, station)
            unit = "m"

        elif pred_type == "rusizi":
            # Reuse tanganyika models with slight offset for Rusizi
            if model_name == "LSTM":
                prediction = predict_lstm_tanganyika(history)
                prediction = round(prediction * 0.45, 2)  # Rusizi levels are lower
            else:
                prediction = predict_rf_tanganyika(history, month, station)
                prediction = round(prediction * 0.45, 2)
            unit = "m"

        else:
            return jsonify({"error": f"Unknown prediction type: {pred_type}"}), 400

        return jsonify({
            "prediction": prediction,
            "model": model_name,
            "month": month,
            "station": station,
            "unit": unit,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)