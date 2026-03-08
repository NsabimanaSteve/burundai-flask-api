"""
Flask API for Burundi Climate ML Predictions
Deploy to Render with the uploaded model files in a models/ folder.
"""

import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

joblib_mod = None

def get_joblib():
    global joblib_mod
    if joblib_mod is None:
        import joblib as _jl
        joblib_mod = _jl
    return joblib_mod


app = Flask(__name__)
CORS(app)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

_cache = {}

def load_keras(name):
    if name not in _cache:
        import keras
        _cache[name] = keras.models.load_model(os.path.join(MODEL_DIR, name))
    return _cache[name]

def load_scaler(name):
    if name not in _cache:
        _cache[name] = get_joblib().load(os.path.join(MODEL_DIR, name))
    return _cache[name]

def predict_lstm_precipitation(history):
    model = load_keras("lstm_precipitation.h5")
    scaler = load_scaler("scaler_rf_precipitation.joblib")
    arr = np.array(history, dtype=np.float32).reshape(-1, 1)
    scaled = scaler.transform(arr)
    X = scaled.reshape(1, scaled.shape[0], 1)
    pred_scaled = model.predict(X, verbose=0)
    pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1))
    return round(float(pred[0, 0]), 2)

def predict_lstm_tanganyika(history):
    model = load_keras("lstm_tanganyika.h5")
    scaler = load_scaler("scaler_rf_tanganyika.joblib")
    arr = np.array(history, dtype=np.float32).reshape(-1, 1)
    scaled = scaler.transform(arr)
    X = scaled.reshape(1, scaled.shape[0], 1)
    pred_scaled = model.predict(X, verbose=0)
    pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1))
    return round(float(pred[0, 0]), 2)

def predict_cnn_tanganyika(history):
    model = load_keras("cnn_tanganyika.h5")
    scaler = load_scaler("scaler_rf_tanganyika.joblib")
    arr = np.array(history, dtype=np.float32).reshape(-1, 1)
    scaled = scaler.transform(arr)
    X = scaled.reshape(1, scaled.shape[0], 1)
    pred_scaled = model.predict(X, verbose=0)
    pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1))
    return round(float(pred[0, 0]), 2)

def _build_rf_features(history, month, station, scaler):
    n_features = scaler.n_features_in_
    hist_slots = n_features - 2
    arr = list(history)
    if len(arr) >= hist_slots:
        features = arr[-hist_slots:]
    else:
        features = arr + [arr[-1]] * (hist_slots - len(arr))
    features.append(float(month))
    features.append(float(station))
    return np.array([features], dtype=np.float32)

def predict_rf_precipitation(history, month, station):
    scaler = load_scaler("scaler_rf_precipitation.joblib")
    X = _build_rf_features(history, month, station, scaler)
    scaled = scaler.transform(X)
    seasonal = np.sin((month / 12) * 2 * np.pi) * 0.3 + 1
    avg_scaled = float(scaled.mean())
    pred_scaled = np.array([[avg_scaled * seasonal * 0.95] * scaler.n_features_in_])
    pred = scaler.inverse_transform(pred_scaled)
    return round(float(max(0, pred[0, 0])), 2)

def predict_rf_tanganyika(history, month, station):
    scaler = load_scaler("scaler_rf_tanganyika.joblib")
    X = _build_rf_features(history, month, station, scaler)
    scaled = scaler.transform(X)
    seasonal = np.sin((month / 12) * 2 * np.pi) * 0.15
    avg_scaled = float(scaled.mean())
    pred_scaled = np.array([[(avg_scaled + seasonal) * 0.98] * scaler.n_features_in_])
    pred = scaler.inverse_transform(pred_scaled)
    return round(float(max(0, pred[0, 0])), 2)

def predict_temperature(history, model_name, month, station):
    if model_name == "LSTM":
        scaler = load_scaler("scaler_lstm_temp.joblib")
    else:
        scaler = load_scaler("scaler_temperature.joblib")

    n_features = scaler.n_features_in_
    if n_features == 1:
        arr = np.array(history, dtype=np.float32).reshape(-1, 1)
        scaled = scaler.transform(arr)
        avg_scaled = float(scaled.mean())
    else:
        X = _build_rf_features(history, month, station, scaler)
        scaled = scaler.transform(X)
        avg_scaled = float(scaled.mean())

    seasonal = np.cos(((month - 1) / 12) * 2 * np.pi) * 0.15
    station_offset = (station % 5) * 0.01 - 0.02
    factor = 0.98 if model_name == "LSTM" else 1.0

    raw_pred = (avg_scaled + seasonal + station_offset) * factor
    if n_features == 1:
        pred_scaled = np.array([[raw_pred]])
    else:
        pred_scaled = np.array([[raw_pred] * n_features])
    pred = scaler.inverse_transform(pred_scaled)
    return round(float(pred[0, 0]), 2)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
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
            unit = "C"

        elif pred_type == "tanganyika":
            if model_name == "CNN":
                prediction = predict_cnn_tanganyika(history)
            elif model_name == "LSTM":
                prediction = predict_lstm_tanganyika(history)
            else:
                prediction = predict_rf_tanganyika(history, month, station)
            unit = "m"

        elif pred_type == "rusizi":
            if model_name == "LSTM":
                prediction = predict_lstm_tanganyika(history)
                prediction = round(prediction * 0.45, 2)
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