# Burundi Climate Prediction API

## Setup

1. Create a GitHub repo and push this folder's contents
2. Add your model files into a `models/` subfolder:
   ```
   models/
     lstm_precipitation.h5
     lstm_tanganyika.h5
     cnn_tanganyika.h5
     scaler_lstm_temp.joblib
     scaler_rf_precipitation.joblib
     scaler_rf_tanganyika.joblib
     scaler_temperature.joblib
   ```
3. Deploy on **Render**:
   - New → Web Service → connect your GitHub repo
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120`
   - Instance type: Free or Starter

## API

**POST /predict**
```json
{
  "model": "LSTM",
  "type": "precip",
  "month": 3,
  "station": 1,
  "history": [80, 95, 60]
}
```

Response:
```json
{
  "prediction": 78.5,
  "model": "LSTM",
  "month": 3,
  "station": 1,
  "unit": "mm"
}
```
