import os
import time
from typing import Any, Dict, List, Union

import numpy as np
from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)


weather_classes: List[str] = [
    "clear", "cloudy", "drizzly", "foggy", "hazy",
    "misty", "rain", "smoky", "thunderstorm"
]

def _load_model() -> Any:
    candidates = [
        "model/model.pkl",   
        "model.pkl",         
    ]
    last_err = None
    for path in candidates:
        if os.path.exists(path):
            try:
                return joblib.load(path)
            except Exception as e:
                last_err = e
    if last_err:
        raise last_err
    raise FileNotFoundError(f"Model file not found. Tried: {', '.join(candidates)}")

model = _load_model()


def _to_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)

def _extract_features_from_payload(payload: Dict[str, Any]) -> np.ndarray:
    """
    Expected fields (form.html): temperature, pressure, humidity, wind_speed, wind_deg,
    optional: rain_1h, rain_3h, snow, clouds
    """
    temperature = _to_float(payload.get("temperature"), 20)
    pressure    = _to_float(payload.get("pressure"),    1013)
    humidity    = _to_float(payload.get("humidity"),      60)
    wind_speed  = _to_float(payload.get("wind_speed"),     5)
    wind_deg    = _to_float(payload.get("wind_deg"),       0)
    rain_1h     = _to_float(payload.get("rain_1h", 0),     0)
    rain_3h     = _to_float(payload.get("rain_3h", 0),     0)
    snow        = _to_float(payload.get("snow", 0),        0)
    clouds      = _to_float(payload.get("clouds", 0),      0)

    return np.array([
        temperature, pressure, humidity,
        wind_speed, wind_deg, rain_1h,
        rain_3h, snow, clouds
    ], dtype=float).reshape(1, -1)

def _get_payload() -> Dict[str, Any]:
    data = request.get_json(silent=True)
    if isinstance(data, dict):
        return data
    return request.form.to_dict()

def classify_weather(features: np.ndarray):
    """
    Predict canonical label + latency (ms). Works for index or string-returning models.
    """
    start = time.time()
    y = model.predict(features)[0]
    latency_ms = round((time.time() - start) * 1000, 2)

    # Model returns an index
    if isinstance(y, (int, np.integer)):
        idx = int(y)
        if not (0 <= idx < len(weather_classes)):
            idx = max(0, min(idx, len(weather_classes) - 1))
        label = weather_classes[idx]
    else:
        # Model returns a string
        label = str(y).strip().lower()
        synonyms = {"rainy": "rain", "hazey": "hazy", "smokey": "smoky"}
        label = synonyms.get(label, label)
        if label not in weather_classes:
            label = "clear"

    return label, latency_ms


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Minimal validation for required form fields
        required = ["temperature", "pressure", "humidity", "wind_speed", "wind_deg"]
        for f in required:
            if f not in request.form or str(request.form[f]).strip() == "":
                return render_template("form.html", error=f"Error: {f} is missing."), 400
        try:
            X = _extract_features_from_payload(request.form)
            prediction, latency = classify_weather(X)
            # Ensure the label is visible in the HTML (integration test looks for it)
            return render_template("result.html", prediction=prediction, latency=latency), 200
        except Exception as e:
            return render_template("form.html", error=f"Error processing input: {e}"), 400

   
    try:
        return render_template("form.html"), 200
    except Exception:
     
        return "Form template not found.", 200

@app.route("/predict", methods=["POST"])
def predict():
    """
    Programmatic endpoint (used by integration tests).
    Returns plain text containing the raw label so it's easily found.
    """
    try:
        payload = _get_payload()
        X = _extract_features_from_payload(payload)
        prediction, latency = classify_weather(X)
        return f"Prediction: {prediction}", 200

	import os
import time
from typing import Any, Dict, List, Union

import numpy as np
from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)


weather_classes: List[str] = [
    "clear", "cloudy", "drizzly", "foggy", "hazy",
    "misty", "rain", "smoky", "thunderstorm"
]


def _load_model() -> Any:
    candidates = [
        "model/model.pkl",   
        "model.pkl",         
    ]
    last_err = None
    for path in candidates:
        if os.path.exists(path):
            try:
                return joblib.load(path)
            except Exception as e:
                last_err = e
    if last_err:
        raise last_err
    raise FileNotFoundError(f"Model file not found. Tried: {', '.join(candidates)}")


model = _load_model()


def _to_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)

def _extract_features_from_payload(payload: Dict[str, Any]) -> np.ndarray:
    """
    Expected fields (form.html): temperature, pressure, humidity, wind_speed, wind_deg,
    optional: rain_1h, rain_3h, snow, clouds
    """
    temperature = _to_float(payload.get("temperature"), 20)
    pressure    = _to_float(payload.get("pressure"),    1013)
    humidity    = _to_float(payload.get("humidity"),      60)
    wind_speed  = _to_float(payload.get("wind_speed"),     5)
    wind_deg    = _to_float(payload.get("wind_deg"),       0)
    rain_1h     = _to_float(payload.get("rain_1h", 0),     0)
    rain_3h     = _to_float(payload.get("rain_3h", 0),     0)
    snow        = _to_float(payload.get("snow", 0),        0)
    clouds      = _to_float(payload.get("clouds", 0),      0)

    return np.array([
        temperature, pressure, humidity,
        wind_speed, wind_deg, rain_1h,
        rain_3h, snow, clouds
    ], dtype=float).reshape(1, -1)

def _get_payload() -> Dict[str, Any]:
    data = request.get_json(silent=True)
    if isinstance(data, dict):
        return data
    return request.form.to_dict()

def classify_weather(features: np.ndarray):
    """
    Predict canonical label + latency (ms). Works for index or string-returning models.
    """
    start = time.time()
    y = model.predict(features)[0]
    latency_ms = round((time.time() - start) * 1000, 2)

    # Model returns an index
    if isinstance(y, (int, np.integer)):
        idx = int(y)
        if not (0 <= idx < len(weather_classes)):
            idx = max(0, min(idx, len(weather_classes) - 1))
        label = weather_classes[idx]
    else:
      
        label = str(y).strip().lower()
        synonyms = {"rainy": "rain", "hazey": "hazy", "smokey": "smoky"}
        label = synonyms.get(label, label)
        if label not in weather_classes:
            label = "clear"

    return label, latency_ms



@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":

		
        required = ["temperature", "pressure", "humidity", "wind_speed", "wind_deg"]
        for f in required:
            if f not in request.form or str(request.form[f]).strip() == "":
                return render_template("form.html", error=f"Error: {f} is missing."), 400
        try:
            X = _extract_features_from_payload(request.form)
            prediction, latency = classify_weather(X)

			
            return render_template("result.html", prediction=prediction, latency=latency), 200
        except Exception as e:
            return render_template("form.html", error=f"Error processing input: {e}"), 400

    try:
        return render_template("form.html"), 200
    except Exception:

		
        return "Form template not found.", 200

@app.route("/predict", methods=["POST"])
def predict():

	
    try:
        payload = _get_payload()
        X = _extract_features_from_payload(payload)
        prediction, latency = classify_weather(X)
        return f"Prediction: {prediction}", 200

    except Exception as e:
        return f"Error processing input: {e}", 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

    except Exception as e:
        return f"Error processing input: {e}", 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
