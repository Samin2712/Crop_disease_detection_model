import os
import joblib
import numpy as np
import cv2

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from skimage.feature import hog
import uvicorn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models_out")

CROP_MODELS = {
    "rice": os.path.join(MODEL_DIR, "rice_model.pkl"),
    "jute": os.path.join(MODEL_DIR, "jute_model.pkl"),
    "potato": os.path.join(MODEL_DIR, "potato_model.pkl"),
    "tomato": os.path.join(MODEL_DIR, "tomato_model.pkl"),
}

_loaded = {}

def softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - np.max(x)
    exps = np.exp(x)
    return exps / (np.sum(exps) + 1e-12)

def extract_features(img_bgr, size=(96, 96)) -> np.ndarray:
    img = cv2.resize(img_bgr, size, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog_feat = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-8)

    return np.concatenate([hog_feat, hist]).astype(np.float32)

def load_bundle(crop: str):
    if crop not in CROP_MODELS:
        raise ValueError(f"Unsupported crop '{crop}'. Use one of: {list(CROP_MODELS.keys())}")

    if crop in _loaded:
        return _loaded[crop]

    path = CROP_MODELS[crop]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found for '{crop}': {path}")

    bundle = joblib.load(path)
    _loaded[crop] = bundle
    return bundle

def predict_image_bytes(crop: str, image_bytes: bytes) -> dict:
    b = load_bundle(crop)
    model = b["model"]
    scaler = b.get("scaler", None)
    le = b["label_encoder"]
    model_name = b.get("model_name", "Unknown")
    feat_info = b.get("feature_info", {})
    img_size = tuple(feat_info.get("img_size", (96, 96)))

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image uploaded.")

    X = extract_features(img, size=img_size).reshape(1, -1)

    if scaler is not None and model_name != "RandomForest":
        X = scaler.transform(X)

    pred_idx = int(model.predict(X)[0])
    pred_label = str(le.inverse_transform([pred_idx])[0])

    probs = None
    conf = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0].astype(float)
        conf = float(np.max(probs))
    elif hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X)).reshape(-1)
        probs = softmax(scores)
        conf = float(np.max(probs))

    topk = []
    if probs is not None:
        order = np.argsort(-probs)[: min(5, len(probs))]
        for i in order:
            topk.append({"label": str(le.classes_[int(i)]), "score": float(probs[int(i)])})

    return {
        "crop": crop,
        "model": model_name,
        "prediction": pred_label,
        "confidence": conf,
        "topk": topk,
    }

app = FastAPI(title="Multi-crop Disease Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "null"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok", "supported_crops": list(CROP_MODELS.keys())}

@app.post("/predict")
async def predict(crop: str = Form(...), image: UploadFile = File(...)):
    data = await image.read()
    try:
        return predict_image_bytes(crop, data)
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)
