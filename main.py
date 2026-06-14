from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import librosa
import cv2
from tensorflow.keras.models import load_model
import tempfile
import os
from pydantic import BaseModel
from datetime import datetime
import asyncio
import gdown
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(
    title="DataForge Emotion Recognition API",
    description="Real-time emotion and language detection from audio",
    version="1.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Labels
EMOTION_MAP = {
    0: "Calm",
    1: "Happy",
    2: "Sad",
    3: "Stressed",
    4: "Excited",
    5: "Angry"
}

LANG_MAP = {
    0: "English",
    1: "Urdu"
}

# Model config
MODEL_PATH = os.getenv("MODEL_PATH", "./best_of_best_model.keras")
MODEL_FILE_ID = os.getenv("MODEL_FILE_ID")

# Global model
best_model = None
model_lock = asyncio.Lock()


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str


class PredictionResponse(BaseModel):
    emotion: str
    emotion_confidence: float
    language: str
    language_confidence: float
    success: bool


def download_model_if_missing():
    """
    Downloads model from Google Drive only if it does not already exist.
    """
    if os.path.exists(MODEL_PATH):
        print(f"Model already exists at: {MODEL_PATH}")
        return

    if not MODEL_FILE_ID:
        raise RuntimeError("MODEL_FILE_ID environment variable is missing")

    print("Downloading model from Google Drive...")

    url = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"

    output = gdown.download(
        url=url,
        output=MODEL_PATH,
        quiet=False,
    )

    if output is None or not os.path.exists(MODEL_PATH):
        raise RuntimeError("Model download failed")

    print(f"Model downloaded to: {MODEL_PATH}")


async def ensure_model_loaded():
    """
    Lazy-loads model.
    Called by /health and /predict.
    """
    global best_model

    if best_model is not None:
        return

    async with model_lock:
        if best_model is not None:
            return

        download_model_if_missing()

        print("Loading model into memory...")
        best_model = load_model(MODEL_PATH, compile=False)
        print("Model loaded successfully")


@app.get("/")
async def root():
    return {
        "message": "DataForge Emotion Recognition API",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health endpoint also warms up the model.

    Frontend should call this when user visits the site.
    """
    try:
        await ensure_model_loaded()

        return HealthResponse(
            status="healthy",
            model_loaded=True,
            timestamp=datetime.utcnow().isoformat()
        )

    except Exception as e:
        print(f"Health check failed: {e}")

        return HealthResponse(
            status="degraded",
            model_loaded=False,
            timestamp=datetime.utcnow().isoformat()
        )


def validate_audio_file(file: UploadFile):
    allowed_extensions = [".wav", ".webm", ".mp3", ".flac", ".m4a", ".ogg"]

    filename = file.filename or ""
    ext = os.path.splitext(filename)[1].lower()

    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )


def preprocess_audio(audio_bytes: bytes, suffix: str = ".wav"):
    """
    Converts audio bytes into mel spectrogram image tensor.
    """
    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        y, sr = librosa.load(tmp_path, sr=16000)

        if len(y) == 0:
            raise ValueError("Audio file is empty or unreadable")

        yt, _ = librosa.effects.trim(y, top_db=30)

        if len(yt) == 0:
            yt = y

        S = librosa.feature.melspectrogram(
            y=yt,
            sr=sr,
            n_mels=128
        )

        S_dB = librosa.power_to_db(S, ref=np.max)

        S_norm = cv2.normalize(
            S_dB,
            None,
            0,
            255,
            cv2.NORM_MINMAX,
            dtype=cv2.CV_8U
        )

        img_resized = cv2.resize(
            S_norm,
            (224, 224),
            interpolation=cv2.INTER_CUBIC
        )

        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)

        X_input = np.expand_dims(img_rgb, axis=0).astype(np.float32)

        return X_input

    except Exception as e:
        raise RuntimeError(f"Audio preprocessing failed: {str(e)}")

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/predict", response_model=PredictionResponse)
async def predict_emotion(file: UploadFile = File(...)):
    """
    Predict emotion and language from audio file.

    Accepts: wav, webm, mp3, flac, m4a, ogg
    Returns: emotion, language, and confidence scores
    """
    try:
        await ensure_model_loaded()

        validate_audio_file(file)

        audio_bytes = await file.read()

        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")

        ext = os.path.splitext(file.filename or "")[1].lower() or ".wav"

        X_input = preprocess_audio(audio_bytes, suffix=ext)

        preds = best_model.predict(X_input, verbose=0)

        emotion_probs = preds[0][0]
        language_probs = preds[1][0]

        emotion_idx = int(np.argmax(emotion_probs))
        emotion_confidence = float(emotion_probs[emotion_idx])

        language_score = float(language_probs[0])
        language_idx = 1 if language_score > 0.5 else 0
        language_confidence = float(max(language_score, 1 - language_score))

        return PredictionResponse(
            emotion=EMOTION_MAP[emotion_idx],
            emotion_confidence=emotion_confidence,
            language=LANG_MAP[language_idx],
            language_confidence=language_confidence,
            success=True
        )

    except HTTPException:
        raise

    except Exception as e:
        print(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )