# DataForge Emotion Recognition API

A FastAPI-based backend service for predicting **emotion** and **language** from uploaded audio files using a trained TensorFlow/Keras model.

The API receives an audio file, preprocesses it into a mel spectrogram, performs model inference, and returns the predicted emotion and language along with confidence scores.

## Features

* Audio-based emotion prediction
* English/Urdu language classification
* TensorFlow/Keras model inference
* Mel spectrogram preprocessing with librosa
* Image-based spectrogram processing using OpenCV
* Lazy model loading for efficient startup
* Google Drive model download using gdown
* Health check endpoint for frontend warm-up
* Docker support for deployment

## Supported Audio Formats

The API supports the following audio file types:

```txt
.wav
.webm
.mp3
.flac
.m4a
.ogg
```

## Tech Stack

* Python
* FastAPI
* TensorFlow/Keras
* librosa
* OpenCV
* NumPy
* gdown
* Docker

## Project Structure

```txt
.
├── main.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
└── README.md
```

## Environment Variables

Create a `.env` file in the root directory for local development:

```env
MODEL_FILE_ID=1jspxrqwsShQp6YuiA0H7NIoYXt0H6xBD
MODEL_PATH=./best_of_best_model.keras
```

| Variable        | Description                                                   |
| --------------- | ------------------------------------------------------------- |
| `MODEL_FILE_ID` | Google Drive file ID of the trained Keras model               |
| `MODEL_PATH`    | Local path where the model will be downloaded and loaded from |

## Installation

Clone the repository:

```bash
git clone git@github.com:Alixoxox/Speech-Recognition-Backend.git
cd Speech-Recognition-Backend
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run Locally

Start the FastAPI server:

```bash
python main.py
```

The API will be available at:

```txt
http://localhost:8000
```

## API Endpoints

### Root Endpoint

```http
GET /
```

Returns basic API information and available routes.

Example response:

```json
{
  "message": "DataForge Emotion Recognition API",
  "endpoints": {
    "health": "/health",
    "predict": "/predict",
    "docs": "/docs"
  }
}
```

### Health Check

```http
GET /health
```

Checks API status and loads the model if it is not already loaded.

Example response:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2026-06-14T12:00:00.000000"
}
```

### Predict Emotion and Language

```http
POST /predict
```

Accepts an audio file using `multipart/form-data`.

Form-data key:

```txt
file
```

Example response:

```json
{
  "emotion": "Happy",
  "emotion_confidence": 0.92,
  "language": "English",
  "language_confidence": 0.88,
  "success": true
}
```

## Interactive API Docs

FastAPI provides interactive documentation at:

```txt
http://localhost:8000/docs
```

## Docker Setup

Build the Docker image:

```bash
docker build -t emotion-api .
```

Run the container:

```bash
docker run -p 8000:8000 ^
  -e MODEL_FILE_ID=1jspxrqwsShQp6YuiA0H7NIoYXt0H6xBD ^
  -e MODEL_PATH=./best_of_best_model.keras ^
  emotion-api
```

For Windows PowerShell:

```bash
docker run -p 8000:8000 `
  -e MODEL_FILE_ID=1jspxrqwsShQp6YuiA0H7NIoYXt0H6xBD `
  -e MODEL_PATH=./best_of_best_model.keras `
  emotion-api
```

## Deployment

For deployment on platforms such as Render or Railway:

1. Push the backend code to GitHub.
2. Add the required environment variables in the platform dashboard.
3. Deploy using the included `Dockerfile` or the platform’s Python runtime.
4. Call `/health` when the frontend loads to prepare the model.
5. Use `/predict` to submit audio files for inference.

Required environment variables:

```txt
MODEL_FILE_ID=1jspxrqwsShQp6YuiA0H7NIoYXt0H6xBD
MODEL_PATH=./best_of_best_model.keras
```

## Model Labels

### Emotion Classes

```txt
Calm
Happy
Sad
Stressed
Excited
Angry
```

### Language Classes

```txt
English
Urdu
```

## Author

**Sufyan Ali**
Computer Systems Engineering Student
NED University
