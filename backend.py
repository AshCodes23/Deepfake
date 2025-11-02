import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware # To allow our frontend to talk to it
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
from mtcnn.mtcnn import MTCNN
import numpy as np
import os
import tempfile
import logging

# --- Configuration ---
MODEL_PATH = 'deepfake_detector_model_v2.keras'
IMG_SIZE = 224
FRAME_SAMPLE_RATE = 30

# --- App Initialization ---
app = FastAPI(title="Deepfake Detection API")

# --- Allow Cross-Origin Requests (CORS) ---
# This is critical to allow your lovable.dev frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (POST, GET, etc.)
    allow_headers=["*"],  # Allows all headers
)

# --- Load Models ---
# We use a simple "global" variable to hold the models
model = None
detector = None

@app.on_event("startup")
def load_models():
    """Load the Keras model and MTCNN detector when the server starts."""
    global model, detector
    try:
        model = load_model(MODEL_PATH)
        detector = MTCNN()
        logging.info(f"Model {MODEL_PATH} loaded successfully.")
        logging.info("MTCNN detector initialized successfully.")
    except Exception as e:
        logging.error(f"Error loading models: {e}")
        raise RuntimeError(f"Could not load models: {e}")

# --- Re-usable Face Extraction Function ---
def extract_faces_from_video(video_file_path):
    processed_faces = []
    try:
        cap = cv2.VideoCapture(video_file_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if frame_count <= 0:
            return []

        frame_indices = np.linspace(0, frame_count - 1, FRAME_SAMPLE_RATE, dtype=int)

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()

            if not ret: continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections = detector.detect_faces(frame_rgb)

            if detections:
                detection = detections[0]
                x, y, w, h = detection['box']
                x, y = max(0, x), max(0, y)
                face = frame[y:y+h, x:x+w]

                if face.size == 0: continue

                resized_face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
                rescaled_face = resized_face / 255.0  # Normalize
                processed_faces.append(rescaled_face)
        cap.release()
        return processed_faces
    except Exception as e:
        logging.error(f"Error extracting faces: {e}")
        return []

# --- API Endpoint ---
@app.post("/analyze/")
async def analyze_video(file: UploadFile = File(...)):
    """
    Receives an uploaded video, processes it, and returns the result.
    """
    if not model or not detector:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    # Save the uploaded file to a temporary path
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
        tfile.write(await file.read())
        temp_video_path = tfile.name

    try:
        # 1. Extract Faces
        faces = extract_faces_from_video(temp_video_path)

        if not faces:
            raise HTTPException(status_code=400, detail="No faces detected in the video.")

        # 2. Run Prediction
        faces_np = np.array(faces)
        predictions = model.predict(faces_np)

        # 3. Aggregate results
        # From training, we know: {'fake': 0, 'real': 1}
        # p[0] is the probability of being 'real'
        real_probabilities = [float(p[0]) for p in predictions]
        avg_real_prob = np.mean(real_probabilities)
        avg_fake_prob = 1 - avg_real_prob

        # 4. Return JSON response
        return {
            "status": "success",
            "faces_detected": len(faces),
            "average_fake_probability": avg_fake_prob,
            "average_real_probability": avg_real_prob,
            "is_deepfake": bool(avg_fake_prob > 0.5)
        }
    except Exception as e:
        logging.error(f"Error during analysis: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

@app.get("/")
def read_root():
    return {"message": "Deepfake Detection API is running."}

# --- Run the App ---
if __name__ == "__main__":
    # Set logging level
    logging.basicConfig(level=logging.INFO)
    # This runs the API server on localhost port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
