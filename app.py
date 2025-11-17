import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
from mtcnn.mtcnn import MTCNN
import numpy as np
import os
import tempfile

# --- Configuration ---
MODEL_PATH = 'deepfake_detector_model_v2.keras'
IMG_SIZE = 224
FRAME_SAMPLE_RATE = 30  # How many frames to sample from the uploaded video
CONF_THRESHOLD = 0.5    # 50% threshold for "fake"

# --- Caching ---
# Cache the model and detector to prevent reloading on every run
@st.cache(allow_output_mutation=True)
def load_deepfake_model():
    """Loads the trained deepfake detection model."""
    try:
        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error(f"Make sure '{MODEL_PATH}' is in the same directory.")
        return None

@st.cache(allow_output_mutation=True)
def get_face_detector():
    """Initializes the MTCNN face detector."""
    return MTCNN()

def extract_faces_from_video(video_file_path, detector, frame_sample_rate):
    """
    Extracts faces from a single video file for prediction.
    This is the INFERENCE version of the preprocessing function.
    
    :param video_file_path: Path to the uploaded video file.
    :param detector: An initialized MTCNN detector.
    :param frame_sample_rate: The number of frames to sample.
    :return: A list of processed face images (numpy arrays).
    """
    processed_faces = []
    
    try:
        cap = cv2.VideoCapture(video_file_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if frame_count <= 0:
            st.warning("Could not read the video file. Is it a valid video?")
            return []

        # Calculate frame indices to sample
        frame_indices = np.linspace(0, frame_count - 1, frame_sample_rate, dtype=int)
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections = detector.detect_faces(frame_rgb)
            
            if detections:
                detection = detections[0]
                x, y, w, h = detection['box']
                x, y = max(0, x), max(0, y)
                
                face = frame[y:y+h, x:x+w]
                
                if face.size == 0:
                    continue
                
                # Resize and rescale for the model
                resized_face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
                rescaled_face = resized_face / 255.0  # Rescale to [0, 1]
                processed_faces.append(rescaled_face)

        cap.release()
        return processed_faces
        
    except Exception as e:
        st.error(f"An error occurred during video processing: {e}")
        return []

# --- Main App ---

st.title("Deepfake Video Detector 🎥")
st.write("Upload a video file to check if it contains a deepfake.")

# Load model and detector
model = load_deepfake_model()
detector = get_face_detector()

if model is None or detector is None:
    st.stop()

uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    
    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
        tfile.write(uploaded_file.read())
        temp_video_path = tfile.name

    st.video(uploaded_file)
    
    with st.spinner('Analyzing video... This may take a moment.'):
        # 1. Extract faces from the uploaded video
        faces = extract_faces_from_video(temp_video_path, detector, FRAME_SAMPLE_RATE)
        
        # Clean up the temporary file
        os.remove(temp_video_path)

        if not faces:
            st.error("No faces were detected in the video. Cannot analyze.")
        else:
            # 2. Run prediction on the extracted faces
            # Convert list of faces to a numpy array for the model
            faces_np = np.array(faces)
            
            # The model expects a batch
            predictions = model.predict(faces_np)
            
            # 3. Aggregate results
            # 'predictions' is a list of [probability_of_fake]
            # (assuming 'fake' is class 0)
            # Let's check our class indices from training.
            # If {'fake': 0, 'real': 1}, then a low score (near 0) is FAKE.
            # If {'fake': 1, 'real': 0}, then a high score (near 1) is FAKE.
            
            # Let's assume class indices were {'fake': 0, 'real': 1}
            # This means a prediction near 1.0 is 'real' and 0.0 is 'fake'
            # Let's re-read the generator logic.
            # It's usually alphabetical: 'fake' -> 0, 'real' -> 1
            # So, prediction = probability of being 'real'.
            
            # Let's define "fake_probability" = 1 - prediction
            fake_probabilities = [1 - p[0] for p in predictions]
            
            # Average the probability of "fake" across all detected faces
            avg_fake_prob = np.mean(fake_probabilities)
            
            # 4. Display verdict
            st.subheader("Analysis Result")
            
            if avg_fake_prob > CONF_THRESHOLD:
                st.error(f"**Result: DEEPFAKE DETECTED**")
                st.write(f"The model is **{avg_fake_prob*100:.2f}%** confident this is a deepfake.")
            else:
                st.success(f"**Result: LIKELY AUTHENTIC**")
                st.write(f"The model is **{(1-avg_fake_prob)*100:.2f}%** confident this is authentic.")
            
            st.info(f"Analyzed {len(faces)} faces from the video. Average 'fake' probability: {avg_fake_prob:.4f}")

            # Optional: Display predictions per frame
            with st.expander("Show detailed frame analysis"):
                st.write(fake_probabilities)