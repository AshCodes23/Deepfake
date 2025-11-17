import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
from mtcnn.mtcnn import MTCNN
import numpy as np
import os
import tempfile
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Deepfake Detector",
    page_icon="📸",
    layout="wide"
)

# --- Configuration ---
# This section was missing, causing the NameError. It's now fixed.
MODEL_PATH = 'deepfake_detector_model_v2.keras'
IMG_SIZE = 224
FRAME_SAMPLE_RATE = 30  # How many frames to sample
CONF_THRESHOLD = 0.5    # 50% threshold

# --- Custom CSS for Instagram UI ---
st.markdown(
    """
    <style>
    /* Main app layout */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Fake Navbar */
    .navbar {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        background-color: #ffffff;
        padding: 1rem 1.5rem;
        border-bottom: 1px solid #dbdbdb;
        z-index: 1000;
        font-weight: 500;
        font-size: 1.2rem;
    }
    
    /* Main content container (centered) */
    .main-container {
        max-width: 600px; /* Controls the width of the center block */
        margin: 60px auto 0 auto; /* 60px top margin to clear the navbar */
        text-align: center;
    }

    /* Instagram Gradient Title */
    .instagram-title {
        font-size: 3.5rem;
        font-weight: 700;
        background: -webkit-linear-gradient(45deg, #f09433 0%, #e6683c 25%, #dc2743 50%, #cc2366 75%, #bc1888 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }

    /* Style the file uploader */
    .stFileUploader {
        margin-top: 1rem;
        margin-bottom: 2rem;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #aaa;
        font-style: italic;
        margin-top: 4rem;
        border-top: 1px solid #eee;
        padding-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Navbar ---
st.markdown('<div class="navbar">Deepfake Shield 🛡️</div>', unsafe_allow_html=True)

# --- Model & Helper Functions (Cached) ---
@st.cache_resource
def load_deepfake_model():
    """Loads the trained deepfake detection model."""
    try:
        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_resource
def get_face_detector():
    """Initializes the MTCNN face detector."""
    return MTCNN()

def extract_faces_from_video(video_file_path, detector, frame_sample_rate):
    """
    Extracts, crops, and resizes faces from a video file.
    """
    processed_faces = []
    try:
        cap = cv2.VideoCapture(video_file_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            st.warning("Could not read the video file. Is it a valid video?")
            return []

        frame_indices = np.linspace(0, frame_count - 1, frame_sample_rate, dtype=int)
        
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
        st.error(f"An error occurred during video processing: {e}")
        return []

# --- Main App Interface (Centered) ---

# We wrap the main content in our custom 'main-container' div
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# 1. The Title
st.markdown('<h1 class="instagram-title">Deepfake Detector</h1>', unsafe_allow_html=True)

# 2. The Uploader (in the center)
uploaded_file = st.file_uploader(
    "Upload a video file to analyze", 
    type=["mp4", "mov", "avi"],
    label_visibility="collapsed"
)

# 3. The Logic and Results Display
if uploaded_file is not None:
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
        tfile.write(uploaded_file.read())
        temp_video_path = tfile.name

    # Display the uploaded video
    st.video(uploaded_file)
    
    with st.spinner('Analyzing... 🕵️‍♂️ This may take a moment.'):
        model = load_deepfake_model()
        detector = get_face_detector()
        
        if model is None or detector is None:
            st.error("Model or face detector failed to load. The app cannot continue.")
            st.stop()

        faces = extract_faces_from_video(temp_video_path, detector, FRAME_SAMPLE_RATE)
        os.remove(temp_video_path)

        if not faces:
            st.error("No faces were detected in the video. Cannot analyze.")
        else:
            faces_np = np.array(faces)
            predictions = model.predict(faces_np)
            
            real_probabilities = [p[0] for p in predictions]
            avg_real_prob = np.mean(real_probabilities)
            avg_fake_prob = 1 - avg_real_prob
            
            st.markdown("---") # Visual separator
            
            if avg_fake_prob > CONF_THRESHOLD:
                st.subheader("Verdict: ❌ Deepfake Detected")
                st.error("Our model has detected a high probability of manipulation.", icon="🚨")
                
                confidence = avg_fake_prob
                st.metric(label="Fake Confidence", value=f"{confidence * 100:.2f}%")
                # We must convert confidence to a standard float
                st.progress(float(confidence))

            else:
                st.subheader("Verdict: ✅ Appears Authentic")
                st.success("Our model indicates this video is likely not a deepfake.", icon="👍")
                
                confidence = avg_real_prob
                st.metric(label="Authentic Confidence", value=f"{confidence * 100:.2f}%")
                # We must convert confidence to a standard float
                st.progress(float(confidence))
            
            st.markdown("---")
            with st.expander("Show Detailed Frame-by-Frame Analysis"):
                st.info(f"Analyzed {len(faces)} faces from the video.")
                st.write("Probability of each frame being 'real':")
                st.json(real_probabilities)

# Close the main-container div
st.markdown('</div>', unsafe_allow_html=True)

# --- Footer ---
st.markdown(
    '<div class="footer">"The truth is rarely pure and never simple." - Oscar Wilde</div>',
    unsafe_allow_html=True
)
