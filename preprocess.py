import os
import cv2
import json
from mtcnn.mtcnn import MTCNN
from tqdm import tqdm
import numpy as np

# --- Configuration ---
# Path to your downloaded FaceForensics++ data (from Recommendation 1)
FFPP_DATA_PATH = './ffpp_data' 
# Path to the JSON files you just downloaded
SPLIT_PATH = './splits'
# Where you want to save the processed face images
OUTPUT_PATH = './processed_data'
# All the manipulation methods you downloaded
MANIPULATION_METHODS = ['Deepfakes', 'Face2Face', 'FaceShifter', 'FaceSwap', 'NeuralTextures']
# The compression level you downloaded
COMPRESSION = 'c23'
# Number of frames to sample per video. 
# A lower number is faster but gives less data. 30 is a good start.
FRAME_SAMPLE_RATE = 30 
# The final size of the face image
IMG_SIZE = 224

# Initialize the MTCNN face detector
detector = MTCNN()

def extract_faces(video_path, output_folder, video_id, frame_sample_rate):
    """
    Extracts faces from a video file and saves them as images.
    
    :param video_path: Path to the input video file.
    :param output_folder: Folder to save the cropped face images.
    :param video_id: The name of the video (used for file naming).
    :param frame_sample_rate: How many frames to extract.
    """
    if not os.path.exists(video_path):
        print(f"Warning: Video not found {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame indices to sample
    if frame_count <= 0:
        print(f"Warning: Could not read video {video_path}")
        return
        
    frame_indices = np.linspace(0, frame_count - 1, frame_sample_rate, dtype=int)
    
    frame_num = 0
    saved_face_count = 0
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if not ret:
            continue
            
        # MTCNN expects RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        detections = detector.detect_faces(frame_rgb)
        
        if detections:
            # Get the first and most confident face
            detection = detections[0]
            x, y, w, h = detection['box']
            
            # Ensure coordinates are valid
            x, y = max(0, x), max(0, y)
            
            # Crop the face
            face = frame[y:y+h, x:x+w]
            
            if face.size == 0:
                continue

            # Resize to standard size
            resized_face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
            
            # Save the face
            output_filename = f"{video_id}_frame{idx}.png"
            output_filepath = os.path.join(output_folder, output_filename)
            cv2.imwrite(output_filepath, resized_face)
            saved_face_count += 1

    cap.release()

def process_split(split_name):
    """
    Processes a whole split (train, val, or test) based on the JSON file.
    """
    print(f"\n--- Processing {split_name} split ---")
    
    # 1. Load the JSON split file
    split_file = os.path.join(SPLIT_PATH, f'{split_name}.json')
    with open(split_file, 'r') as f:
        split_data = json.load(f)

    # 2. Create output directories
    real_output_dir = os.path.join(OUTPUT_PATH, split_name, 'real')
    fake_output_dir = os.path.join(OUTPUT_PATH, split_name, 'fake')
    os.makedirs(real_output_dir, exist_ok=True)
    os.makedirs(fake_output_dir, exist_ok=True)

    # 3. Process videos
    # The JSON contains pairs [id1, id2], meaning id1 was used to fake id2
    for pair in tqdm(split_data, desc=f"Processing {split_name} videos"):
        id1, id2 = pair[0], pair[1]
        
        # --- Process REAL videos ---
        # We process both videos in the pair as real
        
        # Process id1
        real_video_path_1 = os.path.join(FFPP_DATA_PATH, 'original_sequences', 'youtube', COMPRESSION, 'videos', f'{id1}.mp4')
        extract_faces(real_video_path_1, real_output_dir, id1, FRAME_SAMPLE_RATE)
        
        # Process id2
        real_video_path_2 = os.path.join(FFPP_DATA_PATH, 'original_sequences', 'youtube', COMPRESSION, 'videos', f'{id2}.mp4')
        extract_faces(real_video_path_2, real_output_dir, id2, FRAME_SAMPLE_RATE)
        
        # --- Process FAKE videos ---
        # The fakes are named as {id1}_{id2}.mp4 or {id2}_{id1}.mp4
        
        for method in MANIPULATION_METHODS:
            # Fake 1: {id1}_{id2}
            fake_video_name_1 = f'{id1}_{id2}.mp4'
            fake_video_path_1 = os.path.join(FFPP_DATA_PATH, 'manipulated_sequences', method, COMPRESSION, 'videos', fake_video_name_1)
            extract_faces(fake_video_path_1, fake_output_dir, f'{method}_{id1}_{id2}', FRAME_SAMPLE_RATE)

            # Fake 2: {id2}_{id1}
            fake_video_name_2 = f'{id2}_{id1}.mp4'
            fake_video_path_2 = os.path.join(FFPP_DATA_PATH, 'manipulated_sequences', method, COMPRESSION, 'videos', fake_video_name_2)
            extract_faces(fake_video_path_2, fake_output_dir, f'{method}_{id2}_{id1}', FRAME_SAMPLE_RATE)


if __name__ == "__main__":
    # Create the main output directory
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # Process all three splits
    # process_split('train')
    process_split('val')
    process_split('test')
    
    print("\n--- Preprocessing Complete ---")
    print(f"Data saved to: {OUTPUT_PATH}")