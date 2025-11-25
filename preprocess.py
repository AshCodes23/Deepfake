import os
import cv2
import json
from mtcnn.mtcnn import MTCNN
from tqdm import tqdm
import numpy as np
import math

# --- Configuration ---
# Path to your downloaded FaceForensics++ data
FFPP_DATA_PATH = '/home//DeepfakeProject/ffpp_data'  # Update if needed
# Path to the JSON files
SPLIT_PATH = './splits'
# Output path
OUTPUT_PATH = '/home//processed_data'
# Methods
MANIPULATION_METHODS = ['Deepfakes', 'Face2Face', 'FaceShifter', 'FaceSwap', 'NeuralTextures']
COMPRESSION = 'c23'
FRAME_SAMPLE_RATE = 30 
IMG_SIZE = 224

# Initialize the MTCNN face detector
detector = MTCNN()

def align_and_crop_face(frame, detection, target_size):
    """
    1. Calculates the angle between the eyes.
    2. Rotates the image to make eyes horizontal.
    3. Crops the face.
    """
    keypoints = detection['keypoints']
    box = detection['box']
    
    # 1. Get Eye Coordinates
    left_eye = keypoints['left_eye']
    right_eye = keypoints['right_eye']
    
    # 2. Calculate Angle
    # dy is the difference in height, dx is the difference in width
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx)) # Angle of rotation needed
    
    # 3. Get Center of the Face (Nose) to rotate around
    # We use the nose or the center of the bounding box
    face_center = (int(box[0] + box[2]/2), int(box[1] + box[3]/2))
    
    # 4. Create Rotation Matrix
    # Rotate around the center, by 'angle', scale 1.0
    M = cv2.getRotationMatrix2D(face_center, angle, 1.0)
    
    # 5. Rotate the entire Frame
    # (We rotate the whole frame to ensure we don't cut off corners of the face)
    h, w = frame.shape[:2]
    rotated_frame = cv2.warpAffine(frame, M, (w, h))
    
    # 6. Crop the Face from the Rotated Frame
    x, y, w, h = box
    # Ensure crop is within bounds
    x = max(0, x)
    y = max(0, y)
    
    # Add a little padding (10%) to ensure we get the whole chin/forehead after rotation
    padding_w = int(w * 0.1)
    padding_h = int(h * 0.1)
    
    face_crop = rotated_frame[
        max(0, y - padding_h) : min(rotated_frame.shape[0], y + h + padding_h),
        max(0, x - padding_w) : min(rotated_frame.shape[1], x + w + padding_w)
    ]
    
    if face_crop.size == 0:
        return None

    # 7. Resize to target size
    return cv2.resize(face_crop, (target_size, target_size))

def extract_faces(video_path, output_folder, video_id, frame_sample_rate):
    if not os.path.exists(video_path):
        # print(f"Warning: Video not found {video_path}") 
        return

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if frame_count <= 0:
        return
        
    frame_indices = np.linspace(0, frame_count - 1, frame_sample_rate, dtype=int)
    
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
            # Get the most confident face
            detection = detections[0]
            
            # --- NEW TECHNIQUE: ALIGNMENT ---
            # Instead of just cropping, we align then crop
            processed_face = align_and_crop_face(frame, detection, IMG_SIZE)
            
            if processed_face is None:
                continue
            
            # Save the face
            output_filename = f"{video_id}_frame{idx}.png"
            output_filepath = os.path.join(output_folder, output_filename)
            cv2.imwrite(output_filepath, processed_face)

    cap.release()

def process_split(split_name):
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
    for pair in tqdm(split_data, desc=f"Processing {split_name} videos"):
        id1, id2 = pair[0], pair[1]
        
        # Process REAL videos (id1 and id2)
        real_video_path_1 = os.path.join(FFPP_DATA_PATH, 'original_sequences', 'youtube', COMPRESSION, 'videos', f'{id1}.mp4')
        extract_faces(real_video_path_1, real_output_dir, id1, FRAME_SAMPLE_RATE)
        
        real_video_path_2 = os.path.join(FFPP_DATA_PATH, 'original_sequences', 'youtube', COMPRESSION, 'videos', f'{id2}.mp4')
        extract_faces(real_video_path_2, real_output_dir, id2, FRAME_SAMPLE_RATE)
        
        # Process FAKE videos
        for method in MANIPULATION_METHODS:
            fake_video_name_1 = f'{id1}_{id2}.mp4'
            fake_video_path_1 = os.path.join(FFPP_DATA_PATH, 'manipulated_sequences', method, COMPRESSION, 'videos', fake_video_name_1)
            extract_faces(fake_video_path_1, fake_output_dir, f'{method}_{id1}_{id2}', FRAME_SAMPLE_RATE)

            fake_video_name_2 = f'{id2}_{id1}.mp4'
            fake_video_path_2 = os.path.join(FFPP_DATA_PATH, 'manipulated_sequences', method, COMPRESSION, 'videos', fake_video_name_2)
            extract_faces(fake_video_path_2, fake_output_dir, f'{method}_{id2}_{id1}', FRAME_SAMPLE_RATE)


if __name__ == "__main__":
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # Note: If you already processed data, you might want to clear the folder first
    # or just run this to overwrite.
    
    # process_split('train') # Uncomment if you want to re-process training data
    process_split('val')
    process_split('test')
    
    print("\n--- Preprocessing Complete with Face Alignment ---")
