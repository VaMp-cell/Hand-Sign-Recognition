import csv
import copy
import itertools
from collections import Counter, deque
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import numpy as np
import base64
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Import your existing classifiers
from model.keypoint_classifier.keypoint_classifier import KeyPointClassifier
from model.point_history_classifier.point_history_classifier import PointHistoryClassifier

app = FastAPI()

# Initialize classifiers
keypoint_classifier = KeyPointClassifier(score_th=0.75)
point_history_classifier = PointHistoryClassifier(score_th=0.9)

# Load labels
with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
    keypoint_classifier_labels = [row[0] for row in csv.reader(f) if row]
with open('model/point_history_classifier/point_history_classifier_label.csv', encoding='utf-8-sig') as f:
    point_history_classifier_labels = [row[0] for row in csv.reader(f) if row]

# Initialize MediaPipe HandLandmarker for Image Processing
base_options = python.BaseOptions(model_asset_path='model/hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3)
hands = vision.HandLandmarker.create_from_options(options)

# State management for dynamic gestures
# Map session_id to their history queues
sessions = {}

class LandmarkRequest(BaseModel):
    landmarks: List[List[float]]
    point_history: List[List[float]]
    width: int
    height: int

class ImageRequest(BaseModel):
    image_base64: str
    session_id: str = "default"

def get_session(session_id: str):
    if session_id not in sessions:
        sessions[session_id] = {
            "point_history": deque(maxlen=16),
            "finger_gesture_history": deque(maxlen=8)
        }
        # Initialize with empty frames
        for _ in range(16):
            sessions[session_id]["point_history"].append([0.0]*42)
    return sessions[session_id]

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(list(map(abs, temp_landmark_list)))
    if max_value == 0: return [0.0] * 42
    def normalize_(n): return n / max_value
    return list(map(normalize_, temp_landmark_list))

def pre_process_point_history(point_history, width, height):
    temp_point_history = copy.deepcopy(list(point_history))
    base_x, base_y = 0, 0
    for frame_index, frame_landmarks in enumerate(temp_point_history):
        if frame_index == 0:
            if len(frame_landmarks) > 0 and sum(frame_landmarks) != 0:
                base_x, base_y = frame_landmarks[0], frame_landmarks[1]
        for i in range(0, len(frame_landmarks), 2):
            # Calculate against the first frame's wrist point
            if sum(frame_landmarks) != 0:
                frame_landmarks[i] = (frame_landmarks[i] - base_x) / width
                frame_landmarks[i+1] = (frame_landmarks[i+1] - base_y) / height
            else:
                frame_landmarks[i] = 0.0
                frame_landmarks[i+1] = 0.0
    return list(itertools.chain.from_iterable(temp_point_history))

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for _, landmark in enumerate(landmarks):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point

@app.post("/predict_image")
async def predict_image(data: ImageRequest):
    try:
        # 1. Decode Image
        encoded_data = data.image_base64.split(',')[-1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image data")
        
        # 2. Run MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        results = hands.detect(mp_image)
        
        session = get_session(data.session_id)
        point_history = session["point_history"]
        finger_gesture_history = session["finger_gesture_history"]
        
        image_height, image_width, _ = img.shape
        
        if not results.hand_landmarks:
            point_history.append([0.0]*42)
            return {"static_gesture": "Unknown", "dynamic_gesture": "None", "movement_detected": False, "hand_detected": False}
        
        # Take the first hand found
        hand_landmarks = results.hand_landmarks[0]
        
        # Extract pixel coordinates
        landmark_list = calc_landmark_list(img, hand_landmarks)
        
        # 3. Static Classification
        pre_processed_landmarks = pre_process_landmark(landmark_list)
        hand_sign_id = keypoint_classifier(pre_processed_landmarks)
        
        # 4. Point History Updates
        flattened_landmarks = list(itertools.chain.from_iterable(landmark_list))
        point_history.append(flattened_landmarks)
        
        # 5. Dynamic Classification
        pre_processed_history = pre_process_point_history(point_history, image_width, image_height)
        finger_gesture_id = 0
        if len(pre_processed_history) == (16 * 42):
            finger_gesture_id = point_history_classifier(pre_processed_history)
            
        # 6. Geometric Gating (Truth Layer)
        movement_distance = 0
        valid_frames = [p for p in point_history if sum(p) != 0]
        if len(valid_frames) > 0:
            for i in range(42):
                coords = [p[i] for p in valid_frames]
                dist = max(coords) - min(coords)
                if dist > movement_distance:
                    movement_distance = dist

        if movement_distance < 50:
            finger_gesture_id = 0
        elif len(valid_frames) < 10:
            finger_gesture_id = 0
        else:
            if finger_gesture_id == 1:  # Hungry
                if hand_sign_id != 5: finger_gesture_id = 0
                else:
                    wrist_y_start, wrist_y_end = valid_frames[0][1], valid_frames[-1][1]
                    if wrist_y_end < wrist_y_start + 40: finger_gesture_id = 0
            elif finger_gesture_id == 2:  # SOS
                if hand_sign_id != 1: finger_gesture_id = 0
                else:
                    start_dist = (valid_frames[0][0] - valid_frames[0][16])**2 + (valid_frames[0][1] - valid_frames[0][17])**2
                    end_dist = (valid_frames[-1][0] - valid_frames[-1][16])**2 + (valid_frames[-1][1] - valid_frames[-1][17])**2
                    if end_dist > start_dist * 0.7: finger_gesture_id = 0
            elif finger_gesture_id == 3:  # Water
                if hand_sign_id != 6: finger_gesture_id = 0
                else:
                    tip_y_values = [f[17] for f in valid_frames]
                    direction_changes = 0
                    for i in range(2, len(tip_y_values)):
                        prev_delta, curr_delta = tip_y_values[i-1] - tip_y_values[i-2], tip_y_values[i] - tip_y_values[i-1]
                        if prev_delta * curr_delta < 0 and abs(prev_delta) > 3 and abs(curr_delta) > 3:
                            direction_changes += 1
                    if direction_changes < 1: finger_gesture_id = 0

        finger_gesture_history.append(finger_gesture_id)
        most_common_fg_id = Counter(finger_gesture_history).most_common()[0][0]

        return {
            "static_gesture": keypoint_classifier_labels[hand_sign_id] if hand_sign_id != -1 else "Unknown",
            "dynamic_gesture": point_history_classifier_labels[most_common_fg_id] if most_common_fg_id != 0 else "None",
            "movement_detected": movement_distance > 50,
            "hand_detected": True
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
