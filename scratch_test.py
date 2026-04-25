import urllib.request
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2

# Download model
urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task', 'hand_landmarker.task')

# Create detector
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       running_mode=vision.RunningMode.IMAGE,
                                       num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

# Create dummy image
img = np.zeros((100, 100, 3), dtype=np.uint8)
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)

# Detect
detection_result = detector.detect(mp_image)
print("Hand landmarks:", detection_result.hand_landmarks)
print("Handedness:", detection_result.handedness)
