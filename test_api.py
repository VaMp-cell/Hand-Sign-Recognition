import requests
import json
import base64
import cv2
import numpy as np

# URL for the new image endpoint
url = "http://localhost:8000/predict_image"

# Create a dummy image (e.g., black screen) to test the connection
# In reality, this would be a real image loaded via cv2.imread("hand.jpg") or from camera
dummy_img = np.zeros((540, 960, 3), dtype=np.uint8)

# Encode to base64
_, buffer = cv2.imencode('.jpg', dummy_img)
base64_img = base64.b64encode(buffer).decode('utf-8')

payload = {
    "image_base64": f"data:image/jpeg;base64,{base64_img}",
    "session_id": "test_user_1"
}

try:
    print(f"Sending request to {url}...")
    response = requests.post(url, json=payload)
    print(f"Status Code: {response.status_code}")
    print("Response JSON:")
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"Error: Could not connect to the server. Is server.py running?\n{e}")
