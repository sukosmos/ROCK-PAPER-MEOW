# src/hand_predictor.py
import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

def detect_hand_presence(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    return results.multi_hand_landmarks is not None

def detect_hand_label(frame, model):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        # Bounding box 추출
        h, w, _ = frame.shape
        landmarks = results.multi_hand_landmarks[0].landmark
        x_coords = [lm.x for lm in landmarks]
        y_coords = [lm.y for lm in landmarks]
        x_min, x_max = int(min(x_coords) * w), int(max(x_coords) * w)
        y_min, y_max = int(min(y_coords) * h), int(max(y_coords) * h)

        # 약간 margin 추가
        margin = 20
        x_min, x_max = max(x_min - margin, 0), min(x_max + margin, w)
        y_min, y_max = max(y_min - margin, 0), min(y_max + margin, h)

        hand_img = frame[y_min:y_max, x_min:x_max]
        hand_img = cv2.resize(hand_img, (224, 224))  # MobilenetV2 input size
        hand_img = hand_img.astype("float32") / 255.0
        hand_img = np.expand_dims(hand_img, axis=0)

        prediction = model.predict(hand_img)
        label = np.argmax(prediction[0])
        label_map = {0: "paper", 1: "rock", 2: "scissors"}
        return label_map[label]
    
    return None
