# src/hand_predictor
import cv2
import mediapipe as mp
import numpy as np
import joblib

model = joblib.load('../models/hand_classifier.pkl')

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

def detect_hand_label(frame, model):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0]
        data = []
        for lm in landmarks.landmark:
            data.extend([lm.x, lm.y])
        data = np.array(data).reshape(1, -1)
        prediction = model.predict(data)
        return prediction[0]
    else:
        return None

def detect_hand_presence(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    return results.multi_hand_landmarks is not None