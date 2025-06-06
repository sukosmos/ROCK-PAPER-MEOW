# src/predictor.py

import numpy as np
import cv2

CLASS_NAMES = ['paper', 'rock', 'scissors']  # learning order

def preprocess_frame(frame, img_size=(224, 224)):
    resized = cv2.resize(frame, img_size)
    norm = resized / 255.0
    return np.expand_dims(norm, axis=0)

def predict_move(model, frame):
    input_tensor = preprocess_frame(frame)
    preds = model.predict(input_tensor)
    class_idx = np.argmax(preds[0])
    return CLASS_NAMES[class_idx]
