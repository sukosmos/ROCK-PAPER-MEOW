# src/model_loader.py
import joblib


def load_trained_model(path="../models/hand_classifier.pkl"):
    model = joblib.load(path)
    print("Model loaded from", path)
    return model