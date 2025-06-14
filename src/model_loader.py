# src/model_loader.py
from tensorflow.keras.models import load_model

def load_trained_model(path="../models/rps_mobilenetv2_dropout.keras"):
    model = load_model(path)
    print("Model loaded from", path)
    return model

