import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# 경로 설정
model_path = "../models/rps_mobilenetv2_dropout.keras"
val_dir = "../data_split/val"  # validation 이미지가 저장된 폴더

# 모델 로드
model = load_model(model_path)

# ImageDataGenerator로 validation set 준비
val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# 예측
preds = model.predict(val_generator)
predicted_classes = np.argmax(preds, axis=1)
true_classes = val_generator.classes
class_labels = list(val_generator.class_indices.keys())

# 정확도 출력
loss, acc = model.evaluate(val_generator)
print(f"Validation Accuracy: {acc:.4f}")

# Confusion matrix & classification report
print("\nClassification Report:")
print(classification_report(true_classes, predicted_classes, target_names=class_labels))

print("\nConfusion Matrix:")
print(confusion_matrix(true_classes, predicted_classes))
