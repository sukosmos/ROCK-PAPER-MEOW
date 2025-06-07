# src/train_model.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# 설정
csv_path = "../data/landmark_dataset.csv"  
model_output_path = "../models/hand_classifier.pkl"

# CSV 로딩
df = pd.read_csv(csv_path)

# 마지막 열은 label, 나머지는 feature
X = df.iloc[:, :-1].values  # shape: (N, 63)
y = df.iloc[:, -1].values   # shape: (N,)

print(f"loading: {X.shape[0]} samples, {X.shape[1]} features")

# 학습 / 테스트 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 학습
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 평가
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n Accuracy: {acc * 100:.2f}%")
print(classification_report(y_test, y_pred))

# 저장
os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
joblib.dump(model, model_output_path)
print(f"\n 모델 저장 완료: {model_output_path}")
