import os
import csv
import cv2
import mediapipe as mp
from tqdm import tqdm

# 설정
DATA_ROOT = "../data"  # train/test 하위 폴더 포함
SAVE_PATH = "../data/landmark_dataset.csv"
MAX_HANDS = 1

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=MAX_HANDS)
mp_drawing = mp.solutions.drawing_utils

labels = ['rock', 'paper', 'scissors']
header = [f"x{i}" for i in range(1, 22)] + [f"y{i}" for i in range(1, 22)] + ['label']
data = []

for subset in ['train', 'test']:
    for label in labels:
        folder_path = os.path.join(DATA_ROOT, subset, label)
        image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

        for image_name in tqdm(image_files, desc=f"{subset}/{label}"):
            image_path = os.path.join(folder_path, image_name)
            image = cv2.imread(image_path)
            if image is None:
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            if results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks[0].landmark
                x_list = [lm.x for lm in landmarks]
                y_list = [lm.y for lm in landmarks]
                row = x_list + y_list + [label]
                data.append(row)

print(f"{len(data)}개 완료")

# 저장
with open(SAVE_PATH, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(data)

print(f"CSV저장 완료: {SAVE_PATH}")
