from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import os

# 설정
TRAIN_DIR = "../data/train"
TEST_DIR = "../data/test"
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS = 10
MODEL_PATH = "../models/rps_model.h5"

# 1. ImageDataGenerator (validation_split 제거!)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    zoom_range=0.2,
)

test_datagen = ImageDataGenerator(rescale=1./255)

# 2. 데이터 불러오기
train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_gen = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# 3. 사전 학습 모델 (MobileNetV2)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # freeze

# 4. 커스텀 헤드
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(64, activation='relu')(x)
predictions = Dense(3, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# 5. 컴파일
model.compile(optimizer=Adam(1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 6. 체크포인트 저장
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
checkpoint = ModelCheckpoint(MODEL_PATH, save_best_only=True)

# 7. 학습
model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=test_gen,   # <-- test_gen 사용
    callbacks=[checkpoint]
)

print(f"success: {MODEL_PATH}")
