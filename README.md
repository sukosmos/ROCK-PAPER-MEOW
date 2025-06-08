![image](https://github.com/user-attachments/assets/6da9f9c0-8125-4e0a-9018-20816d5a2871)

# ROCK-PAPER-MEOW 😻

Webcam으로 고양이와 가위바위보 승부! 😼
<br>
kaggle의 가위바위보 image dataset을 학습한 MobileNetV2 기반 모델이 랜덤하게 패를 내는 고양이와의 승부를 판별합니다.



<br>

---

## Installation

### 1. Python 환경 설정

conda 가상환경에서 개발

```bash
conda create -n rps_env python=3.9
conda activate rps_env
pip install opencv-python mediapipe pygame tensorflow==2.13.0 numpy
```

<br>


### 2. 모델 & 데이터

* `models/rps_mobilenetv2_dropout.keras` → 학습된 모델 파일
* `assets/images/` → PNG 이미지
* `assets/sound/` → WAV 사운드
  * path 절대경로로 수정필요

<br>


### 3. 실행

```bash
python src/camera_game.py
```

<br>



<br>




## Tech Stack

* Python 3.9
* OpenCV
* MediaPipe
* TensorFlow / Keras
* pygame


<br>



## Dataset

  Kaggle의 [손 이미지 데이터셋](https://www.kaggle.com/datasets/alexandredj/rock-paper-scissors-dataset)을 사용 (paper, rock, scissors로 폴더 분류)
  
  
<br>



## Model Training

* 사용 모델: MobileNetV2 (Keras 기반)
* 입력 데이터: kaggle 손 이미지
* 클래스: `paper`, `rock`, `scissors`
* 학습 결과: accuracy ~97% 달성
    
    ![image](https://github.com/user-attachments/assets/1689d260-91e1-4fbd-893e-17eb765e77c9)

  

모델 저장: `rps_mobilenetv2_dropout.keras`
모델 로드: `model_loader.py`

<br>


## OpenCV + MediaPipe

OpenCV로 webcam의 실시간 frame 수집

MediaPipe로 손 존재 여부 판단 `detect_hand_presence()`
  ```python
def detect_hand_presence(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    return results.multi_hand_landmarks is not None
  ```

모델 추론 `detect_hand_label`

<br>


- Mediapipe에서 손 위치 추출

```python
h, w, _ = frame.shape
landmarks = results.multi_hand_landmarks[0].landmark
x_coords = [lm.x for lm in landmarks]
y_coords = [lm.y for lm in landmarks]
x_min, x_max = int(min(x_coords) * w), int(max(x_coords) * w)
y_min, y_max = int(min(y_coords) * h), int(max(y_coords) * h)
```
- 이미지 전처리
  
```python
hand_img = frame[y_min:y_max, x_min:x_max]  # ROI 자르기
hand_img = cv2.resize(hand_img, (224, 224))  # MobileNetV2 입력 사이즈
hand_img = hand_img.astype("float32") / 255.0  # 정규화
hand_img = np.expand_dims(hand_img, axis=0)  # 배치 차원 추가
```

- 모델 추론
```python
prediction = model.predict(hand_img)
label = np.argmax(prediction[0])
label_map = {0: "paper", 1: "rock", 2: "scissors"}
return label_map[label]
```

<br>


## FSM 
Finite State machine기반으로 게임 로직이 구성됨

esc 누르면 언제든지 종료


![image](https://github.com/user-attachments/assets/5c735923-4e95-42e9-9641-738f55b00c47)

<br>


### MAIN

![image](https://github.com/user-attachments/assets/e9d8eec2-71f6-412d-98d5-c6992659cbd7)


초기 상태이며 게임 시작 화면을 보여줌

사용자에게 "Press SPACE to start" 메시지를 띄움

SPACE 입력 시 → `STATE_READY`

<br>


### READY

![image](https://github.com/user-attachments/assets/910e7b43-6ca0-443e-8343-fb3d58a2218f)


사용자 손 존재 여부 확인

손이 감지되면 → `STATE_COUNTDOWN`

<br>


### COUNTDOWN

![image](https://github.com/user-attachments/assets/89ad881d-599b-45ea-9187-a51a4f79f655)


3초 동안 숫자 카운트 표시 (Ready... 3, Ready... 2, Ready... 1)

4초 후 프레임을 캡처하고 snapshot 저장

`detect_hand_label()`을 통해 trained model이 사용자의 손 제스처 예측 (rock-paper-scissors)

고양이는 랜덤한 손 모양 선택

```python
# camera_game.py
 cat_move = random.choice(["rock", "paper", "scissors"])
```

이후 `STATE_SNAPSHOT`으로 이동

<br>


### SNAPSHOT

![image](https://github.com/user-attachments/assets/c49a5e65-c06f-4241-8bc2-3dc1161399cb)


캡처된 이미지와 고양이 손 이미지 표시

사용자의 손 제스처도 함께 보여줌

2초간 정지 상태 유지 후 → `STATE_RESULT`

<br>


### RESULT

![image](https://github.com/user-attachments/assets/8a7e1b39-f563-453e-8a14-14186b59467d)



승패를 판단하고 점수 업데이트

결과에 따라 win/lose/draw 고양이 이미지 출력

2초 후 

  - 누군가 3점 도달 시 → `STATE_GAME_OVER`

  - 그렇지 않으면 → `STATE_COUNTDOWN` (다음 라운드)



<br>


### GAMEOVER

![image](https://github.com/user-attachments/assets/c59025d0-39e1-4047-aa45-5a78dedb2ed6)


게임 최종 승자 메시지와 이미지 출력

You win 또는 Cat wins

15초 후 점수 초기화, STATE_MAIN으로 돌아감

esc 누르면 종료



<br>
<br>




## Pygame - sound

OpenCV에 사운드를 지원하기 위해 Pygame 사용

음악 재생 주기가 짧으므로 `pre_load()`와 cache 사용으로 효율적 자원 관리


<br>

<br>




## Demo Video

https://www.youtube.com/watch?v=Ml9RkVhMupY


<br>


<br>



## Issue
https://github.com/sukosmos/Rock-Paper-Scissors-with-cat/issues?q=is%3Aissue%20state%3Aclosed



<br>


<br>


# 한계 및 개선사항 
## 모델 성능

- 모델이 화면상 오른쪽 손의 인식 성능이 더 우수함. webcam은 좌우반전이 적용되므로 왼손 사용시 판단 성능이 뛰어남
- 배경이 복잡할 시 성능 대폭 저하, 손이 잘 보이도록 배경을 설정해야함
- 더 좋은 성능을 기대할 수 있는 training 방법 찾아보기

## gamelogic - state=COUNTDOWN
- 가위바위보 카운트에 손을 내면 인식이 느린 issue 발생
- `if elapsed >= 3.5:` 에서 `if elapsed >= 4:`로 시간 조정
- 개선 완료

## audio_player

- state=ready일 때 소리가 안 나는 오류가 있음
- audio play가 많을수록 재생 지연 발생

## UX

- 더 깔끔한 UX 디자인 필요

## 적용성

- 현재 python OpenCV 환경에서만 사용가능
- 웹/앱 등에도 사용할 수 있도록 개발 



<br><br>

---

# 참고

- Kaggle Rock-Paper-Scissors-Dataset: https://www.kaggle.com/datasets/alexandredj/rock-paper-scissors-dataset
- MobileNet: https://arxiv.org/abs/1801.04381
- MediaPipe-hands: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/hands.md
- ChatGPT

<br><br>

