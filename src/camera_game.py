import cv2
import time
import pygame
import os
from model_loader import load_trained_model
from hand_predictor import detect_hand_label
from game_logic import get_computer_winning_move
from scene_renderer import render_scene
from audio_player import play_sound

# 상태 정의
STATE_MAIN = "main"
STATE_READY = "ready"
STATE_GAME = "game"
STATE_MATCH_RESULT = "result"
STATE_GAME_OVER = "over"

# 변수 초기화
state = STATE_MAIN
player_score = 0
cat_score = 0
win_score = 3
countdown_time = 5
last_state_change = time.time()
last_countdown_played = -1  # 추가: 사운드 중복 방지용

# 이미지 로딩
image_dir = "../assets/images"
images = {
    "main": cv2.imread(os.path.join(image_dir, "cat_main.png")),
    "wait": cv2.imread(os.path.join(image_dir, "cat_wait.png")),
    "match": cv2.imread(os.path.join(image_dir, "cat_match.png")),
    "win": cv2.imread(os.path.join(image_dir, "cat_win.png")),
    "lose": cv2.imread(os.path.join(image_dir, "cat_lose.png")),
    "gamewin": cv2.imread(os.path.join(image_dir, "cat_gamewin.png")),
    "gamelose": cv2.imread(os.path.join(image_dir, "cat_gamelose.png")),
    "rock": cv2.imread(os.path.join(image_dir, "cat_rock.png")),
    "paper": cv2.imread(os.path.join(image_dir, "cat_paper.png")),
    "scissors": cv2.imread(os.path.join(image_dir, "cat_scissors.png")),
}

# 사운드 경로
sound_dir = "../assets/sounds"

# 모델 로드
model = load_trained_model("../models/hand_classifier.pkl")

# 카메라 시작
cap = cv2.VideoCapture(0)
cv2.namedWindow("Rock Paper Scissors")

# 메인 루프
while True:
    ret, frame = cap.read()
    if not ret:
        break

    key = cv2.waitKey(1) & 0xFF
    current_time = time.time()

    if state == STATE_MAIN:
        display = render_scene(frame, images["main"], message="Press SPACE to start", player_score=player_score, cat_score=cat_score)
        if detect_hand_label(frame, model) and key == ord(' '):
            state = STATE_READY
            last_state_change = current_time

    elif state == STATE_READY:
        label = detect_hand_label(frame, model)
        msg = "Show your hand to start!"
        display = render_scene(frame, images["wait"], message=msg, player_score=player_score, cat_score=cat_score)
        if label is not None:
            state = STATE_GAME
            last_state_change = current_time
            last_countdown_played = -1  # 초기화

    elif state == STATE_GAME:
        elapsed = current_time - last_state_change
        countdown_remaining = countdown_time - int(elapsed)

        # 카운트다운 사운드 (1회만 재생)
        if countdown_remaining > 0 and countdown_remaining != last_countdown_played:
            last_countdown_played = countdown_remaining
            sound_path = os.path.join(sound_dir, "rock-paper-scissors.wav")
            play_sound(sound_path)

        display = render_scene(frame, images["match"], message=f"Ready... {countdown_remaining}", player_score=player_score, cat_score=cat_score)

        if elapsed >= countdown_time:
            snapshot = frame.copy()
            player_move = detect_hand_label(snapshot, model)
            if player_move is None:
                state = STATE_READY
                continue

            cat_move = get_computer_winning_move(player_move)

            display = render_scene(snapshot, images[cat_move], message=f"You: {player_move}", player_score=player_score, cat_score=cat_score)

            # 결과 비교
            if player_move == cat_move:
                pass
            elif get_computer_winning_move(cat_move) == player_move:
                player_score += 1
            else:
                cat_score += 1

            state = STATE_MATCH_RESULT
            last_state_change = current_time

    elif state == STATE_MATCH_RESULT:
        display = render_scene(frame, images["win" if cat_score < player_score else "lose"],
                               message="Next round...", player_score=player_score, cat_score=cat_score)

        if player_score >= win_score or cat_score >= win_score:
            state = STATE_GAME_OVER
            last_state_change = current_time
        elif current_time - last_state_change > 2:
            state = STATE_GAME
            last_state_change = current_time
            last_countdown_played = -1  # 초기화

    elif state == STATE_GAME_OVER:
        if player_score >= win_score:
            display = render_scene(frame, images["gamewin"], message="You win", player_score=player_score, cat_score=cat_score)
            play_sound(os.path.join(sound_dir, "playerwin.wav"))
        else:
            display = render_scene(frame, images["gamelose"], message="Cat wins", player_score=player_score, cat_score=cat_score)
            play_sound(os.path.join(sound_dir, "catwin.wav"))

        if current_time - last_state_change > 5:
            player_score, cat_score = 0, 0
            state = STATE_MAIN

    # ESC 누르면 종료
    if key == 27:
        break

    cv2.imshow("Rock Paper Scissors", display)

cap.release()
cv2.destroyAllWindows()

