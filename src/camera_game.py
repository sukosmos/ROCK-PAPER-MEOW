# src/camera_game.py 

import cv2
import time
import os
import random
from model_loader import load_trained_model
from hand_predictor import detect_hand_label, detect_hand_presence
from game_logic import get_winner
from scene_renderer import render_scene, render_snapshot_result
from audio_player import play_sound

# STATE
STATE_MAIN = "main"
STATE_READY = "ready"
STATE_COUNTDOWN = "countdown"
STATE_SNAPSHOT = "snapshot"
STATE_RESULT = "result"
STATE_GAME_OVER = "over"

# 초기 변수
state = STATE_MAIN
player_score = 0
cat_score = 0
win_score = 3
countdown_time = 3  # 실제 표시되는 건 3초, snapshot은 3.5초
last_state_change = time.time()
snapshot = None
player_move = None
cat_move = None
result_start_time = None

# path
image_dir = "../assets/images"
sound_dir = "assets/sounds"

# image(투명 배경 유지)
images = {name: cv2.imread(os.path.join(image_dir, f"cat_{name}.png"), cv2.IMREAD_UNCHANGED)
          for name in ["main", "wait", "match", "win", "lose", "gamewin", "gamelose",
                       "rock", "paper", "scissors", "draw"]}

# model 
model = load_trained_model()

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
    elapsed = current_time - last_state_change

    if state == STATE_MAIN:
        display = render_scene(frame, images["main"], "Press SPACE to start", player_score, cat_score)
        if key == ord(' '):
            state = STATE_READY
            last_state_change = current_time

    elif state == STATE_READY:
        display = render_scene(frame, images["wait"], "Show your hand to start!", player_score, cat_score)
        if detect_hand_presence(frame):
            state = STATE_COUNTDOWN
            last_state_change = current_time

    elif state == STATE_COUNTDOWN:
        seconds_left = countdown_time - int(elapsed)
        display = render_scene(frame, images["match"], f"Ready... {seconds_left}", player_score, cat_score)
        if elapsed >= 3.5:
            snapshot = frame.copy()
            player_move = detect_hand_label(snapshot, model)
            cat_move = random.choice(["rock", "paper", "scissors"])
            if player_move is None:
                state = STATE_READY
            else:
                state = STATE_SNAPSHOT
            last_state_change = time.time()

    elif state == STATE_SNAPSHOT:
        display = render_snapshot_result(snapshot, images[cat_move], player_move)
        if time.time() - last_state_change > 2:
            state = STATE_RESULT
            last_state_change = time.time()

    elif state == STATE_RESULT:
        if result_start_time is None:
            # 최초 진입 시에만 점수 반영
            winner = get_winner(player_move, cat_move)
    
            if winner == "player":
                player_score += 1
                winner_state = "player win"
                result_image="lose"
            elif winner == "cat":
                cat_score += 1
                winner_state = "cat win"
                result_image="win"
            else:
                winner_state = "draw"
                result_image="draw"
    
            result_start_time = current_time  # 시간 기록
    
        display = render_scene(snapshot, images[result_image], f"{winner_state}", player_score, cat_score)
    
        if current_time - result_start_time > 2:
            if player_score >= win_score or cat_score >= win_score:
                state = STATE_GAME_OVER
            else:
                state = STATE_COUNTDOWN
            last_state_change = current_time
            result_start_time = None  # 초기화


    elif state == STATE_GAME_OVER:
        if player_score >= win_score:
            display = render_scene(frame, images["gamelose"], "You win", player_score, cat_score)
        else:
            display = render_scene(frame, images["gamewin"], "Cat wins", player_score, cat_score)

        if elapsed > 5:
            player_score = 0
            cat_score = 0
            state = STATE_MAIN

    # ESC 종료
    if key == 27:
        break

    cv2.imshow("Rock Paper Scissors", display)

cap.release()
cv2.destroyAllWindows()


