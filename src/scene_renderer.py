# src/scene_renderer.py

import cv2
import numpy as np

def resize_with_aspect_ratio(image, target_height):
    """
    이미지 비율 유지하며 target_height에 맞게 resize
    """
    h, w = image.shape[:2]
    scale = target_height / h
    new_w = int(w * scale)
    resized = cv2.resize(image, (new_w, target_height), interpolation=cv2.INTER_AREA)
    return resized

def overlay_transparent(background, overlay, x, y):
    """
    배경 위에 알파 채널 가진 overlay 이미지를 지정 위치(x,y)에 투명하게 덮어씌움
    """
    if overlay.shape[2] == 3:
        # 알파 채널이 없으면 그대로 덮어씌움
        background[y:y+overlay.shape[0], x:x+overlay.shape[1]] = overlay
        return background

    bgr = overlay[:, :, :3]
    alpha = overlay[:, :, 3:] / 255.0

    h, w = bgr.shape[:2]
    roi = background[y:y+h, x:x+w]

    blended = (1.0 - alpha) * roi + alpha * bgr
    background[y:y+h, x:x+w] = blended.astype(np.uint8)

    return background

def render_scene(frame, cat_img, message="", player_score=0, cat_score=0):
    display = frame.copy()

    if cat_img is not None:
        resized_cat = resize_with_aspect_ratio(cat_img, 250)
        display = overlay_transparent(display, resized_cat, 10, 10)

    if message:
        text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
        center_x = (display.shape[1] - text_size[0]) // 2
        cv2.putText(display, message, (center_x, 400),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    score_text = f"You {player_score} : {cat_score} Cat"
    cv2.putText(display, score_text, (display.shape[1] - 300, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    return display

def render_snapshot_result(snapshot, cat_img, player_move):
    display = snapshot.copy()

    if cat_img is not None:
        resized_cat = resize_with_aspect_ratio(cat_img, 250)
        display = overlay_transparent(display, resized_cat, 10, 10)

    if player_move:
        text = f"You: {player_move}"
    else:
        text = "No hand detected. Try again."

    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
    center_x = (display.shape[1] - text_size[0]) // 2
    cv2.putText(display, text, (center_x, 400),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    return display
