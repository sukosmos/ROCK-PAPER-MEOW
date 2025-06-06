# src/scene_renderer.py
import cv2
import numpy as np

def resize_with_aspect_ratio(image, target_height):
    """이미지의 가로세로 비율을 유지하면서 높이를 맞추고, 알파 포함 유지"""
    h, w = image.shape[:2]
    scale = target_height / h
    new_w, new_h = int(w * scale), target_height
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized

def overlay_transparent(background, overlay, x, y):
    """배경 위에 알파 채널 가진 overlay 이미지를 지정 위치(x,y)에 투명하게 덮어씌움"""
    bh, bw = background.shape[:2]
    oh, ow = overlay.shape[:2]

    if x + ow > bw or y + oh > bh:
        return background  # 범위 벗어나면 생략

    # 알파 채널 없는 경우
    if overlay.shape[2] < 4:
        background[y:y+oh, x:x+ow] = overlay
        return background

    # 분리
    bgr_overlay = overlay[:, :, :3]
    alpha_mask = overlay[:, :, 3:] / 255.0

    roi = background[y:y+oh, x:x+ow]

    # 오버레이
    blended = (1.0 - alpha_mask) * roi + alpha_mask * bgr_overlay
    background[y:y+oh, x:x+ow] = blended.astype(np.uint8)

    return background

def render_scene(frame, cat_img, message="", player_score=0, cat_score=0):
    display = frame.copy()

    # cat image (with transparency)
    if cat_img is not None:
        resized_cat = resize_with_aspect_ratio(cat_img, target_height=250)
        display = overlay_transparent(display, resized_cat, 10, 10)

    # message 중앙 출력 (높이 70% 지점)
    if message:
        text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
        center_x = (display.shape[1] - text_size[0]) // 2
        msg_y = int(display.shape[0] * 0.7)
        cv2.putText(display, message, (center_x, msg_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    # score 우측 상단
    score_text = f"You {player_score} : {cat_score} Cat"
    cv2.putText(display, score_text, (display.shape[1] - 300, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    return display
