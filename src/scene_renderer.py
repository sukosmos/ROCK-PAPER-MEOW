#src/scene_renderer
import cv2
import numpy as np

def overlay_image(background, overlay, position=(0, 0)):
    x, y = position
    h, w = overlay.shape[:2]
    background[y:y+h, x:x+w] = cv2.resize(overlay, (w, h))
    return background

def render_scene(frame, cat_img, message="", player_score=0, cat_score=0):
    # cat image, text, score
    display = frame.copy()

    #cat
    if cat_img is not None:
        display[0:cat_img.shape[0], 0:cat_img.shape[1]] = cat_img

    # message
    if message:
        text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
        center_x = (display.shape[1] - text_size[0]) // 2
        cv2.putText(display, message, (center_x, 400),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    # score
    score_text = f"You {player_score} : {cat_score} Cat"
    cv2.putText(display, score_text, (display.shape[1] - 300, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    return display
