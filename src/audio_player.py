# src/audio_player.py
import pygame
import os

pygame.mixer.init()

def play_sound(file_path):
    if os.path.exists(file_path):
        try:
            sound = pygame.mixer.Sound(file_path)
            sound.play()
        except pygame.error as e:
            print("Sound play error:", e)
    else:
        print("Sound file not found:", file_path)

