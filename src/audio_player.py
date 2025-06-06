#src/audio_player
import pygame

pygame.mixer.init()

def play_sound(file_path):
    try:
        sound = pygame.mixer.Sound(file_path)
        sound.play()
    except Exception as e:
        print(f"[Sound Error] {e}")
