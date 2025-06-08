import pygame
import os

# 오디오 시스템 초기화 (작은 버퍼로 지연 최소화)
pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)

# 전역 캐시 및 현재 재생 추적용
_sound_cache = {}
_current_channel = None
_current_sound_file = None

# main.wav 전용 반복 재생용 채널
_main_channel = None

# 필요한 경우 사전 로드할 파일 목록
SOUND_FILES = {
    "countdown": "countdown.wav",
    # 필요시 여기에 계속 추가
}

def preload_sounds(base_path):
    """
    사운드를 메모리로 사전 로드하여 재생 지연 최소화.
    """
    for key, filename in SOUND_FILES.items():
        path = os.path.join(base_path, filename)
        if os.path.exists(path):
            _sound_cache[key] = pygame.mixer.Sound(path)
            print(f"[AUDIO] 로드 완료: {path}")
        else:
            print(f"[AUDIO] 누락된 파일: {path}")

def play_sound(path):
    """
    지정된 경로의 사운드를 재생. 이미 재생 중이면 재생하지 않음.
    """
    global _current_channel, _current_sound_file

    if not os.path.exists(path):
        print(f"[AUDIO] 경로 오류: {path}")
        return

    # 이미 재생 중이면 중복 재생 방지
    if _current_channel and _current_channel.get_busy() and _current_sound_file == path:
        return

    stop_sound()

    try:
        sound = pygame.mixer.Sound(path)
        _current_channel = sound.play()
        _current_sound_file = path
    except Exception as e:
        print(f"[AUDIO] 재생 실패: {e}")

def play_main_loop(path, force=False):
    global _main_channel
    if not os.path.exists(path):
        print(f"[AUDIO] main.wav 경로 오류: {path}")
        return

    if force or not _main_channel or not _main_channel.get_busy():
        try:
            sound = pygame.mixer.Sound(path)
            _main_channel = sound.play(loops=-1)
            print(f"[AUDIO] 루프 재생 시작: {path}")
        except Exception as e:
            print(f"[AUDIO] 루프 재생 실패: {e}")


def stop_sound():
    global _current_channel, _current_sound_file
    if _current_channel:
        _current_channel.stop()
    _current_channel = None
    _current_sound_file = None

def stop_main_loop():
    global _main_channel
    if _main_channel:
        _main_channel.stop()
    _main_channel = None
