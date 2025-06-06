# src/game_logic.py

def get_computer_winning_move(player_move):
    win_map = {
        'rock': 'paper',
        'paper': 'scissors',
        'scissors': 'rock'
    }
    return win_map.get(player_move, 'rock')  # default to rock
