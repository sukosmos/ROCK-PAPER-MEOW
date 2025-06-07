# src/game_logic.py

def get_winner(player_move, cat_move):
    """승자 판정 로직"""
    if player_move == cat_move:
        return "draw"
    elif (player_move == "rock" and cat_move == "scissors") or \
         (player_move == "paper" and cat_move == "rock") or \
         (player_move == "scissors" and cat_move == "paper"):
        return "player"
    else:
        return "cat"
