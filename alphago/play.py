import torch
import numpy as np
from board import GoBoard
from model import PolicyValueNet
from mcts import MCTS
import sys

def play():
    size = 9
    board = GoBoard(size=size)
    model = PolicyValueNet(size=size)
    # model.load_state_dict(torch.load('model.pth')) # Load if exists
    model.eval()
    
    mcts = MCTS(model)
    
    print("Welcome to AlphaGo (Simplified)!")
    print(f"Board size: {size}x{size}")
    print("You are Black (B), AI is White (W).")
    print("Enter moves as 'row col' (e.g., '3 4') or 'pass'.")
    
    while True:
        print("\n" + str(board))
        if board.current_player == 1:
            # Human turn
            move_str = input("Your move (B): ").strip().lower()
            if move_str == 'pass':
                board.move(None, None)
            elif move_str == 'quit':
                break
            else:
                try:
                    r, c = map(int, move_str.split())
                    if not board.move(r, c):
                        print("Illegal move!")
                        continue
                except ValueError:
                    print("Invalid input! Use 'row col' or 'pass' or 'quit'.")
                    continue
        else:
            # AI turn
            print("AI is thinking...")
            probs = mcts.search(board, num_simulations=100)
            action = np.argmax(probs)
            
            if action == size * size:
                print("AI passes.")
                board.move(None, None)
            else:
                r, c = divmod(action, size)
                print(f"AI plays: {r} {c}")
                board.move(r, c)
        
        # Check for game end (two consecutive passes)
        if len(board.history) >= 2 and board.history[-1] is None and board.history[-2] is None:
            print("\nGame Over!")
            print(str(board))
            winner = board.get_winner()
            if winner == 1:
                print("Black wins!")
            else:
                print("White wins!")
            break

if __name__ == "__main__":
    play()


