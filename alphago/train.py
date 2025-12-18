import torch
import torch.optim as optim
import torch.nn as nn
from board import GoBoard
from model import PolicyValueNet, state_to_tensor
from mcts import MCTS
import numpy as np
import random
from collections import deque

class Trainer:
    def __init__(self, size=9):
        self.size = size
        self.model = PolicyValueNet(size=size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.memory = deque(maxlen=2000)
        self.batch_size = 32

    def self_play(self, num_games=1):
        for _ in range(num_games):
            board = GoBoard(self.size)
            mcts = MCTS(self.model)
            game_data = []
            
            while True:
                probs = mcts.search(board, num_simulations=50)
                
                # Store state and probs
                state_tensor = state_to_tensor(board).squeeze(0)
                game_data.append([state_tensor, probs, board.current_player])
                
                # Choose move (stochastic for training)
                action = np.random.choice(len(probs), p=probs)
                
                if action == self.size * self.size:
                    board.move(None, None)
                else:
                    r, c = divmod(action, self.size)
                    board.move(r, c)
                
                if len(board.history) >= 2 and board.history[-1] is None and board.history[-2] is None:
                    break
                if len(board.history) > 200: # Max moves
                    break
            
            # Label data with winner
            winner = board.get_winner()
            for i in range(len(game_data)):
                # If winner == current_player at that step, value is 1, else -1
                game_data[i].append(1 if winner == game_data[i][2] else -1)
                self.memory.append(game_data[i])

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states = torch.stack([b[0] for b in batch])
        target_probs = torch.tensor(np.array([b[1] for b in batch]), dtype=torch.float32)
        target_values = torch.tensor(np.array([b[3] for b in batch]), dtype=torch.float32).unsqueeze(1)
        
        self.optimizer.zero_grad()
        out_probs, out_values = self.model(states)
        
        # Loss: Cross-entropy for policy, MSE for value
        policy_loss = -torch.mean(torch.sum(target_probs * torch.log(out_probs + 1e-8), dim=1))
        value_loss = F.mse_loss(out_values, target_values)
        
        loss = policy_loss + value_loss
        loss.backward()
        self.optimizer.step()
        return loss.item()

if __name__ == "__main__":
    import torch.nn.functional as F
    trainer = Trainer(size=9)
    print("Starting self-play and training loop...")
    for i in range(10):
        trainer.self_play(num_games=1)
        loss = trainer.train_step()
        print(f"Iteration {i}, Loss: {loss}")
    
    # Save model
    torch.save(trainer.model.state_dict(), "model.pth")
    print("Model saved to model.pth")


