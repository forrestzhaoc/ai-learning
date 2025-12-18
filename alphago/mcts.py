import numpy as np
import torch
import math

class MCTSNode:
    def __init__(self, parent=None, prior=0):
        self.parent = parent
        self.children = {}  # action -> MCTSNode
        self.visit_count = 0
        self.value_sum = 0
        self.prior = prior

    @property
    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def select_child(self, c_puct):
        best_score = -float('inf')
        best_action = -1
        best_child = None

        for action, child in self.children.items():
            # AlphaZero PUCT formula: Q + c_puct * P * (sqrt(parent_N) / (1 + child_N))
            score = child.value + c_puct * child.prior * (math.sqrt(self.visit_count) / (1 + child.visit_count))
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def expand(self, state, action_probs):
        for action, prob in action_probs:
            if action not in self.children:
                self.children[action] = MCTSNode(parent=self, prior=prob)

    def is_expanded(self):
        return len(self.children) > 0

class MCTS:
    def __init__(self, model, c_puct=1.4):
        self.model = model
        self.c_puct = c_puct

    def search(self, state, num_simulations=400):
        root = MCTSNode()

        for _ in range(num_simulations):
            node = root
            search_state = state.copy()

            # 1. Selection
            while node.is_expanded():
                action, node = node.select_child(self.c_puct)
                if action == search_state.size * search_state.size:
                    search_state.move(None, None)
                else:
                    r, c = divmod(action, search_state.size)
                    search_state.move(r, c)

            # 2. Expansion & Evaluation
            from model import state_to_tensor
            with torch.no_grad():
                probs, value = self.model(state_to_tensor(search_state))
                probs = probs.squeeze(0).cpu().numpy()
                value = value.item()

            # Mask illegal moves
            legal_moves = search_state.get_legal_moves()
            legal_indices = []
            for move in legal_moves:
                if move is None:
                    legal_indices.append(search_state.size * search_state.size)
                else:
                    legal_indices.append(move[0] * search_state.size + move[1])
            
            mask = np.zeros_like(probs)
            mask[legal_indices] = 1
            probs = probs * mask
            if probs.sum() > 0:
                probs /= probs.sum()
            else:
                # If no legal moves (shouldn't happen with Pass), use uniform over legal
                probs[legal_indices] = 1.0 / len(legal_indices)

            node.expand(search_state, enumerate(probs))

            # 3. Backpropagation
            # Note: value is from current player's perspective, but we need to alternate
            # AlphaZero usually predicts value for the current state.
            curr_value = value
            while node is not None:
                node.value_sum += curr_value
                node.visit_count += 1
                # Flip value for parent (opponent's turn)
                curr_value = -curr_value
                node = node.parent

        # Return action probabilities based on visit counts
        counts = [0] * (state.size * state.size + 1)
        for action, child in root.children.items():
            counts[action] = child.visit_count
        
        counts = np.array(counts)
        if counts.sum() > 0:
            probs = counts / counts.sum()
        else:
            probs = np.zeros_like(counts)
            probs[state.size * state.size] = 1.0 # Default to pass if nothing
            
        return probs


