# AlphaGo Simplified (AlphaZero Style)

This is a simplified implementation of an AlphaGo-like algorithm using the AlphaZero approach (Monte Carlo Tree Search + Policy-Value Network).

## Structure
- `board.py`: Go game logic (9x9 by default).
- `model.py`: ResNet-based Policy-Value Network.
- `mcts.py`: Monte Carlo Tree Search algorithm.
- `play.py`: Interface to play against the AI.
- `train.py`: Self-play training loop.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## How to Play
Run the following command to play against the AI:
```bash
python3 play.py
```
*Note: Without training, the AI will make moves based on random weights and MCTS simulations.*

## Training
To train the AI through self-play:
```bash
python3 train.py
```
This will generate `model.pth`, which `play.py` can load if you uncomment the loading line.


