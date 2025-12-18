import numpy as np

class GoBoard:
    def __init__(self, size=9):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)  # 0: empty, 1: black, -1: white
        self.current_player = 1  # 1 for black, -1 for white
        self.ko_state = None
        self.history = []

    def copy(self):
        new_board = GoBoard(self.size)
        new_board.board = np.copy(self.board)
        new_board.current_player = self.current_player
        new_board.ko_state = self.ko_state
        new_board.history = self.history.copy()
        return new_board

    def get_legal_moves(self):
        moves = []
        for r in range(self.size):
            for c in range(self.size):
                if self.is_legal_move(r, c, self.current_player):
                    moves.append((r, c))
        moves.append(None)  # Pass move
        return moves

    def is_legal_move(self, r, c, player):
        if self.board[r, c] != 0:
            return False
        
        # Check Ko rule (simplified: can't immediately recapture if it leads to previous state)
        if self.ko_state == (r, c):
            return False

        # Try move
        temp_board = np.copy(self.board)
        temp_board[r, c] = player
        
        # Check if move captures or has liberties
        captured = self._get_captured(temp_board, r, c, player)
        if len(captured) > 0:
            return True
        
        if self._get_liberties(temp_board, r, c) > 0:
            return True
        
        return False

    def move(self, r, c):
        if r is None:  # Pass
            self.current_player *= -1
            self.history.append(None)
            self.ko_state = None
            return True

        if not self.is_legal_move(r, c, self.current_player):
            return False

        captured = self._get_captured(self.board, r, c, self.current_player)
        self.board[r, c] = self.current_player
        
        for cr, cc in captured:
            self.board[cr, cc] = 0

        # Simple Ko detection: if exactly one stone was captured and the placed stone now has 1 liberty
        if len(captured) == 1 and self._get_liberties(self.board, r, c) == 1:
            self.ko_state = captured[0]
        else:
            self.ko_state = None

        self.current_player *= -1
        self.history.append((r, c))
        return True

    def _get_neighbors(self, r, c):
        neighbors = []
        if r > 0: neighbors.append((r-1, c))
        if r < self.size - 1: neighbors.append((r+1, c))
        if c > 0: neighbors.append((r, c-1))
        if c < self.size - 1: neighbors.append((r, c+1))
        return neighbors

    def _get_group(self, board, r, c):
        color = board[r, c]
        group = {(r, c)}
        stack = [(r, c)]
        while stack:
            curr_r, curr_c = stack.pop()
            for nr, nc in self._get_neighbors(curr_r, curr_c):
                if board[nr, nc] == color and (nr, nc) not in group:
                    group.add((nr, nc))
                    stack.append((nr, nc))
        return group

    def _get_liberties(self, board, r, c):
        group = self._get_group(board, r, c)
        liberties = set()
        for gr, gc in group:
            for nr, nc in self._get_neighbors(gr, gc):
                if board[nr, nc] == 0:
                    liberties.add((nr, nc))
        return len(liberties)

    def _get_captured(self, board, r, c, player):
        captured_all = []
        temp_board = np.copy(board)
        temp_board[r, c] = player
        
        for nr, nc in self._get_neighbors(r, c):
            if temp_board[nr, nc] == -player:
                if self._get_liberties(temp_board, nr, nc) == 0:
                    group = self._get_group(temp_board, nr, nc)
                    captured_all.extend(list(group))
                    # Remove from temp_board to avoid double counting if multiple groups are captured
                    for gr, gc in group:
                        temp_board[gr, gc] = 0
        return captured_all

    def get_winner(self):
        # Simplified scoring: area scoring
        black_score = np.sum(self.board == 1)
        white_score = np.sum(self.board == -1)
        
        # Add territory (simplified: only counting empty spots surrounded by one color)
        # This is very rough, real Go scoring is complex.
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r, c] == 0:
                    owner = self._get_territory_owner(r, c)
                    if owner == 1: black_score += 1
                    elif owner == -1: white_score += 1
        
        # Komi for white (standard is 6.5 or 7.5, let's use 6.5)
        white_score += 6.5
        
        return 1 if black_score > white_score else -1

    def _get_territory_owner(self, r, c):
        group = {(r, c)}
        stack = [(r, c)]
        colors_seen = set()
        while stack:
            curr_r, curr_c = stack.pop()
            for nr, nc in self._get_neighbors(curr_r, curr_c):
                if self.board[nr, nc] == 0:
                    if (nr, nc) not in group:
                        group.add((nr, nc))
                        stack.append((nr, nc))
                else:
                    colors_seen.add(self.board[nr, nc])
        
        if len(colors_seen) == 1:
            return list(colors_seen)[0]
        return 0

    def __str__(self):
        res = "   " + " ".join([str(i) for i in range(self.size)]) + "\n"
        for r in range(self.size):
            line = f"{r:2} "
            for c in range(self.size):
                if self.board[r, c] == 1: line += "B "
                elif self.board[r, c] == -1: line += "W "
                else: line += ". "
            res += line + "\n"
        return res


