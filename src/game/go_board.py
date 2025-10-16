import numpy as np
from typing import List, Tuple, Optional
import copy

class GoBoard:
    """
    Efficient 9x9 Go board implementation with minimal memory footprint.
    Focus on performance for self-play generation.
    """
    
    def __init__(self, size=9):
        self.size = size
        self.board = np.zeros((size, size), dtype=np.int8)  # 0=empty, 1=black, -1=white
        self.current_player = 1  # Black starts
        self.ko_point = None     # For ko rule
        self.history = []        # For superko and network input
        self.capture_counts = [0, 0]  # [black_captures, white_captures]
        self.move_count = 0      # Track total moves made  # ADD THIS LINE
        
        # Precompute neighbors for efficiency
        self._precompute_neighbors()
    
    def _precompute_neighbors(self):
        """Precompute neighbor positions for faster liberty counting"""
        self.neighbors = {}
        for i in range(self.size):
            for j in range(self.size):
                adjacent = []
                for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self.size and 0 <= nj < self.size:
                        adjacent.append((ni, nj))
                self.neighbors[(i, j)] = adjacent
    
    def get_legal_moves(self) -> List[Tuple[int, int]]:
        """Get all legal moves, including pass (represented as (-1, -1))"""
        moves = [(-1, -1)]  # Pass is always legal
        
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i, j] == 0:  # Empty point
                    if self._is_legal_move(i, j):
                        moves.append((i, j))
        return moves
    
    def _is_legal_move(self, i: int, j: int) -> bool:
        """Check if a move is legal (respects ko and suicide rules)"""
        # Check if point is empty and not ko
        if self.board[i, j] != 0 or (i, j) == self.ko_point:
            return False
        
        # Make a test board to check suicide
        test_board = copy.deepcopy(self)
        test_board.board[i, j] = self.current_player
        
        # Check if the move captures any stones
        captures_any = False
        for ni, nj in self.neighbors[(i, j)]:
            if test_board.board[ni, nj] == -self.current_player:
                if test_board._count_liberties(ni, nj) == 0:
                    captures_any = True
                    break
        
        # If no captures, check if the new stone has liberties
        if not captures_any:
            if test_board._count_liberties(i, j) == 0:
                return False  # Suicide
        
        return True
    
    def _count_liberties(self, i: int, j: int) -> int:
        """Count liberties of a connected group using flood fill"""
        color = self.board[i, j]
        if color == 0:
            return 0
            
        visited = set()
        liberties = set()
        stack = [(i, j)]
        
        while stack:
            ci, cj = stack.pop()
            if (ci, cj) in visited:
                continue
            visited.add((ci, cj))
            
            for ni, nj in self.neighbors[(ci, cj)]:
                if self.board[ni, nj] == 0:
                    liberties.add((ni, nj))
                elif self.board[ni, nj] == color and (ni, nj) not in visited:
                    stack.append((ni, nj))
        
        return len(liberties)
    
    def make_move(self, move: Tuple[int, int]) -> bool:
        """Make a move, returns True if successful"""
        if move == (-1, -1):  # Pass
            self.history.append(copy.deepcopy(self.board))
            self.current_player = -self.current_player
            self.ko_point = None
            self.move_count += 1  # ADD THIS LINE
            return True
            
        i, j = move
        if not self._is_legal_move(i, j):
            return False
        
        # Place stone
        self.board[i, j] = self.current_player
        
        # Capture opponent stones
        captured_stones = []
        for ni, nj in self.neighbors[(i, j)]:
            if self.board[ni, nj] == -self.current_player:
                if self._count_liberties(ni, nj) == 0:
                    # Capture this group
                    self._remove_group(ni, nj)
                    # Count the captured stones (we'll track the first one for ko)
                    captured_stones.append((ni, nj))
        
        # Update ko - simple ko rule: single stone capture that recreates previous position
        self.ko_point = None
        if len(captured_stones) == 1:
            # Check if this creates a ko situation
            cap_i, cap_j = captured_stones[0]
            # For simplicity, we'll set ko at the captured point
            self.ko_point = (cap_i, cap_j)
        
        # Update capture counts
        if captured_stones:
            if self.current_player == 1:
                self.capture_counts[0] += len(captured_stones)
            else:
                self.capture_counts[1] += len(captured_stones)
        
        # Save to history and switch player
        self.history.append(copy.deepcopy(self.board))
        self.current_player = -self.current_player
        self.move_count += 1  # ADD THIS LINE
        return True
    
    def _remove_group(self, i: int, j: int):
        """Remove a connected group of stones"""
        color = self.board[i, j]
        stack = [(i, j)]
        
        while stack:
            ci, cj = stack.pop()
            if self.board[ci, cj] == color:
                self.board[ci, cj] = 0
                for ni, nj in self.neighbors[(ci, cj)]:
                    if self.board[ni, nj] == color:
                        stack.append((ni, nj))
    
    def get_game_ended(self) -> float:
        """
        Returns:
            0 if game not ended
            1 if current player won
            -1 if current player lost
        """
        # Simple implementation: game ends after two consecutive passes
        if len(self.history) >= 2:
            # Check if last two moves were passes
            if (len(self.history) >= 2 and 
                np.array_equal(self.history[-1], self.history[-2])):
                return self._calculate_winner()
        return 0
    
    def _calculate_winner(self) -> float:
        """Calculate winner using territory scoring"""
        # Simple scoring: count stones + captures
        black_stones = np.sum(self.board == 1)
        white_stones = np.sum(self.board == -1)
        black_score = black_stones + self.capture_counts[0]
        white_score = white_stones + self.capture_counts[1] + 6.5  # Komi
        
        if black_score > white_score:
            return 1  # Black wins
        else:
            return -1  # White wins
    
    def get_network_input(self) -> np.ndarray:
        """Convert board to neural network input format (17 channels)"""
        # Current board and history for the last 8 moves
        input_planes = np.zeros((17, self.size, self.size), dtype=np.float32)
        
        # Fill history planes (8 for each color)
        history_len = min(8, len(self.history))
        for i in range(history_len):
            historic_board = self.history[-(i+1)]
            # Black stones
            input_planes[2*i] = (historic_board == 1).astype(np.float32)
            # White stones  
            input_planes[2*i + 1] = (historic_board == -1).astype(np.float32)
        
        # Current player plane (all 1s if black, all 0s if white)
        input_planes[16] = np.full((self.size, self.size), 1 if self.current_player == 1 else 0)
        
        return input_planes

# Test the implementation
if __name__ == "__main__":
    board = GoBoard(9)
    print("Board initialized successfully")
    print(f"Legal moves: {len(board.get_legal_moves())}")