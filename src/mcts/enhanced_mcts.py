import math
import numpy as np
import copy
import time
from typing import Dict, List, Tuple
import torch
import torch.nn.functional as F

class EnhancedMCTS:
    """MCTS with Go-specific heuristics for faster learning"""
    
    def __init__(self, model, c_puct=1.0, num_simulations=100, enable_heuristics=True):
        self.model = model
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.enable_heuristics = enable_heuristics
        
    def get_heuristic_score(self, board, move: Tuple[int, int]) -> float:
        """Comprehensive Go heuristics with AGGRESSIVE komi-overcoming focus"""
        if move == (-1, -1):  # Pass
            return -2.0  # HEAVILY PENALIZE passes for Black (aggressive play)
            
        i, j = move
        current_player = board.current_player
        score = 0.0
        
        # AGGRESSIVE WEIGHTS for Black to overcome komi
        player_multiplier = 2.0 if current_player == 1 else 1.0  # Black gets double bonuses
        
        # 1. LIBERTY COUNT - Survival but with aggressive twist
        liberty_score = self._get_liberty_score(board, move, current_player)
        score += liberty_score * 2.0
        
        # 2. TERRITORY POTENTIAL - HEAVY weight for Black
        territory_score = self._get_territory_score(board, move, current_player)
        score += territory_score * (2.0 if current_player == 1 else 1.0)  # Black bonus
        
        # 3. CAPTURE THREATS - CRITICAL for aggressive play
        capture_score = self._get_capture_score(board, move, current_player)
        score += capture_score * (3.0 if current_player == 1 else 1.5)  # Black gets double
        
        # 4. STRATEGIC POSITIONS - Standard weight
        strategic_score = self._get_strategic_score(board, move, current_player)
        score += strategic_score * 0.8
        
        # 5. CONNECTION STRENGTH - Standard weight
        connection_score = self._get_connection_score(board, move, current_player)
        score += connection_score * 1.0
        
        # 6. AGGRESSIVE BONUS: Stone count awareness
        aggressive_bonus = self._get_aggressive_bonus(board, move, current_player)
        score += aggressive_bonus * player_multiplier
        
        return max(-5.0, min(5.0, score))

    def _get_aggressive_bonus(self, board, move: Tuple[int, int], player: int) -> float:
        """Bonus for moves that help overcome komi (Black gets heavy bonuses)"""
        if player == -1:  # White - play normally
            return 0.0
            
        # BLACK-ONLY AGGRESSIVE BONUSES
        i, j = move
        
        # Bonus for claiming corners and sides (easy territory)
        if (i == 0 or i == board.size-1) and (j == 0 or j == board.size-1):
            return 1.5  # Corner claim bonus
        
        # Bonus for center control in early game
        if board.move_count < 20 and 3 <= i <= 5 and 3 <= j <= 5:
            return 1.0
            
        # Bonus for moves that threaten multiple White groups
        threat_bonus = 0.0
        for ni, nj in [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]:
            if 0 <= ni < board.size and 0 <= nj < board.size:
                if board.board[ni, nj] == -1:  # Adjacent to White
                    threat_bonus += 0.3
                    
        return threat_bonus
    
    def _get_liberty_score(self, board, move: Tuple[int, int], player: int) -> float:
        """Score based on liberties - survival is paramount"""
        # Test the move
        test_board = copy.deepcopy(board)
        if not test_board.make_move(move):
            return -10.0  # Illegal move
            
        # Count liberties for the new stone
        i, j = move
        liberties = 0
        for ni, nj in [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]:
            if 0 <= ni < board.size and 0 <= nj < board.size:
                if test_board.board[ni, nj] == 0:
                    liberties += 1
        
        # More liberties = better survival
        if liberties == 0:
            return -5.0  # Suicide move - heavily penalize
        elif liberties == 1:
            return -2.0   # Atari - very dangerous
        elif liberties == 2:
            return 0.5    # Reasonable
        else:
            return 1.0    # Safe
    
    def _get_territory_score(self, board, move: Tuple[int, int], player: int) -> float:
        """FIXED: Proper 9x9 territory scoring"""
        i, j = move
        size = board.size
        
        # CORNERS: Highest value in 9x9 Go
        if (i == 0 or i == size-1) and (j == 0 or j == size-1):
            return 1.0  # Corner - maximum territory
        
        # STAR POINTS: High strategic value  
        star_points = [(2,2), (2,6), (6,2), (6,6)]
        if (i,j) in star_points:
            return 0.8  # Star points
        
        # CENTER: Important for influence
        if (i == size//2 and j == size//2):
            return 0.7  # Center
        
        # EDGES: Medium value
        if (i == 0 or i == size-1 or j == 0 or j == size-1):
            return 0.4  # Edge
        
        # MIDDLE: Lowest initial territory value
        return 0.1
    
    def _get_capture_score(self, board, move: Tuple[int, int], player: int) -> float:
        """Score based on capture opportunities"""
        i, j = move
        capture_potential = 0.0
        
        # Check each adjacent opponent group
        for ni, nj in [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]:
            if 0 <= ni < board.size and 0 <= nj < board.size:
                if board.board[ni, nj] == -player:
                    # Check if this opponent group is in atari
                    if self._count_liberties(board, ni, nj) == 1:
                        capture_potential += 2.0  # Immediate capture
                    elif self._count_liberties(board, ni, nj) == 2:
                        capture_potential += 0.5  # Potential capture
        
        return capture_potential
    
    def _get_strategic_score(self, board, move: Tuple[int, int], player: int) -> float:
        """FIXED: Strategic positions for 9x9"""
        i, j = move
        size = board.size
        
        # For 9x9, key strategic points
        if size == 9:
            key_points = [(2,2), (2,6), (6,2), (6,6), (4,4)]  # Star points + center
            if (i,j) in key_points:
                return 1.0
        
        return 0.0
    
    def _get_connection_score(self, board, move: Tuple[int, int], player: int) -> float:
        """Score based on connecting with friendly stones"""
        i, j = move
        connection_bonus = 0.0
        
        # Check adjacent friendly stones
        friendly_neighbors = 0
        for ni, nj in [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]:
            if 0 <= ni < board.size and 0 <= nj < board.size:
                if board.board[ni, nj] == player:
                    friendly_neighbors += 1
                    connection_bonus += 0.3  # Bonus per connection
        
        # Extra bonus for forming strong groups
        if friendly_neighbors >= 2:
            connection_bonus += 0.5
        
        return connection_bonus
    
    def _count_liberties(self, board, i: int, j: int) -> int:
        """Count liberties of a stone group (simplified)"""
        if board.board[i, j] == 0:
            return 0
            
        visited = set()
        liberties = 0
        stack = [(i, j)]
        color = board.board[i, j]
        
        while stack:
            ci, cj = stack.pop()
            if (ci, cj) in visited:
                continue
            visited.add((ci, cj))
            
            for ni, nj in [(ci-1, cj), (ci+1, cj), (ci, cj-1), (ci, cj+1)]:
                if 0 <= ni < board.size and 0 <= nj < board.size:
                    if board.board[ni, nj] == 0:
                        liberties += 1
                    elif board.board[ni, nj] == color and (ni, nj) not in visited:
                        stack.append((ni, nj))
        
        return liberties

    def search(self, root_state, temperature=1.0):
        """Enhanced MCTS search with heuristics - IMPROVED for wins"""
        # Use neural network for policy and value if available, otherwise use heuristics
        if self.model and hasattr(self.model, 'eval'):
            with torch.no_grad():
                state_tensor = torch.tensor(root_state.get_network_input()).unsqueeze(0).float()
                
                # FIX: Handle both GreenNet (2 outputs) and StateOfArtGreenNet (4 outputs)
                model_output = self.model(state_tensor)
                
                # Unpack model outputs correctly
                if len(model_output) == 2:
                    policy_logits, value_estimate = model_output
                elif len(model_output) >= 4:
                    policy_logits, value_estimate, win_prob, territory = model_output
                else:
                    # Fallback if model output doesn't match expected formats
                    policy_logits = model_output[0]
                    value_estimate = torch.tensor([[0.0]])  # Neutral value fallback
                
                base_policy = F.softmax(policy_logits, dim=1).numpy()[0]
                value = value_estimate.item() if hasattr(value_estimate, 'item') else 0.0
        else:
            # Fallback to heuristic policy and neutral value
            base_policy = self._heuristic_fallback_policy(root_state)
            value = 0.0
        
        # Enhance policy with heuristics - INCREASED heuristic influence
        legal_moves = root_state.get_legal_moves()
        enhanced_scores = np.zeros_like(base_policy)
        
        for move in legal_moves:
            if move == (-1, -1):
                move_idx = base_policy.shape[0] - 1
            else:
                i, j = move
                move_idx = i * root_state.size + j
            
            heuristic_bonus = 0.0
            if self.enable_heuristics:
                heuristic_bonus = self.get_heuristic_score(root_state, move)
            
            # INCREASED heuristic influence for better moves
            scale_factor = 1.0 + (heuristic_bonus * 0.5)  # Increased from 0.2 to 0.5
            enhanced_scores[move_idx] = base_policy[move_idx] * max(0.01, scale_factor)
        
        # FIX: More exploration early, more exploitation later
        if temperature != 1.0:
            enhanced_scores = np.power(enhanced_scores, 1.0 / temperature)
        
        # FIX: Ensure all probabilities are positive and normalized
        enhanced_scores = np.maximum(enhanced_scores, 1e-8)
        enhanced_policy = enhanced_scores / enhanced_scores.sum()
        
        return enhanced_policy, value
    
    def _heuristic_fallback_policy(self, board):
        """Fallback policy when no neural network is available - FIXED probabilities"""
        policy_size = board.size * board.size + 1
        policy = np.ones(policy_size) * 1e-8  # Start with small positive values
        legal_moves = board.get_legal_moves()
        
        if not legal_moves:
            return policy / policy.sum()  # Normalize anyway
            
        # Score each legal move
        move_scores = []
        for move in legal_moves:
            score = max(0.01, self.get_heuristic_score(board, move) + 5.0)  # Shift to positive
            move_scores.append((move, score))
        
        # Convert to probabilities
        total_score = sum(score for _, score in move_scores)
        
        for move, score in move_scores:
            if move == (-1, -1):
                policy[-1] = score / total_score
            else:
                i, j = move
                policy[i * board.size + j] = score / total_score
        
        return policy

def generate_enhanced_self_play_game(model=None, board_size=5, use_heuristics=True):
    """Generate self-play games using enhanced MCTS with heuristics - FIXED probability handling"""
    from src.game.go_board import GoBoard
    
    board = GoBoard(board_size)
    game_history = []
    mcts = EnhancedMCTS(model, enable_heuristics=use_heuristics)
    
    move_count = 0
    consecutive_passes = 0
    
    while board.get_game_ended() == 0 and move_count < board_size * board_size * 2:
        # Get enhanced policy
        policy, _ = mcts.search(board)
        
        # FIX: Validate policy before using
        if np.any(policy < 0) or abs(policy.sum() - 1.0) > 1e-6:
            print(f"⚠️  Invalid policy detected! Sum: {policy.sum()}, Min: {policy.min()}")
            # Fallback to uniform over legal moves
            legal_moves = board.get_legal_moves()
            policy = np.ones_like(policy) * 1e-8
            uniform_prob = 1.0 / len(legal_moves) if legal_moves else 1.0
            for move in legal_moves:
                if move == (-1, -1):
                    policy[-1] = uniform_prob
                else:
                    i, j = move
                    policy[i * board_size + j] = uniform_prob
            policy = policy / policy.sum()  # Renormalize
        
        # Choose move (balance exploration vs exploitation)
        try:
            if move_count < 10:  # More exploration early
                move_idx = np.random.choice(len(policy), p=policy)
            else:  # More exploitation later
                move_idx = np.argmax(policy)
        except ValueError as e:
            print(f"❌ Probability error: {e}")
            print(f"   Policy sum: {policy.sum()}, min: {policy.min()}, max: {policy.max()}")
            # Fallback to first legal move
            legal_moves = board.get_legal_moves()
            if legal_moves:
                move = legal_moves[0]
                if move == (-1, -1):
                    move_idx = board_size * board_size
                else:
                    i, j = move
                    move_idx = i * board_size + j
            else:
                break
        
        # Convert to move
        if move_idx == board_size * board_size:
            move = (-1, -1)
            consecutive_passes += 1
        else:
            i = move_idx // board_size
            j = move_idx % board_size
            move = (i, j)
            consecutive_passes = 0
        
        # Make move
        if not board.make_move(move):
            print(f"❌ Illegal move attempted: {move}")
            break
            
        # Store data
        game_history.append({
            'state': board.get_network_input(),
            'policy': policy,
            'value': 0.0
        })
        
        move_count += 1
        
        # End game if both players pass consecutively
        if consecutive_passes >= 2:
            break
    
    # Score game
    result = _score_go_game(board)
    print(f"🎯 Enhanced game: {move_count} moves, result: {result}")
    return game_history, result

def _score_go_game(board):
    """Score Go game with territory counting - FIXED komi for board size"""
    # Simplified scoring for training
    black_area = np.sum(board.board == 1)
    white_area = np.sum(board.board == -1)
    
    # Adjust komi based on board size
    if board.size == 5:
        komi = 2.5  # Reduced komi for 5x5
    elif board.size == 9:
        komi = 6.5  # Standard komi for 9x9
    else:
        komi = 6.5  # Default
    
    # Add captured stones
    black_score = black_area + board.capture_counts[0]
    white_score = white_area + board.capture_counts[1] + komi
    
    return 1 if black_score > white_score else -1