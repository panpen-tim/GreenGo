"""
SYSTEMATIC HYPERPARAMETER OPTIMIZATION
Focused on pushing from 40% to 50%+ win rates
"""

import torch
import numpy as np
from src.model.state_of_art_greennet import StateOfArtGreenNet
from src.mcts.enhanced_mcts import generate_enhanced_self_play_game

class HyperparameterOptimizer:
    def __init__(self):
        self.best_win_rate = 0.40  # Current proven baseline
        self.best_params = None
        self.best_model = None
        
    def optimize_learning_rates(self):
        """Test different learning rates systematically"""
        learning_rates = [0.001, 0.0005, 0.0001, 0.00005]
        win_rates = []
        
        print("🎯 SYSTEMATIC LEARNING RATE OPTIMIZATION")
        print("=" * 50)
        
        for lr in learning_rates:
            print(f"\nTesting LR: {lr}")
            win_rate = self._train_and_test_lr(lr, epochs=20)
            win_rates.append(win_rate)
            
            if win_rate > self.best_win_rate:
                self.best_win_rate = win_rate
                print(f"🎉 NEW BEST: {win_rate:.1%} with LR={lr}")
                torch.save(self.best_model.state_dict(), f"optimized_lr_{lr}.pth")
            else:
                print(f"Current: {win_rate:.1%} (Best: {self.best_win_rate:.1%})")
        
        return dict(zip(learning_rates, win_rates))
    
    def _train_and_test_lr(self, lr, epochs=20):
        """Train with specific LR and test performance"""
        model = StateOfArtGreenNet(board_size=9, channels=96, num_blocks=8)
        
        # Load proven champion as starting point
        try:
            model.load_state_dict(torch.load("aggressive_trained.pth"))
        except:
            print("⚠️  Starting from scratch")
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # Simple training with proven aggressive strategy
        training_data = self._create_optimized_training_data()
        
        for epoch in range(epochs):
            total_loss = 0
            np.random.shuffle(training_data)
            
            for i in range(0, len(training_data), 8):
                batch = training_data[i:i+8]
                if len(batch) < 4:
                    continue
                
                states = torch.stack([torch.tensor(d['state'], dtype=torch.float32) for d in batch])
                value_targets = torch.tensor([d['value_target'] for d in batch], dtype=torch.float32).view(-1, 1)
                
                optimizer.zero_grad()
                policy, value, win_prob, territory = model(states)
                
                # Conservative loss - avoid over-optimism
                loss = torch.nn.functional.mse_loss(value, value_targets)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
        
        self.best_model = model
        return self._test_win_rate(model)
    
    def _create_optimized_training_data(self):
        """Enhanced training data based on proven winning patterns"""
        from src.game.go_board import GoBoard
        import copy
        
        training_data = []
        
        # PROVEN WINNING PATTERNS (from strategic_analysis.py)
        
        # Pattern 1: Corner control + center influence
        board = GoBoard(9)
        winning_moves = [
            (0,0), (0,8), (8,0), (8,8),  # Corners first
            (2,2), (2,6), (6,2), (6,6),  # Approach moves
            (4,4)  # Center
        ]
        
        for i, move in enumerate(winning_moves):
            if board.make_move(move):
                black_stones = np.sum(board.board == 1)
                advantage = min(0.8, 0.1 + (black_stones / 50.0))  # Conservative scaling
                
                training_data.append({
                    'state': copy.deepcopy(board.get_network_input()),
                    'value_target': advantage,  # MODERATE values (0.1-0.8)
                    'weight': 1.5
                })
        
        # Pattern 2: Capture sequences
        capture_data = self._create_capture_sequences()
        training_data.extend(capture_data)
        
        # Pattern 3: Territory consolidation
        territory_data = self._create_territory_sequences()  
        training_data.extend(territory_data)
        
        print(f"Created {len(training_data)} optimized training examples")
        return training_data
    
    def _create_capture_sequences(self):
        """Training data that rewards safe captures"""
        from src.game.go_board import GoBoard
        import copy
        
        training_data = []
        
        # Safe capture scenario
        board = GoBoard(9)
        # Black surrounds white stone
        moves = [(1,0), (0,1), (1,2), (2,1), (1,1)]  # White in atari
        for move in moves:
            board.make_move(move)
        
        # Before capture - opportunity
        training_data.append({
            'state': copy.deepcopy(board.get_network_input()),
            'value_target': 0.3,  # MODERATE confidence
            'weight': 1.2
        })
        
        # After capture - success
        board.make_move((0,0))  # Capture
        training_data.append({
            'state': copy.deepcopy(board.get_network_input()),
            'value_target': 0.6,  # Good but not overconfident
            'weight': 2.0
        })
        
        return training_data
    
    def _create_territory_sequences(self):
        """Training data for territory building"""
        from src.game.go_board import GoBoard
        import copy
        
        training_data = []
        board = GoBoard(9)
        
        # Build corner territory efficiently
        corner_build = [
            (0,0), (0,1), (1,0),  # Small corner
            (0,2), (2,0), (2,2)   # Expand corner
        ]
        
        for i, move in enumerate(corner_build):
            if board.make_move(move):
                territory_size = min(10, i * 2)  # Scale with progress
                value = 0.1 + (territory_size / 15.0)  # 0.1-0.7 range
                
                training_data.append({
                    'state': copy.deepcopy(board.get_network_input()),
                    'value_target': value,
                    'weight': 1.3
                })
        
        return training_data
    
    def _test_win_rate(self, model, num_games=5):
        """Test win rate with current model"""
        wins = 0
        
        for i in range(num_games):
            try:
                game_result = generate_enhanced_self_play_game(
                    model, board_size=9, use_heuristics=True
                )
                
                if isinstance(game_result, tuple):
                    game_history, result = game_result
                else:
                    result = -1
                    
                if result > 0:
                    wins += 1
            except Exception as e:
                print(f"Game {i} failed: {e}")
        
        return wins / num_games

def main():
    print("🎯 SYSTEMATIC HYPERPARAMETER OPTIMIZATION")
    print("Goal: Push from 40% to 50%+ win rates")
    print("Strategy: Conservative value targets, proven patterns")
    print("=" * 60)
    
    optimizer = HyperparameterOptimizer()
    
    # Phase 1: Learning rate optimization
    lr_results = optimizer.optimize_learning_rates()
    
    print("\n📊 LEARNING RATE RESULTS:")
    for lr, win_rate in lr_results.items():
        print(f"LR {lr}: {win_rate:.1%}")
    
    print(f"\n🏆 BEST OVERALL: {optimizer.best_win_rate:.1%}")
    
    if optimizer.best_win_rate > 0.40:
        torch.save(optimizer.best_model.state_dict(), "hyperparameter_optimized.pth")
        print("💾 Optimized model saved: hyperparameter_optimized.pth")

if __name__ == "__main__":
    main()