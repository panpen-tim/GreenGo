"""
AGGRESSIVE KOMI TRAINING - Teach Black to overcome 6.5 komi through aggressive play
FIXED: Proper loss function and training data generation
"""

import torch
import torch.nn.functional as F
import numpy as np
import copy
from src.model.state_of_art_greennet import StateOfArtGreenNet
from src.game.go_board import GoBoard

class AggressiveKomiTrainer:
    def __init__(self):
        self.model = StateOfArtGreenNet(board_size=9, channels=96, num_blocks=8)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        # AGGRESSIVE HYPERPARAMETERS
        self.target_stones = 44  # Black needs 44+ stones to overcome komi
        self.komi = 6.5
        
    def create_winning_training_data(self):
        """Create training data that teaches winning strategies"""
        print("🎯 CREATING WINNING TRAINING DATA")
        
        training_data = []
        
        # STRATEGY 1: Black dominates corners and center
        board = GoBoard(9)
        winning_sequence = [
            # Black takes all 4 corners
            (0, 0), (0, 8), (8, 0), (8, 8),
            # Black takes center
            (4, 4), (4, 5), (5, 4), (5, 5),
            # Black expands influence
            (2, 2), (2, 6), (6, 2), (6, 6)
        ]
        
        for i, move in enumerate(winning_sequence):
            if board.make_move(move):
                # Calculate advantage based on stones placed
                black_stones = np.sum(board.board == 1)
                advantage = min(1.0, black_stones / 40.0)  # Scale to 0-1
                
                training_data.append({
                    'state': copy.deepcopy(board.get_network_input()),
                    'value_target': 0.1 + advantage * 0.9,  # Reward progress
                    'weight': 2.0
                })
        
        # STRATEGY 2: Capture training
        capture_data = self.create_capture_training_data()
        training_data.extend(capture_data)
        
        print(f"Created {len(training_data)} winning training examples")
        return training_data
    
    def create_capture_training_data(self):
        """Create training data that rewards capturing"""
        training_data = []
        
        # Simple capture scenario
        board = GoBoard(9)
        # Setup: Black surrounds a white stone
        setup_moves = [
            (1, 0), (0, 1), (1, 2), (2, 1),  # Black surrounds
            (1, 1)  # White stone in middle
        ]
        
        for move in setup_moves:
            board.make_move(move)
        
        # Position before capture - opportunity
        training_data.append({
            'state': copy.deepcopy(board.get_network_input()),
            'value_target': 0.3,  # Capture opportunity
            'weight': 1.5
        })
        
        # Execute capture
        board.make_move((0, 0))  # Black captures
        
        # Position after capture - success!
        training_data.append({
            'state': copy.deepcopy(board.get_network_input()),
            'value_target': 0.8,  # Big reward for capture
            'weight': 3.0
        })
        
        return training_data
    
    def train_aggressive_strategy(self, epochs=30):
        """Train the model to play aggressively"""
        print("🎯 AGGRESSIVE KOMI TRAINING")
        
        training_data = self.create_winning_training_data()
        
        for epoch in range(epochs):
            total_loss = 0
            batches = 0
            
            # Shuffle and batch
            np.random.shuffle(training_data)
            
            for i in range(0, len(training_data), 8):
                batch = training_data[i:i+8]
                if len(batch) < 4:
                    continue
                
                states = torch.stack([torch.tensor(d['state'], dtype=torch.float32) for d in batch])
                value_targets = torch.tensor([d['value_target'] for d in batch], dtype=torch.float32).view(-1, 1)
                weights = torch.tensor([d.get('weight', 1.0) for d in batch], dtype=torch.float32)
                
                self.optimizer.zero_grad()
                policy, value, win_prob, territory = self.model(states)
                
                # Simple weighted MSE loss
                loss = F.mse_loss(value, value_targets)
                weighted_loss = (loss * weights).mean()
                
                weighted_loss.backward()
                self.optimizer.step()
                
                total_loss += weighted_loss.item()
                batches += 1
            
            if epoch % 5 == 0:
                avg_loss = total_loss / batches if batches > 0 else 0
                print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
                
                # Test value predictions
                self._test_value_predictions()
    
    def _test_value_predictions(self):
        """Test if model is learning positive value predictions"""
        test_boards = [
            ("Empty", GoBoard(9)),
            ("Black Advantage", self._create_black_advantage_board()),
            ("White Advantage", self._create_white_advantage_board())
        ]
        
        print("  Value Predictions:")
        for name, board in test_boards:
            state_tensor = torch.tensor(board.get_network_input()).unsqueeze(0).float()
            with torch.no_grad():
                policy, value, win_prob, territory = self.model(state_tensor)
                black_stones = np.sum(board.board == 1)
                white_stones = np.sum(board.board == -1)
                print(f"    {name}: {value.item():.3f} (B:{black_stones} W:{white_stones})")
    
    def _create_black_advantage_board(self):
        """Create board with Black advantage"""
        board = GoBoard(9)
        # Black controls corners
        black_moves = [(0,0), (0,8), (8,0), (8,8), (4,4)]
        for move in black_moves:
            board.make_move(move)
        return board
    
    def _create_white_advantage_board(self):
        """Create board with White advantage"""
        board = GoBoard(9)
        # White controls corners
        board.current_player = -1  # White to move first
        white_moves = [(0,0), (0,8), (8,0), (8,8), (4,4)]
        for move in white_moves:
            board.make_move(move)
        return board

def test_model_performance():
    """Test the trained model with self-play"""
    print("\n🎯 TESTING MODEL PERFORMANCE")
    
    from src.mcts.enhanced_mcts import generate_enhanced_self_play_game
    
    # Load the trained model
    model = StateOfArtGreenNet(board_size=9, channels=96, num_blocks=8)
    try:
        model.load_state_dict(torch.load("aggressive_trained.pth"))
        print("✅ Loaded trained model")
    except:
        print("⚠️  Using fresh model")
    
    wins = 0
    total_games = 3
    
    for i in range(total_games):
        try:
            game_result = generate_enhanced_self_play_game(
                model, board_size=9, use_heuristics=True
            )
            
            if isinstance(game_result, tuple) and len(game_result) == 2:
                game_history, result = game_result
            else:
                game_history = game_result
                result = -1
                
            if result > 0:
                wins += 1
                print(f"🎉 Game {i+1}: BLACK WINS!")
                
                # Analyze the winning game
                final_board = GoBoard(9)
                for step in game_history[-10:]:  # Last 10 moves
                    if 'state' in step:
                        # Count stones from final state
                        state = step['state']
                        black_stones = np.sum(state[0]) * 81
                        white_stones = np.sum(state[1]) * 81
                        print(f"  Final stones - Black: {black_stones:.0f}, White: {white_stones:.0f}")
                        break
            else:
                print(f"Game {i+1}: result = {result}")
                
        except Exception as e:
            print(f"Game {i+1} failed: {e}")
    
    win_rate = wins / total_games
    print(f"🎯 FINAL WIN RATE: {win_rate:.1%}")
    return win_rate

def main():
    print("🔥 AGGRESSIVE KOMI TRAINING ACTIVATED")
    print("=" * 60)
    print("Strategy: TEACH BLACK TO OVERCOME 6.5 KOMI")
    print("          Focus on corner control and captures")
    print("          Reward stone accumulation")
    print("=" * 60)
    
    trainer = AggressiveKomiTrainer()
    
    # Train aggressive strategy
    trainer.train_aggressive_strategy(epochs=30)
    
    # Save model
    torch.save(trainer.model.state_dict(), "aggressive_trained.pth")
    print("💾 Model saved: aggressive_trained.pth")
    
    # Test performance
    win_rate = test_model_performance()
    
    if win_rate > 0:
        print("🎉 BREAKTHROUGH: Model achieved wins!")
    else:
        print("⚠️  Model still needs improvement")

if __name__ == "__main__":
    main()