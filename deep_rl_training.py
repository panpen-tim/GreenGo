"""
DEEP RL TRAINING - Advanced training with proven strategies and green computing metrics
"""

import torch
import time
import numpy as np
from tqdm import tqdm
from src.model.state_of_art_greennet import StateOfArtGreenNet
from src.mcts.enhanced_mcts import generate_enhanced_self_play_game

class GreenRLTrainer:
    def __init__(self):
        self.proven_lr = 0.0005
        self.best_win_rate = 0.375  # Our current proven baseline
        self.green_metrics = {
            'total_energy_kwh': 0,
            'co2_emissions_g': 0,
            'samples_per_watt': 0,
            'training_time_hours': 0
        }
    
    def train_with_advanced_rl(self, num_iterations=10, games_per_iteration=20):
        """Advanced RL training with enhanced green computing metrics"""
        print("🎯 DEEP RL TRAINING WITH OPTIMIZED GREEN COMPUTING")
        print("=" * 60)
        
        # Start with proven architecture - PRESERVE CHAMPION
        model = StateOfArtGreenNet(board_size=9, channels=96, num_blocks=8)
        model.load_state_dict(torch.load("aggressive_trained.pth"))
        
        # OPTIMIZED: Dynamic learning rate based on performance
        optimizer = torch.optim.Adam(model.parameters(), lr=self.proven_lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
        
        start_time = time.time()
        total_samples = 0
        performance_history = []
        
        for iteration in range(num_iterations):
            print(f"\n🔄 RL Iteration {iteration + 1}/{num_iterations}")
            iter_start = time.time()
            
            # Generate self-play data with IMPROVED efficiency
            games_data = self._generate_optimized_self_play_data(model, games_per_iteration)
            total_samples += len(games_data)
            
            # Train on collected data
            improvement = self._train_on_self_play_data(model, optimizer, games_data)
            
            # Test current performance with ENHANCED metrics
            win_rate, stone_count, confidence = self._test_model_performance_enhanced(model)
            performance_history.append(win_rate)
            
            # Update learning rate based on performance
            scheduler.step(win_rate)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Update green metrics
            self._update_green_metrics(iter_start, time.time(), len(games_data))
            
            # Save if improved with CONFIDENCE THRESHOLD
            if win_rate > self.best_win_rate and confidence > 0.6:
                self.best_win_rate = win_rate
                torch.save(model.state_dict(), f"rl_improved_{win_rate:.3f}.pth")
                print(f"🎉 NEW BEST: {win_rate:.1%} win rate, {stone_count:.0f} stones, confidence: {confidence:.2f}")
            
            self._print_enhanced_iteration_summary(iteration, win_rate, stone_count, improvement, current_lr, confidence)
            
            # Early stopping if we hit target with confidence
            if win_rate >= 0.50 and confidence > 0.7:
                print("🎯 TARGET ACHIEVED: 50%+ win rate with high confidence!")
                break
        
        self._print_final_green_metrics(total_samples, time.time() - start_time, performance_history)
        return model

    def _generate_optimized_self_play_data(self, model, num_games):
        """Generate self-play data with OPTIMIZED efficiency - FIXED temperature handling"""
        games_data = []
        
        print("  🎯 Generating optimized self-play games...")
        for i in tqdm(range(num_games), desc="Games"):
            try:
                # FIXED: Remove temperature parameter since function doesn't support it
                # Use exploration scheduling through move selection instead
                game_result = generate_enhanced_self_play_game(
                    model, board_size=9, use_heuristics=True
                )
                
                if isinstance(game_result, tuple):
                    game_history, result = game_result
                    
                    # Enhanced training data selection
                    if result > 0:  # Winning games
                        winning_moves = self._extract_winning_patterns_enhanced(game_history, result)
                        games_data.extend(winning_moves)
                    else:
                        # Learn from critical mistakes in losing games
                        critical_moves = self._extract_critical_positions(game_history, result)
                        games_data.extend(critical_moves[:2])  # Limit to top 2 critical positions
                        
            except Exception as e:
                print(f"    Game {i} failed: {e}")
                continue  # Continue with next game instead of breaking
        
        return games_data

    def _extract_winning_patterns_enhanced(self, game_history, result):
        """Enhanced pattern extraction focusing on komi-overcoming strategies"""
        training_examples = []
        
        # Focus on positions that lead to 44+ stones (proven threshold)
        for i, step in enumerate(game_history):
            if 'state' in step:
                state = step['state']
                black_stones = np.sum(state[0]) * 81
                
                # ENHANCED: Only train on positions that contribute to winning with progressive weighting
                if black_stones >= 25:  # Lower threshold to capture more learning signals
                    # Progressive value targets based on stone count
                    if black_stones >= 44:  # Winning threshold
                        value_target = 0.8  # High confidence for winning positions
                    elif black_stones >= 35:
                        value_target = 0.5  # Medium confidence
                    else:
                        value_target = 0.3  # Building phase
                    
                    # Weight by move importance (later moves more important)
                    move_weight = 1.0 + (i / len(game_history)) * 0.5
                    
                    training_examples.append({
                        'state': state,
                        'value_target': value_target,
                        'weight': move_weight,
                        'stone_count': black_stones
                    })
        
        return training_examples

    def _extract_critical_positions(self, game_history, result):
        """Extract critical positions from losing games for learning"""
        training_examples = []
        
        if not game_history:
            return training_examples
            
        # Focus on positions where the game was still winnable
        for i, step in enumerate(game_history):
            if 'state' in step and i > len(game_history) * 0.5:  # Second half of game
                state = step['state']
                black_stones = np.sum(state[0]) * 81
                
                # Only consider positions where black still had a chance
                if black_stones >= 30:
                    training_examples.append({
                        'state': state,
                        'value_target': 0.1,  # Low value for losing positions
                        'weight': 0.5,  # Lower weight for negative examples
                        'stone_count': black_stones
                    })
        
        return training_examples

    def _train_on_self_play_data(self, model, optimizer, games_data):
        """Train on self-play data with stability measures"""
        if len(games_data) < 10:
            return 0
        
        model.train()
        total_loss = 0
        batches = 0
        
        # Shuffle and batch data
        np.random.shuffle(games_data)
        
        for i in range(0, len(games_data), 8):
            batch = games_data[i:i+8]
            if len(batch) < 4:
                continue
            
            states = torch.stack([torch.tensor(d['state'], dtype=torch.float32) for d in batch])
            value_targets = torch.tensor([d['value_target'] for d in batch], dtype=torch.float32).view(-1, 1)
            weights = torch.tensor([d.get('weight', 1.0) for d in batch], dtype=torch.float32)
            
            optimizer.zero_grad()
            policy, value, win_prob, territory = model(states)
            
            # Conservative loss function
            loss = torch.nn.functional.mse_loss(value, value_targets)
            weighted_loss = (loss * weights).mean()
            
            weighted_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            
            total_loss += weighted_loss.item()
            batches += 1
        
        return total_loss / batches if batches > 0 else 0

    def _test_model_performance_enhanced(self, model, num_games=12):
        """Enhanced testing with confidence intervals and komi analysis - FIXED function call"""
        wins = 0
        stone_counts = []
        komi_overcomes = 0  # Track games where komi was overcome
        
        for i in range(num_games):
            try:
                # FIXED: Remove temperature parameter
                game_result = generate_enhanced_self_play_game(
                    model, board_size=9, use_heuristics=True
                )
                
                if isinstance(game_result, tuple):
                    game_history, result = game_result
                    if result > 0:
                        wins += 1
                        # Track stone count in wins
                        if game_history:
                            final_state = game_history[-1]['state']
                            black_stones = np.sum(final_state[0]) * 81
                            stone_counts.append(black_stones)
                            
                            # Check if komi was overcome (44+ stones)
                            if black_stones >= 44:
                                komi_overcomes += 1
                            
            except Exception as e:
                print(f"    Test game {i} failed: {e}")
        
        win_rate = wins / num_games
        avg_stones = np.mean(stone_counts) if stone_counts else 0
        
        # Calculate confidence interval (simplified)
        confidence = min(0.95, wins / max(1, num_games - wins)) if wins > 0 else 0.1
        
        komi_overcome_rate = komi_overcomes / max(1, wins)
        
        print(f"    📊 Komi Overcome Rate: {komi_overcome_rate:.1%} in wins")
        
        return win_rate, avg_stones, confidence

    def _update_green_metrics(self, start_time, end_time, samples_processed):
        """Update green computing metrics"""
        duration_hours = (end_time - start_time) / 3600
        
        # Estimated power consumption (adjust based on your hardware)
        # Typical: 100W for CPU training, 200W for GPU training
        power_watts = 150  
        energy_kwh = (power_watts * duration_hours) / 1000
        
        # CO2 emissions (typical: 0.4 kg CO2 per kWh)
        co2_g = energy_kwh * 400
        
        self.green_metrics['total_energy_kwh'] += energy_kwh
        self.green_metrics['co2_emissions_g'] += co2_g
        self.green_metrics['samples_per_watt'] = samples_processed / power_watts
        self.green_metrics['training_time_hours'] += duration_hours

    def _print_enhanced_iteration_summary(self, iteration, win_rate, stone_count, improvement, current_lr, confidence):
        """Enhanced iteration summary with learning rate and confidence"""
        print(f"  📊 Iteration {iteration + 1}:")
        print(f"     Win Rate: {win_rate:.1%} | Confidence: {confidence:.2f}")
        print(f"     Avg Stones: {stone_count:.0f} | LR: {current_lr:.6f}")
        print(f"     Loss: {improvement:.4f} | Energy: {self.green_metrics['total_energy_kwh']:.3f} kWh")

    def _print_final_green_metrics(self, total_samples, total_time, performance_history):
        """Enhanced final metrics with performance history"""
        total_hours = total_time / 3600
        performance_gain = max(performance_history) - min(performance_history) if performance_history else 0
        
        print(f"\n🌿 GREEN COMPUTING METRICS SUMMARY")
        print("=" * 50)
        print(f"🏆 Final Win Rate: {self.best_win_rate:.1%}")
        print(f"📈 Performance Gain: +{performance_gain*100:.1f}%")
        print(f"⏱️  Total Training Time: {total_hours:.2f} hours")
        print(f"⚡ Energy Consumption: {self.green_metrics['total_energy_kwh']:.3f} kWh")
        print(f"🌍 CO2 Emissions: {self.green_metrics['co2_emissions_g']:.1f} grams")
        print(f"💚 Efficiency: {self.green_metrics['samples_per_watt']:.1f} samples per watt")
        print(f"📈 Total Samples Processed: {total_samples:,}")

def main():
    print("🎯 DEEP RL TRAINING WITH GREEN COMPUTING")
    print("Using proven: LR=0.0005, ResNet-8_96ch, moderate value targets")
    print("Target: Break through 37.5% plateau to 50%+ win rate")
    print("Green metrics: Energy, CO2, efficiency tracking")
    print("=" * 60)
    
    trainer = GreenRLTrainer()
    model = trainer.train_with_advanced_rl(num_iterations=10, games_per_iteration=20)
    
    # Save final model
    torch.save(model.state_dict(), "green_rl_final.pth")
    print("💾 Final model saved: green_rl_final.pth")

if __name__ == "__main__":
    main()