# 🍃 GreenGo: Energy-Efficient Go AI

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)  
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)  

> **Professional Go AI implementation with green computing focus, featuring full GTP protocol benchmarking against world-class KataGo**

## 🎯 Project Highlights

- **🤖 Neural Architecture**: ResNet-based with squeeze-excitation & multi-scale value heads  
- **🌱 Green Computing**: Energy & CO2 tracking (0.008 kWh for 3 professional games)  
- **🏆 Professional Benchmarking**: Full GTP implementation against KataGo 9x9  
- **🔬 Research Rigor**: Systematic architecture search & ensemble optimization  

## 📊 Key Results

| Metric | Value | Context |
|--------|-------|---------|
| Internal Win Rate | 58.3% | Self-play evaluation |
| KataGo Benchmark | 0% (0/3) | **Complete games** against world-class AI |
| Energy Efficiency | 0.008 kWh | For 3 full 81-move games |
| CO2 Emissions | 3.3g | Environmentally conscious AI |

## 🏗️ Architecture

```python
# State-of-the-art components
- Squeeze-Excitation blocks for channel attention
- Multi-scale value head with territory estimation  
- Enhanced MCTS with Go-specific heuristics
- Full GTP protocol implementation
```

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/panpen-tim/GreenGo.git
cd GreenGo
pip install -r requirements.txt
```

### Basic Usage

```python
from src.model.state_of_art_greennet import StateOfArtGreenNet
from src.mcts.enhanced_mcts import EnhancedMCTS
from src.game.go_board import GoBoard

# Initialize AI
model = StateOfArtGreenNet(board_size=9, channels=96, num_blocks=8)
mcts = EnhancedMCTS(model)
board = GoBoard(9)

# Play a move
policy, value = mcts.search(board)
```

### Professional Benchmarking

```python
python final_katago_benchmark.py
```

## 📁 Project Structure

```text
GreenGo/
├── src/
│   ├── game/go_board.py          # Efficient 9x9 board implementation
│   ├── model/state_of_art_greennet.py  # Neural architecture
│   └── mcts/enhanced_mcts.py     # MCTS with Go heuristics
├── benchmarks/
│   └── katago_9x9_benchmark_results.json  # Professional results
├── research_manifest.json        # Complete project context
├── final_katago_benchmark.py     # Professional benchmarking
└── requirements.txt
```

## 🔬 Technical Innovations

### 1. Green Computing Focus

```python
# Energy and CO2 tracking throughout
energy_kwh = (150 * duration) / 3600 / 1000
co2_g = energy_kwh * 400
```

### 2. Professional-Grade Architecture

- **ResNet-8/12/16 variants** with systematic evaluation  
- **Squeeze-Excitation** for channel-wise attention  
- **Multi-scale value heads** for territory estimation  
- **Ensemble optimization** with weighted combinations  

### 3. Robust Engineering

- **Full GTP** protocol implementation  
- **Real-time move tracking** with proper error handling  
- **JSON serialization** for reproducible results  

## 📈 Development Journey

### Phase 1: Architecture Exploration

- Systematically tested ResNet variants (8-16 layers)  
- Identified optimal depth/width trade-offs  
- Achieved 58.3% win rate with RL-optimized training  

### Phase 2: Professional Benchmarking

- Implemented complete GTP protocol  
- Successfully played 3 full games against KataGo  
- Established green computing metrics baseline  

### Phase 3: Production Readiness

- Code cleanup and documentation  
- Error handling and robustness  
- Portfolio optimization  

## 🌱 Environmental Impact

| Component | Energy | CO2 Equivalent |
|--------|-------|---------|
| Training (Breakthrough) | 0.101 kWh | 40.5g |
| Benchmarking (3 games) | 0.008 kWh | 3.3g |
| Total Project | 0.109 kWh | 43.8g |
> Equivalent to charging a smartphone 3 times ⚡

## 🎯 Employment Value Proposition

This project demonstrates:  

- 🔧 Full-Stack ML: From neural architecture to production benchmarking  
- 🌍 Green Leadership: Environmentally conscious AI development  
- 📊 Professional Standards: Honest benchmarking against world-class systems  
- 🚀 Engineering Rigor: Robust implementation with proper protocols  

## 🤝 Contributing

This project is part of my professional portfolio. While primarily demonstrating individual capability, I welcome discussions about:  

- Green computing in AI  
- Go AI development  
- Professional benchmarking methodologies  

## 📜 License

MIT License - feel free to reference this implementation in your own work.

## 📫 Connect

- GitHub: [panpen-tim](https://github.com/panpen-tim)  
- LinkedIn: [Timothy Leung](https://linkedin.com/in/timothy-leung-3928ba234)  

---

Built with 🌱 green computing principles and professional engineering standards.