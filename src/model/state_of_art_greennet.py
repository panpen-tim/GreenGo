"""
STATE-OF-THE-ART GO ARCHITECTURE
Modern neural components for professional-level play
Honors existing file structure - placed in src/model/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SqueezeExcitation(nn.Module):
    """Channel-wise attention mechanism"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y

class ResidualSEBlock(nn.Module):
    """Residual block with squeeze-excitation"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SqueezeExcitation(channels)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.se(x)
        x += residual
        return F.relu(x)

class MultiScaleValueHead(nn.Module):
    """Advanced value head with multi-scale processing"""
    def __init__(self, channels, board_size=9):
        super().__init__()
        
        # Global context branch
        self.global_conv = nn.Conv2d(channels, channels//2, 3, padding=1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Local pattern branch  
        self.local_conv = nn.Conv2d(channels, channels//2, 1)
        
        # Territory estimation
        self.territory_net = nn.Sequential(
            nn.Linear(channels, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()
        )
        
        # Win probability
        self.win_net = nn.Sequential(
            nn.Linear(channels, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Global features
        global_feat = F.relu(self.global_conv(x))
        global_pooled = self.global_pool(global_feat).squeeze(-1).squeeze(-1)
        
        # Local features
        local_feat = F.relu(self.local_conv(x))
        local_pooled = F.adaptive_avg_pool2d(local_feat, (1,1)).squeeze(-1).squeeze(-1)
        
        combined = torch.cat([global_pooled, local_pooled], dim=1)
        
        territory = self.territory_net(combined)
        win_prob = self.win_net(combined)
        
        # Combine territory and win probability
        final_value = territory * (1 + 0.5 * win_prob)
        return final_value, win_prob, territory

class StateOfArtGreenNet(nn.Module):
    """Modern Go architecture with advanced components"""
    
    def __init__(self, board_size=9, channels=96, num_blocks=8):
        super().__init__()
        self.board_size = board_size
        self.channels = channels
        
        # Input: 17 channels (8 history + current player)
        self.input_conv = nn.Conv2d(17, channels, 3, padding=1)
        
        # Advanced residual blocks with SE
        self.blocks = nn.ModuleList([
            ResidualSEBlock(channels) for _ in range(num_blocks)
        ])
        
        # Self-attention across spatial positions
        self.attention = nn.MultiheadAttention(
            embed_dim=channels, num_heads=8, batch_first=True
        )
        
        # Policy head
        self.policy_conv = nn.Conv2d(channels, 2, 1)
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size + 1)
        
        # Advanced value head
        self.value_head = MultiScaleValueHead(channels, board_size)
        
        print(f"🔬 StateOfArtGreenNet Parameters: {sum(p.numel() for p in self.parameters()):,}")
        
    def forward(self, x):
        # Input shape: [batch, 17, 9, 9]
        x = F.relu(self.input_conv(x))
        
        # Residual blocks with SE
        for block in self.blocks:
            x = block(x)
        
        # Attention
        batch, channels, h, w = x.shape
        x_flat = x.view(batch, channels, h * w).transpose(1, 2)
        attended, _ = self.attention(x_flat, x_flat, x_flat)
        attended = attended.transpose(1, 2).view(batch, channels, h, w)
        
        # Policy head
        policy = self.policy_conv(attended)
        policy = policy.view(batch, -1)
        policy = self.policy_fc(policy)
        
        # Advanced value head
        value, win_prob, territory = self.value_head(attended)
        
        return policy, value, win_prob, territory

# Test the architecture
if __name__ == "__main__":
    model = StateOfArtGreenNet()
    print("✅ StateOfArtGreenNet created successfully!")