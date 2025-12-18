import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class PolicyValueNet(nn.Module):
    def __init__(self, size, num_res_blocks=4, channels=64):
        super(PolicyValueNet, self).__init__()
        self.size = size
        
        # Initial convolution
        self.conv_in = nn.Conv2d(3, channels, kernel_size=3, padding=1)  # 3 channels: player, opponent, empty
        self.bn_in = nn.BatchNorm2d(channels)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([ResBlock(channels) for _ in range(num_res_blocks)])
        
        # Policy head
        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * size * size, size * size + 1)  # +1 for pass move
        
        # Value head
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(size * size, channels)
        self.value_fc2 = nn.Linear(channels, 1)

    def forward(self, x):
        # x shape: (batch_size, 3, size, size)
        out = F.relu(self.bn_in(self.conv_in(x)))
        for block in self.res_blocks:
            out = block(out)
            
        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(out)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)
        # Note: Log-softmax is often better for training, but we'll use softmax for prediction
        p = F.softmax(p, dim=1)
        
        # Value head
        v = F.relu(self.value_bn(self.value_conv(out)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))
        
        return p, v

def state_to_tensor(board):
    # Create 3-channel input: current player's stones, opponent's stones, empty
    tensor = torch.zeros((3, board.size, board.size))
    player = board.current_player
    tensor[0] = torch.tensor(board.board == player, dtype=torch.float32)
    tensor[1] = torch.tensor(board.board == -player, dtype=torch.float32)
    tensor[2] = torch.tensor(board.board == 0, dtype=torch.float32)
    return tensor.unsqueeze(0)  # Add batch dimension


