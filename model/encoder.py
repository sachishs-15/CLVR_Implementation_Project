import torch.nn as nn
import torch

import math

class Encoder(nn.Module):

    def __init__(self, input_resolution, input_channels, output_channels = 64):
        super(Encoder, self).__init__()
        
        # log 2 of input resolution
        layers_n = math.log2(input_resolution)

        conv_layers = []
        next_channels = 4

        for i in range(int(layers_n)): # conv layers

            conv_layers.append(nn.Conv2d(input_channels, next_channels, 3, 2, 1))
            conv_layers.append(nn.ReLU())
            input_channels = next_channels
            next_channels *= 2

        self.encoder = nn.Sequential(*conv_layers)

        flatten_size = int(2 ** (layers_n + 1)) # flatten size for linear layer
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(flatten_size, output_channels)
        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.encoder(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.relu(x)

        return x

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.ReLU()
        )

    def forward(self, x):

        x = self.mlp(x)
        return x
    
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, conditioning_frames = 0):
        super(LSTM, self).__init__()
    
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

        self.out = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.ReLU()
        )
        self.conditioning_frames = conditioning_frames

    def forward(self, x):

        x, h = self.lstm(x) # gettings the hidden state
        x = x.squeeze(0)
        x = self.out(x)[self.conditioning_frames:]
        return x
    
class RewardMLP(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, heads = 1):
        super(RewardMLP, self).__init__()
        
        self.heads = nn.ModuleList([MLP(input_size, hidden_size, output_size) for _ in range(heads)])
    
    def forward(self, x):

        x = torch.cat([head(x) for head in self.heads], dim = 1)
        return x

class CNNEncoder(nn.Module):  # for 64 dim input

    def __init__(self, input_channels):
        super(CNNEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 16, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 2, 1),
            nn.ReLU(),
        )

        self.flatten = nn.Flatten()

    def forward(self, x):

        x = self.encoder(x)
        x = self.flatten(x)
        
        return x
    
class ImageRecEncoder(nn.Module):

    def __init__(self):
        super(ImageRecEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # Output: 16x32x32
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # Output: 32x16x16
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # Output: 64x8x8
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # Output: 128x4x4
            nn.ReLU(),
            nn.Flatten(),  # Output: 2048
            nn.Linear(2048, 64),  # Output: 64
            nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        return x
