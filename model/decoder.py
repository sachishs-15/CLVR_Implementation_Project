# to change the decoder architecture

import torch
import math
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, input_channels, output_resolution, output_channels=1):
        super(Decoder, self).__init__()

        
        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(64, 256, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 512, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),
            #nn.BatchNorm2d(512),

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            #nn.BatchNorm2d(256),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            #nn.BatchNorm2d(128),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            #nn.BatchNorm2d(64),

            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),            
        )

    def forward(self, x):
        x = x.unsqueeze(2).unsqueeze(2)
        x = self.decoder(x)
        return x
    
class ImageRecDecoder(nn.Module):
    
    def __init__(self):
        super(ImageRecDecoder, self).__init__()
        
        self.decoder = nn.Sequential(

                nn.Linear(64, 2048),  # Output: 2048
                nn.ReLU(),
                nn.Unflatten(1, (128, 4, 4)),  # Input: 2048
                nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),  # Output: 64x8x8
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # Output: 32x16x16
                nn.ReLU(),
                nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # Output: 16x32x32
                nn.ReLU(),
                nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),  # Output: 1x64x64
            )
    
    def forward(self, x):
        
        x = self.decoder(x)
        return x