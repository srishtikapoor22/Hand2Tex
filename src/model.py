import torch
import torch.nn as nn
from typing import Optional

class SimpleMathOCR(nn.Module):
    def __init__(self, num_classes=100, use_decoder=False, input_size=128):
        super().__init__()
        self.use_decoder = use_decoder
        self.input_size = input_size
        
        # Enhanced encoder architecture
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Calculate encoder output dimensions
        if input_size == 128:
            self.encoder_output_dim = 128 * 16 * 16  # 32768
        elif input_size == 64:
            self.encoder_output_dim = 128 * 8 * 8    # 8192
        else:
            # Fallback calculation
            self.encoder_output_dim = 128 * 8 * 8
        
        if not use_decoder:
            # Original simple linear decoder
            self.decoder = nn.Linear(self.encoder_output_dim, num_classes)
        else:
            # When using external decoder, just provide encoder output
            self.decoder = None

    def forward(self, x, target: Optional[torch.Tensor] = None):
        # Encode the image
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten to [batch_size, encoder_output_dim]
        
        if not self.use_decoder:
            # Use simple linear decoder
            return self.decoder(x)
        else:
            # Return encoder output for external decoder
            return x

    def get_encoder_output_dim(self) -> int:
        """Get the dimension of encoder output."""
        return self.encoder_output_dim
