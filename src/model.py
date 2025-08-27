import torch
import torch.nn as nn

class SimpleMathOCR(nn.Module):
    def __init__(self, num_classes):
        super(SimpleMathOCR, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # [batch, 1, 64, 64] -> [batch, 16, 64, 64]
            nn.ReLU(),
            nn.MaxPool2d(2),  # [batch, 16, 32, 32]
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # [batch, 32, 32, 32]
            nn.ReLU(),
            nn.MaxPool2d(2),  # [batch, 32, 16, 16]
            nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=1),  # [batch, 128, 16, 16]
            nn.ReLU(),
            nn.MaxPool2d(2)   # [batch, 128, 8, 8]
        )
        self.decoder = nn.Linear(128 * 8 * 8, num_classes)  # 128 * 8 * 8 = 8192

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten to [batch, 8192]
        return self.decoder(x)

if __name__ == "__main__":
    model = SimpleMathOCR(num_classes=82)
    sample_input = torch.randn(4, 1, 64, 64)  # Batch of 4 grayscale 64x64 images
    output = model(sample_input)
    print(f"Output shape: {output.shape}")  # Should be [4, 82]