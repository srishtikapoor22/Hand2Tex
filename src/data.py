from PIL import Image
from torchvision import transforms
import os
import torch
import torch.utils.data import Dataset, Dataloader
from PIL import Image
from typing import Tuple, Optional
import yaml



def preprocess_image(img_path: str):
    transform=transforms.Compose([
        transforms.Resize((128,128)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,),(0.5,))
    ])
    img=Image.open(img_path).convert("RGB")
    return transform(img).unsqueeze(0)