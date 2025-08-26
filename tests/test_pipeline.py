import torch
from src.data import preprocess_image
from src.model import SimpleMathOCR

def test_pipeline():
    img = preprocess_image("data/sample.png")
    model = SimpleMathOCR(num_classes=5)
    out = model(img)
    assert isinstance(out, torch.Tensor)
    assert out.shape[1] == 5
