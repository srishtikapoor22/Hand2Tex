import torch

model = SimpleMathOCR(num_classes=10)
img_tensor = preprocess_image("data/sample.png")

with torch.no_grad():
    preds = model(img_tensor)
print(preds)
