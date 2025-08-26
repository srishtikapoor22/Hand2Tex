from PIL import Image
from torchvision import transforms

def preprocess_image(img_path: str):
    transform=transforms.Compose([
        transforms.Resize((128,128)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,),(0.5,))
    ])
    img=Image.open(img_path).convert("RGB")
    return transform(img).unsqueeze(0)