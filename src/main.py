import yaml
from pathlib import Path
import torch
def load_config(config_path: str) -> dict:
    config_file=Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_file,"r") as file:
        config=yaml.safe_load(file)
    return config

if __name__=="__main__":
    config=load_config("configs/config.yaml")
    print(config)
    if "dataset_path" in config:
        print("Dataset path:", config["dataset_path"])
    if "batch_size" in config:
        print("Batch size:", config["batch_size"])

model = SimpleMathOCR(num_classes=10)
img_tensor = preprocess_image("data/sample.png")

with torch.no_grad():
    preds = model(img_tensor)
print(preds)