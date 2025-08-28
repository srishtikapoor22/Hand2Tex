#!/usr/bin/env python3
"""
Inference script for SimpleMathOCR.

This script takes an input image, runs it through the trained model pipeline,
and outputs the predicted LaTeX string.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import torch
import yaml

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.data import preprocess_image
from src.model import SimpleMathOCR
from src.decoder import create_decoder
from src.symbol_map import get_symbol_mapper


class MathOCRInference:
    """Inference pipeline for Math OCR."""
    
    def __init__(self, model_path: str, config_path: str = "configs/config.yaml"):
        """
        Initialize the inference pipeline.
        
        Args:
            model_path: Path to the trained model checkpoint
            config_path: Path to the configuration file
        """
        self.model_path = model_path
        self.config_path = config_path
        self.config = self._load_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self.symbol_mapper = get_symbol_mapper()
        self.model = self._load_model()
        self.decoder = self._load_decoder()
        
        print(f"Initialized inference pipeline on device: {self.device}")
    
    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        config_file = Path(self.config_path)
        
        if not config_file.exists():
            print(f"Warning: Config file not found at {self.config_path}, using defaults")
            return {
                "model": {"use_decoder": False, "hidden_dim": 256},
                "data": {"num_classes": 45, "input_size": 128},
                "decoder_type": "rnn",
                "max_length": 50,
                "num_layers": 2
            }
        
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)
        
        # Set defaults for inference-specific parameters
        config.setdefault("decoder_type", "rnn")
        config.setdefault("max_length", 50)
        config.setdefault("num_layers", 2)
        
        return config
    
    def _load_model(self) -> SimpleMathOCR:
        """Load the trained model."""
        # Get model configuration
        use_decoder = self.config.get("model", {}).get("use_decoder", False)
        num_classes = self.config.get("data", {}).get("num_classes", 45)
        input_size = self.config.get("data", {}).get("input_size", 128)
        
        model = SimpleMathOCR(
            num_classes=num_classes,
            use_decoder=use_decoder,
            input_size=input_size
        )
        
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"Loaded model from {self.model_path}")
        else:
            print(f"Warning: Model checkpoint not found at {self.model_path}, using untrained model")
        
        model.to(self.device)
        model.eval()
        return model
    
    def _load_decoder(self):
        """Load the decoder if using external decoder architecture."""
        use_decoder = self.config.get("model", {}).get("use_decoder", False)
        
        if not use_decoder:
            return None
        
        # Get decoder configuration
        input_dim = self.model.get_encoder_output_dim()
        hidden_dim = self.config.get("model", {}).get("hidden_dim", 256)
        num_classes = self.config.get("data", {}).get("num_classes", 45)
        max_length = self.config.get("max_length", 50)
        decoder_type = self.config.get("decoder_type", "rnn")
        num_layers = self.config.get("num_layers", 2)
        
        decoder = create_decoder(
            decoder_type=decoder_type,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            max_length=max_length,
            num_layers=num_layers
        )
        
        # Load decoder weights if available
        decoder_path = self.model_path.replace('.pth', '_decoder.pth')
        if os.path.exists(decoder_path):
            checkpoint = torch.load(decoder_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                decoder.load_state_dict(checkpoint['model_state_dict'])
            else:
                decoder.load_state_dict(checkpoint)
            print(f"Loaded decoder from {decoder_path}")
        else:
            print(f"Warning: Decoder checkpoint not found at {decoder_path}, using untrained decoder")
        
        decoder.to(self.device)
        decoder.eval()
        return decoder
    
    def predict(self, image_path: str) -> str:
        """
        Run inference on an input image.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Predicted LaTeX string
        """
        # Check if image exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Preprocess image
        print(f"Processing image: {image_path}")
        img_tensor = preprocess_image(image_path)
        img_tensor = img_tensor.to(self.device)
        
        # Run inference
        with torch.no_grad():
            if self.decoder:
                # Use external decoder
                encoder_output = self.model(img_tensor)
                decoder_output = self.decoder(encoder_output)
                latex_str = self.symbol_mapper.tensor_to_latex(decoder_output)
            else:
                # Use basic model
                outputs = self.model(img_tensor)
                # Convert class predictions to LaTeX
                predicted_class = torch.argmax(outputs, dim=-1).item()
                latex_str = self.symbol_mapper.id_to_symbol(predicted_class)
        
        return latex_str
    
    def predict_batch(self, image_paths: list) -> list:
        """
        Run inference on multiple images.
        
        Args:
            image_paths: List of image paths
            
        Returns:
            List of predicted LaTeX strings
        """
        results = []
        for image_path in image_paths:
            try:
                latex = self.predict(image_path)
                results.append(latex)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append("")
        return results


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Math OCR Inference")
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("--model", "-m", default="model.pth", help="Path to model checkpoint")
    parser.add_argument("--config", "-c", default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--output", "-o", help="Output file path (optional)")
    parser.add_argument("--batch", "-b", nargs="+", help="Process multiple images")
    
    args = parser.parse_args()
    
    try:
        # Initialize inference pipeline
        inference = MathOCRInference(args.model, args.config)
        
        if args.batch:
            # Batch processing
            print(f"Processing {len(args.batch)} images...")
            results = inference.predict_batch(args.batch)
            
            for image_path, latex in zip(args.batch, results):
                print(f"{image_path}: {latex}")
                
                # Save individual results
                if args.output:
                    output_dir = Path(args.output)
                    output_dir.mkdir(exist_ok=True)
                    output_file = output_dir / f"{Path(image_path).stem}.txt"
                    with open(output_file, "w") as f:
                        f.write(latex)
        else:
            # Single image processing
            latex = inference.predict(args.image_path)
            print(f"Predicted LaTeX: {latex}")
            
            # Save to file if specified
            if args.output:
                with open(args.output, "w") as f:
                    f.write(latex)
                print(f"Result saved to: {args.output}")
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
