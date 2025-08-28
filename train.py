#!/usr/bin/env python3
"""
Training script for SimpleMathOCR with both basic and decoder architectures.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import logging
from typing import Dict, Tuple
import time
from datetime import datetime
from pathlib import Path

from src.data_kaggle import create_kaggle_data_loaders, create_dummy_data_loaders
from src.model import SimpleMathOCR
from src.decoder import create_decoder
from src.symbol_map import get_symbol_mapper
from src.utils import save_model_checkpoint


def setup_logging(config: Dict) -> logging.Logger:
    """Set up logging configuration"""
    os.makedirs(config['logging']['log_dir'], exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(config['logging']['log_dir'], f"training_{timestamp}.log")
    
    logging.basicConfig(
        level=getattr(logging, config['logging']['level']),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also print to console
        ]
    )
    
    return logging.getLogger(__name__)


class Trainer:
    """Training class to organize training logic"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = setup_logging(config)
        self.device = torch.device(config['training']['device'])
        
        # Create directories
        os.makedirs(config['paths']['model_save_dir'], exist_ok=True)
        
        # Initialize model
        if config['model']['use_decoder']:
            # Use external decoder architecture
            self.model = SimpleMathOCR(
                num_classes=config['data']['num_classes'],
                use_decoder=True,
                input_size=config['data']['input_size']
            ).to(self.device)
            
            # Initialize decoder
            self.decoder = create_decoder(
                decoder_type=config['decoder_type'],
                input_dim=self.model.get_encoder_output_dim(),
                hidden_dim=config['model']['hidden_dim'],
                num_classes=config['data']['num_classes'],
                max_length=config['max_length'],
                num_layers=config['num_layers']
            ).to(self.device)
            
            self.logger.info(f"Initialized model with external decoder")
            self.logger.info(f"Encoder output dim: {self.model.get_encoder_output_dim()}")
        else:
            # Use basic model
            self.model = SimpleMathOCR(
                num_classes=config['data']['num_classes'],
                use_decoder=False,
                input_size=config['data']['input_size']
            ).to(self.device)
            self.decoder = None
            self.logger.info(f"Initialized basic model")
        
        # Initialize training components
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config['training']['learning_rate']
        )
        
        # Initialize decoder optimizer if using external decoder
        if self.decoder:
            self.decoder_optimizer = optim.Adam(
                self.decoder.parameters(),
                lr=config['training']['learning_rate']
            )
        
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")
        if self.decoder:
            self.logger.info(f"Decoder parameters: {sum(p.numel() for p in self.decoder.parameters())}")
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        if self.decoder:
            self.decoder.train()
            
        total_loss = 0.0
        num_batches = len(train_loader)
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images, targets = images.to(self.device), targets.to(self.device)
            
            # Simplify targets if needed
            if targets.dim() > 1:
                targets = targets[:, 0]
            
            # Forward pass
            self.optimizer.zero_grad()
            if self.decoder:
                self.decoder_optimizer.zero_grad()
            
            if self.decoder:
                # Use external decoder
                encoder_output = self.model(images)
                outputs = self.decoder(encoder_output)
                # For sequence output, we need to handle it differently
                if outputs.dim() == 3:  # [batch, seq_len, vocab_size]
                    # For now, take the first token prediction
                    outputs = outputs[:, 0, :]  # [batch, vocab_size]
            else:
                # Use basic model
                outputs = self.model(images)
            
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            if self.decoder:
                self.decoder_optimizer.step()
            
            total_loss += loss.item()
            
            # Log every 10 batches
            if batch_idx % 10 == 0:
                self.logger.info(f'Epoch {epoch}, Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate the model"""
        self.model.eval()
        if self.decoder:
            self.decoder.eval()
            
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(self.device), targets.to(self.device)
                
                # Simplify targets like in training
                if targets.dim() > 1:
                    targets = targets[:, 0]
                
                if self.decoder:
                    # Use external decoder
                    encoder_output = self.model(images)
                    outputs = self.decoder(encoder_output)
                    if outputs.dim() == 3:
                        outputs = outputs[:, 0, :]
                else:
                    # Use basic model
                    outputs = self.model(images)
                
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100 * correct / total
        return avg_loss, accuracy
    
    def save_checkpoint(self, epoch: int, train_loss: float, val_loss: float, val_accuracy: float):
        """Save model checkpoint"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save main model
        model_path = os.path.join(
            self.config['paths']['model_save_dir'], 
            f"model_epoch_{epoch}_{timestamp}.pth"
        )
        save_model_checkpoint(
            self.model, 
            self.optimizer, 
            epoch, 
            train_loss, 
            model_path
        )
        
        # Save decoder if using external decoder
        if self.decoder:
            decoder_path = os.path.join(
                self.config['paths']['model_save_dir'], 
                f"decoder_epoch_{epoch}_{timestamp}.pth"
            )
            save_model_checkpoint(
                self.decoder,
                self.decoder_optimizer,
                epoch,
                val_loss,
                decoder_path
            )
        
        self.logger.info(f"Saved checkpoint: {model_path}")
        if self.decoder:
            self.logger.info(f"Saved decoder checkpoint: {decoder_path}")
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Main training loop"""
        self.logger.info("Starting training...")
        
        best_val_loss = float('inf')
        
        for epoch in range(1, self.config['training']['epochs'] + 1):
            start_time = time.time()
            
            # Training
            train_loss = self.train_epoch(train_loader, epoch)
            
            # Validation
            val_loss, val_accuracy = self.validate(val_loader)
            
            epoch_time = time.time() - start_time
            
            # Logging
            self.logger.info(
                f"Epoch {epoch}/{self.config['training']['epochs']} "
                f"({epoch_time:.2f}s): "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val Accuracy: {val_accuracy:.2f}%"
            )
            
            # Save checkpoint
            if epoch % self.config['training']['save_every'] == 0:
                self.save_checkpoint(epoch, train_loss, val_loss, val_accuracy)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, train_loss, val_loss, val_accuracy)
                self.logger.info(f"New best model saved with validation loss: {val_loss:.4f}")


def main():
    """Main training function"""
    # Load configuration
    config_path = "configs/config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create trainer
    trainer = Trainer(config)
    
    # Create data loaders
    try:
        # Try to use Kaggle dataset
        train_loader, val_loader, test_loader = create_kaggle_data_loaders(
            data_dir=config['data']['kaggle_data_dir'],
            batch_size=config['training']['batch_size'],
            max_classes=config['data']['num_classes'],
            k_per_class=5,
            total_cap=200
        )
        trainer.logger.info("Using Kaggle dataset")
    except Exception as e:
        # Fall back to dummy data
        trainer.logger.warning(f"Could not load Kaggle dataset: {e}")
        trainer.logger.info("Using dummy dataset for testing")
        train_loader, val_loader, test_loader = create_dummy_data_loaders(
            batch_size=config['training']['batch_size'],
            num_samples=100,
            num_classes=config['data']['num_classes']
        )
    
    # Start training
    trainer.train(train_loader, val_loader)
    
    trainer.logger.info("Training completed!")


if __name__ == "__main__":
    main()
