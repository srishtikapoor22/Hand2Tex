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

from src.data_kaggle import create_kaggle_data_loaders
from src.model import SimpleMathOCR

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
        
        
        os.makedirs(config['paths']['model_save_dir'], exist_ok=True)
        
        
        self.model = SimpleMathOCR(num_classes=config['data']['num_classes']).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config['training']['learning_rate']
        )
        
        self.logger.info(f"Initialized model with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()  
        total_loss = 0.0
        num_batches = len(train_loader)
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            
            images, targets = images.to(self.device), targets.to(self.device)
            
            
            if targets.dim() > 1:
                targets = targets[:, 0]  
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Log every 10 batches
            if batch_idx % 10 == 0:
                self.logger.info(f'Epoch {epoch}, Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate the model"""
        self.model.eval()  # Set to evaluation mode
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(self.device), targets.to(self.device)
                
                # Simplify targets like in training
                if targets.dim() > 1:
                    targets = targets[:, 0]
                
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def save_model(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }
        
        # Regular checkpoint
        checkpoint_path = os.path.join(
            self.config['paths']['model_save_dir'], 
            f'checkpoint_epoch_{epoch}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Best model
        if is_best:
            best_path = os.path.join(
                self.config['paths']['model_save_dir'], 
                'best_model.pth'
            )
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model at epoch {epoch}")
    
    def train(self):
        """Main training loop"""
        self.logger.info("Starting training...")
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_kaggle_data_loaders(self.config)

        self.logger.info(f"Created data loaders: {len(train_loader)} train batches, {len(val_loader)} val batches")
        
        best_val_acc = 0.0
        
        for epoch in range(1, self.config['training']['epochs'] + 1):
            start_time = time.time()
            
            # Training
            train_loss = self.train_epoch(train_loader, epoch)
            
            # Validation
            val_loss, val_acc = self.validate(val_loader)
            
            epoch_time = time.time() - start_time
            
            # Logging
            self.logger.info(
                f'Epoch {epoch}: Train Loss: {train_loss:.4f}, '
                f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, '
                f'Time: {epoch_time:.2f}s'
            )
            
            # Save model
            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc
            
            if epoch % self.config['training']['save_every'] == 0 or is_best:
                self.save_model(epoch, is_best)
        
        self.logger.info(f"Training completed! Best validation accuracy: {best_val_acc:.2f}%")

def main():
    """Main training script"""
    # Load configuration
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create and run trainer
    trainer = Trainer(config)
    trainer.train()

if __name__ == "__main__":
    main()