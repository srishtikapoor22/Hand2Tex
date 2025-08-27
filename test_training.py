#!/usr/bin/env python3
"""
🚀 FULL TRAINING PIPELINE TEST
Tests the complete training pipeline with Kaggle dataset
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from pathlib import Path
import logging
from typing import Dict, Tuple
import time
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_header(title: str):
    """Print a formatted header"""
    print(f"\n🚀 {title}")
    print("=" * 50)

def print_success(message: str):
    """Print success message"""
    print(f"✅ {message}")

def print_error(message: str):
    """Print error message"""
    print(f"❌ {message}")

def print_info(message: str):
    """Print info message"""
    print(f"📊 {message}")

def check_prerequisites():
    """Check if all prerequisites are met"""
    print_header("PREREQUISITES CHECK")
    
    # Check Python version
    python_version = sys.version_info
    print_info(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check required packages
    required_packages = ['torch', 'torchvision', 'PIL', 'numpy']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print_success(f"{package} ✓")
        except ImportError:
            print_error(f"{package} ✗")
            missing_packages.append(package)
    
    if missing_packages:
        print_error(f"Missing packages: {missing_packages}")
        print_info("Install with: pip install " + " ".join(missing_packages))
        return False
    
    # Check dataset
    data_path = Path("data/kaggle_math")
    if not data_path.exists():
        print_error("Kaggle dataset not found at data/kaggle_math")
        return False
    
    print_success("All prerequisites met!")
    return True

def get_device():
    """Get the best available device"""
    print_header("DEVICE SELECTION")
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print_success(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print_info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print_info("Using CPU (GPU not available)")
    
    return device

class SimpleCNN(nn.Module):
    """Simple CNN model for handwritten math symbol recognition"""
    
    def __init__(self, num_classes=82, input_channels=1):
        super(SimpleCNN, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

def create_dataset_and_loaders(batch_size=32, num_workers=2):
    """Create dataset and data loaders"""
    print_header("DATASET CREATION")
    
    data_path = "data/kaggle_math"
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64)),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    val_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Load full dataset
    full_dataset = datasets.ImageFolder(
        root=data_path,
        transform=None  # We'll apply transforms after splitting
    )
    
    num_classes = len(full_dataset.classes)
    total_samples = len(full_dataset)
    
    print_success(f"Dataset loaded: {total_samples} samples, {num_classes} classes")
    print_info(f"Classes: {full_dataset.classes[:10]}..." if len(full_dataset.classes) > 10 else f"Classes: {full_dataset.classes}")
    
    # Split dataset
    train_size = int(0.8 * total_samples)
    val_size = total_samples - train_size
    
    train_indices, val_indices = random_split(
        range(total_samples), 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print_info(f"Train samples: {train_size}")
    print_info(f"Validation samples: {val_size}")
    
    # Create datasets with transforms
    class TransformDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, indices, transform):
            self.dataset = dataset
            self.indices = indices
            self.transform = transform
        
        def __len__(self):
            return len(self.indices)
        
        def __getitem__(self, idx):
            actual_idx = self.indices[idx]
            path, label = self.dataset.samples[actual_idx]
            
            # Load image
            from PIL import Image
            image = Image.open(path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
    
    train_dataset = TransformDataset(full_dataset, train_indices, train_transform)
    val_dataset = TransformDataset(full_dataset, val_indices, val_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    print_success(f"Data loaders created: {len(train_loader)} train batches, {len(val_loader)} val batches")
    
    return train_loader, val_loader, num_classes

def test_model_creation_and_forward_pass(num_classes, device):
    """Test model creation and forward pass"""
    print_header("MODEL CREATION & FORWARD PASS TEST")
    
    try:
        # Create model
        model = SimpleCNN(num_classes=num_classes)
        model = model.to(device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print_success(f"Model created successfully")
        print_info(f"Total parameters: {total_params:,}")
        print_info(f"Trainable parameters: {trainable_params:,}")
        
        # Test forward pass
        batch_size = 4
        test_input = torch.randn(batch_size, 1, 64, 64).to(device)
        
        model.eval()
        with torch.no_grad():
            output = model(test_input)
        
        expected_shape = (batch_size, num_classes)
        if output.shape == expected_shape:
            print_success(f"Forward pass successful: {output.shape}")
            return model
        else:
            print_error(f"Wrong output shape: got {output.shape}, expected {expected_shape}")
            return None
            
    except Exception as e:
        print_error(f"Model creation/forward pass failed: {e}")
        return None

def test_single_training_step(model, train_loader, device):
    """Test a single training step"""
    print_header("SINGLE TRAINING STEP TEST")
    
    try:
        # Setup training components
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        model.train()
        
        # Get one batch
        data_iter = iter(train_loader)
        inputs, labels = next(data_iter)
        inputs, labels = inputs.to(device), labels.to(device)
        
        print_info(f"Batch input shape: {inputs.shape}")
        print_info(f"Batch labels shape: {labels.shape}")
        print_info(f"Label range: {labels.min().item()} to {labels.max().item()}")
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        print_info(f"Output shape: {outputs.shape}")
        print_info(f"Loss: {loss.item():.4f}")
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        print_success("Single training step completed successfully!")
        return True
        
    except Exception as e:
        print_error(f"Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_validation_step(model, val_loader, device):
    """Test validation step"""
    print_header("VALIDATION STEP TEST")
    
    try:
        criterion = nn.CrossEntropyLoss()
        model.eval()
        
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            # Test on first few batches
            for batch_idx, (inputs, labels) in enumerate(val_loader):
                if batch_idx >= 3:  # Test first 3 batches
                    break
                
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = val_loss / min(3, len(val_loader))
        accuracy = 100 * correct / total if total > 0 else 0
        
        print_success("Validation step completed!")
        print_info(f"Average validation loss: {avg_loss:.4f}")
        print_info(f"Accuracy on test batches: {accuracy:.2f}%")
        
        return True
        
    except Exception as e:
        print_error(f"Validation step failed: {e}")
        return False

def run_mini_training_loop(model, train_loader, val_loader, device, num_epochs=2):
    """Run a mini training loop to test everything"""
    print_header("MINI TRAINING LOOP TEST")
    
    try:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        print_info(f"Running {num_epochs} epochs with limited batches...")
        
        for epoch in range(num_epochs):
            print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
            
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                if batch_idx >= 10:  # Limit to 10 batches for testing
                    break
                
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                if batch_idx % 5 == 0:
                    print_info(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            # Calculate training metrics
            avg_train_loss = train_loss / min(10, len(train_loader))
            train_accuracy = 100 * train_correct / train_total if train_total > 0 else 0
            
            print_success(f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_idx, (inputs, labels) in enumerate(val_loader):
                    if batch_idx >= 5:  # Limit validation batches too
                        break
                    
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            avg_val_loss = val_loss / min(5, len(val_loader))
            val_accuracy = 100 * val_correct / val_total if val_total > 0 else 0
            
            print_info(f"Epoch {epoch + 1} - Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        
        print_success("🎉 Mini training loop completed successfully!")
        return True
        
    except Exception as e:
        print_error(f"Mini training loop failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_saving(model):
    """Test model saving and loading"""
    print_header("MODEL SAVING & LOADING TEST")
    
    try:
        # Create models directory
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        # Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = models_dir / f"test_model_{timestamp}.pth"
        
        # Save model state
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'timestamp': timestamp,
            'model_class': 'SimpleCNN',
            'test_run': True
        }
        
        torch.save(checkpoint, model_path)
        print_success(f"Model saved to: {model_path}")
        
        # Test loading
        loaded_checkpoint = torch.load(model_path, map_location='cpu')
        print_success("Model loaded successfully")
        print_info(f"Saved at: {loaded_checkpoint['timestamp']}")
        
        # Test loading into model
        model.load_state_dict(loaded_checkpoint['model_state_dict'])
        print_success("Model state loaded successfully")
        
        return True
        
    except Exception as e:
        print_error(f"Model saving/loading failed: {e}")
        return False

def run_full_pipeline_test():
    """Run the complete training pipeline test"""
    print_header("FULL TRAINING PIPELINE TEST")
    print_info("This will test the complete training pipeline with your Kaggle data")
    
    tests_passed = 0
    total_tests = 7
    
    # Test 1: Prerequisites
    if not check_prerequisites():
        print_error("Prerequisites not met. Please fix the issues above.")
        return False
    tests_passed += 1
    
    # Test 2: Device selection
    device = get_device()
    tests_passed += 1
    
    # Test 3: Dataset and loaders
    try:
        train_loader, val_loader, num_classes = create_dataset_and_loaders(
            batch_size=8,  # Smaller batch size for testing
            num_workers=0   # Avoid multiprocessing issues on Windows
        )
        tests_passed += 1
    except Exception as e:
        print_error(f"Dataset creation failed: {e}")
        return False
    
    # Test 4: Model creation
    model = test_model_creation_and_forward_pass(num_classes, device)
    if model is None:
        return False
    tests_passed += 1
    
    # Test 5: Single training step
    if not test_single_training_step(model, train_loader, device):
        return False
    tests_passed += 1
    
    # Test 6: Validation step
    if not test_validation_step(model, val_loader, device):
        return False
    tests_passed += 1
    
    # Test 7: Mini training loop
    if not run_mini_training_loop(model, train_loader, val_loader, device):
        return False
    tests_passed += 1
    
    # Bonus test: Model saving
    model_saving_success = test_model_saving(model)
    
    # Final results
    print_header("PIPELINE TEST RESULTS")
    
    if tests_passed == total_tests:
        print_success("🎉 ALL PIPELINE TESTS PASSED!")
        print("")
        print_info("Your training pipeline is ready!")
        print_info("You can now start full training with:")
        print("  python -m src.train")
        print("")
        
        if model_saving_success:
            print_success("✨ Bonus: Model saving/loading works perfectly!")
        
        print_info("System is ready for production training! 🚀")
        return True
    else:
        print_error(f"❌ {total_tests - tests_passed}/{total_tests} tests failed")
        print_info("Please fix the issues above before starting full training")
        return False

if __name__ == "__main__":
    print("🚀 Starting Full Training Pipeline Test...")
    print("This will test your complete Neural OCR training setup")
    
    success = run_full_pipeline_test()
    
    if success:
        print("\n🎉 SUCCESS! Your training pipeline is ready!")
    else:
        print("\n❌ Some tests failed. Please check the output above.")
    
    sys.exit(0 if success else 1)