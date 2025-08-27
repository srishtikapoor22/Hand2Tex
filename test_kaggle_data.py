#!/usr/bin/env python3
"""
🧪 FIXED KAGGLE DATASET INTEGRATION TEST
Tests the Kaggle handwritten math symbols dataset integration
"""

import os
import sys
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import random

# Add src to Python path so we can import our modules
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from src.data_kaggle import load_kaggle_dataset, get_dataset_info, create_kaggle_data_loaders
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("📝 Creating basic test without src imports...")
    
    # Fallback: test without imports
    import torch
    from torchvision import datasets, transforms
    
    def load_kaggle_dataset_fallback(data_path="data/kaggle_math", batch_size=4):
        """Fallback dataset loader"""
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        dataset = datasets.ImageFolder(root=data_path, transform=transform)
        
        # Split dataset
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
        
        return train_loader, val_loader, len(dataset.classes), {
            'classes': dataset.classes,
            'total_samples': len(dataset)
        }

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_header(title: str):
    """Print a formatted header"""
    print(f"\n🧪 {title}")
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

def test_kaggle_directory_structure() -> bool:
    """Test if Kaggle dataset has proper structure"""
    print_header("KAGGLE DATASET STRUCTURE TEST")
    
    kaggle_path = Path("data/kaggle_math")
    
    if not kaggle_path.exists():
        print_error(f"Kaggle dataset not found at {kaggle_path}")
        print_info("Expected structure:")
        print("  data/kaggle_math/")
        print("    ├── 0/")
        print("    ├── 1/")
        print("    ├── plus/")
        print("    ├── minus/")
        print("    └── ...")
        print("\n📝 To fix this:")
        print("  1. Make sure you've downloaded the Kaggle dataset")
        print("  2. Extract it to the data/kaggle_math/ directory")
        print("  3. Each symbol class should be in its own folder")
        return False
    
    print_success(f"Found Kaggle dataset at {kaggle_path}")
    
    # Get all class directories
    class_dirs = [d for d in kaggle_path.iterdir() if d.is_dir()]
    
    if len(class_dirs) == 0:
        print_error("No class directories found")
        print_info("The dataset directory exists but contains no class folders")
        return False
    
    print_success(f"Found {len(class_dirs)} class directories")
    
    # Show sample classes
    sample_classes = [d.name for d in sorted(class_dirs)[:10]]
    print_info(f"Sample classes: {sample_classes}")
    
    return True

def count_dataset_samples(kaggle_path: Path) -> Dict[str, int]:
    """Count samples in each class"""
    print_header("DATASET SAMPLE COUNT")
    
    class_counts = {}
    total_samples = 0
    
    for class_dir in kaggle_path.iterdir():
        if class_dir.is_dir():
            # Count image files
            image_files = (
                list(class_dir.glob("*.jpg")) + 
                list(class_dir.glob("*.png")) + 
                list(class_dir.glob("*.jpeg")) +
                list(class_dir.glob("*.JPG")) +
                list(class_dir.glob("*.PNG"))
            )
            count = len(image_files)
            class_counts[class_dir.name] = count
            total_samples += count
    
    print_info(f"Total samples across all classes: {total_samples:,}")
    print_info(f"Unique classes: {len(class_counts)}")
    
    if total_samples == 0:
        print_error("No image files found in any class directory!")
        print_info("Make sure your class directories contain .jpg or .png files")
        return {}
    
    # Show top 10 classes by sample count
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    print_info("Top 10 classes by sample count:")
    for class_name, count in sorted_classes[:10]:
        print(f"  {class_name}: {count:,} samples")
    
    # Check for classes with very few samples
    small_classes = [(name, count) for name, count in class_counts.items() if count < 5]
    if small_classes:
        print_error(f"Found {len(small_classes)} classes with <5 samples:")
        for name, count in small_classes[:5]:  # Show first 5
            print(f"  {name}: {count} samples")
        print_info("Classes with few samples may cause training issues")
    
    return class_counts

def test_pytorch_compatibility():
    """Test PyTorch and dependencies"""
    print_header("PYTORCH COMPATIBILITY TEST")
    
    try:
        import torch
        import torchvision
        from PIL import Image
        
        print_success(f"PyTorch version: {torch.__version__}")
        print_success(f"TorchVision version: {torchvision.__version__}")
        
        # Test CUDA
        if torch.cuda.is_available():
            print_success(f"CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print_info("CUDA not available, will use CPU")
        
        return True
        
    except ImportError as e:
        print_error(f"Missing dependency: {e}")
        print_info("Install with: pip install torch torchvision pillow")
        return False

def test_dataset_loading():
    """Test actual dataset loading"""
    print_header("DATASET LOADING TEST")
    
    try:
        # Try to use our custom loader first
        try:
            train_loader, val_loader, num_classes, class_info = load_kaggle_dataset(
                batch_size=4,
                num_workers=0  # Avoid multiprocessing issues
            )
            print_success("Used custom load_kaggle_dataset function")
            
        except NameError:
            # Fallback to basic loading
            print_info("Using fallback dataset loading...")
            train_loader, val_loader, num_classes, class_info = load_kaggle_dataset_fallback()
        
        print_success(f"Dataset loaded successfully!")
        print_info(f"Training batches: {len(train_loader)}")
        print_info(f"Validation batches: {len(val_loader)}")
        print_info(f"Number of classes: {num_classes}")
        print_info(f"Total samples: {class_info.get('total_samples', 'Unknown')}")
        
        # Test loading one batch
        try:
            for batch_idx, (images, labels) in enumerate(train_loader):
                print_success(f"Successfully loaded batch: {images.shape}")
                print_info(f"Label range: {labels.min()} to {labels.max()}")
                break
        except Exception as e:
            print_error(f"Failed to load batch: {e}")
            return False
        
        return True
        
    except Exception as e:
        print_error(f"Dataset loading failed: {e}")
        return False

def diagnose_common_issues():
    """Diagnose common setup issues"""
    print_header("COMMON ISSUES DIAGNOSIS")
    
    issues_found = []
    
    # Check if data directory exists
    data_dir = Path("data")
    if not data_dir.exists():
        issues_found.append("❌ 'data' directory not found")
        print_error("'data' directory not found")
        print_info("Create it with: mkdir data")
    else:
        print_success("'data' directory exists")
    
    # Check if kaggle_math exists
    kaggle_dir = Path("data/kaggle_math")
    if not kaggle_dir.exists():
        issues_found.append("❌ 'data/kaggle_math' directory not found")
        print_error("'data/kaggle_math' directory not found")
        print_info("This should contain your downloaded Kaggle dataset")
    else:
        print_success("'data/kaggle_math' directory exists")
    
    # Check src directory
    src_dir = Path("src")
    if not src_dir.exists():
        issues_found.append("❌ 'src' directory not found")
        print_error("'src' directory not found")
    else:
        print_success("'src' directory exists")
    
    # Check __init__.py files
    src_init = Path("src/__init__.py")
    if not src_init.exists():
        issues_found.append("⚠️  'src/__init__.py' missing")
        print_info("Consider creating src/__init__.py (touch src/__init__.py)")
    
    # Check permissions
    try:
        test_file = Path("test_write.tmp")
        test_file.write_text("test")
        test_file.unlink()
        print_success("Write permissions OK")
    except Exception:
        issues_found.append("❌ Write permission issues")
        print_error("Cannot write files in current directory")
    
    if not issues_found:
        print_success("🎉 No common issues found!")
    else:
        print_error(f"Found {len(issues_found)} potential issues:")
        for issue in issues_found:
            print(f"  {issue}")
    
    return len(issues_found) == 0

def run_all_tests():
    """Run all Kaggle dataset tests"""
    print_header("KAGGLE DATASET INTEGRATION TEST")
    
    tests_passed = 0
    total_tests = 5
    
    # Test 1: Common issues diagnosis
    print_info("Running diagnostic checks first...")
    diagnose_common_issues()
    
    # Test 2: PyTorch compatibility
    if test_pytorch_compatibility():
        tests_passed += 1
    
    # Test 3: Directory structure
    if test_kaggle_directory_structure():
        tests_passed += 1
    else:
        print("\n🚨 CRITICAL: Fix dataset structure before proceeding")
        print("📋 Quick fix checklist:")
        print("  1. Download Kaggle handwritten math symbols dataset")
        print("  2. Extract to data/kaggle_math/")
        print("  3. Verify structure: data/kaggle_math/0/, data/kaggle_math/1/, etc.")
        return False
    
    kaggle_path = Path("data/kaggle_math")
    
    # Test 4: Count samples
    class_counts = count_dataset_samples(kaggle_path)
    if class_counts:
        tests_passed += 1
        
        # Test 5: Dataset loading
        if test_dataset_loading():
            tests_passed += 1
    
    # Final results
    print_header("TEST RESULTS")
    
    if tests_passed >= 4:  # Allow some flexibility
        print_success("🎉 KAGGLE TESTS PASSED!")
        print_info("Dataset is ready for training!")
        
        print("\n📋 Next steps:")
        print("  1. Run: python test_training.py")
        print("  2. If that passes, run: python -m src.train")
        
        return True
    else:
        print_error(f"❌ {total_tests - tests_passed}/{total_tests} tests failed")
        print_info("Please fix the issues above before proceeding")
        
        print("\n🔧 Quick troubleshooting:")
        print("  • Make sure Kaggle dataset is properly extracted")
        print("  • Check that image files are in class subdirectories")
        print("  • Verify PyTorch installation: pip install torch torchvision")
        print("  • Try running: python -c 'import torch; print(torch.__version__)'")
        
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)