import torch
from typing import Optional

def load_model_checkpoint(model_path: str, model_class, **kwargs):
    """
    Load a model from checkpoint.
    
    Args:
        model_path: Path to the checkpoint file
        model_class: Model class to instantiate
        **kwargs: Arguments to pass to model constructor
        
    Returns:
        Loaded model instance
    """
    model = model_class(**kwargs)
    
    if torch.cuda.is_available():
        checkpoint = torch.load(model_path)
    else:
        checkpoint = torch.load(model_path, map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    return model

def save_model_checkpoint(model, optimizer, epoch, loss, save_path: str):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss
        save_path: Path to save the checkpoint
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    torch.save(checkpoint, save_path)
