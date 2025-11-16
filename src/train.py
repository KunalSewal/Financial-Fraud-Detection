"""
Training utilities for fraud detection models.
"""

import os
import time
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from .evaluate import compute_metrics

def train_epoch(
    model: nn.Module,
    data_dict: Dict,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    use_graph: bool = True
) -> Tuple[float, Dict]:
    """
    Train model for one epoch.
    
    Args:
        model: Neural network model
        data_dict: Dictionary containing data
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        use_graph: Whether to use graph structure
        
    Returns:
        Tuple of (loss, metrics)
    """
    model.train()
    
    # Move data to device
    x = data_dict["node_features"].to(device)
    y = data_dict["labels"].to(device)
    train_mask = data_dict["train_mask"].to(device)
    
    # Forward pass
    optimizer.zero_grad()
    
    if use_graph:
        edge_index = data_dict["edge_index"].to(device)
        out = model(x, edge_index)
    else:
        out = model(x)
    
    # Compute loss only on training nodes
    loss = criterion(out[train_mask], y[train_mask])
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    # Compute metrics
    with torch.no_grad():
        pred = out[train_mask].argmax(dim=1)
        metrics = compute_metrics(
            y[train_mask].cpu().numpy(),
            pred.cpu().numpy(),
            out[train_mask].cpu().numpy()
        )
    
    return loss.item(), metrics


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data_dict: Dict,
    device: torch.device,
    split: str = "val",
    use_graph: bool = True
) -> Tuple[float, Dict]:
    """
    Evaluate model on validation or test set.
    
    Args:
        model: Neural network model
        data_dict: Dictionary containing data
        device: Device to evaluate on
        split: Which split to evaluate ("val" or "test")
        use_graph: Whether to use graph structure
        
    Returns:
        Tuple of (loss, metrics)
    """
    model.eval()
    
    # Move data to device
    x = data_dict["node_features"].to(device)
    y = data_dict["labels"].to(device)
    mask = data_dict[f"{split}_mask"].to(device)
    
    # Forward pass
    if use_graph:
        edge_index = data_dict["edge_index"].to(device)
        out = model(x, edge_index)
    else:
        out = model(x)
    
    # Compute loss
    criterion = nn.CrossEntropyLoss()
    loss = criterion(out[mask], y[mask])
    
    # Compute metrics
    pred = out[mask].argmax(dim=1)
    metrics = compute_metrics(
        y[mask].cpu().numpy(),
        pred.cpu().numpy(),
        out[mask].cpu().numpy()
    )
    
    return loss.item(), metrics


def train_model(
    model: nn.Module,
    data_dict: Dict,
    config: Dict,
    device: torch.device,
    save_path: Optional[str] = None
) -> Dict:
    """
    Complete training loop with validation and early stopping.
    
    Args:
        model: Neural network model
        data_dict: Dictionary containing data
        config: Training configuration
        device: Device to train on
        save_path: Path to save best model
        
    Returns:
        Dictionary containing training history
    """
    # Extract config
    num_epochs = config.get("num_epochs", 100)
    learning_rate = config.get("learning_rate", 0.001)
    weight_decay = config.get("weight_decay", 5e-4)
    patience = config.get("patience", 20)
    use_graph = config.get("use_graph", True)
    
    # Initialize optimizer and criterion
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Handle class imbalance
    class_counts = torch.bincount(data_dict["labels"])
    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum()
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_metrics": [],
        "val_metrics": [],
        "epoch_times": []
    }
    
    # Early stopping
    best_val_loss = float("inf")
    best_val_f1 = 0.0
    patience_counter = 0
    best_model_state = None
    
    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70)
    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Weight decay: {weight_decay}")
    print("=" * 70 + "\n")
    
    # Training loop
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Train
        train_loss, train_metrics = train_epoch(
            model, data_dict, optimizer, criterion, device, use_graph
        )
        
        # Validate
        val_loss, val_metrics = evaluate_model(
            model, data_dict, device, split="val", use_graph=use_graph
        )
        
        epoch_time = time.time() - start_time
        
        # Save history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_metrics"].append(train_metrics)
        history["val_metrics"].append(val_metrics)
        history["epoch_times"].append(epoch_time)
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Train F1: {train_metrics['f1']:.4f} | "
                  f"Val F1: {val_metrics['f1']:.4f} | "
                  f"Time: {epoch_time:.2f}s")
        
        # Early stopping check
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics
            }
            
            # Save best model
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(best_model_state, save_path)
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state['model_state_dict'])
    
    print("\n" + "=" * 70)
    print("Training Complete")
    print("=" * 70)
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Best Val F1: {best_val_f1:.4f}")
    print(f"Total Time: {sum(history['epoch_times']):.2f}s")
    
    # Final evaluation on test set
    print("\nEvaluating on test set...")
    test_loss, test_metrics = evaluate_model(
        model, data_dict, device, split="test", use_graph=use_graph
    )
    
    print("\nTest Set Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f}")
    print(f"  F1 Score: {test_metrics['f1']:.4f}")
    print(f"  ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print("=" * 70 + "\n")
    
    # Add test results to history
    history["test_loss"] = test_loss
    history["test_metrics"] = test_metrics
    history["best_epoch"] = best_model_state['epoch'] if best_model_state else num_epochs - 1
    
    return history


def load_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device):
    """
    Load model from checkpoint.
    
    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"  Val Loss: {checkpoint['val_loss']:.4f}")
    print(f"  Val F1: {checkpoint['val_metrics']['f1']:.4f}")
    return checkpoint


class EarlyStopping:
    """Early stopping utility class."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = "min"):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait
            min_delta: Minimum change to qualify as improvement
            mode: "min" or "max" depending on metric
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Check if should stop.
        
        Args:
            score: Current score
            
        Returns:
            True if should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == "min":
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop

