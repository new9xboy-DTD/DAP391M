"""
Training utilities for ViXNet
Includes training functions, evaluation, and metrics
"""

import os
import time
import json
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from config import Config


def train_one_epoch(model, train_loader, criterion, optimizer, scaler, epoch, stage_name):
    """
    Train the model for one epoch
    
    Args:
        model: ViXNet model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        scaler: GradScaler for mixed precision
        epoch: Current epoch number
        stage_name: Name of training stage
        
    Returns:
        Dictionary with training metrics
    """
    model.train()
    
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(train_loader, desc=f"{stage_name} - Epoch {epoch} [TRAIN]")
    
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(Config.DEVICE)
        labels = labels.to(Config.DEVICE)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if Config.MIXED_PRECISION and scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            
            # Gradient clipping
            if Config.GRADIENT_CLIP > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRADIENT_CLIP)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            if Config.GRADIENT_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRADIENT_CLIP)
            
            optimizer.step()
        
        # Get predictions
        _, preds = torch.max(outputs, 1)
        
        # Update metrics
        running_loss += loss.item() * images.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        if (batch_idx + 1) % Config.LOG_INTERVAL == 0:
            current_loss = running_loss / len(all_labels)
            current_acc = accuracy_score(all_labels, all_preds)
            pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'acc': f'{current_acc:.4f}'
            })
    
    # Calculate epoch metrics
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return {
        'loss': epoch_loss,
        'accuracy': epoch_acc
    }


def validate(model, val_loader, criterion, epoch=None, stage_name="Validation"):
    """
    Validate the model
    
    Args:
        model: ViXNet model
        val_loader: Validation data loader
        criterion: Loss function
        epoch: Current epoch number (optional)
        stage_name: Name for progress bar
        
    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    desc = f"{stage_name} - Epoch {epoch} [VAL]" if epoch else f"{stage_name} [VAL]"
    pbar = tqdm(val_loader, desc=desc)
    
    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE)
            
            # Mixed precision inference
            if Config.MIXED_PRECISION:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            # Get predictions and probabilities
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            # Update metrics
            running_loss += loss.item() * images.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    # Only compute binary metrics if we have both classes
    unique_labels = np.unique(all_labels)
    if len(unique_labels) > 1:
        epoch_precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
        epoch_recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
        epoch_f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
    else:
        epoch_precision = 0.0
        epoch_recall = 0.0
        epoch_f1 = 0.0
    
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        'loss': epoch_loss,
        'accuracy': epoch_acc,
        'precision': epoch_precision,
        'recall': epoch_recall,
        'f1': epoch_f1,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }


def save_checkpoint(model, optimizer, epoch, metrics, stage, is_best=False, filename=None):
    """
    Save model checkpoint
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        metrics: Validation metrics
        stage: Training stage (1 or 2)
        is_best: Whether this is the best model
        filename: Custom filename (optional)
    """
    os.makedirs(Config.SAVE_DIR, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'stage': stage,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': {
            'loss': float(metrics['loss']),
            'accuracy': float(metrics['accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1': float(metrics['f1']),
            'confusion_matrix': metrics['confusion_matrix'].tolist()
        },
        'config': {
            'img_size': Config.IMG_SIZE,
            'xception_dim': Config.XCEPTION_DIM,
            'vit_dim': Config.VIT_DIM,
            'fusion_dim': Config.FUSION_DIM,
            'num_classes': Config.NUM_CLASSES
        },
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Save regular checkpoint
    if filename is None:
        filename = f'checkpoint_stage{stage}_epoch{epoch}.pth'
    
    checkpoint_path = os.path.join(Config.SAVE_DIR, filename)
    torch.save(checkpoint, checkpoint_path)
    print(f"💾 Saved checkpoint: {checkpoint_path}")
    
    # Save best model
    if is_best:
        best_path = os.path.join(Config.SAVE_DIR, f'best_model_stage{stage}.pth')
        torch.save(checkpoint, best_path)
        print(f"🏆 Saved best model: {best_path}")
        
        # Also save as overall best
        overall_best_path = os.path.join(Config.SAVE_DIR, 'best_model.pth')
        torch.save(checkpoint, overall_best_path)
        print(f"🏆 Saved overall best: {overall_best_path}")


def load_checkpoint(model, checkpoint_path, optimizer=None):
    """
    Load model checkpoint
    
    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint file
        optimizer: Optimizer to load state into (optional)
        
    Returns:
        Dictionary with checkpoint info
    """
    print(f"\n📂 Loading checkpoint from: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE, weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"✅ Loaded checkpoint from epoch {checkpoint['epoch']}, stage {checkpoint.get('stage', 'N/A')}")
    print(f"   Validation accuracy: {checkpoint['metrics']['accuracy']:.4f}")
    
    return checkpoint


def print_metrics(train_metrics, val_metrics, test_metrics=None, epoch=None):
    """
    Print training, validation, and test metrics in a formatted way
    
    Args:
        train_metrics: Training metrics dictionary
        val_metrics: Validation metrics dictionary
        test_metrics: Test metrics dictionary (optional)
        epoch: Current epoch number (optional)
    """
    print("\n" + "="*70)
    if epoch:
        print(f"📊 METRICS - EPOCH {epoch}")
    else:
        print("📊 METRICS")
    print("="*70)
    
    print("\n🎯 TRAINING:")
    print(f"   Loss: {train_metrics['loss']:.4f}")
    print(f"   Accuracy: {train_metrics['accuracy']:.4f}")
    
    print("\n✅ VALIDATION:")
    print(f"   Loss: {val_metrics['loss']:.4f}")
    print(f"   Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"   Precision: {val_metrics['precision']:.4f}")
    print(f"   Recall: {val_metrics['recall']:.4f}")
    print(f"   F1-Score: {val_metrics['f1']:.4f}")
    print(f"\n   Confusion Matrix:")
    print(f"   {val_metrics['confusion_matrix']}")
    
    if test_metrics:
        print("\n🧪 TEST:")
        print(f"   Loss: {test_metrics['loss']:.4f}")
        print(f"   Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"   Precision: {test_metrics['precision']:.4f}")
        print(f"   Recall: {test_metrics['recall']:.4f}")
        print(f"   F1-Score: {test_metrics['f1']:.4f}")
        print(f"\n   Confusion Matrix:")
        print(f"   {test_metrics['confusion_matrix']}")
    
    print("="*70)


def convert_to_json_serializable(obj):
    """
    Recursively convert numpy types to Python native types for JSON serialization
    
    Args:
        obj: Object to convert (can be dict, list, numpy type, etc.)
        
    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


def save_training_history(history, filename='training_history.json'):
    """
    Save training history to JSON file
    
    Args:
        history: List of epoch histories
        filename: Output filename
    """
    filepath = os.path.join(Config.SAVE_DIR, filename)
    
    # Convert all numpy types to JSON-serializable Python types
    json_serializable_history = convert_to_json_serializable(history)
    
    with open(filepath, 'w') as f:
        json.dump(json_serializable_history, f, indent=2)
    
    print(f"💾 Saved training history: {filepath}")


class EarlyStopping:
    """
    Early stopping to stop training when validation metric stops improving
    """
    
    def __init__(self, patience=7, min_delta=0.0, mode='max'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for accuracy (higher is better), 'min' for loss (lower is better)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, metric):
        """
        Check if should stop training
        
        Args:
            metric: Current metric value
            
        Returns:
            Boolean indicating if training should stop
        """
        if self.mode == 'max':
            score = metric
        else:
            score = -metric
        
        if self.best_score is None:
            self.best_score = score
            return False
        
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False


if __name__ == "__main__":
    print("Training utilities loaded successfully!")
    print(f"Device: {Config.DEVICE}")
    print(f"Mixed precision: {Config.MIXED_PRECISION}")
