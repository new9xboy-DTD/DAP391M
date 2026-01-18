"""
Main training script for ViXNet
Implements 2-stage training strategy as described in the paper
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim

from model import create_vixnet
from config import Config
from dataset import create_data_loaders, check_dataset_availability
from utils import (
    train_one_epoch, validate, save_checkpoint, load_checkpoint,
    print_metrics, save_training_history, EarlyStopping, check_stage1_complete
)


def get_optimizer(model, lr, weight_decay):
    """
    Create optimizer for training
    
    Args:
        model: Model to optimize
        lr: Learning rate
        weight_decay: Weight decay for regularization
        
    Returns:
        Optimizer
    """
    trainable_params = model.get_trainable_params()
    
    if Config.OPTIMIZER.lower() == 'adamw':
        optimizer = optim.AdamW(
            trainable_params,
            lr=lr,
            weight_decay=weight_decay
        )
    elif Config.OPTIMIZER.lower() == 'sgd':
        optimizer = optim.SGD(
            trainable_params,
            lr=lr,
            momentum=Config.MOMENTUM,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {Config.OPTIMIZER}")
    
    return optimizer


def get_scheduler(optimizer, total_epochs):
    """
    Create learning rate scheduler
    
    Args:
        optimizer: Optimizer
        total_epochs: Total number of epochs
        
    Returns:
        Scheduler
    """
    if Config.SCHEDULER == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_epochs
        )
    elif Config.SCHEDULER == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=Config.STEP_SIZE,
            gamma=Config.GAMMA
        )
    elif Config.SCHEDULER == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=Config.GAMMA,
            patience=Config.STEP_SIZE
        )
    else:
        scheduler = None
    
    return scheduler


def train_stage(
    model, 
    train_loader, 
    val_loader, 
    test_loader,
    stage, 
    stage_config,
    start_epoch=1
):
    """
    Train model for one stage
    
    Args:
        model: ViXNet model
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        stage: Stage number (1 or 2)
        stage_config: Configuration for this stage
        start_epoch: Starting epoch number
        
    Returns:
        Tuple of (model, training_history)
    """
    print("\n" + "="*70)
    print(f"🚀 {stage_config['name'].upper()}")
    print("="*70)
    print(f"   Epochs: {stage_config['epochs']}")
    print(f"   Learning rate: {stage_config['lr']}")
    print(f"   Batch size: {stage_config['batch_size']}")
    print(f"   Weight decay: {stage_config['weight_decay']}")
    print("="*70)
    
    # Setup training components
    criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)
    optimizer = get_optimizer(model, stage_config['lr'], stage_config['weight_decay'])
    scheduler = get_scheduler(optimizer, stage_config['epochs'])
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if Config.MIXED_PRECISION else None
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=Config.PATIENCE,
        min_delta=Config.MIN_DELTA,
        mode='max'  # Maximize accuracy
    )
    
    # Training history
    history = []
    best_val_acc = 0.0
    
    # Training loop
    for epoch in range(start_epoch, start_epoch + stage_config['epochs']):
        print(f"\n{'='*70}")
        print(f"📅 STAGE {stage} - EPOCH {epoch}/{start_epoch + stage_config['epochs'] - 1}")
        print(f"{'='*70}")
        
        # Train one epoch
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            epoch, f"Stage {stage}"
        )
        
        # Validate
        val_metrics = validate(
            model, val_loader, criterion, epoch, f"Stage {stage}"
        )
        
        # Test on test set if enabled
        test_metrics = None
        if Config.TEST_AFTER_EPOCH and test_loader is not None:
            print(f"\n🧪 Testing on test set...")
            test_metrics = validate(
                model, test_loader, criterion, epoch, f"Stage {stage} - Test"
            )
        
        # Print metrics
        print_metrics(train_metrics, val_metrics, test_metrics, epoch)
        
        # Update learning rate
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['loss'])
            else:
                scheduler.step()
            
            current_lr = optimizer.param_groups[0]['lr']
            print(f"\n📈 Learning rate: {current_lr:.2e}")
        
        # Save history
        history_entry = {
            'epoch': epoch,
            'stage': stage,
            'train': train_metrics,
            'val': val_metrics,
            'lr': optimizer.param_groups[0]['lr']
        }
        if test_metrics:
            history_entry['test'] = test_metrics
        history.append(history_entry)
        
        # Check if best model
        is_best = val_metrics['accuracy'] > best_val_acc
        if is_best:
            best_val_acc = val_metrics['accuracy']
            print(f"\n🎉 New best model! Validation accuracy: {best_val_acc:.4f}")
        
        # Save checkpoint
        if Config.SAVE_EVERY_EPOCH or is_best:
            save_checkpoint(
                model, optimizer, epoch, val_metrics, stage, is_best
            )
        
        # Early stopping check
        if early_stopping(val_metrics['accuracy']):
            print(f"\n⚠️  Early stopping triggered after {epoch} epochs!")
            print(f"   Best validation accuracy: {best_val_acc:.4f}")
            break
    
    print(f"\n✅ {stage_config['name']} completed!")
    print(f"🏆 Best validation accuracy: {best_val_acc:.4f}")
    
    return model, history


def train_vixnet():
    """
    Main training function implementing 2-stage training strategy
    
    Stage 1: Freeze feature extractors, train fusion + classifier
    Stage 2: Unfreeze high-level layers, fine-tune with low LR
    """
    
    print("\n" + "="*70)
    print("🚀 VIXNET TRAINING")
    print("="*70)
    
    # Print configuration
    Config.print_config()
    
    # Check dataset availability
    dataset_available = check_dataset_availability()
    
    if not dataset_available:
        print("\n⚠️  Dataset not available!")
        print("   Please ensure the dataset is available at the specified paths.")
        print("   Expected structure:")
        print(f"   {Config.DATA_DIR}/")
        print(f"   ├── Train/")
        print(f"   │   ├── Fake/")
        print(f"   │   └── Real/")
        print(f"   ├── Validation/")
        print(f"   │   ├── Fake/")
        print(f"   │   └── Real/")
        print(f"   └── Test/")
        print(f"       ├── Fake/")
        print(f"       └── Real/")
        return
    
    # Create data loaders for Stage 1
    print("\n" + "="*70)
    print("📦 PREPARING DATA LOADERS")
    print("="*70)
    
    stage1_config = Config.get_stage_config(1)
    data_loaders = create_data_loaders(
        batch_size=stage1_config['batch_size']
    )
    
    if data_loaders is None:
        print("❌ Failed to create data loaders!")
        return
    
    train_loader = data_loaders['train']
    val_loader = data_loaders['val']
    test_loader = data_loaders['test']
    
    # Create model
    print("\n" + "="*70)
    print("🏗️  INITIALIZING MODEL")
    print("="*70)
    
    model = create_vixnet(pretrained=True, num_classes=Config.NUM_CLASSES)
    model = model.to(Config.DEVICE)
    
    # ==================== STAGE 1: TRAIN FUSION + CLASSIFIER ====================
    
    print("\n" + "="*70)
    print("🎯 STAGE 1: TRAINING FUSION + CLASSIFIER")
    print("="*70)
    
    # Check if Stage 1 is already complete
    stage1_complete = check_stage1_complete()
    
    if stage1_complete:
        print("\n✅ Stage 1 already complete! Found all 5 epoch checkpoints.")
        print("   Skipping Stage 1 training and loading best Stage 1 model...")
        
        # Load best model from Stage 1
        best_stage1_path = os.path.join(Config.SAVE_DIR, 'best_model_stage1.pth')
        load_checkpoint(model, best_stage1_path)
        
        # Load existing Stage 1 history if available
        stage1_history_path = os.path.join(Config.SAVE_DIR, 'stage1_history.json')
        if os.path.exists(stage1_history_path):
            try:
                with open(stage1_history_path, 'r') as f:
                    stage1_history = json.load(f)
                print(f"   Loaded existing Stage 1 history with {len(stage1_history)} epochs")
            except (json.JSONDecodeError, IOError) as e:
                print(f"   ⚠️  Warning: Could not load Stage 1 history: {e}")
                print("   Continuing without history...")
                stage1_history = []
        else:
            stage1_history = []
            print("   Note: Stage 1 history file not found, continuing without history")
    else:
        print("\n🔄 Stage 1 checkpoints not found or incomplete. Starting Stage 1 training...")
        
        # Freeze feature extractors
        model.freeze_feature_extractors()
        
        # Train Stage 1
        model, stage1_history = train_stage(
            model, train_loader, val_loader, test_loader,
            stage=1,
            stage_config=stage1_config,
            start_epoch=1
        )
        
        # Save Stage 1 history
        save_training_history(stage1_history, 'stage1_history.json')
    
    # ==================== STAGE 2: FINE-TUNE HIGH-LEVEL LAYERS ====================
    
    print("\n" + "="*70)
    print("🎯 STAGE 2: FINE-TUNING HIGH-LEVEL LAYERS")
    print("="*70)
    
    # Load best model from Stage 1
    best_stage1_path = os.path.join(Config.SAVE_DIR, 'best_model_stage1.pth')
    if os.path.exists(best_stage1_path):
        print("\n📂 Loading best model from Stage 1...")
        load_checkpoint(model, best_stage1_path)
    else:
        print("\n⚠️  Best Stage 1 model not found, continuing with current model...")
    
    # Unfreeze high-level layers
    model.unfreeze_high_level_layers()
    
    # Create data loaders for Stage 2 (may have different batch size)
    stage2_config = Config.get_stage_config(2)
    if stage2_config['batch_size'] != stage1_config['batch_size']:
        print("\n📦 Creating new data loaders for Stage 2...")
        data_loaders = create_data_loaders(
            batch_size=stage2_config['batch_size']
        )
        train_loader = data_loaders['train']
        val_loader = data_loaders['val']
        test_loader = data_loaders['test']
    
    # Train Stage 2
    start_epoch = stage1_config['epochs'] + 1
    model, stage2_history = train_stage(
        model, train_loader, val_loader, test_loader,
        stage=2,
        stage_config=stage2_config,
        start_epoch=start_epoch
    )
    
    # Save Stage 2 history
    save_training_history(stage2_history, 'stage2_history.json')
    
    # Combine histories
    full_history = stage1_history + stage2_history
    save_training_history(full_history, 'full_training_history.json')
    
    # ==================== FINAL EVALUATION ====================
    
    print("\n" + "="*70)
    print("🏁 FINAL EVALUATION")
    print("="*70)
    
    # Load best overall model
    best_model_path = os.path.join(Config.SAVE_DIR, 'best_model.pth')
    if os.path.exists(best_model_path):
        print("\n📂 Loading best overall model...")
        checkpoint = load_checkpoint(model, best_model_path)
        
        # Final test evaluation
        print("\n🧪 Final evaluation on test set...")
        criterion = nn.CrossEntropyLoss()
        test_metrics = validate(model, test_loader, criterion, stage_name="Final Test")
        
        print("\n" + "="*70)
        print("🏆 FINAL TEST RESULTS")
        print("="*70)
        print(f"   Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"   Precision: {test_metrics['precision']:.4f}")
        print(f"   Recall: {test_metrics['recall']:.4f}")
        print(f"   F1-Score: {test_metrics['f1']:.4f}")
        print(f"\n   Confusion Matrix:")
        print(f"   {test_metrics['confusion_matrix']}")
        print("="*70)
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETED!")
    print("="*70)
    print(f"💾 Models saved in: {Config.SAVE_DIR}")
    print(f"📊 Training history saved")
    print("="*70)


if __name__ == "__main__":
    try:
        train_vixnet()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user!")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
