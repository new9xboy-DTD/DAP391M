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

from model import create_vit_only, create_vixnet, create_vixnet_cross_attention, create_xception_only
from config import Config
from dataset import create_data_loaders, check_dataset_availability
from utils import (
    train_one_epoch, validate, save_checkpoint, load_checkpoint,
    print_metrics, save_training_history, EarlyStopping, check_stage1_complete,
    FocalLoss
)


def get_criterion():
    """
    Get the loss function based on configuration.
    
    Returns:
        Loss function (FocalLoss or CrossEntropyLoss)
    """
    class_weights = torch.tensor(Config.WEIGHT_DATASET).to(Config.DEVICE)
    
    if Config.USE_FOCAL_LOSS:
        print(f"\n📊 Using Focal Loss (gamma={Config.FOCAL_GAMMA}, label_smoothing={Config.LABEL_SMOOTHING})")
        criterion = FocalLoss(
            alpha=class_weights.tolist(),
            gamma=Config.FOCAL_GAMMA,
            label_smoothing=Config.LABEL_SMOOTHING,
            reduction='mean'
        )
    else:
        print(f"\n📊 Using CrossEntropyLoss (label_smoothing={Config.LABEL_SMOOTHING})")
        criterion = nn.CrossEntropyLoss(
            label_smoothing=Config.LABEL_SMOOTHING,
            weight=class_weights
        )
    
    return criterion


def get_optimizer(model, lr, weight_decay):
    """
    Create optimizer for training with proper weight decay handling.
    
    IMPORTANT: Weight decay is NOT applied to:
    - Bias parameters
    - LayerNorm/BatchNorm parameters (weight and bias)
    
    This is crucial for training stability, especially for ViT.
    
    Args:
        model: Model to optimize
        lr: Learning rate
        weight_decay: Weight decay for regularization
        
    Returns:
        Optimizer
    """
    # Separate parameters into decay and no_decay groups
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Skip weight decay for bias and normalization layers
        if 'bias' in name:
            no_decay_params.append(param)
        elif 'norm' in name.lower():  # LayerNorm, BatchNorm, etc.
            no_decay_params.append(param)
        elif 'bn' in name.lower():  # BatchNorm alternative naming
            no_decay_params.append(param)
        elif 'ln' in name.lower():  # LayerNorm alternative naming
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    # Create parameter groups
    param_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
    
    # Log parameter counts
    n_decay = sum(p.numel() for p in decay_params)
    n_no_decay = sum(p.numel() for p in no_decay_params)
    print(f"\n⚙️  Optimizer parameter groups:")
    print(f"   With weight decay ({weight_decay}): {n_decay:,} params")
    print(f"   Without weight decay: {n_no_decay:,} params")
    
    if Config.OPTIMIZER.lower() == 'adamw':
        optimizer = optim.AdamW(
            param_groups,
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8
        )
    elif Config.OPTIMIZER.lower() == 'sgd':
        optimizer = optim.SGD(
            param_groups,
            lr=lr,
            momentum=Config.MOMENTUM
        )
    else:
        raise ValueError(f"Unknown optimizer: {Config.OPTIMIZER}")
    
    return optimizer


def get_scheduler(optimizer, total_epochs, lr=1e-3):
    """
    Create learning rate scheduler
    
    Args:
        optimizer: Optimizer
        total_epochs: Total number of epochs
        lr: learing rate of stage
        
    Returns:
        Scheduler
    """
    if Config.SCHEDULER == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_epochs,
            eta_min=lr * 0.001
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
    criterion = get_criterion()
    optimizer = get_optimizer(model, stage_config['lr'], stage_config['weight_decay'])
    scheduler = get_scheduler(optimizer, stage_config['epochs'], stage_config['lr'])
    
    # Mixed precision scaler
    scaler = torch.amp.GradScaler() if Config.MIXED_PRECISION else None
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=Config.PATIENCE,
        min_delta=Config.MIN_DELTA,
        mode='max'  # Maximize AUC
    )
    
    # Training history
    history = []
    best_val_auc = 0.0
    
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
        
        # Check if best model (based on AUC)
        is_best = val_metrics['auc'] > best_val_auc
        if is_best:
            best_val_auc = val_metrics['auc']
            print(f"\n🎉 New best model! Validation AUC: {best_val_auc:.4f}")
        
        # Save checkpoint
        if Config.SAVE_EVERY_EPOCH or is_best:
            save_checkpoint(
                model, optimizer, epoch, val_metrics, stage, is_best
            )
        
        # Early stopping check (based on AUC)
        if early_stopping(val_metrics['auc']):
            print(f"\n⚠️  Early stopping triggered after {epoch} epochs!")
            print(f"   Best validation AUC: {best_val_auc:.4f}")
            break
    
    print(f"\n✅ {stage_config['name']} completed!")
    print(f"🏆 Best validation AUC: {best_val_auc:.4f}")
    
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
    
    # Set model-specific checkpoint directory
    Config.set_model_checkpoint_dir('vixnet')
    
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
            except json.JSONDecodeError as e:
                print(f"   ⚠️  Warning: Stage 1 history file is corrupted (invalid JSON)")
                print(f"   Error details: {e}")
                print("   Continuing without history...")
                stage1_history = []
            except IOError as e:
                print(f"   ⚠️  Warning: Cannot read Stage 1 history file (permission/access issue)")
                print(f"   Error details: {e}")
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
        print(f"   AUC: {test_metrics['auc']:.4f}")
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
    
def train_model(model_name="vixnet"):
    """
    Main training function implementing 2-stage training strategy
    Args:
        model_name: Name of the model to train ("vixnet", "xception", "vit", or "vixnet_cross_attention")
    """

    print("\n" + "="*70)
    print(f"TRAINING: {model_name.upper()}")
    print("="*70)
    
    # Set model-specific checkpoint directory
    Config.set_model_checkpoint_dir(model_name)
    
    # Print config
    Config.print_config()
    
    #check dataset availability
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
    
    #Create data loaders for stage 1
    print("\n" + "="*70)
    print("📦 PREPARING DATA LOADERS")
    print("="*70)
    
    stage1_config = Config.get_stage_config(1, "Stage 1: Classifier Training")
    data_loaders = create_data_loaders(
        batch_size=stage1_config['batch_size']
    )
    
    if data_loaders is None:
        print("❌ Failed to create data loaders!")
        return
    
    train_loader = data_loaders['train']
    val_loader = data_loaders['val']
    test_loader = data_loaders['test']
    
    #Create model
    print("\n" + "="*70)
    print("🏗️  INITIALIZING MODEL")
    print("="*70)
    
    if model_name.lower() == "vixnet":
        model = create_vixnet(pretrained=True, num_classes=Config.NUM_CLASSES)
    elif model_name.lower() == "xception":
        model = create_xception_only(pretrained=True, num_classes=Config.NUM_CLASSES)
    elif model_name.lower() == "vit":
        model = create_vit_only(pretrained=True, num_classes=Config.NUM_CLASSES)
    elif model_name.lower() == "vixnet_cross_attention":
        model = create_vixnet_cross_attention(pretrained=True, num_classes=Config.NUM_CLASSES)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    model = model.to(Config.DEVICE)
    
    # ==================== STAGE 1: TRAIN FUSION + CLASSIFIER ====================
    
    print("\n" + "="*70)
    print("🎯 STAGE 1: TRAINING CLASSIFIER")
    print("="*70)
    
    model.freeze_feature_extractors()
    
    _, stage1_history = train_stage(
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
        print(f"   AUC: {test_metrics['auc']:.4f}")
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


def get_optimizer_with_param_groups(model, param_groups, weight_decay):
    """
    Create optimizer with different parameter groups (different LRs).
    
    IMPORTANT: Weight decay is NOT applied to:
    - Bias parameters
    - LayerNorm/BatchNorm parameters
    
    Args:
        model: Model to optimize
        param_groups: List of parameter groups with 'params' and 'lr'
        weight_decay: Weight decay for regularization
        
    Returns:
        Optimizer
    """
    # Process each param group to separate decay/no_decay
    processed_groups = []
    
    for pg in param_groups:
        group_name = pg.get('name', 'unnamed')
        group_lr = pg.get('lr', 1e-4)
        
        decay_params = []
        no_decay_params = []
        
        for param in pg['params']:
            # We need to find the parameter name - this is a workaround
            # since we only have the param tensor, not the name
            decay_params.append(param)  # Default to decay
        
        # For simplicity, we'll use a different approach:
        # Check if param is 1D (likely bias/norm) vs 2D+ (weights)
        decay_params = []
        no_decay_params = []
        
        for param in pg['params']:
            if param.dim() == 1:  # Bias and norm parameters are typically 1D
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        if decay_params:
            processed_groups.append({
                'params': decay_params,
                'lr': group_lr,
                'weight_decay': weight_decay,
                'name': f'{group_name}_decay'
            })
        
        if no_decay_params:
            processed_groups.append({
                'params': no_decay_params,
                'lr': group_lr,
                'weight_decay': 0.0,
                'name': f'{group_name}_no_decay'
            })
    
    if Config.OPTIMIZER.lower() == 'adamw':
        optimizer = optim.AdamW(processed_groups, betas=(0.9, 0.999), eps=1e-8)
    elif Config.OPTIMIZER.lower() == 'sgd':
        optimizer = optim.SGD(processed_groups, momentum=Config.MOMENTUM)
    else:
        raise ValueError(f"Unknown optimizer: {Config.OPTIMIZER}")
    
    return optimizer


def train_stage_3stage(
    model, 
    train_loader, 
    val_loader, 
    test_loader,
    stage, 
    stage_config,
    start_epoch=1,
    use_param_groups=False
):
    """
    Train model for one stage (supports 3-stage training with param groups)
    
    Args:
        model: Model
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        stage: Stage number (1, 2, or 3)
        stage_config: Configuration for this stage
        start_epoch: Starting epoch number
        use_param_groups: Whether to use different LRs for different components
        
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
    criterion = get_criterion()
    
    # Create optimizer with or without param groups
    if use_param_groups and hasattr(model, 'get_param_groups'):
        print("\n📊 Using different learning rates for components...")
        param_groups = model.get_param_groups(
            lr_head=Config.LR_HEAD,
            lr_cnn=Config.LR_CNN,
            lr_vit=Config.LR_VIT
        )
        optimizer = get_optimizer_with_param_groups(model, param_groups, stage_config['weight_decay'])
    else:
        optimizer = get_optimizer(model, stage_config['lr'], stage_config['weight_decay'])
    
    scheduler = get_scheduler(optimizer, stage_config['epochs'], stage_config['lr'])
    scaler = torch.amp.GradScaler() if Config.MIXED_PRECISION else None
    
    early_stopping = EarlyStopping(
        patience=Config.PATIENCE,
        min_delta=Config.MIN_DELTA,
        mode='max'  # Maximize AUC
    )
    
    history = []
    best_val_auc = 0.0
    
    for epoch in range(start_epoch, start_epoch + stage_config['epochs']):
        print(f"\n{'='*70}")
        print(f"📅 STAGE {stage} - EPOCH {epoch}/{start_epoch + stage_config['epochs'] - 1}")
        print(f"{'='*70}")
        
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            epoch, f"Stage {stage}"
        )
        
        val_metrics = validate(
            model, val_loader, criterion, epoch, f"Stage {stage}"
        )
        
        test_metrics = None
        if Config.TEST_AFTER_EPOCH and test_loader is not None:
            print(f"\n🧪 Testing on test set...")
            test_metrics = validate(
                model, test_loader, criterion, epoch, f"Stage {stage} - Test"
            )
        
        print_metrics(train_metrics, val_metrics, test_metrics, epoch)
        
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['loss'])
            else:
                scheduler.step()
            
            # Print LR for each param group
            print(f"\n📈 Learning rates:")
            for i, pg in enumerate(optimizer.param_groups):
                name = pg.get('name', f'group_{i}')
                print(f"   {name}: {pg['lr']:.2e}")
        
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
        
        is_best = val_metrics['auc'] > best_val_auc
        if is_best:
            best_val_auc = val_metrics['auc']
            print(f"\n🎉 New best model! Validation AUC: {best_val_auc:.4f}")
        
        if Config.SAVE_EVERY_EPOCH or is_best:
            save_checkpoint(
                model, optimizer, epoch, val_metrics, stage, is_best
            )
        
        if early_stopping(val_metrics['auc']):
            print(f"\n⚠️  Early stopping triggered after {epoch} epochs!")
            print(f"   Best validation AUC: {best_val_auc:.4f}")
            break
    
    print(f"\n✅ {stage_config['name']} completed!")
    print(f"🏆 Best validation AUC: {best_val_auc:.4f}")
    
    return model, history


def train_vixnet_cross_attention_3stage():
    """
    3-stage training for ViXNet Cross-Attention:
    Stage 1: Train head (fusion + classifier) only
    Stage 2: Unfreeze Xception, train with different LRs
    Stage 3: Unfreeze ViT, train with different LRs
    
    Labels: Real=0, Fake=1
    """
    
    print("\n" + "="*70)
    print("🚀 VIXNET CROSS-ATTENTION 3-STAGE TRAINING")
    print("="*70)
    print("   📋 Label mapping: Real=0, Fake=1")
    print("="*70)
    
    # Set model-specific checkpoint directory
    Config.set_model_checkpoint_dir('vixnet_cross_attention')
    
    Config.print_config()
    
    dataset_available = check_dataset_availability()
    
    if not dataset_available:
        print("\n⚠️  Dataset not available!")
        return
    
    # Create data loaders
    print("\n" + "="*70)
    print("📦 PREPARING DATA LOADERS")
    print("="*70)
    
    stage1_config = Config.get_stage_config(1, "Stage 1: Head Training")
    data_loaders = create_data_loaders(batch_size=stage1_config['batch_size'])
    
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
    
    model = create_vixnet_cross_attention(pretrained=True, num_classes=Config.NUM_CLASSES)
    model = model.to(Config.DEVICE)
    
    full_history = []
    
    # ==================== STAGE 1: TRAIN HEAD ====================
    
    print("\n" + "="*70)
    print("🎯 STAGE 1: TRAINING HEAD (Fusion + Classifier)")
    print("="*70)
    
    model.freeze_feature_extractors()
    
    _, stage1_history = train_stage_3stage(
        model, train_loader, val_loader, test_loader,
        stage=1,
        stage_config=stage1_config,
        start_epoch=1,
        use_param_groups=False
    )
    
    save_training_history(stage1_history, 'stage1_history.json')
    full_history.extend(stage1_history)
    
    # ==================== STAGE 2: UNFREEZE XCEPTION ====================
    
    print("\n" + "="*70)
    print("🎯 STAGE 2: FINE-TUNING XCEPTION")
    print("="*70)
    
    # Load best model from Stage 1
    best_stage1_path = os.path.join(Config.SAVE_DIR, 'best_model_stage1.pth')
    if os.path.exists(best_stage1_path):
        print("\n📂 Loading best model from Stage 1...")
        load_checkpoint(model, best_stage1_path)
    
    model.unfreeze_xception_layers(unfreeze_ratio=0.20)
    
    stage2_config = Config.get_stage_config(2, "Stage 2: Xception Fine-tuning")
    
    if stage2_config['batch_size'] != stage1_config['batch_size']:
        print("\n📦 Creating new data loaders for Stage 2...")
        data_loaders = create_data_loaders(batch_size=stage2_config['batch_size'])
        train_loader = data_loaders['train']
        val_loader = data_loaders['val']
        test_loader = data_loaders['test']
    
    start_epoch = stage1_config['epochs'] + 1
    _, stage2_history = train_stage_3stage(
        model, train_loader, val_loader, test_loader,
        stage=2,
        stage_config=stage2_config,
        start_epoch=start_epoch,
        use_param_groups=True
    )
    
    save_training_history(stage2_history, 'stage2_history.json')
    full_history.extend(stage2_history)
    
    # ==================== STAGE 3: UNFREEZE VIT ====================
    
    print("\n" + "="*70)
    print("🎯 STAGE 3: FINE-TUNING VIT")
    print("="*70)
    
    # Load best model from Stage 2
    best_stage2_path = os.path.join(Config.SAVE_DIR, 'best_model_stage2.pth')
    if os.path.exists(best_stage2_path):
        print("\n📂 Loading best model from Stage 2...")
        load_checkpoint(model, best_stage2_path)
    
    model.unfreeze_vit_layers(num_blocks=2)
    
    stage3_config = Config.get_stage_config(3, "Stage 3: ViT Fine-tuning")
    
    if stage3_config['batch_size'] != stage2_config['batch_size']:
        print("\n📦 Creating new data loaders for Stage 3...")
        data_loaders = create_data_loaders(batch_size=stage3_config['batch_size'])
        train_loader = data_loaders['train']
        val_loader = data_loaders['val']
        test_loader = data_loaders['test']
    
    start_epoch = stage1_config['epochs'] + stage2_config['epochs'] + 1
    _, stage3_history = train_stage_3stage(
        model, train_loader, val_loader, test_loader,
        stage=3,
        stage_config=stage3_config,
        start_epoch=start_epoch,
        use_param_groups=True
    )
    
    save_training_history(stage3_history, 'stage3_history.json')
    full_history.extend(stage3_history)
    
    # Save full history
    save_training_history(full_history, 'full_training_history.json')
    
    # ==================== FINAL EVALUATION ====================
    
    print("\n" + "="*70)
    print("🏁 FINAL EVALUATION")
    print("="*70)
    
    best_model_path = os.path.join(Config.SAVE_DIR, 'best_model.pth')
    if os.path.exists(best_model_path):
        print("\n📂 Loading best overall model...")
        load_checkpoint(model, best_model_path)
        
        print("\n🧪 Final evaluation on test set...")
        criterion = nn.CrossEntropyLoss()
        test_metrics = validate(model, test_loader, criterion, stage_name="Final Test")
        
        print("\n" + "="*70)
        print("🏆 FINAL TEST RESULTS")
        print("="*70)
        print(f"   Labels: Real=0, Fake=1")
        print(f"   AUC: {test_metrics['auc']:.4f}")
        print(f"   Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"   Precision: {test_metrics['precision']:.4f}")
        print(f"   Recall: {test_metrics['recall']:.4f}")
        print(f"   F1-Score: {test_metrics['f1']:.4f}")
        print(f"\n   Confusion Matrix:")
        print(f"   {test_metrics['confusion_matrix']}")
        print("="*70)
    
    print("\n" + "="*70)
    print("✅ 3-STAGE TRAINING COMPLETED!")
    print("="*70)
    print(f"💾 Models saved in: {Config.SAVE_DIR}")
    print(f"📊 Training history saved")
    print("="*70)


def resume_training_vixnet_cross_attention(checkpoint_path=None, resume_stage=None, resume_epoch=None):
    """
    Resume training for ViXNet Cross-Attention from a checkpoint.
    
    IMPORTANT: This function properly handles:
    1. Loading model weights from checkpoint
    2. Setting correct freeze/unfreeze state based on stage
    3. NOT loading optimizer state (fresh optimizer for stability)
    4. Proper stage transitions
    
    Args:
        checkpoint_path: Path to checkpoint file. If None, will auto-detect latest checkpoint.
        resume_stage: Stage to resume from (1, 2, or 3). If None, will auto-detect from checkpoint.
        resume_epoch: Epoch to start from. If None, will auto-detect from checkpoint.
    
    Labels: Real=0, Fake=1
    """
    
    print("\n" + "="*70)
    print("🔄 RESUME TRAINING - VIXNET CROSS-ATTENTION")
    print("="*70)
    print("   📋 Label mapping: Real=0, Fake=1")
    print("="*70)
    
    # Set model-specific checkpoint directory
    Config.set_model_checkpoint_dir('vixnet_cross_attention')
    
    Config.print_config()
    
    # Auto-detect checkpoint if not provided
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint()
        if checkpoint_path is None:
            print("\n❌ No checkpoint found! Please provide a checkpoint path or train from scratch.")
            return
    
    if not os.path.exists(checkpoint_path):
        print(f"\n❌ Checkpoint not found: {checkpoint_path}")
        return
    
    # Load checkpoint to get info
    print(f"\n📂 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE, weights_only=False)
    
    checkpoint_epoch = checkpoint.get('epoch', 0)
    checkpoint_stage = checkpoint.get('stage', 1)
    checkpoint_metrics = checkpoint.get('metrics', {})
    
    print(f"\n📊 Checkpoint Info:")
    print(f"   Stage: {checkpoint_stage}")
    print(f"   Epoch: {checkpoint_epoch}")
    print(f"   Validation Accuracy: {checkpoint_metrics.get('accuracy', 'N/A')}")
    
    # Determine resume point
    if resume_stage is None:
        resume_stage = checkpoint_stage
    if resume_epoch is None:
        resume_epoch = checkpoint_epoch + 1
    
    print(f"\n🎯 Resume Point:")
    print(f"   Starting from Stage: {resume_stage}")
    print(f"   Starting from Epoch: {resume_epoch}")
    
    # Check dataset
    dataset_available = check_dataset_availability()
    if not dataset_available:
        print("\n⚠️  Dataset not available!")
        return
    
    # Create model - use pretrained=False since we're loading weights
    print("\n" + "="*70)
    print("🏗️  INITIALIZING MODEL")
    print("="*70)
    
    model = create_vixnet_cross_attention(pretrained=False, num_classes=Config.NUM_CLASSES)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(Config.DEVICE)
    
    print("✅ Model weights loaded successfully from checkpoint!")
    
    # IMPORTANT: Set correct freeze/unfreeze state based on resume stage
    # This ensures the model has the same trainable parameters as when it was saved
    print(f"\n🔧 Setting model state for Stage {resume_stage}...")
    
    if resume_stage == 1:
        # Stage 1: Only head is trainable
        model.freeze_feature_extractors()
        print("   ✅ Feature extractors frozen (Stage 1 mode)")
    elif resume_stage == 2:
        # Stage 2: Head + Xception high-level layers trainable
        model.freeze_feature_extractors()  # First freeze all
        model.unfreeze_xception_layers(unfreeze_ratio=0.20)  # Then unfreeze Xception
        print("   ✅ Xception layers unfrozen (Stage 2 mode)")
    elif resume_stage == 3:
        # Stage 3: Head + Xception + ViT high-level layers trainable
        model.freeze_feature_extractors()  # First freeze all
        model.unfreeze_xception_layers(unfreeze_ratio=0.20)  # Unfreeze Xception
        model.unfreeze_vit_layers(num_blocks=2)  # Unfreeze ViT
        print("   ✅ Xception and ViT layers unfrozen (Stage 3 mode)")
    
    # Print trainable parameters count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   📊 Trainable params: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # Get stage configs
    stage1_config = Config.get_stage_config(1, "Stage 1: Head Training")
    stage2_config = Config.get_stage_config(2, "Stage 2: Xception Fine-tuning")
    stage3_config = Config.get_stage_config(3, "Stage 3: ViT Fine-tuning")
    
    # Calculate epoch ranges for each stage
    stage1_end = stage1_config['epochs']
    stage2_end = stage1_end + stage2_config['epochs']
    stage3_end = stage2_end + stage3_config['epochs']
    
    full_history = []
    
    # Load existing history if available
    full_history_path = os.path.join(Config.SAVE_DIR, 'full_training_history.json')
    if os.path.exists(full_history_path):
        try:
            with open(full_history_path, 'r') as f:
                full_history = json.load(f)
            print(f"\n📜 Loaded existing training history with {len(full_history)} entries")
        except Exception as e:
            print(f"\n⚠️  Could not load training history: {e}")
            full_history = []
    
    # ==================== RESUME FROM APPROPRIATE STAGE ====================
    # NOTE: Model freeze/unfreeze state has already been set above based on resume_stage
    # We DON'T call freeze/unfreeze again in each stage block to avoid messing up the state
    
    if resume_stage == 1:
        # Resume Stage 1
        if resume_epoch <= stage1_end:
            print("\n" + "="*70)
            print("🎯 RESUMING STAGE 1: TRAINING HEAD (Fusion + Classifier)")
            print("="*70)
            
            # Create data loaders
            data_loaders = create_data_loaders(batch_size=stage1_config['batch_size'])
            if data_loaders is None:
                print("❌ Failed to create data loaders!")
                return
            train_loader, val_loader, test_loader = data_loaders['train'], data_loaders['val'], data_loaders['test']
            
            # NOTE: freeze_feature_extractors() already called above
            
            # Calculate remaining epochs
            remaining_epochs = stage1_end - resume_epoch + 1
            stage1_resume_config = stage1_config.copy()
            stage1_resume_config['epochs'] = remaining_epochs
            
            _, stage1_history = train_stage_3stage(
                model, train_loader, val_loader, test_loader,
                stage=1,
                stage_config=stage1_resume_config,
                start_epoch=resume_epoch,
                use_param_groups=False
            )
            
            save_training_history(stage1_history, 'stage1_history_resumed.json')
            full_history.extend(stage1_history)
            
            # Prepare for Stage 2: load best model and change freeze state
            best_stage1_path = os.path.join(Config.SAVE_DIR, 'best_model_stage1.pth')
            if os.path.exists(best_stage1_path):
                print("\n📂 Loading best model from Stage 1 for Stage 2...")
                load_checkpoint(model, best_stage1_path)
                model.freeze_feature_extractors()
                model.unfreeze_xception_layers(unfreeze_ratio=0.20)
            
            # Continue to Stage 2
            resume_stage = 2
            resume_epoch = stage1_end + 1
    
    if resume_stage == 2:
        # Resume Stage 2
        if resume_epoch <= stage2_end:
            print("\n" + "="*70)
            print("🎯 RESUMING STAGE 2: FINE-TUNING XCEPTION")
            print("="*70)
            
            # NOTE: If we jumped here from stage 1 completion above, model is already prepared
            # If we started directly at stage 2 (from user input), freeze state was set at the top
            
            # Create data loaders
            data_loaders = create_data_loaders(batch_size=stage2_config['batch_size'])
            if data_loaders is None:
                print("❌ Failed to create data loaders!")
                return
            train_loader, val_loader, test_loader = data_loaders['train'], data_loaders['val'], data_loaders['test']
            
            # NOTE: unfreeze state already set at the top of function
            
            # Calculate remaining epochs
            remaining_epochs = stage2_end - resume_epoch + 1
            stage2_resume_config = stage2_config.copy()
            stage2_resume_config['epochs'] = remaining_epochs
            
            _, stage2_history = train_stage_3stage(
                model, train_loader, val_loader, test_loader,
                stage=2,
                stage_config=stage2_resume_config,
                start_epoch=resume_epoch,
                use_param_groups=True
            )
            
            save_training_history(stage2_history, 'stage2_history_resumed.json')
            full_history.extend(stage2_history)
            
            # Prepare for Stage 3: load best model and change freeze state
            best_stage2_path = os.path.join(Config.SAVE_DIR, 'best_model_stage2.pth')
            if os.path.exists(best_stage2_path):
                print("\n📂 Loading best model from Stage 2 for Stage 3...")
                load_checkpoint(model, best_stage2_path)
                model.freeze_feature_extractors()
                model.unfreeze_xception_layers(unfreeze_ratio=0.20)
                model.unfreeze_vit_layers(num_blocks=2)
            
            # Continue to Stage 3
            resume_stage = 3
            resume_epoch = stage2_end + 1
    
    if resume_stage == 3:
        # Resume Stage 3
        if resume_epoch <= stage3_end:
            print("\n" + "="*70)
            print("🎯 RESUMING STAGE 3: FINE-TUNING VIT")
            print("="*70)
            
            # NOTE: If we jumped here from stage 2 completion above, model is already prepared
            # If we started directly at stage 3 (from user input), freeze state was set at the top
            
            # Create data loaders
            data_loaders = create_data_loaders(batch_size=stage3_config['batch_size'])
            if data_loaders is None:
                print("❌ Failed to create data loaders!")
                return
            train_loader, val_loader, test_loader = data_loaders['train'], data_loaders['val'], data_loaders['test']
            
            # NOTE: unfreeze state already set at the top of function
            
            # Calculate remaining epochs  
            remaining_epochs = stage3_end - resume_epoch + 1
            stage3_resume_config = stage3_config.copy()
            stage3_resume_config['epochs'] = remaining_epochs
            
            _, stage3_history = train_stage_3stage(
                model, train_loader, val_loader, test_loader,
                stage=3,
                stage_config=stage3_resume_config,
                start_epoch=resume_epoch,
                use_param_groups=True
            )
            
            save_training_history(stage3_history, 'stage3_history_resumed.json')
            full_history.extend(stage3_history)
    
    # Save full history
    save_training_history(full_history, 'full_training_history.json')
    
    # ==================== FINAL EVALUATION ====================
    
    print("\n" + "="*70)
    print("🏁 FINAL EVALUATION")
    print("="*70)
    
    # Create test loader if not exists
    if 'test_loader' not in dir():
        data_loaders = create_data_loaders(batch_size=stage3_config['batch_size'])
        test_loader = data_loaders['test']
    
    best_model_path = os.path.join(Config.SAVE_DIR, 'best_model.pth')
    if os.path.exists(best_model_path):
        print("\n📂 Loading best overall model...")
        load_checkpoint(model, best_model_path)
        
        print("\n🧪 Final evaluation on test set...")
        criterion = nn.CrossEntropyLoss()
        test_metrics = validate(model, test_loader, criterion, stage_name="Final Test")
        
        print("\n" + "="*70)
        print("🏆 FINAL TEST RESULTS")
        print("="*70)
        print(f"   Labels: Real=0, Fake=1")
        print(f"   AUC: {test_metrics['auc']:.4f}")
        print(f"   Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"   Precision: {test_metrics['precision']:.4f}")
        print(f"   Recall: {test_metrics['recall']:.4f}")
        print(f"   F1-Score: {test_metrics['f1']:.4f}")
        print(f"\n   Confusion Matrix:")
        print(f"   {test_metrics['confusion_matrix']}")
        print("="*70)
    
    print("\n" + "="*70)
    print("✅ RESUME TRAINING COMPLETED!")
    print("="*70)
    print(f"💾 Models saved in: {Config.SAVE_DIR}")
    print(f"📊 Training history saved")
    print("="*70)


def find_latest_checkpoint():
    """
    Find the latest checkpoint file in the checkpoint directory.
    
    Returns:
        Path to the latest checkpoint or None if no checkpoint found.
    """
    if not os.path.exists(Config.SAVE_DIR):
        return None
    
    checkpoints = []
    for filename in os.listdir(Config.SAVE_DIR):
        if filename.startswith('checkpoint_stage') and filename.endswith('.pth'):
            filepath = os.path.join(Config.SAVE_DIR, filename)
            # Extract stage and epoch from filename
            try:
                parts = filename.replace('.pth', '').split('_')
                stage = int(parts[1].replace('stage', ''))
                epoch = int(parts[2].replace('epoch', ''))
                checkpoints.append((filepath, stage, epoch))
            except (IndexError, ValueError):
                continue
    
    if not checkpoints:
        # Try to find best_model.pth or any .pth file
        for filename in ['best_model_stage3.pth', 'best_model_stage2.pth', 'best_model_stage1.pth', 'best_model.pth']:
            filepath = os.path.join(Config.SAVE_DIR, filename)
            if os.path.exists(filepath):
                return filepath
        return None
    
    # Sort by stage then epoch and return latest
    checkpoints.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return checkpoints[0][0]


def list_available_checkpoints():
    """
    List all available checkpoints with their info.
    
    Returns:
        List of checkpoint info dictionaries.
    """
    if not os.path.exists(Config.SAVE_DIR):
        return []
    
    checkpoints = []
    for filename in os.listdir(Config.SAVE_DIR):
        if filename.endswith('.pth'):
            filepath = os.path.join(Config.SAVE_DIR, filename)
            try:
                checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
                info = {
                    'filename': filename,
                    'path': filepath,
                    'epoch': checkpoint.get('epoch', 'N/A'),
                    'stage': checkpoint.get('stage', 'N/A'),
                    'accuracy': checkpoint.get('metrics', {}).get('accuracy', 'N/A'),
                    'timestamp': checkpoint.get('timestamp', 'N/A')
                }
                checkpoints.append(info)
            except Exception as e:
                print(f"⚠️  Could not load {filename}: {e}")
                continue
    
    return checkpoints


def interactive_resume_training():
    """
    Interactive interface for resuming training.
    """
    print("\n" + "="*70)
    print("🔄 RESUME TRAINING - INTERACTIVE MODE")
    print("="*70)
    
    # List available checkpoints
    checkpoints = list_available_checkpoints()
    
    if not checkpoints:
        print("\n❌ No checkpoints found!")
        print(f"   Checkpoint directory: {Config.SAVE_DIR}")
        return
    
    print(f"\n📁 Found {len(checkpoints)} checkpoint(s):")
    print("-" * 70)
    
    for i, ckpt in enumerate(checkpoints):
        acc_str = f"{ckpt['accuracy']:.4f}" if isinstance(ckpt['accuracy'], float) else ckpt['accuracy']
        print(f"   {i+1}. {ckpt['filename']}")
        print(f"      Stage: {ckpt['stage']}, Epoch: {ckpt['epoch']}, Accuracy: {acc_str}")
        print(f"      Timestamp: {ckpt['timestamp']}")
    
    print("-" * 70)
    print("   0. Auto-detect latest checkpoint")
    print("-" * 70)
    
    try:
        choice = input("\nSelect checkpoint (0 for auto): ").strip()
        
        if choice == '0' or choice == '':
            checkpoint_path = None
            print("\n🔍 Auto-detecting latest checkpoint...")
        else:
            idx = int(choice) - 1
            if 0 <= idx < len(checkpoints):
                checkpoint_path = checkpoints[idx]['path']
            else:
                print("\n❌ Invalid choice!")
                return
        
        # Ask for resume options
        print("\n📋 Resume Options:")
        print("   1. Continue from checkpoint (auto-detect stage/epoch)")
        print("   2. Specify stage and epoch manually")
        
        option = input("\nSelect option (1-2): ").strip()
        
        if option == '2':
            stage_input = input("Enter stage (1-3): ").strip()
            epoch_input = input("Enter epoch: ").strip()
            resume_stage = int(stage_input) if stage_input else None
            resume_epoch = int(epoch_input) if epoch_input else None
        else:
            resume_stage = None
            resume_epoch = None
        
        # Start resume training
        resume_training_vixnet_cross_attention(
            checkpoint_path=checkpoint_path,
            resume_stage=resume_stage,
            resume_epoch=resume_epoch
        )
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Cancelled by user!")
    except ValueError as e:
        print(f"\n❌ Invalid input: {e}")


def train_vixnet_3stage():
    """
    3-stage training for ViXNet (concatenation fusion):
    Stage 1: Train head (fusion + classifier) only
    Stage 2: Unfreeze Xception, train with different LRs
    Stage 3: Unfreeze ViT, train with different LRs
    
    Labels: Real=0, Fake=1
    """
    
    print("\n" + "="*70)
    print("🚀 VIXNET 3-STAGE TRAINING")
    print("="*70)
    print("   📋 Label mapping: Real=0, Fake=1")
    print("="*70)
    
    # Set model-specific checkpoint directory
    Config.set_model_checkpoint_dir('vixnet')
    
    Config.print_config()
    
    dataset_available = check_dataset_availability()
    
    if not dataset_available:
        print("\n⚠️  Dataset not available!")
        return
    
    # Create data loaders
    print("\n" + "="*70)
    print("📦 PREPARING DATA LOADERS")
    print("="*70)
    
    stage1_config = Config.get_stage_config(1, "Stage 1: Head Training")
    data_loaders = create_data_loaders(batch_size=stage1_config['batch_size'])
    
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
    
    full_history = []
    
    # ==================== STAGE 1: TRAIN HEAD ====================
    
    print("\n" + "="*70)
    print("🎯 STAGE 1: TRAINING HEAD (Fusion + Classifier)")
    print("="*70)
    
    model.freeze_feature_extractors()
    
    _, stage1_history = train_stage_3stage(
        model, train_loader, val_loader, test_loader,
        stage=1,
        stage_config=stage1_config,
        start_epoch=1,
        use_param_groups=False
    )
    
    save_training_history(stage1_history, 'stage1_history.json')
    full_history.extend(stage1_history)
    
    # ==================== STAGE 2: UNFREEZE XCEPTION ====================
    
    print("\n" + "="*70)
    print("🎯 STAGE 2: FINE-TUNING XCEPTION")
    print("="*70)
    
    # Load best model from Stage 1
    best_stage1_path = os.path.join(Config.SAVE_DIR, 'best_model_stage1.pth')
    if os.path.exists(best_stage1_path):
        print("\n📂 Loading best model from Stage 1...")
        load_checkpoint(model, best_stage1_path)
    
    model.unfreeze_xception_layers(unfreeze_ratio=0.20)
    
    stage2_config = Config.get_stage_config(2, "Stage 2: Xception Fine-tuning")
    
    if stage2_config['batch_size'] != stage1_config['batch_size']:
        print("\n📦 Creating new data loaders for Stage 2...")
        data_loaders = create_data_loaders(batch_size=stage2_config['batch_size'])
        train_loader = data_loaders['train']
        val_loader = data_loaders['val']
        test_loader = data_loaders['test']
    
    start_epoch = stage1_config['epochs'] + 1
    _, stage2_history = train_stage_3stage(
        model, train_loader, val_loader, test_loader,
        stage=2,
        stage_config=stage2_config,
        start_epoch=start_epoch,
        use_param_groups=True
    )
    
    save_training_history(stage2_history, 'stage2_history.json')
    full_history.extend(stage2_history)
    
    # ==================== STAGE 3: UNFREEZE VIT ====================
    
    print("\n" + "="*70)
    print("🎯 STAGE 3: FINE-TUNING VIT")
    print("="*70)
    
    # Load best model from Stage 2
    best_stage2_path = os.path.join(Config.SAVE_DIR, 'best_model_stage2.pth')
    if os.path.exists(best_stage2_path):
        print("\n📂 Loading best model from Stage 2...")
        load_checkpoint(model, best_stage2_path)
    
    model.unfreeze_vit_layers(num_blocks=2)
    
    stage3_config = Config.get_stage_config(3, "Stage 3: ViT Fine-tuning")
    
    if stage3_config['batch_size'] != stage2_config['batch_size']:
        print("\n📦 Creating new data loaders for Stage 3...")
        data_loaders = create_data_loaders(batch_size=stage3_config['batch_size'])
        train_loader = data_loaders['train']
        val_loader = data_loaders['val']
        test_loader = data_loaders['test']
    
    start_epoch = stage1_config['epochs'] + stage2_config['epochs'] + 1
    _, stage3_history = train_stage_3stage(
        model, train_loader, val_loader, test_loader,
        stage=3,
        stage_config=stage3_config,
        start_epoch=start_epoch,
        use_param_groups=True
    )
    
    save_training_history(stage3_history, 'stage3_history.json')
    full_history.extend(stage3_history)
    
    # Save full history
    save_training_history(full_history, 'full_training_history.json')
    
    # ==================== FINAL EVALUATION ====================
    
    print("\n" + "="*70)
    print("🏁 FINAL EVALUATION")
    print("="*70)
    
    best_model_path = os.path.join(Config.SAVE_DIR, 'best_model.pth')
    if os.path.exists(best_model_path):
        print("\n📂 Loading best overall model...")
        load_checkpoint(model, best_model_path)
        
        print("\n🧪 Final evaluation on test set...")
        criterion = nn.CrossEntropyLoss()
        test_metrics = validate(model, test_loader, criterion, stage_name="Final Test")
        
        print("\n" + "="*70)
        print("🏆 FINAL TEST RESULTS")
        print("="*70)
        print(f"   Labels: Real=0, Fake=1")
        print(f"   AUC: {test_metrics['auc']:.4f}")
        print(f"   Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"   Precision: {test_metrics['precision']:.4f}")
        print(f"   Recall: {test_metrics['recall']:.4f}")
        print(f"   F1-Score: {test_metrics['f1']:.4f}")
        print(f"\n   Confusion Matrix:")
        print(f"   {test_metrics['confusion_matrix']}")
        print("="*70)
    
    print("\n" + "="*70)
    print("✅ VIXNET 3-STAGE TRAINING COMPLETED!")
    print("="*70)
    print(f"💾 Models saved in: {Config.SAVE_DIR}")
    print(f"📊 Training history saved")
    print("="*70)


if __name__ == "__main__":
    try:
        print("Select training mode:")
        print("1. ViXNet (2-stage, default)")
        print("2. Xception Only (2-stage)")
        print("3. ViT Only (2-stage)")
        print("4. ViXNet Cross-Attention (2-stage)")
        print("5. ViXNet Cross-Attention (3-stage) ⭐")
        print("6. ViXNet (3-stage) ⭐")
        print("7. Resume Training - ViXNet Cross-Attention 🔄")
        user_input = input("Enter choice (1-7): ").strip()
        
        if user_input == '2':
            train_model(model_name="xception")
        elif user_input == '3':
            train_model(model_name="vit")
        elif user_input == '4':
            train_model(model_name="vixnet_cross_attention")
        elif user_input == '5':
            train_vixnet_cross_attention_3stage()
        elif user_input == '6':
            train_vixnet_3stage()
        elif user_input == '7':
            interactive_resume_training()
        else:
            train_model(model_name="vixnet")
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user!")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
