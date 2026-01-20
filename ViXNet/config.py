"""
Configuration file for ViXNet training
Contains all hyperparameters and settings
"""

import os
import torch


class Config:
    """
    Configuration class for ViXNet model and training
    """
    
    # ==================== DATA PATHS ====================
    DATA_DIR = os.path.join("D:/Repo/DAP391M", "Dataset")
    TRAIN_DIR = os.path.join(DATA_DIR, "Train")
    VAL_DIR = os.path.join(DATA_DIR, "Validation")
    TEST_DIR = os.path.join(DATA_DIR, "Test")
    
    # Multiple dataset configurations
    DATASETS = {
        'default': {
            'name': 'Default Dataset',
            'path': os.path.join("D:/Repo/DAP391M", "Dataset"),
            'train': os.path.join("D:/Repo/DAP391M", "Dataset", "Train"),
            'val': os.path.join("D:/Repo/DAP391M", "Dataset", "Validation"),
            'test': os.path.join("D:/Repo/DAP391M", "Dataset", "Test")
        },
        # Add more datasets here in the future
        # 'dataset2': {
        #     'name': 'Another Dataset',
        #     'path': '/path/to/dataset2',
        #     'train': '/path/to/dataset2/Train',
        #     'val': '/path/to/dataset2/Validation', 
        #     'test': '/path/to/dataset2/Test'
        # }
    }
    
    # ==================== MODEL PARAMETERS ====================
    IMG_SIZE = 224  # ViT and Xception both work well with 224x224
    NUM_CLASSES = 2  # Binary classification: Real/Fake
    
    # Model architecture dimensions
    XCEPTION_DIM = 2048  # Xception output dimension
    VIT_DIM = 768  # ViT output dimension
    FUSION_DIM = 512  # Fusion layer dimension
    VIT_MODEL_NAME = 'vit_base_patch16_224'  # ViT model variant
    
    # ==================== TRAINING PARAMETERS ====================
    
    # Stage 1: Feature extractor frozen
    STAGE1_EPOCHS = 5
    STAGE1_BATCH_SIZE = 32
    STAGE1_LR = 1e-3  # Higher learning rate for new layers
    STAGE1_WEIGHT_DECAY = 0.01
    
    # Stage 2: Fine-tuning high-level layers
    STAGE2_EPOCHS = 20
    STAGE2_BATCH_SIZE = 32
    STAGE2_LR = 1e-5  # Very low learning rate for fine-tuning
    STAGE2_WEIGHT_DECAY = 0.01
    
    # ==================== OPTIMIZATION ====================
    OPTIMIZER = 'adamw'  # AdamW optimizer
    MOMENTUM = 0.9  # For SGD (if used)
    SCHEDULER = 'cosine'  # Learning rate scheduler: 'cosine', 'step', 'plateau'
    
    # Scheduler parameters
    STEP_SIZE = 5  # For StepLR
    GAMMA = 0.5  # LR decay factor
    T_MAX = 30  # For CosineAnnealingLR (total epochs)
    
    # ==================== REGULARIZATION ====================
    DROPOUT = 0.5  # Dropout rate in classifier
    LABEL_SMOOTHING = 0.1  # Label smoothing for CrossEntropyLoss
    
    # ==================== DATA LOADING ====================
    NUM_WORKERS = 8  # Number of workers for data loading
    PIN_MEMORY = True  # Pin memory for faster GPU transfer
    
    # ==================== TRAINING SETTINGS ====================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MIXED_PRECISION = True  # Use mixed precision training (FP16)
    GRADIENT_CLIP = 1.0  # Gradient clipping value
    
    # ==================== CHECKPOINT & LOGGING ====================
    SAVE_DIR = "checkpoints"  # Directory to save models
    LOG_INTERVAL = 50  # Log every N batches
    SAVE_EVERY_EPOCH = True  # Save checkpoint after every epoch
    
    # ==================== EARLY STOPPING ====================
    PATIENCE = 5  # Early stopping patience (epochs)
    MIN_DELTA = 0.001  # Minimum improvement to reset patience
    
    # ==================== TESTING ====================
    TEST_AFTER_EPOCH = False  # Test on test set after each epoch
    
    # ==================== AUGMENTATION ====================
    # Data augmentation probabilities
    HORIZONTAL_FLIP_PROB = 0.5
    ROTATION_DEGREES = 15
    COLOR_JITTER_BRIGHTNESS = 0.2
    COLOR_JITTER_CONTRAST = 0.2
    COLOR_JITTER_SATURATION = 0.2
    COLOR_JITTER_HUE = 0.1
    
    # Advanced augmentations
    USE_RANDOM_ERASING = True  # Random erasing augmentation
    RANDOM_ERASING_PROB = 0.3
    
    @classmethod
    def print_config(cls):
        """Print all configuration parameters"""
        print("="*70)
        print("⚙️  VIXNET CONFIGURATION")
        print("="*70)
        
        print("\n📁 DATA:")
        print(f"   Image size: {cls.IMG_SIZE}x{cls.IMG_SIZE}")
        print(f"   Number of classes: {cls.NUM_CLASSES}")
        print(f"   Train dir: {cls.TRAIN_DIR}")
        print(f"   Val dir: {cls.VAL_DIR}")
        print(f"   Test dir: {cls.TEST_DIR}")
        
        print("\n🏗️  MODEL:")
        print(f"   Xception dim: {cls.XCEPTION_DIM}")
        print(f"   ViT dim: {cls.VIT_DIM}")
        print(f"   Fusion dim: {cls.FUSION_DIM}")
        print(f"   ViT model: {cls.VIT_MODEL_NAME}")
        
        print("\n🎯 TRAINING - STAGE 1:")
        print(f"   Epochs: {cls.STAGE1_EPOCHS}")
        print(f"   Batch size: {cls.STAGE1_BATCH_SIZE}")
        print(f"   Learning rate: {cls.STAGE1_LR}")
        print(f"   Weight decay: {cls.STAGE1_WEIGHT_DECAY}")
        
        print("\n🎯 TRAINING - STAGE 2:")
        print(f"   Epochs: {cls.STAGE2_EPOCHS}")
        print(f"   Batch size: {cls.STAGE2_BATCH_SIZE}")
        print(f"   Learning rate: {cls.STAGE2_LR}")
        print(f"   Weight decay: {cls.STAGE2_WEIGHT_DECAY}")
        
        print("\n⚡ OPTIMIZATION:")
        print(f"   Optimizer: {cls.OPTIMIZER}")
        print(f"   Scheduler: {cls.SCHEDULER}")
        print(f"   Gradient clip: {cls.GRADIENT_CLIP}")
        print(f"   Mixed precision: {cls.MIXED_PRECISION}")
        
        print("\n💾 SYSTEM:")
        print(f"   Device: {cls.DEVICE}")
        print(f"   Num workers: {cls.NUM_WORKERS}")
        print(f"   Save dir: {cls.SAVE_DIR}")
        
        print("="*70)
    
    @classmethod
    def get_stage_config(cls, stage=1):
        """
        Get configuration for specific training stage
        
        Args:
            stage: 1 or 2
            
        Returns:
            Dictionary with stage-specific config
        """
        if stage == 1:
            return {
                'epochs': cls.STAGE1_EPOCHS,
                'batch_size': cls.STAGE1_BATCH_SIZE,
                'lr': cls.STAGE1_LR,
                'weight_decay': cls.STAGE1_WEIGHT_DECAY,
                'name': 'Stage 1: Fusion Training'
            }
        elif stage == 2:
            return {
                'epochs': cls.STAGE2_EPOCHS,
                'batch_size': cls.STAGE2_BATCH_SIZE,
                'lr': cls.STAGE2_LR,
                'weight_decay': cls.STAGE2_WEIGHT_DECAY,
                'name': 'Stage 2: Fine-tuning'
            }
        else:
            raise ValueError(f"Invalid stage: {stage}. Must be 1 or 2.")
    
    @classmethod
    def get_dataset_config(cls, dataset_key='default'):
        """
        Get dataset configuration by key
        
        Args:
            dataset_key: Key for dataset in DATASETS dict
            
        Returns:
            Dictionary with dataset paths
        """
        if dataset_key not in cls.DATASETS:
            raise ValueError(f"Dataset '{dataset_key}' not found. Available: {list(cls.DATASETS.keys())}")
        return cls.DATASETS[dataset_key]
    
    @classmethod
    def list_available_datasets(cls):
        """
        List all available datasets
        
        Returns:
            List of dictionaries with dataset info
        """
        return [
            {
                'key': key,
                'name': config['name'],
                'path': config['path']
            }
            for key, config in cls.DATASETS.items()
        ]


if __name__ == "__main__":
    # Print configuration
    Config.print_config()
    
    # Test stage configs
    print("\n" + "="*70)
    print("Testing stage configurations:")
    print("="*70)
    
    stage1_config = Config.get_stage_config(1)
    print(f"\n{stage1_config['name']}:")
    for key, value in stage1_config.items():
        if key != 'name':
            print(f"   {key}: {value}")
    
    stage2_config = Config.get_stage_config(2)
    print(f"\n{stage2_config['name']}:")
    for key, value in stage2_config.items():
        if key != 'name':
            print(f"   {key}: {value}")
