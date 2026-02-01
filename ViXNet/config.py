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
    DATA_DIR = os.path.join("D:\\Repo\\DAP391m", "FaceForensics_new")
    TRAIN_DIR = os.path.join(DATA_DIR, "train")
    VAL_DIR = os.path.join(DATA_DIR, "val")
    TEST_DIR = os.path.join(DATA_DIR, "test")
    
    # Multiple dataset configurations
    DATASETS = {
        'default': {
            'name': 'Default Dataset',
            'path': os.path.join("D:\\Repo\\DAP391m", "FaceForensics_new"),
            'train': os.path.join("D:\\Repo\\DAP391m", "FaceForensics_new", "train"),
            'val': os.path.join("D:\\Repo\\DAP391m", "FaceForensics_new", "val"),
            'test': os.path.join("D:\\Repo\\DAP391m", "FaceForensics_new", "test")
        },
        'celeb': {
            'name': 'CelebDF (V2)',
            'path': os.path.join("D:\\Repo\\DAP391m", "Celeb_V2"),
            'train': os.path.join("D:\\Repo\\DAP391m", "Celeb_V2", "Train"),
            'val': os.path.join("D:\\Repo\\DAP391m", "Celeb_V2", "Validation"),
            'test': os.path.join("D:\\Repo\\DAP391m", "Celeb_V2", "Test")
        },
        'wilddeepfake': {
            'name': 'WildDeepfake',
            'path': os.path.join("D:\\Repo\\DAP391m", "wilddeepfake"),
            'train': os.path.join("D:\\Repo\\DAP391m", "wilddeepfake", "train"),
            'val': os.path.join("D:\\Repo\\DAP391m", "wilddeepfake", "val"),
            'test': os.path.join("D:\\Repo\\DAP391m", "wilddeepfake_20k")
        },
        'DFDC': {
            'name': 'DFDC',
            'path': os.path.join("D:\\Repo\\DAP391m", "real_vs_fake", "real-vs-fake"),
            'train': os.path.join("D:\\Repo\\DAP391m", "real_vs_fake", "real-vs-fake", "train"),
            'val': os.path.join("D:\\Repo\\DAP391m", "real_vs_fake", "real-vs-fake", "valid"),
            'test': os.path.join("D:\\Repo\\DAP391m", "real_vs_fake", "real-vs-fake", "test")
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
    
    # Stage 1: Feature extractor frozen (train head/classifier only)
    STAGE1_EPOCHS = 5
    STAGE1_BATCH_SIZE = 48  # Can go higher since only head is trained
    STAGE1_LR = 1e-3  # High LR for fresh head layers
    STAGE1_WEIGHT_DECAY = 1e-4
    
    # Stage 2: Fine-tuning high-level layers (Xception + head)
    STAGE2_EPOCHS = 8
    STAGE2_BATCH_SIZE = 32  # Lower for fine-tuning (more memory)
    STAGE2_LR = 5e-5  # Lower LR for fine-tuning
    STAGE2_WEIGHT_DECAY = 1e-4
    
    # Stage 3: Fine-tuning ViT layers (for 3-stage training)
    STAGE3_EPOCHS = 8
    STAGE3_BATCH_SIZE = 16  # ViT + Xception unfrozen = more memory
    STAGE3_LR = 2e-5  # Very low LR for ViT fine-tuning
    STAGE3_WEIGHT_DECAY = 1e-4
    
    # Learning rates for different components (3-stage training)
    # Used in stage 2 & 3 when use_param_groups=True
    LR_HEAD = 1e-5      # Very low LR for head (already well-trained)
    LR_CNN = 5e-5       # Moderate LR for Xception CNN
    LR_VIT = 2e-5       # Low LR for ViT (pretrained is good)
    
    # ==================== OPTIMIZATION ====================
    OPTIMIZER = 'adamw'  # AdamW optimizer
    MOMENTUM = 0.9  # For SGD (if used)
    SCHEDULER = 'cosine'  # Learning rate scheduler: 'cosine', 'step', 'plateau'
    WEIGHT_DATASET = [1.0, 1.0]  # Class weights for imbalanced dataset
    GRADIENT_ACCUMULATION = 2  # Accumulate gradients for effective larger batch
    
    # Scheduler parameters
    STEP_SIZE = 5  # For StepLR
    GAMMA = 0.5  # LR decay factor
    T_MAX = 15  # For CosineAnnealingLR (total epochs)
    
    # ==================== REGULARIZATION ====================
    DROPOUT = 0.5  # Dropout rate in classifier
    LABEL_SMOOTHING = 0.1  # Label smoothing for CrossEntropyLoss
    
    # ==================== DATA LOADING ====================
    NUM_WORKERS = 4 # Reduced - too many can slow down on Windows
    PIN_MEMORY = True  # Pin memory for faster GPU transfer
    SAMPLES_PER_EPOCH = None  # Use subset each epoch (None = use all)
    
    # ==================== TRAINING SETTINGS ====================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MIXED_PRECISION = True  # Use mixed precision training (FP16)
    GRADIENT_CLIP = 1.0  # Gradient clipping value
    
    # ==================== CHECKPOINT & LOGGING ====================
    # Base checkpoint directory
    CHECKPOINT_BASE_DIR = "checkpoints"
    
    # Model-specific checkpoint directories
    CHECKPOINT_DIRS = {
        'vixnet': os.path.join(CHECKPOINT_BASE_DIR, 'vixnet'),
        'vixnet_cross_attention': os.path.join(CHECKPOINT_BASE_DIR, 'vixnet_cross_attention'),
        'xception': os.path.join(CHECKPOINT_BASE_DIR, 'xception_only'),
        'vit': os.path.join(CHECKPOINT_BASE_DIR, 'vit_only'),
    }
    
    # Current save directory - default to vixnet_cross_attention
    SAVE_DIR = os.path.join(CHECKPOINT_BASE_DIR, 'vixnet_cross_attention')
    
    # Active model name (for tracking which model is being trained)
    ACTIVE_MODEL = None
    
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
    COLOR_JITTER_HUE = 0.02
    
    # Advanced augmentations
    USE_RANDOM_ERASING = True  # Random erasing augmentation
    RANDOM_ERASING_PROB = 0.15
    
    # Blur augmentation
    USE_GAUSSIAN_BLUR = True
    GAUSSIAN_BLUR_PROB = 0.3
    GAUSSIAN_BLUR_KERNEL = [3, 5, 7]  # Kernel sizes
    GAUSSIAN_BLUR_SIGMA = (0.1, 2.0)  # Sigma range
    
    # Noise augmentation
    USE_GAUSSIAN_NOISE = True
    GAUSSIAN_NOISE_PROB = 0.3
    GAUSSIAN_NOISE_STD = (0.01, 0.05)  # Noise std range
    
    # Random downscale (compression artifacts simulation)
    USE_RANDOM_DOWNSCALE = True
    RANDOM_DOWNSCALE_PROB = 0.3
    RANDOM_DOWNSCALE_RANGE = (0.5, 0.9)  # Scale range before upscaling back
    
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
        
        print("\n🎯 TRAINING - STAGE 3:")
        print(f"   Epochs: {cls.STAGE3_EPOCHS}")
        print(f"   Batch size: {cls.STAGE3_BATCH_SIZE}")
        print(f"   Learning rate: {cls.STAGE3_LR}")
        print(f"   Weight decay: {cls.STAGE3_WEIGHT_DECAY}")
        
        print("\n📊 COMPONENT LEARNING RATES:")
        print(f"   Head LR: {cls.LR_HEAD}")
        print(f"   CNN LR: {cls.LR_CNN}")
        print(f"   ViT LR: {cls.LR_VIT}")
        
        print("\n⚡ OPTIMIZATION:")
        print(f"   Optimizer: {cls.OPTIMIZER}")
        print(f"   Scheduler: {cls.SCHEDULER}")
        print(f"   Gradient clip: {cls.GRADIENT_CLIP}")
        print(f"   Mixed precision: {cls.MIXED_PRECISION}")
        
        print("\n💾 SYSTEM:")
        print(f"   Device: {cls.DEVICE}")
        print(f"   Num workers: {cls.NUM_WORKERS}")
        print(f"   Save dir: {cls.SAVE_DIR}")
        print(f"   Active model: {cls.ACTIVE_MODEL}")
        
        print("="*70)
    
    @classmethod
    def set_model_checkpoint_dir(cls, model_name):
        """
        Set the checkpoint directory based on model type.
        
        Args:
            model_name: Name of the model ('vixnet', 'vixnet_cross_attention', 'xception', 'vit')
        """
        model_name_lower = model_name.lower()
        
        if model_name_lower in cls.CHECKPOINT_DIRS:
            cls.SAVE_DIR = cls.CHECKPOINT_DIRS[model_name_lower]
        else:
            # Default to a new folder with the model name
            cls.SAVE_DIR = os.path.join(cls.CHECKPOINT_BASE_DIR, model_name_lower)
        
        cls.ACTIVE_MODEL = model_name_lower
        
        # Create directory if it doesn't exist
        os.makedirs(cls.SAVE_DIR, exist_ok=True)
        
        print(f"\n📁 Checkpoint directory set to: {cls.SAVE_DIR}")
        return cls.SAVE_DIR
    
    @classmethod
    def get_checkpoint_dir(cls, model_name=None):
        """
        Get the checkpoint directory for a specific model.
        
        Args:
            model_name: Name of the model. If None, returns current SAVE_DIR.
            
        Returns:
            Path to checkpoint directory
        """
        if model_name is None:
            return cls.SAVE_DIR
        
        model_name_lower = model_name.lower()
        if model_name_lower in cls.CHECKPOINT_DIRS:
            return cls.CHECKPOINT_DIRS[model_name_lower]
        return os.path.join(cls.CHECKPOINT_BASE_DIR, model_name_lower)
    
    @classmethod
    def list_model_checkpoints(cls):
        """
        List all available model checkpoint directories and their contents.
        
        Returns:
            Dictionary with model names and their checkpoint info
        """
        result = {}
        for model_name, checkpoint_dir in cls.CHECKPOINT_DIRS.items():
            if os.path.exists(checkpoint_dir):
                files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
                result[model_name] = {
                    'path': checkpoint_dir,
                    'checkpoints': files,
                    'count': len(files)
                }
            else:
                result[model_name] = {
                    'path': checkpoint_dir,
                    'checkpoints': [],
                    'count': 0
                }
        return result
    
    @classmethod
    def get_stage_config(cls, stage=1, name=None):
        """
        Get configuration for specific training stage
        
        Args:
            stage: 1 or 2
            name: name of stage
        Returns:
            Dictionary with stage-specific config
        """
        if stage == 1:
            return {
                'epochs': cls.STAGE1_EPOCHS,
                'batch_size': cls.STAGE1_BATCH_SIZE,
                'lr': cls.STAGE1_LR,
                'weight_decay': cls.STAGE1_WEIGHT_DECAY,
                'name': name or 'Stage 1: Fusion Training'
            }
        elif stage == 2:
            return {
                'epochs': cls.STAGE2_EPOCHS,
                'batch_size': cls.STAGE2_BATCH_SIZE,
                'lr': cls.STAGE2_LR,
                'weight_decay': cls.STAGE2_WEIGHT_DECAY,
                'name': name or 'Stage 2: Xception Fine-tuning'
            }
        elif stage == 3:
            return {
                'epochs': cls.STAGE3_EPOCHS,
                'batch_size': cls.STAGE3_BATCH_SIZE,
                'lr': cls.STAGE3_LR,
                'weight_decay': cls.STAGE3_WEIGHT_DECAY,
                'name': name or 'Stage 3: ViT Fine-tuning'
            }
        else:
            raise ValueError(f"Invalid stage: {stage}. Must be 1, 2 or 3.")
    
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
