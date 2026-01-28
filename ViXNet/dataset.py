"""
Data loading and preprocessing utilities for ViXNet
"""

import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image

from config import Config


def get_data_transforms(stage='train'):
    """
    Get data transformations for different stages
    
    Args:
        stage: 'train', 'val', or 'test'
        
    Returns:
        torchvision.transforms.Compose object
    """
    
    if stage == 'train':
        # Training augmentations for better generalization
        transform_list = [
            transforms.RandomResizedCrop(Config.IMG_SIZE, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
            transforms.RandomHorizontalFlip(p=Config.HORIZONTAL_FLIP_PROB),
            transforms.RandomRotation(Config.ROTATION_DEGREES),
            transforms.ColorJitter(
                brightness=Config.COLOR_JITTER_BRIGHTNESS,
                contrast=Config.COLOR_JITTER_CONTRAST,
                saturation=Config.COLOR_JITTER_SATURATION,
                hue=Config.COLOR_JITTER_HUE
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                std=[0.229, 0.224, 0.225]
            ),
        ]
        
        # Add random erasing if enabled
        if Config.USE_RANDOM_ERASING:
            transform_list.append(
                transforms.RandomErasing(
                    p=Config.RANDOM_ERASING_PROB,
                    scale=(0.02, 0.15),
                    ratio=(0.3, 3.3),
                    value='random'
                )
            )
        
        return transforms.Compose(transform_list)
    
    else:
        # Validation/Test transformations (no augmentation)
        return transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])


def create_data_loaders(batch_size=None, num_workers=None):
    """
    Create DataLoaders for train, validation, and test sets
    
    Args:
        batch_size: Batch size (uses Config.STAGE1_BATCH_SIZE if None)
        num_workers: Number of workers (uses Config.NUM_WORKERS if None)
        
    Returns:
        Dictionary containing train_loader, val_loader, test_loader, and class_names
    """
    
    if batch_size is None:
        batch_size = Config.STAGE1_BATCH_SIZE
    
    if num_workers is None:
        num_workers = Config.NUM_WORKERS
    
    print(f"\n📂 Loading datasets...")
    print(f"   Batch size: {batch_size}")
    print(f"   Num workers: {num_workers}")
    
    # Check if dataset directories exist
    if not os.path.exists(Config.TRAIN_DIR):
        print(f"⚠️  Warning: Training directory not found: {Config.TRAIN_DIR}")
        print("   Dataset may not be available in this environment.")
        return None
    
    # Get transforms
    train_transform = get_data_transforms('train')
    val_transform = get_data_transforms('val')
    test_transform = get_data_transforms('test')
    
    # Create datasets
    try:
        train_dataset = datasets.ImageFolder(
            root=Config.TRAIN_DIR,
            transform=train_transform
        )
        
        val_dataset = datasets.ImageFolder(
            root=Config.VAL_DIR,
            transform=val_transform
        )
        
        test_dataset = datasets.ImageFolder(
            root=Config.TEST_DIR,
            transform=test_transform
        )
        
        # Print dataset info
        print(f"\n✅ Datasets loaded successfully!")
        print(f"   Training samples: {len(train_dataset):,}")
        print(f"   Validation samples: {len(val_dataset):,}")
        print(f"   Test samples: {len(test_dataset):,}")
        print(f"   Classes: {train_dataset.classes}")
        print(f"   Class to index mapping: {train_dataset.class_to_idx}")
        
    except Exception as e:
        print(f"❌ Error loading datasets: {str(e)}")
        print("   Please ensure dataset is available at the specified paths.")
        return None
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=Config.PIN_MEMORY,
        drop_last=True  # Drop incomplete batches for batch normalization
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=Config.PIN_MEMORY
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=Config.PIN_MEMORY
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'class_names': train_dataset.classes,
        'class_to_idx': train_dataset.class_to_idx
    }


def check_dataset_availability():
    """
    Check if dataset is available and print information
    
    Returns:
        Boolean indicating if dataset is available
    """
    print("\n" + "="*70)
    print("🔍 CHECKING DATASET AVAILABILITY")
    print("="*70)
    
    dirs_to_check = [
        ('Train', Config.TRAIN_DIR),
        ('Validation', Config.VAL_DIR),
        ('Test', Config.TEST_DIR)
    ]
    
    all_exist = True
    
    for name, path in dirs_to_check:
        exists = os.path.exists(path)
        status = "✅" if exists else "❌"
        print(f"{status} {name}: {path}")
        
        if exists:
            # Count subdirectories (classes)
            try:
                subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
                print(f"   Classes found: {subdirs}")
                
                # Count files in each class
                for subdir in subdirs:
                    subdir_path = os.path.join(path, subdir)
                    num_files = len([f for f in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, f))])
                    print(f"   - {subdir}: {num_files:,} images")
            except Exception as e:
                print(f"   Error reading directory: {str(e)}")
        
        all_exist = all_exist and exists
    
    print("="*70)
    
    if all_exist:
        print("✅ All dataset directories found!")
    else:
        print("⚠️  Some dataset directories are missing.")
        print("   The model can still be initialized, but training requires the dataset.")
    
    print("="*70)
    
    return all_exist


if __name__ == "__main__":
    """
    Test data loading
    """
    print("="*70)
    print("Testing Data Loading")
    print("="*70)
    
    # Check dataset availability
    dataset_available = check_dataset_availability()
    
    if dataset_available:
        # Try to create data loaders
        print("\n📦 Creating data loaders...")
        data_loaders = create_data_loaders(batch_size=4)
        
        if data_loaders is not None:
            # Test loading a batch
            print("\n🧪 Testing batch loading...")
            train_loader = data_loaders['train']
            
            for images, labels in train_loader:
                print(f"   Batch shape: {images.shape}")
                print(f"   Labels shape: {labels.shape}")
                print(f"   Labels: {labels}")
                print(f"   Image range: [{images.min():.3f}, {images.max():.3f}]")
                break
            
            print("\n✅ Data loading test successful!")
        else:
            print("\n⚠️  Could not create data loaders.")
    else:
        print("\n⚠️  Dataset not available. Skipping data loader test.")
    
    print("\n" + "="*70)
