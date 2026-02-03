"""
Data loading and preprocessing utilities for ViXNet
"""

import os
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import functional as F
from PIL import Image, ImageFilter
import numpy as np

from config import Config


# Class mapping: Real=0, Fake=1
CLASS_TO_IDX = {'Real': 0, 'Fake': 1}
IDX_TO_CLASS = {0: 'Real', 1: 'Fake'}


# ==================== CUSTOM AUGMENTATION TRANSFORMS ====================

class GaussianNoise(nn.Module):
    """
    Add Gaussian noise to image tensor
    Helps model be robust to sensor noise and compression artifacts
    """
    def __init__(self, std_range=(0.01, 0.05), p=0.5):
        super().__init__()
        self.std_range = std_range
        self.p = p
    
    def forward(self, img):
        if random.random() < self.p:
            std = random.uniform(*self.std_range)
            noise = torch.randn_like(img) * std
            img = img + noise
            img = torch.clamp(img, 0, 1)  # Keep values in valid range
        return img
    
    def __repr__(self):
        return f"{self.__class__.__name__}(std_range={self.std_range}, p={self.p})"


class RandomDownscale(nn.Module):
    """
    Randomly downscale then upscale image to simulate compression artifacts
    Helps model be robust to low-quality deepfakes
    """
    def __init__(self, scale_range=(0.5, 0.9), p=0.5):
        super().__init__()
        self.scale_range = scale_range
        self.p = p
    
    def forward(self, img):
        if random.random() < self.p:
            # img is a PIL Image or Tensor
            if isinstance(img, torch.Tensor):
                # Convert to PIL for resize
                img = F.to_pil_image(img)
                was_tensor = True
            else:
                was_tensor = False
            
            original_size = img.size  # (W, H)
            scale = random.uniform(*self.scale_range)
            new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
            
            # Downscale then upscale back (use LANCZOS for better quality)
            img = img.resize(new_size, Image.Resampling.BILINEAR)
            img = img.resize(original_size, Image.Resampling.BILINEAR)
            
            if was_tensor:
                img = F.to_tensor(img)
        
        return img
    
    def __repr__(self):
        return f"{self.__class__.__name__}(scale_range={self.scale_range}, p={self.p})"


class RandomGaussianBlur(nn.Module):
    """
    Apply Gaussian blur with random kernel size
    Helps model be robust to blurry/out-of-focus deepfakes
    """
    def __init__(self, kernel_sizes=[3, 5, 7], sigma=(0.1, 2.0), p=0.5):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.sigma = sigma
        self.p = p
    
    def forward(self, img):
        if random.random() < self.p:
            kernel_size = random.choice(self.kernel_sizes)
            sigma = random.uniform(*self.sigma)
            
            if isinstance(img, torch.Tensor):
                # Use torchvision's gaussian blur for tensors
                img = F.gaussian_blur(img, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])
            else:
                # PIL Image
                img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        
        return img
    
    def __repr__(self):
        return f"{self.__class__.__name__}(kernel_sizes={self.kernel_sizes}, sigma={self.sigma}, p={self.p})"


class RandomJPEGCompression(nn.Module):
    """
    Apply random JPEG compression to simulate real-world compression artifacts.
    
    This is CRITICAL for cross-dataset generalization because:
    - Different datasets use different compression levels
    - FF++ uses specific compression, Celeb-DF uses different compression
    - Real-world images have varying JPEG quality
    
    Works on PIL Images (before ToTensor).
    """
    def __init__(self, quality_range=(30, 100), p=0.8):
        """
        Args:
            quality_range: Tuple of (min_quality, max_quality) for JPEG compression.
                          Lower quality = more compression artifacts.
                          Recommended: (30, 100) for strong augmentation.
            p: Probability of applying compression.
        """
        super().__init__()
        self.quality_range = quality_range
        self.p = p
    
    def forward(self, img):
        if random.random() < self.p:
            # Handle tensor input
            if isinstance(img, torch.Tensor):
                img = F.to_pil_image(img)
                was_tensor = True
            else:
                was_tensor = False
            
            # Random quality in range
            quality = random.randint(*self.quality_range)
            
            # Save to buffer with JPEG compression and reload
            from io import BytesIO
            buffer = BytesIO()
            
            # Ensure RGB mode for JPEG
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            img.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            img = Image.open(buffer).copy()  # .copy() to detach from buffer
            buffer.close()
            
            if was_tensor:
                img = F.to_tensor(img)
        
        return img
    
    def __repr__(self):
        return f"{self.__class__.__name__}(quality_range={self.quality_range}, p={self.p})"


class RandomMotionBlur(nn.Module):
    """
    Apply random motion blur to simulate camera movement or subject motion.
    
    Helps model generalize to blurry/motion-affected deepfake videos.
    Motion blur is directional (horizontal, vertical, or diagonal).
    
    Works on PIL Images (before ToTensor).
    """
    def __init__(self, kernel_range=(3, 9), p=0.3):
        """
        Args:
            kernel_range: Tuple of (min_kernel, max_kernel) for blur kernel size.
                         Larger kernel = more motion blur. Must be odd numbers.
            p: Probability of applying motion blur.
        """
        super().__init__()
        self.kernel_range = kernel_range
        self.p = p
    
    def forward(self, img):
        if random.random() < self.p:
            # Handle tensor input
            if isinstance(img, torch.Tensor):
                img = F.to_pil_image(img)
                was_tensor = True
            else:
                was_tensor = False
            
            # Random kernel size (ensure odd)
            kernel_size = random.randint(self.kernel_range[0] // 2, self.kernel_range[1] // 2) * 2 + 1
            
            # Create motion blur kernel
            # Random angle for motion direction
            angle = random.choice([0, 45, 90, 135])  # Horizontal, diagonal, vertical
            
            kernel = self._create_motion_kernel(kernel_size, angle)
            
            # Apply convolution using numpy (PIL Kernel only supports 3x3 and 5x5)
            img = self._apply_kernel(img, kernel)
            
            if was_tensor:
                img = F.to_tensor(img)
        
        return img
    
    def _apply_kernel(self, img, kernel):
        """
        Apply convolution kernel to PIL image using numpy.
        
        Args:
            img: PIL Image
            kernel: numpy array kernel
            
        Returns:
            PIL Image with kernel applied
        """
        from scipy import ndimage
        
        # Convert to numpy array
        img_array = np.array(img, dtype=np.float32)
        
        # Apply kernel to each channel
        if len(img_array.shape) == 3:
            result = np.zeros_like(img_array)
            for c in range(img_array.shape[2]):
                result[:, :, c] = ndimage.convolve(img_array[:, :, c], kernel, mode='reflect')
        else:
            result = ndimage.convolve(img_array, kernel, mode='reflect')
        
        # Clip values and convert back to uint8
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return Image.fromarray(result)
    
    def _create_motion_kernel(self, size, angle):
        """
        Create a motion blur kernel with given size and angle.
        
        Args:
            size: Kernel size (odd number)
            angle: Angle in degrees (0=horizontal, 90=vertical)
            
        Returns:
            numpy array of shape (size, size)
        """
        kernel = np.zeros((size, size), dtype=np.float32)
        center = size // 2
        
        if angle == 0:  # Horizontal motion
            kernel[center, :] = 1.0
        elif angle == 90:  # Vertical motion
            kernel[:, center] = 1.0
        elif angle == 45:  # Diagonal (top-left to bottom-right)
            for i in range(size):
                kernel[i, i] = 1.0
        elif angle == 135:  # Diagonal (top-right to bottom-left)
            for i in range(size):
                kernel[i, size - 1 - i] = 1.0
        else:
            # Default to horizontal
            kernel[center, :] = 1.0
        
        # Normalize
        kernel = kernel / kernel.sum()
        
        return kernel
    
    def __repr__(self):
        return f"{self.__class__.__name__}(kernel_range={self.kernel_range}, p={self.p})"


class DeepfakeDataset(datasets.ImageFolder):
    """
    Custom ImageFolder dataset with fixed class mapping: Real=0, Fake=1
    """
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)
        
        # Override class_to_idx to ensure Real=0, Fake=1
        self.class_to_idx = CLASS_TO_IDX.copy()
        
        # Rebuild samples with correct mapping
        self.samples = []
        self.targets = []
        
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = os.path.join(root, class_name)
            if os.path.isdir(class_dir):
                for fname in os.listdir(class_dir):
                    fpath = os.path.join(class_dir, fname)
                    if os.path.isfile(fpath) and self._is_valid_file(fpath):
                        self.samples.append((fpath, class_idx))
                        self.targets.append(class_idx)
        
        self.classes = ['Real', 'Fake']  # Real=0, Fake=1
    
    def _is_valid_file(self, path):
        """Check if file is a valid image"""
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.gif'}
        return os.path.splitext(path)[1].lower() in valid_extensions


def get_data_transforms(stage='train'):
    """
    Get data transformations for different stages.
    
    Training augmentations are designed for maximum cross-dataset generalization,
    especially from FF++ → Celeb-DF v2. Key augmentations for generalization:
    - JPEG compression: Different datasets use different compression levels
    - Motion blur: Simulates real-world video artifacts
    - Gaussian blur/noise: Handles quality variations
    - Random downscale: Simulates low-resolution sources
    - Grayscale: Forces model to learn non-color features
    
    Args:
        stage: 'train', 'val', or 'test'
        
    Returns:
        torchvision.transforms.Compose object
    """
    
    if stage == 'train':
        # ============================================================
        # TRAINING AUGMENTATIONS FOR CROSS-DATASET GENERALIZATION
        # ============================================================
        # Order matters! PIL-based augmentations before ToTensor.
        
        transform_list = [
            # 1. Spatial transforms (PIL)
            transforms.RandomResizedCrop(Config.IMG_SIZE, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
            transforms.RandomHorizontalFlip(p=Config.HORIZONTAL_FLIP_PROB),
            transforms.RandomRotation(Config.ROTATION_DEGREES),
            
            # 2. Color transforms (PIL)
            transforms.ColorJitter(
                brightness=Config.COLOR_JITTER_BRIGHTNESS,
                contrast=Config.COLOR_JITTER_CONTRAST,
                saturation=Config.COLOR_JITTER_SATURATION,
                hue=Config.COLOR_JITTER_HUE
            ),
        ]
        
        # 3. Grayscale augmentation (PIL) - forces non-color feature learning
        if Config.USE_RANDOM_GRAYSCALE:
            transform_list.append(
                transforms.RandomGrayscale(p=Config.RANDOM_GRAYSCALE_PROB)
            )
        
        # 4. JPEG Compression (PIL) - CRITICAL for cross-dataset generalization
        # Different datasets use different compression, this makes model robust
        if Config.USE_JPEG_COMPRESSION:
            transform_list.append(
                RandomJPEGCompression(
                    quality_range=Config.JPEG_QUALITY_RANGE,
                    p=Config.JPEG_COMPRESSION_PROB
                )
            )
        
        # 5. Random Downscale (PIL) - simulates low-resolution sources
        if Config.USE_RANDOM_DOWNSCALE:
            transform_list.append(
                RandomDownscale(
                    scale_range=Config.RANDOM_DOWNSCALE_RANGE,
                    p=Config.RANDOM_DOWNSCALE_PROB
                )
            )
        
        # 6. Motion Blur (PIL) - simulates camera/subject movement
        if Config.USE_MOTION_BLUR:
            transform_list.append(
                RandomMotionBlur(
                    kernel_range=Config.MOTION_BLUR_KERNEL_RANGE,
                    p=Config.MOTION_BLUR_PROB
                )
            )
        
        # 7. Gaussian Blur (PIL) - handles out-of-focus/blurry images
        if Config.USE_GAUSSIAN_BLUR:
            transform_list.append(
                RandomGaussianBlur(
                    kernel_sizes=Config.GAUSSIAN_BLUR_KERNEL,
                    sigma=Config.GAUSSIAN_BLUR_SIGMA,
                    p=Config.GAUSSIAN_BLUR_PROB
                )
            )
        
        # ============================================================
        # Convert to Tensor
        # ============================================================
        transform_list.append(transforms.ToTensor())
        
        # 8. Gaussian Noise (Tensor) - handles sensor noise
        if Config.USE_GAUSSIAN_NOISE:
            transform_list.append(
                GaussianNoise(
                    std_range=Config.GAUSSIAN_NOISE_STD,
                    p=Config.GAUSSIAN_NOISE_PROB
                )
            )
        
        # 9. Normalize with ImageNet stats
        transform_list.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        )
        
        # 10. Random Erasing (Tensor, after normalization) - occlusion robustness
        if Config.USE_RANDOM_ERASING:
            transform_list.append(
                transforms.RandomErasing(
                    p=Config.RANDOM_ERASING_PROB,
                    scale=(0.02, 0.15),
                    ratio=(0.3, 3.3)
                )
            )
        
        return transforms.Compose(transform_list)
    
    else:
        # ============================================================
        # VALIDATION/TEST TRANSFORMS
        # ============================================================
        # Use Resize + CenterCrop to preserve aspect ratio better
        # This is more robust than direct resize to square
        return transforms.Compose([
            transforms.Resize(Config.IMG_SIZE + 32),  # Resize shorter edge
            transforms.CenterCrop(Config.IMG_SIZE),   # Center crop to target size
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])


def create_data_loaders(batch_size=None, num_workers=None, samples_per_epoch=None):
    """
    Create DataLoaders for train, validation, and test sets
    
    Args:
        batch_size: Batch size (uses Config.STAGE1_BATCH_SIZE if None)
        num_workers: Number of workers (uses Config.NUM_WORKERS if None)
        samples_per_epoch: Number of samples per epoch (None = use all, uses Config.SAMPLES_PER_EPOCH)
        
    Returns:
        Dictionary containing train_loader, val_loader, test_loader, and class_names
    """
    from torch.utils.data import RandomSampler
    
    if batch_size is None:
        batch_size = Config.STAGE1_BATCH_SIZE
    
    if num_workers is None:
        num_workers = Config.NUM_WORKERS
    
    if samples_per_epoch is None:
        samples_per_epoch = getattr(Config, 'SAMPLES_PER_EPOCH', None)
    
    print(f"\n📂 Loading datasets...")
    print(f"   Batch size: {batch_size}")
    print(f"   Num workers: {num_workers}")
    if samples_per_epoch:
        print(f"   Samples per epoch: {samples_per_epoch:,}")
    
    # Check if dataset directories exist
    if not os.path.exists(Config.TRAIN_DIR):
        print(f"⚠️  Warning: Training directory not found: {Config.TRAIN_DIR}")
        print("   Dataset may not be available in this environment.")
        return None
    
    # Get transforms
    train_transform = get_data_transforms('train')
    val_transform = get_data_transforms('val')
    test_transform = get_data_transforms('test')
    
    # Create datasets with fixed class mapping (Real=0, Fake=1)
    try:
        train_dataset = DeepfakeDataset(
            root=Config.TRAIN_DIR,
            transform=train_transform
        )
        
        val_dataset = DeepfakeDataset(
            root=Config.VAL_DIR,
            transform=val_transform
        )
        
        test_dataset = DeepfakeDataset(
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
        print(f"   ⚠️  Label mapping: Real=0, Fake=1")
        
    except Exception as e:
        print(f"❌ Error loading datasets: {str(e)}")
        print("   Please ensure dataset is available at the specified paths.")
        return None
    
    # Create sampler for subset training (faster epochs)
    train_sampler = None
    shuffle_train = True
    if samples_per_epoch and samples_per_epoch < len(train_dataset):
        train_sampler = RandomSampler(
            train_dataset, 
            replacement=True, 
            num_samples=samples_per_epoch
        )
        shuffle_train = False  # Sampler handles randomization
        print(f"   🚀 Using RandomSampler: {samples_per_epoch:,} samples/epoch (faster training)")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=Config.PIN_MEMORY,
        drop_last=True,  # Drop incomplete batches for batch normalization
        persistent_workers=num_workers > 0  # Keep workers alive between epochs
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
