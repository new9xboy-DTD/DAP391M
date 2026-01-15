"""
=====================================================================
TRAINING SCRIPT - HUẤN LUYỆN CÁC MÔ-ĐUN
=====================================================================
Mô tả:
    Script này điều phối quá trình huấn luyện của tất cả các modules
    trong hệ thống nhận diện deepfake.
    
    Quy trình huấn luyện:
    1. Phase 1 - Unsupervised (chỉ dùng ảnh thật):
       - Train VQ-VAE để tokenize ảnh
       - Train Transformer để học phân phối token
       - Train DDPM để học khử nhiễu
       
    2. Phase 2 - Supervised (dùng cả Real và Fake):
       - Train CNN/ViT classifier
       - Train GCN (nếu có landmarks)
       
    3. Phase 3 - Fusion:
       - Train Fusion module để kết hợp các scores
       
Cách sử dụng:
    python train.py --phase 1  # Train unsupervised modules
    python train.py --phase 2  # Train supervised modules  
    python train.py --phase 3  # Train fusion module
    python train.py --phase all  # Train tất cả
    
    # Train trên Google Colab (dùng MyDrive cho checkpoints và logs)
    python train.py --phase all --colab
    
    # Train với dataset từ Kaggle (path là đường dẫn dataset)
    python train.py --phase all --colab --data_dir /kaggle/input/dataset
    
Tác giả: DAP391M Team
Phiên bản: 1.0
=====================================================================
"""

import os
import sys
import argparse
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

# Import các modules
from config import (
    DataConfig, VQVAEConfig, TransformerConfig, 
    DiffusionConfig, CNNViTConfig, GCNConfig, 
    FusionConfig, TrainingConfig, create_directories, print_config
)
from data_preprocessing import create_dataloaders
from vqvae_module import VQVAE, vqvae_loss
from transformer_module import ImageGPT, TransformerTrainer
from diffusion_module import DDPM
from cnn_vit_module import CNNViTClassifier
from gcn_module import GCNModule
from fusion_module import DeepfakeFusionModule


# =====================================================================
# UTILITY FUNCTIONS
# =====================================================================

def set_seed(seed):
    """
    Đặt random seed để đảm bảo reproducibility
    
    Args:
        seed: Giá trị seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # Deterministic mode (có thể làm chậm)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(model, optimizer, epoch, loss, path, extra_info=None):
    """
    Lưu checkpoint của model
    
    Args:
        model: Model cần lưu
        optimizer: Optimizer state
        epoch: Epoch hiện tại
        loss: Loss value
        path: Đường dẫn file
        extra_info: Thông tin bổ sung
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    if extra_info:
        checkpoint.update(extra_info)
    
    torch.save(checkpoint, path)
    print(f"💾 Đã lưu checkpoint: {path}")


def load_checkpoint(model, path, optimizer=None, device='cpu'):
    """
    Load checkpoint
    
    Args:
        model: Model để load weights vào
        path: Đường dẫn checkpoint
        optimizer: Optional optimizer để load state
        device: Device
        
    Returns:
        epoch: Epoch của checkpoint
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"📂 Đã load checkpoint từ: {path}")
    print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
    
    return checkpoint.get('epoch', 0)


class AverageMeter:
    """
    Helper class để tính running average
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# =====================================================================
# PHASE 1: UNSUPERVISED TRAINING
# =====================================================================

def train_vqvae(num_epochs=None, save_every=10):
    """
    Train VQ-VAE model
    
    VQ-VAE học cách tokenize ảnh thành discrete tokens.
    Chỉ dùng ảnh thật (unsupervised).
    
    Args:
        num_epochs: Số epochs (None = dùng config)
        save_every: Lưu checkpoint sau mỗi N epochs
    """
    print("\n" + "=" * 70)
    print("🎯 TRAINING VQ-VAE")
    print("=" * 70)
    
    if num_epochs is None:
        num_epochs = VQVAEConfig.NUM_EPOCHS
    
    device = TrainingConfig.DEVICE
    
    # Data loaders (chỉ ảnh thật)
    print("\n📂 Loading data...")
    dataloaders = create_dataloaders(mode='vqvae')
    
    if dataloaders is None:
        print("❌ Không thể load data. Kiểm tra đường dẫn dataset.")
        return None
    
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    
    # Model
    print("\n🏗️  Initializing model...")
    model = VQVAE().to(device)
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=VQVAEConfig.LEARNING_RATE,
        weight_decay=0.01
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # TensorBoard
    writer = SummaryWriter(os.path.join(DataConfig.LOG_DIR, 'vqvae'))
    
    # Training loop
    print("\n🎯 Starting training...")
    best_loss = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'='*50}")
        
        # Training
        model.train()
        train_loss = AverageMeter()
        train_recon = AverageMeter()
        train_vq = AverageMeter()
        
        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, (images, _) in enumerate(pbar):
            images = images.to(device)
            
            # Forward
            recon, vq_loss_val, perplexity, _ = model(images)
            
            # Compute loss
            total_loss, loss_dict = vqvae_loss(
                recon, images, vq_loss_val,
                recon_weight=VQVAEConfig.RECON_LOSS_WEIGHT,
                vq_weight=VQVAEConfig.VQ_LOSS_WEIGHT
            )
            
            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Update metrics
            train_loss.update(loss_dict['total'])
            train_recon.update(loss_dict['recon'])
            train_vq.update(loss_dict['vq'])
            
            pbar.set_postfix({
                'loss': f'{train_loss.avg:.4f}',
                'recon': f'{train_recon.avg:.4f}',
                'ppl': f'{perplexity.item():.1f}'
            })
        
        # Validation
        model.eval()
        val_loss = AverageMeter()
        
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(device)
                recon, vq_loss_val, _, _ = model(images)
                total_loss, _ = vqvae_loss(recon, images, vq_loss_val)
                val_loss.update(total_loss.item())
        
        # Update scheduler
        scheduler.step(val_loss.avg)
        
        # Logging
        print(f"\n📊 Train Loss: {train_loss.avg:.4f} (Recon: {train_recon.avg:.4f}, VQ: {train_vq.avg:.4f})")
        print(f"📊 Val Loss: {val_loss.avg:.4f}")
        
        writer.add_scalar('Loss/train', train_loss.avg, epoch)
        writer.add_scalar('Loss/val', val_loss.avg, epoch)
        writer.add_scalar('Loss/recon', train_recon.avg, epoch)
        writer.add_scalar('Loss/vq', train_vq.avg, epoch)
        
        # Save checkpoint
        if epoch % save_every == 0 or val_loss.avg < best_loss:
            save_path = os.path.join(
                DataConfig.CHECKPOINT_DIR,
                f'vqvae_epoch_{epoch}.pth'
            )
            save_checkpoint(model, optimizer, epoch, val_loss.avg, save_path)
            
            if val_loss.avg < best_loss:
                best_loss = val_loss.avg
                best_path = os.path.join(DataConfig.CHECKPOINT_DIR, 'vqvae_best.pth')
                save_checkpoint(model, optimizer, epoch, best_loss, best_path)
    
    writer.close()
    print(f"\n✅ VQ-VAE training hoàn thành! Best loss: {best_loss:.4f}")
    
    return model


def train_transformer(vqvae_path=None, num_epochs=None, save_every=10):
    """
    Train Transformer (GPT-like) model
    
    Transformer học phân phối của các image tokens từ VQ-VAE.
    
    Args:
        vqvae_path: Đường dẫn đến trained VQ-VAE
        num_epochs: Số epochs
        save_every: Lưu checkpoint mỗi N epochs
    """
    print("\n" + "=" * 70)
    print("🎯 TRAINING TRANSFORMER")
    print("=" * 70)
    
    if num_epochs is None:
        num_epochs = TransformerConfig.NUM_EPOCHS
    
    device = TrainingConfig.DEVICE
    
    # Load VQ-VAE
    print("\n📂 Loading VQ-VAE...")
    vqvae = VQVAE().to(device)
    
    if vqvae_path and os.path.exists(vqvae_path):
        load_checkpoint(vqvae, vqvae_path, device=device)
    else:
        print("⚠️  VQ-VAE checkpoint không tìm thấy. Dùng random weights.")
    
    vqvae.eval()  # Freeze VQ-VAE
    
    # Data loaders
    dataloaders = create_dataloaders(mode='vqvae')
    if dataloaders is None:
        print("❌ Không thể load data.")
        return None
    
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    
    # Model
    print("\n🏗️  Initializing Transformer...")
    model = ImageGPT().to(device)
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=TransformerConfig.LEARNING_RATE,
        weight_decay=0.01
    )
    
    # Trainer
    trainer = TransformerTrainer(
        model, optimizer, device,
        warmup_steps=TransformerConfig.WARMUP_STEPS
    )
    
    # TensorBoard
    writer = SummaryWriter(os.path.join(DataConfig.LOG_DIR, 'transformer'))
    
    # Training loop
    print("\n🎯 Starting training...")
    best_loss = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'='*50}")
        
        model.train()
        train_loss = AverageMeter()
        
        pbar = tqdm(train_loader, desc="Training")
        for images, _ in pbar:
            images = images.to(device)
            
            # Get tokens from VQ-VAE
            with torch.no_grad():
                _, token_indices = vqvae.encode(images)
            
            # Train transformer
            loss = trainer.train_step(
                token_indices,
                label_smoothing=TransformerConfig.LABEL_SMOOTHING
            )
            
            train_loss.update(loss)
            pbar.set_postfix({'loss': f'{train_loss.avg:.4f}'})
        
        # Validation
        model.eval()
        val_loss = AverageMeter()
        val_perplexity = AverageMeter()
        
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(device)
                _, token_indices = vqvae.encode(images)
                
                loss = model.compute_loss(token_indices)
                perplexity = model.compute_perplexity(token_indices)
                
                val_loss.update(loss.item())
                val_perplexity.update(perplexity.mean().item())
        
        # Logging
        print(f"\n📊 Train Loss: {train_loss.avg:.4f}")
        print(f"📊 Val Loss: {val_loss.avg:.4f}, Perplexity: {val_perplexity.avg:.2f}")
        
        writer.add_scalar('Loss/train', train_loss.avg, epoch)
        writer.add_scalar('Loss/val', val_loss.avg, epoch)
        writer.add_scalar('Perplexity/val', val_perplexity.avg, epoch)
        
        # Save checkpoint
        if epoch % save_every == 0 or val_loss.avg < best_loss:
            save_path = os.path.join(
                DataConfig.CHECKPOINT_DIR,
                f'transformer_epoch_{epoch}.pth'
            )
            save_checkpoint(model, optimizer, epoch, val_loss.avg, save_path)
            
            if val_loss.avg < best_loss:
                best_loss = val_loss.avg
                best_path = os.path.join(DataConfig.CHECKPOINT_DIR, 'transformer_best.pth')
                save_checkpoint(model, optimizer, epoch, best_loss, best_path)
    
    writer.close()
    print(f"\n✅ Transformer training hoàn thành! Best loss: {best_loss:.4f}")
    
    return model


def train_ddpm(num_epochs=None, save_every=10):
    """
    Train DDPM model
    
    DDPM học cách khử nhiễu ảnh, chỉ dùng ảnh thật.
    
    Args:
        num_epochs: Số epochs
        save_every: Lưu checkpoint mỗi N epochs
    """
    print("\n" + "=" * 70)
    print("🎯 TRAINING DDPM")
    print("=" * 70)
    
    if num_epochs is None:
        num_epochs = DiffusionConfig.NUM_EPOCHS
    
    device = TrainingConfig.DEVICE
    
    # Data loaders (chỉ ảnh thật)
    dataloaders = create_dataloaders(mode='unsupervised')
    if dataloaders is None:
        print("❌ Không thể load data.")
        return None
    
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    
    # Model
    print("\n🏗️  Initializing DDPM...")
    model = DDPM().to(device)
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=DiffusionConfig.LEARNING_RATE,
        weight_decay=0.01
    )
    
    # TensorBoard
    writer = SummaryWriter(os.path.join(DataConfig.LOG_DIR, 'ddpm'))
    
    # Training loop
    print("\n🎯 Starting training...")
    best_loss = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'='*50}")
        
        model.train()
        train_loss = AverageMeter()
        
        pbar = tqdm(train_loader, desc="Training")
        for images, _ in pbar:
            images = images.to(device)
            
            # Forward
            loss = model(images)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss.update(loss.item())
            pbar.set_postfix({'loss': f'{train_loss.avg:.4f}'})
        
        # Validation
        model.eval()
        val_loss = AverageMeter()
        
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(device)
                loss = model(images)
                val_loss.update(loss.item())
        
        # Logging
        print(f"\n📊 Train Loss: {train_loss.avg:.4f}")
        print(f"📊 Val Loss: {val_loss.avg:.4f}")
        
        writer.add_scalar('Loss/train', train_loss.avg, epoch)
        writer.add_scalar('Loss/val', val_loss.avg, epoch)
        
        # Save checkpoint
        if epoch % save_every == 0 or val_loss.avg < best_loss:
            save_path = os.path.join(
                DataConfig.CHECKPOINT_DIR,
                f'ddpm_epoch_{epoch}.pth'
            )
            save_checkpoint(model, optimizer, epoch, val_loss.avg, save_path)
            
            if val_loss.avg < best_loss:
                best_loss = val_loss.avg
                best_path = os.path.join(DataConfig.CHECKPOINT_DIR, 'ddpm_best.pth')
                save_checkpoint(model, optimizer, epoch, best_loss, best_path)
    
    writer.close()
    print(f"\n✅ DDPM training hoàn thành! Best loss: {best_loss:.4f}")
    
    return model


# =====================================================================
# PHASE 2: SUPERVISED TRAINING
# =====================================================================

def train_cnn_vit(num_epochs=None, save_every=5):
    """
    Train CNN/ViT classifier
    
    Supervised training với cả ảnh Real và Fake.
    
    Args:
        num_epochs: Số epochs
        save_every: Lưu checkpoint mỗi N epochs
    """
    print("\n" + "=" * 70)
    print("🎯 TRAINING CNN/ViT CLASSIFIER")
    print("=" * 70)
    
    if num_epochs is None:
        num_epochs = CNNViTConfig.NUM_EPOCHS
    
    device = TrainingConfig.DEVICE
    
    # Data loaders (supervised)
    dataloaders = create_dataloaders(mode='supervised')
    if dataloaders is None:
        print("❌ Không thể load data.")
        return None
    
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    
    # Model
    print("\n🏗️  Initializing CNN/ViT Classifier...")
    model = CNNViTClassifier(
        extractor_type='cnn',
        backbone_name=CNNViTConfig.BACKBONE_NAME,
        pretrained=CNNViTConfig.PRETRAINED
    ).to(device)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer với learning rate khác nhau cho backbone và head
    backbone_params = list(model.feature_extractor.parameters())
    head_params = list(model.classifier.parameters())
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': CNNViTConfig.BACKBONE_LR},
        {'params': head_params, 'lr': CNNViTConfig.LEARNING_RATE}
    ], weight_decay=0.01)
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    
    # TensorBoard
    writer = SummaryWriter(os.path.join(DataConfig.LOG_DIR, 'cnn_vit'))
    
    # Training loop
    print("\n🎯 Starting training...")
    best_acc = 0.0
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'='*50}")
        
        # Training
        model.train()
        train_loss = AverageMeter()
        train_acc = AverageMeter()
        
        pbar = tqdm(train_loader, desc="Training")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward
            logits = model(images)
            loss = criterion(logits, labels)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Accuracy
            preds = torch.argmax(logits, dim=1)
            acc = (preds == labels).float().mean()
            
            train_loss.update(loss.item())
            train_acc.update(acc.item())
            
            pbar.set_postfix({
                'loss': f'{train_loss.avg:.4f}',
                'acc': f'{train_acc.avg:.4f}'
            })
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = AverageMeter()
        val_acc = AverageMeter()
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                logits = model(images)
                loss = criterion(logits, labels)
                
                preds = torch.argmax(logits, dim=1)
                acc = (preds == labels).float().mean()
                
                val_loss.update(loss.item())
                val_acc.update(acc.item())
        
        # Logging
        print(f"\n📊 Train - Loss: {train_loss.avg:.4f}, Acc: {train_acc.avg:.4f}")
        print(f"📊 Val   - Loss: {val_loss.avg:.4f}, Acc: {val_acc.avg:.4f}")
        
        writer.add_scalar('Loss/train', train_loss.avg, epoch)
        writer.add_scalar('Loss/val', val_loss.avg, epoch)
        writer.add_scalar('Accuracy/train', train_acc.avg, epoch)
        writer.add_scalar('Accuracy/val', val_acc.avg, epoch)
        
        # Save checkpoint
        if epoch % save_every == 0 or val_acc.avg > best_acc:
            save_path = os.path.join(
                DataConfig.CHECKPOINT_DIR,
                f'cnn_vit_epoch_{epoch}.pth'
            )
            save_checkpoint(model, optimizer, epoch, val_loss.avg, save_path,
                          extra_info={'accuracy': val_acc.avg})
            
            if val_acc.avg > best_acc:
                best_acc = val_acc.avg
                best_path = os.path.join(DataConfig.CHECKPOINT_DIR, 'cnn_vit_best.pth')
                save_checkpoint(model, optimizer, epoch, val_loss.avg, best_path,
                              extra_info={'accuracy': best_acc})
    
    writer.close()
    print(f"\n✅ CNN/ViT training hoàn thành! Best accuracy: {best_acc:.4f}")
    
    return model


# =====================================================================
# MAIN
# =====================================================================

def main():
    """
    Main function - Entry point
    """
    parser = argparse.ArgumentParser(description='Train Deepfake Detection Models')
    parser.add_argument('--phase', type=str, default='all',
                       choices=['1', '2', '3', 'all', 'vqvae', 'transformer', 'ddpm', 'cnn_vit'],
                       help='Training phase')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (override config)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda/cpu)')
    parser.add_argument('--colab', action='store_true',
                       help='Enable Google Colab mode (use MyDrive paths for checkpoints and logs)')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Custom dataset directory path (e.g., path from Kaggle clone)')
    
    args = parser.parse_args()
    
    # Configure paths for Google Colab if enabled
    if args.colab:
        # Save checkpoints and outputs to MyDrive
        DataConfig.CHECKPOINT_DIR = "/content/drive/MyDrive/checkpoints"
        DataConfig.LOG_DIR = "/content/drive/MyDrive/logs"
        DataConfig.RESULTS_DIR = "/content/drive/MyDrive/results"
        
        # Use default Colab dataset path if --data_dir is not specified
        if not args.data_dir:
            DataConfig.DATASET_ROOT = "/content/drive/MyDrive/datasets/faces"
            DataConfig.TRAIN_DIR = os.path.join(DataConfig.DATASET_ROOT, "Train")
            DataConfig.VAL_DIR = os.path.join(DataConfig.DATASET_ROOT, "Validation")
            DataConfig.TEST_DIR = os.path.join(DataConfig.DATASET_ROOT, "Test")
        
        print("\n☁️  Google Colab mode enabled!")
        print(f"   → Checkpoints: {DataConfig.CHECKPOINT_DIR}")
        print(f"   → Logs: {DataConfig.LOG_DIR}")
    
    # Override dataset path if --data_dir is specified
    if args.data_dir:
        DataConfig.DATASET_ROOT = args.data_dir
        DataConfig.TRAIN_DIR = os.path.join(DataConfig.DATASET_ROOT, "Train")
        DataConfig.VAL_DIR = os.path.join(DataConfig.DATASET_ROOT, "Validation")
        DataConfig.TEST_DIR = os.path.join(DataConfig.DATASET_ROOT, "Test")
        print(f"\n📂 Custom dataset directory: {DataConfig.DATASET_ROOT}")
    
    # Print banner
    print("\n" + "=" * 70)
    print("🚀 HỆ THỐNG HUẤN LUYỆN NHẬN DIỆN DEEPFAKE")
    print("   Sequence Modeling + Diffusion + GCN Pipeline")
    print("=" * 70)
    
    # Setup
    set_seed(TrainingConfig.SEED)
    create_directories()
    
    if args.device:
        TrainingConfig.DEVICE = torch.device(args.device)
    
    print(f"\n⚙️  Device: {TrainingConfig.DEVICE}")
    print(f"⚙️  Phase: {args.phase}")
    
    # Run training based on phase
    if args.phase in ['1', 'all', 'vqvae']:
        train_vqvae(num_epochs=args.epochs)
    
    if args.phase in ['1', 'all', 'transformer']:
        vqvae_path = os.path.join(DataConfig.CHECKPOINT_DIR, 'vqvae_best.pth')
        train_transformer(vqvae_path=vqvae_path, num_epochs=args.epochs)
    
    if args.phase in ['1', 'all', 'ddpm']:
        train_ddpm(num_epochs=args.epochs)
    
    if args.phase in ['2', 'all', 'cnn_vit']:
        train_cnn_vit(num_epochs=args.epochs)
    
    print("\n" + "=" * 70)
    print("✅ HOÀN THÀNH!")
    print("=" * 70)


if __name__ == "__main__":
    main()
