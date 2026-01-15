"""
Hệ thống nhận diện ảnh deepfake sử dụng CNN + Transformer
Dataset: 190k ảnh (fake_số.jpg và real_số.jpg)
Cấu trúc: Train/Validation/Test với các thư mục Fake/ và Real/
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os
import json
from datetime import datetime


# ======================== THIẾT LẬP CẤU HÌNH ========================

class Config:
    """Cấu hình các tham số cho quá trình training"""
    
    # Đường dẫn dữ liệu
    DATA_DIR = "Dataset"  # Thư mục chứa dữ liệu
    TRAIN_DIR = os.path.join(DATA_DIR, "Train")
    VAL_DIR = os.path.join(DATA_DIR, "Validation")
    TEST_DIR = os.path.join(DATA_DIR, "Test")
    
    # Tham số model
    IMG_SIZE = 256  # Kích thước ảnh đầu vào (256x256 theo mô tả)
    BATCH_SIZE = 32  # Số ảnh trong mỗi batch
    NUM_EPOCHS = 50  # Số epoch training
    LEARNING_RATE = 0.0001  # Tốc độ học
    NUM_CLASSES = 2  # 2 classes: Fake và Real
    
    # Tham số CNN backbone
    CNN_MODEL = "efficientnet_b0"  # Sử dụng EfficientNet làm CNN backbone
    PRETRAINED = True  # Sử dụng pretrained weights từ ImageNet
    
    # Tham số Transformer
    D_MODEL = 512  # Dimension của embedding trong Transformer
    NHEAD = 8  # Số attention heads
    NUM_TRANSFORMER_LAYERS = 4  # Số lớp Transformer encoder
    DIM_FEEDFORWARD = 2048  # Dimension của feedforward network
    DROPOUT = 0.1  # Dropout rate
    
    # Tham số training
    NUM_WORKERS = 4  # Số workers để load data
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SAVE_DIR = "checkpoints"  # Thư mục lưu model
    LOG_INTERVAL = 100  # Log sau mỗi N batches
    
    # Early stopping
    PATIENCE = 10  # Dừng sớm nếu không cải thiện sau N epochs


# ======================== DATA PREPROCESSING ========================

def get_data_transforms():
    """
    Định nghĩa các phép biến đổi dữ liệu cho training và validation
    
    Returns:
        dict: Dictionary chứa transforms cho train và val/test
    """
    
    # Data augmentation cho training set để tăng tính đa dạng
    train_transforms = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),  # Resize về kích thước chuẩn
        transforms.RandomHorizontalFlip(p=0.5),  # Lật ngang ngẫu nhiên
        transforms.RandomRotation(10),  # Xoay ngẫu nhiên ±10 độ
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Thay đổi màu sắc
        transforms.ToTensor(),  # Chuyển sang tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Chuẩn hóa theo ImageNet
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Không augmentation cho validation/test, chỉ preprocessing cơ bản
    val_transforms = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    return {'train': train_transforms, 'val': val_transforms}


def create_data_loaders():
    """
    Tạo DataLoader cho train, validation và test sets
    
    Returns:
        dict: Dictionary chứa các DataLoader
    """
    transforms_dict = get_data_transforms()
    
    # Tạo datasets từ cấu trúc thư mục (ImageFolder tự động label theo tên thư mục)
    train_dataset = datasets.ImageFolder(
        root=Config.TRAIN_DIR,
        transform=transforms_dict['train']
    )
    
    val_dataset = datasets.ImageFolder(
        root=Config.VAL_DIR,
        transform=transforms_dict['val']
    )
    
    test_dataset = datasets.ImageFolder(
        root=Config.TEST_DIR,
        transform=transforms_dict['val']
    )
    
    # Tạo DataLoaders với batch size và số workers đã định nghĩa
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,  # Shuffle data trong training
        num_workers=Config.NUM_WORKERS,
        pin_memory=True  # Tăng tốc độ transfer data lên GPU
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    print(f"📊 Số lượng ảnh training: {len(train_dataset)}")
    print(f"📊 Số lượng ảnh validation: {len(val_dataset)}")
    print(f"📊 Số lượng ảnh test: {len(test_dataset)}")
    print(f"📊 Class mapping: {train_dataset.class_to_idx}")
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'classes': train_dataset.classes
    }


# ======================== MODEL ARCHITECTURE ========================

class CNNTransformerModel(nn.Module):
    """
    Kiến trúc Hybrid CNN + Transformer cho nhận diện deepfake
    
    Kiến trúc:
    1. CNN Backbone (EfficientNet): Trích xuất features từ ảnh
    2. Feature Projection: Chuyển đổi CNN features sang dimension của Transformer
    3. Positional Encoding: Thêm thông tin vị trí cho các features
    4. Transformer Encoder: Học mối quan hệ giữa các features
    5. Classification Head: Dự đoán Fake/Real
    """
    
    def __init__(self):
        super(CNNTransformerModel, self).__init__()
        
        # 1. CNN Backbone - Sử dụng EfficientNet pretrained
        print(f"🏗️  Khởi tạo CNN backbone: {Config.CNN_MODEL}")
        self.cnn_backbone = timm.create_model(
            Config.CNN_MODEL,
            pretrained=Config.PRETRAINED,
            num_classes=0,  # Không dùng classification head của EfficientNet
            global_pool=''  # Giữ lại spatial features thay vì global pooling
        )
        
        # Lấy số channels output của CNN
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, Config.IMG_SIZE, Config.IMG_SIZE)
            cnn_output = self.cnn_backbone(dummy_input)
            self.cnn_feature_dim = cnn_output.shape[1]  # Channels
            self.spatial_size = cnn_output.shape[2] * cnn_output.shape[3]  # H x W
            print(f"📐 CNN output shape: {cnn_output.shape}")
            print(f"📐 Feature dimension: {self.cnn_feature_dim}, Spatial size: {self.spatial_size}")
        
        # 2. Feature Projection - Chuyển CNN features sang dimension của Transformer
        self.feature_projection = nn.Linear(self.cnn_feature_dim, Config.D_MODEL)
        
        # 3. Positional Encoding - Thêm thông tin vị trí cho các spatial features
        self.positional_encoding = nn.Parameter(
            torch.randn(1, self.spatial_size, Config.D_MODEL)
        )
        
        # 4. Transformer Encoder - Học mối quan hệ global giữa các features
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=Config.D_MODEL,
            nhead=Config.NHEAD,
            dim_feedforward=Config.DIM_FEEDFORWARD,
            dropout=Config.DROPOUT,
            activation='gelu',  # GELU activation thường tốt hơn ReLU
            batch_first=True  # Input shape: (batch, seq, feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=Config.NUM_TRANSFORMER_LAYERS
        )
        
        # 5. Classification Head
        self.classifier = nn.Sequential(
            nn.LayerNorm(Config.D_MODEL),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(Config.D_MODEL, Config.D_MODEL // 2),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(Config.D_MODEL // 2, Config.NUM_CLASSES)
        )
        
        # Class token - Học representation tổng thể của ảnh (giống BERT's [CLS] token)
        self.class_token = nn.Parameter(torch.randn(1, 1, Config.D_MODEL))
        
        print(f"✅ Model khởi tạo thành công!")
        print(f"📊 Tổng số parameters: {sum(p.numel() for p in self.parameters()):,}")
        print(f"📊 Trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
    
    def forward(self, x):
        """
        Forward pass của model
        
        Args:
            x: Input images với shape (batch_size, 3, H, W)
            
        Returns:
            logits: Output predictions với shape (batch_size, num_classes)
        """
        batch_size = x.shape[0]
        
        # 1. Trích xuất features bằng CNN
        # Output shape: (batch_size, channels, H', W')
        cnn_features = self.cnn_backbone(x)
        
        # 2. Reshape features thành sequence
        # (batch_size, channels, H', W') -> (batch_size, H'*W', channels)
        cnn_features = cnn_features.flatten(2).permute(0, 2, 1)
        
        # 3. Project CNN features sang dimension của Transformer
        # (batch_size, seq_len, channels) -> (batch_size, seq_len, d_model)
        features = self.feature_projection(cnn_features)
        
        # 4. Thêm positional encoding
        features = features + self.positional_encoding
        
        # 5. Thêm class token vào đầu sequence
        # Expand class token cho cả batch
        class_tokens = self.class_token.expand(batch_size, -1, -1)
        # Concat: (batch_size, 1 + seq_len, d_model)
        features = torch.cat([class_tokens, features], dim=1)
        
        # 6. Pass qua Transformer Encoder
        # Transformer học mối quan hệ giữa các spatial features
        transformer_output = self.transformer_encoder(features)
        
        # 7. Lấy class token output (token đầu tiên) để classification
        cls_output = transformer_output[:, 0]
        
        # 8. Classification
        logits = self.classifier(cls_output)
        
        return logits


# ======================== TRAINING FUNCTIONS ========================

def train_one_epoch(model, train_loader, criterion, optimizer, epoch):
    """
    Train model trong một epoch
    
    Args:
        model: Model cần train
        train_loader: DataLoader cho training data
        criterion: Loss function
        optimizer: Optimizer
        epoch: Epoch hiện tại
        
    Returns:
        dict: Metrics của epoch (loss, accuracy)
    """
    model.train()  # Chuyển model sang training mode
    
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    # Progress bar cho training
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{Config.NUM_EPOCHS} [TRAIN]")
    
    for batch_idx, (images, labels) in enumerate(pbar):
        # Chuyển data lên GPU/CPU
        images = images.to(Config.DEVICE)
        labels = labels.to(Config.DEVICE)
        
        # Forward pass
        optimizer.zero_grad()  # Reset gradients
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass và optimization
        loss.backward()  # Tính gradients
        optimizer.step()  # Update weights
        
        # Tính predictions
        _, preds = torch.max(outputs, 1)
        
        # Lưu kết quả để tính metrics
        running_loss += loss.item() * images.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        if (batch_idx + 1) % Config.LOG_INTERVAL == 0:
            current_loss = running_loss / ((batch_idx + 1) * Config.BATCH_SIZE)
            pbar.set_postfix({'loss': f'{current_loss:.4f}'})
    
    # Tính metrics cho toàn bộ epoch
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return {
        'loss': epoch_loss,
        'accuracy': epoch_acc
    }


def validate(model, val_loader, criterion, epoch=None):
    """
    Validate model trên validation set
    
    Args:
        model: Model cần validate
        val_loader: DataLoader cho validation data
        criterion: Loss function
        epoch: Epoch hiện tại (optional)
        
    Returns:
        dict: Metrics của validation (loss, accuracy, precision, recall, f1)
    """
    model.eval()  # Chuyển model sang evaluation mode
    
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    # Progress bar cho validation
    desc = f"Epoch {epoch}/{Config.NUM_EPOCHS} [VAL]" if epoch else "VALIDATION"
    pbar = tqdm(val_loader, desc=desc)
    
    with torch.no_grad():  # Không tính gradients trong validation
        for images, labels in pbar:
            # Chuyển data lên GPU/CPU
            images = images.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Tính predictions
            _, preds = torch.max(outputs, 1)
            
            # Lưu kết quả
            running_loss += loss.item() * images.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Tính metrics
    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_precision = precision_score(all_labels, all_preds, average='binary')
    epoch_recall = recall_score(all_labels, all_preds, average='binary')
    epoch_f1 = f1_score(all_labels, all_preds, average='binary')
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        'loss': epoch_loss,
        'accuracy': epoch_acc,
        'precision': epoch_precision,
        'recall': epoch_recall,
        'f1': epoch_f1,
        'confusion_matrix': cm
    }


def save_checkpoint(model, optimizer, epoch, metrics, is_best=False):
    """
    Lưu checkpoint của model
    
    Args:
        model: Model cần lưu
        optimizer: Optimizer state
        epoch: Epoch hiện tại
        metrics: Metrics của epoch
        is_best: Có phải best model hay không
    """
    os.makedirs(Config.SAVE_DIR, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': {
            'img_size': Config.IMG_SIZE,
            'd_model': Config.D_MODEL,
            'nhead': Config.NHEAD,
            'num_layers': Config.NUM_TRANSFORMER_LAYERS,
            'cnn_model': Config.CNN_MODEL
        }
    }
    
    # Lưu checkpoint thường
    checkpoint_path = os.path.join(Config.SAVE_DIR, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f"💾 Đã lưu checkpoint: {checkpoint_path}")
    
    # Lưu best model
    if is_best:
        best_path = os.path.join(Config.SAVE_DIR, 'best_model.pth')
        torch.save(checkpoint, best_path)
        print(f"🏆 Đã lưu best model: {best_path}")


def train_model():
    """
    Hàm chính để train model CNN + Transformer
    """
    print("="*70)
    print("🚀 BẮT ĐẦU TRAINING MODEL NHẬN DIỆN DEEPFAKE")
    print("="*70)
    print(f"📱 Device: {Config.DEVICE}")
    print(f"🖼️  Image size: {Config.IMG_SIZE}x{Config.IMG_SIZE}")
    print(f"📦 Batch size: {Config.BATCH_SIZE}")
    print(f"🔄 Number of epochs: {Config.NUM_EPOCHS}")
    print(f"📚 Learning rate: {Config.LEARNING_RATE}")
    print("="*70)
    
    # 1. Tạo data loaders
    print("\n📂 Đang load dữ liệu...")
    data_loaders = create_data_loaders()
    train_loader = data_loaders['train']
    val_loader = data_loaders['val']
    
    # 2. Khởi tạo model
    print("\n🏗️  Đang khởi tạo model...")
    model = CNNTransformerModel().to(Config.DEVICE)
    
    # 3. Loss function và Optimizer
    # Sử dụng CrossEntropyLoss cho binary classification
    criterion = nn.CrossEntropyLoss()
    
    # Sử dụng AdamW optimizer (Adam với weight decay)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=0.01  # L2 regularization
    )
    
    # Learning rate scheduler - Giảm learning rate khi validation loss không giảm
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,  # Giảm LR xuống 50%
        patience=5,  # Đợi 5 epochs
        verbose=True
    )
    
    # 4. Training loop
    print("\n🎯 Bắt đầu training...\n")
    
    best_val_acc = 0.0
    patience_counter = 0
    training_history = []
    
    for epoch in range(1, Config.NUM_EPOCHS + 1):
        print(f"\n{'='*70}")
        print(f"📅 EPOCH {epoch}/{Config.NUM_EPOCHS}")
        print(f"{'='*70}")
        
        # Train
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, epoch)
        
        # Update learning rate scheduler
        scheduler.step(val_metrics['loss'])
        
        # In metrics
        print(f"\n📊 TRAIN - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
        print(f"📊 VAL   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
        print(f"📊 VAL   - Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}")
        print(f"📊 Confusion Matrix:\n{val_metrics['confusion_matrix']}")
        
        # Lưu history
        history_entry = {
            'epoch': epoch,
            'train': train_metrics,
            'val': val_metrics,
            'lr': optimizer.param_groups[0]['lr']
        }
        training_history.append(history_entry)
        
        # Kiểm tra best model
        is_best = val_metrics['accuracy'] > best_val_acc
        if is_best:
            best_val_acc = val_metrics['accuracy']
            patience_counter = 0
            print(f"🎉 New best model! Validation accuracy: {best_val_acc:.4f}")
        else:
            patience_counter += 1
            print(f"⏳ No improvement. Patience: {patience_counter}/{Config.PATIENCE}")
        
        # Lưu checkpoint
        if epoch % 5 == 0 or is_best:  # Lưu mỗi 5 epochs hoặc khi là best model
            save_checkpoint(model, optimizer, epoch, val_metrics, is_best)
        
        # Early stopping
        if patience_counter >= Config.PATIENCE:
            print(f"\n⚠️  Early stopping triggered! Không cải thiện sau {Config.PATIENCE} epochs.")
            break
    
    # 5. Lưu training history
    history_path = os.path.join(Config.SAVE_DIR, 'training_history.json')
    with open(history_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        for entry in training_history:
            if 'confusion_matrix' in entry['val']:
                entry['val']['confusion_matrix'] = entry['val']['confusion_matrix'].tolist()
        json.dump(training_history, f, indent=2)
    print(f"\n💾 Đã lưu training history: {history_path}")
    
    print("\n" + "="*70)
    print("✅ HOÀN THÀNH TRAINING!")
    print(f"🏆 Best validation accuracy: {best_val_acc:.4f}")
    print("="*70)
    
    return model, training_history


def evaluate_on_test_set(model_path=None):
    """
    Đánh giá model trên test set
    
    Args:
        model_path: Đường dẫn đến model checkpoint (mặc định dùng best model)
    """
    print("\n" + "="*70)
    print("🧪 ĐÁNH GIÁ MODEL TRÊN TEST SET")
    print("="*70)
    
    # Load test data
    data_loaders = create_data_loaders()
    test_loader = data_loaders['test']
    
    # Load model
    if model_path is None:
        model_path = os.path.join(Config.SAVE_DIR, 'best_model.pth')
    
    print(f"\n📂 Đang load model từ: {model_path}")
    checkpoint = torch.load(model_path, map_location=Config.DEVICE)
    
    model = CNNTransformerModel().to(Config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate
    criterion = nn.CrossEntropyLoss()
    test_metrics = validate(model, test_loader, criterion)
    
    # In kết quả
    print(f"\n📊 TEST RESULTS:")
    print(f"   Loss: {test_metrics['loss']:.4f}")
    print(f"   Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"   Precision: {test_metrics['precision']:.4f}")
    print(f"   Recall: {test_metrics['recall']:.4f}")
    print(f"   F1-Score: {test_metrics['f1']:.4f}")
    print(f"\n📊 Confusion Matrix:")
    print(test_metrics['confusion_matrix'])
    
    # Lưu test results
    test_results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_path': model_path,
        'metrics': {
            'loss': float(test_metrics['loss']),
            'accuracy': float(test_metrics['accuracy']),
            'precision': float(test_metrics['precision']),
            'recall': float(test_metrics['recall']),
            'f1': float(test_metrics['f1']),
            'confusion_matrix': test_metrics['confusion_matrix'].tolist()
        }
    }
    
    results_path = os.path.join(Config.SAVE_DIR, 'test_results.json')
    with open(results_path, 'w') as f:
        json.dump(test_results, f, indent=2)
    print(f"\n💾 Đã lưu test results: {results_path}")
    
    print("="*70)


# ======================== MAIN FUNCTION ========================

if __name__ == "__main__":
    """
    Main function - Entry point của chương trình
    
    Để chạy training:
        python deepfake_detection.py
    
    Lưu ý:
    - Dataset phải có cấu trúc: Dataset/Train|Validation|Test/Fake|Real/
    - Cần GPU để training nhanh (có thể dùng CPU nhưng sẽ chậm)
    - Model sẽ được lưu trong thư mục checkpoints/
    """
    
    try:
        # Training model
        model, history = train_model()
        
        # Đánh giá trên test set sau khi training xong
        print("\n" + "="*70)
        response = input("🤔 Bạn có muốn đánh giá model trên test set không? (y/n): ")
        if response.lower() == 'y':
            evaluate_on_test_set()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training bị dừng bởi user!")
    except Exception as e:
        print(f"\n\n❌ Lỗi trong quá trình training: {str(e)}")
        import traceback
        traceback.print_exc()