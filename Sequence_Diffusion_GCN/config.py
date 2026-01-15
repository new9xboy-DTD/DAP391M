"""
=====================================================================
CẤU HÌNH HỆ THỐNG NHẬN DIỆN DEEPFAKE
=====================================================================
Mô tả:
    File này chứa tất cả cấu hình và hyperparameters cho hệ thống nhận
    diện deepfake đa mô-đun bao gồm:
    - VQ-VAE + Transformer (Sequence Modeling)
    - DDPM (Back-in-Time Diffusion)
    - CNN/ViT + GCN (Feature Extraction + Graph Analysis)
    
Tác giả: DAP391M Team
Phiên bản: 1.0
=====================================================================
"""

import os
import torch


class DataConfig:
    """
    Cấu hình liên quan đến dữ liệu và đường dẫn
    
    Lưu ý: Đường dẫn DATASET_ROOT có thể được thay đổi tùy theo vị trí 
    dataset của bạn. Cấu trúc thư mục yêu cầu:
        DATASET_ROOT/
        ├── Train/
        │   ├── Fake/
        │   └── Real/
        ├── Validation/
        │   ├── Fake/
        │   └── Real/
        └── Test/
            ├── Fake/
            └── Real/
    """
    
    # ================= ĐƯỜNG DẪN DỮ LIỆU =================
    # Thay đổi đường dẫn này để trỏ đến dataset của bạn
    DATASET_ROOT = "Dataset"  # Sẽ được cập nhật sau khi có dataset
    
    # Các thư mục con
    TRAIN_DIR = os.path.join(DATASET_ROOT, "Train")
    VAL_DIR = os.path.join(DATASET_ROOT, "Validation")  
    TEST_DIR = os.path.join(DATASET_ROOT, "Test")
    
    # Thư mục lưu model và kết quả
    CHECKPOINT_DIR = "checkpoints"
    LOG_DIR = "logs"
    RESULTS_DIR = "results"
    
    # ================= THAM SỐ ẢNH =================
    IMG_SIZE = 256  # Kích thước ảnh đầu vào (256x256)
    IMG_CHANNELS = 3  # Số kênh màu (RGB)
    
    # ================= THAM SỐ DATA LOADING =================
    BATCH_SIZE = 16  # Kích thước batch (giảm nếu thiếu VRAM)
    NUM_WORKERS = 4  # Số worker để load dữ liệu song song


class VQVAEConfig:
    """
    Cấu hình cho mô-đun VQ-VAE (Vector Quantized Variational AutoEncoder)
    
    VQ-VAE chịu trách nhiệm:
    1. Mã hóa ảnh thành các vector liên tục
    2. Lượng tử hóa (quantize) các vector thành token rời rạc
    3. Giải mã token trở lại ảnh
    
    Điều này cho phép ta xử lý ảnh như một "câu" gồm các "từ" (token),
    tương tự như trong xử lý ngôn ngữ tự nhiên (NLP).
    """
    
    # ================= KIẾN TRÚC ENCODER/DECODER =================
    # Số kênh ẩn trong các lớp convolution
    HIDDEN_DIMS = [64, 128, 256, 512]
    
    # Kích thước của mỗi embedding vector trong codebook
    EMBEDDING_DIM = 64
    
    # Số lượng embedding vectors trong codebook (vocabulary size)
    # Càng nhiều thì biểu diễn càng chi tiết nhưng khó học hơn
    NUM_EMBEDDINGS = 512
    
    # ================= THAM SỐ QUANTIZATION =================
    # Hệ số commitment loss - cân bằng giữa reconstruction và codebook
    COMMITMENT_COST = 0.25
    
    # Decay rate cho EMA update của codebook (0 = không dùng EMA)
    DECAY = 0.99
    
    # ================= THAM SỐ HUẤN LUYỆN =================
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 100
    
    # Hệ số các thành phần loss
    RECON_LOSS_WEIGHT = 1.0  # Trọng số reconstruction loss
    VQ_LOSS_WEIGHT = 1.0  # Trọng số vector quantization loss


class TransformerConfig:
    """
    Cấu hình cho mô-đun Transformer tự quy hồi (GPT-like)
    
    Transformer này học phân phối xác suất của các token ảnh từ VQ-VAE.
    Khi suy luận, ta tính perplexity của một ảnh - nếu perplexity cao,
    ảnh đó có khả năng là deepfake (không phù hợp phân phối ảnh thật).
    
    Cơ chế hoạt động:
    1. Nhận chuỗi token từ VQ-VAE
    2. Dự đoán token tiếp theo dựa trên các token trước đó
    3. Tính xác suất/perplexity của toàn bộ chuỗi
    """
    
    # ================= KIẾN TRÚC TRANSFORMER =================
    # Kích thước embedding (phải khớp với EMBEDDING_DIM của VQ-VAE)
    D_MODEL = 512
    
    # Số attention heads
    NHEAD = 8
    
    # Số lớp Transformer decoder
    NUM_LAYERS = 6
    
    # Kích thước feedforward network
    DIM_FEEDFORWARD = 2048
    
    # Dropout rate để regularization
    DROPOUT = 0.1
    
    # ================= THAM SỐ SEQUENCE =================
    # Độ dài tối đa của chuỗi token
    # Với ảnh 256x256 và downsampling 4x của VQ-VAE, ta có 64x64 = 4096 tokens
    # Có thể giảm bằng cách tăng downsampling hoặc dùng patch
    MAX_SEQ_LEN = 1024  # Giới hạn để tiết kiệm bộ nhớ
    
    # Kích thước vocabulary (phải khớp với NUM_EMBEDDINGS của VQ-VAE)
    VOCAB_SIZE = VQVAEConfig.NUM_EMBEDDINGS
    
    # ================= THAM SỐ HUẤN LUYỆN =================
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 50
    
    # Warmup steps cho learning rate scheduler
    WARMUP_STEPS = 4000
    
    # Label smoothing để cải thiện generalization
    LABEL_SMOOTHING = 0.1


class DiffusionConfig:
    """
    Cấu hình cho mô-đun DDPM (Denoising Diffusion Probabilistic Model)
    
    Ý tưởng "Back-in-Time Diffusion":
    1. Huấn luyện DDPM CHỈ trên ảnh thật
    2. Model học cách khử nhiễu (denoise) ảnh thật
    3. Khi test, cho ảnh qua một bước khử nhiễu
    4. Tính sai số giữa ảnh gốc và ảnh đã khử nhiễu
    
    Ảnh deepfake thường có "dấu vết nhân tạo" mà model không học được,
    nên sai số sẽ lớn hơn so với ảnh thật.
    
    Tham khảo: Grabovski et al. (2024)
    """
    
    # ================= THAM SỐ DIFFUSION =================
    # Số bước diffusion (T trong công thức)
    NUM_TIMESTEPS = 1000
    
    # Schedule cho beta (variance schedule)
    # 'linear': β tăng tuyến tính
    # 'cosine': β theo hàm cosine (thường tốt hơn)
    BETA_SCHEDULE = 'cosine'
    
    # Giá trị min và max của beta (cho linear schedule)
    BETA_START = 1e-4
    BETA_END = 0.02
    
    # ================= KIẾN TRÚC U-NET =================
    # Số kênh cơ bản của U-Net
    BASE_CHANNELS = 64
    
    # Bội số kênh ở mỗi level (xuống/lên)
    CHANNEL_MULTS = [1, 2, 4, 8]
    
    # Số attention heads trong các lớp attention
    NUM_HEADS = 4
    
    # Số ResNet blocks ở mỗi level
    NUM_RES_BLOCKS = 2
    
    # Dropout rate
    DROPOUT = 0.1
    
    # ================= THAM SỐ HUẤN LUYỆN =================
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 100
    
    # EMA decay cho model (giúp ổn định)
    EMA_DECAY = 0.9999
    
    # ================= THAM SỐ ANOMALY DETECTION =================
    # Số bước reverse diffusion khi tính anomaly score
    # Không cần chạy toàn bộ T bước, chỉ cần một số bước nhỏ
    ANOMALY_TIMESTEPS = 100
    
    # Loại anomaly metric
    # 'mse': Mean Squared Error
    # 'lpips': Learned Perceptual Image Patch Similarity
    ANOMALY_METRIC = 'mse'


class CNNViTConfig:
    """
    Cấu hình cho mô-đun CNN/ViT trích xuất đặc trưng
    
    Sử dụng các model pretrained mạnh như:
    - EfficientNet, ResNet (CNN)
    - ViT, Swin Transformer (Vision Transformer)
    - CLIP, DINO (Self-supervised pretrained)
    
    Các đặc trưng từ model này sẽ được:
    1. Kết hợp với GCN để phân tích cấu trúc khuôn mặt
    2. Đưa qua classifier để phân loại trực tiếp
    """
    
    # ================= LỰA CHỌN BACKBONE =================
    # Tên model từ thư viện timm
    # Các lựa chọn phổ biến:
    # - 'efficientnet_b0', 'efficientnet_b4'
    # - 'resnet50', 'resnet101'
    # - 'vit_base_patch16_224', 'vit_large_patch16_224'
    # - 'swin_base_patch4_window7_224'
    BACKBONE_NAME = 'efficientnet_b4'
    
    # Sử dụng pretrained weights
    PRETRAINED = True
    
    # Freeze backbone (không train) trong giai đoạn đầu
    FREEZE_BACKBONE = False
    
    # ================= FEATURE EXTRACTION =================
    # Lấy features từ layer nào
    # 'last': Chỉ lấy features cuối cùng
    # 'multi': Lấy features từ nhiều levels (multi-scale)
    FEATURE_LEVEL = 'last'
    
    # Kích thước output feature vector
    FEATURE_DIM = 512
    
    # ================= CLASSIFIER HEAD =================
    # Số hidden units trong MLP classifier
    CLASSIFIER_HIDDEN = [512, 256]
    
    # Dropout trong classifier
    CLASSIFIER_DROPOUT = 0.3
    
    # ================= THAM SỐ HUẤN LUYỆN =================
    LEARNING_RATE = 1e-4
    
    # Learning rate cho backbone (thường nhỏ hơn)
    BACKBONE_LR = 1e-5
    
    NUM_EPOCHS = 50


class GCNConfig:
    """
    Cấu hình cho mô-đun GCN (Graph Convolutional Network)
    
    GCN xử lý facial landmarks như một đồ thị:
    - Nodes: Các điểm mốc trên khuôn mặt (landmarks)
    - Edges: Mối quan hệ/khoảng cách giữa các landmarks
    
    Deepfake thường có bất thường trong cấu trúc khuôn mặt
    mà GCN có thể phát hiện được.
    
    Tham khảo: Samad & Bandhu (2025)
    """
    
    # ================= FACIAL LANDMARKS =================
    # Số điểm landmark sử dụng
    # 68: Dlib standard landmarks
    # 468: MediaPipe face mesh
    NUM_LANDMARKS = 68
    
    # Phương pháp detect landmarks
    # 'dlib': Dlib shape predictor
    # 'mediapipe': Google MediaPipe
    LANDMARK_DETECTOR = 'dlib'
    
    # ================= KIẾN TRÚC GCN =================
    # Kích thước đặc trưng đầu vào cho mỗi node
    # Thường là tọa độ (x, y) hoặc (x, y, z) + các đặc trưng bổ sung
    INPUT_DIM = 2
    
    # Các hidden dimensions của GCN layers
    HIDDEN_DIMS = [64, 128, 256]
    
    # Kích thước output features
    OUTPUT_DIM = 256
    
    # Loại GCN layer
    # 'gcn': Standard GCN (Kipf & Welling)
    # 'gat': Graph Attention Network
    # 'sage': GraphSAGE
    GCN_TYPE = 'gat'
    
    # Số attention heads (cho GAT)
    NUM_HEADS = 4
    
    # Dropout rate
    DROPOUT = 0.3
    
    # ================= THAM SỐ HUẤN LUYỆN =================
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 50


class FusionConfig:
    """
    Cấu hình cho mô-đun Fusion (Kết hợp các scores)
    
    Fusion module kết hợp các anomaly scores và features từ:
    1. Transformer perplexity score
    2. Diffusion anomaly score
    3. CNN/ViT features
    4. GCN structural features
    
    Để ra quyết định cuối cùng: Ảnh là Real hay Fake
    """
    
    # ================= PHƯƠNG PHÁP FUSION =================
    # Cách kết hợp các scores:
    # 'weighted_sum': Tổng có trọng số
    # 'mlp': Qua một MLP nhỏ
    # 'attention': Attention-based fusion
    FUSION_METHOD = 'mlp'
    
    # Trọng số cho mỗi module (dùng cho weighted_sum)
    # Tổng các trọng số nên = 1.0
    WEIGHTS = {
        'transformer': 0.25,  # Perplexity score
        'diffusion': 0.25,    # Diffusion anomaly score
        'cnn_vit': 0.30,      # CNN/ViT classification
        'gcn': 0.20           # GCN structural analysis
    }
    
    # ================= MLP FUSION =================
    # Kích thước các lớp ẩn trong MLP fusion
    MLP_HIDDEN = [256, 128]
    
    # Dropout
    MLP_DROPOUT = 0.3
    
    # ================= NGƯỠNG PHÂN LOẠI =================
    # Ngưỡng để phân loại ảnh là deepfake
    # Nếu anomaly score > threshold -> Fake
    THRESHOLD = 0.5
    
    # ================= THAM SỐ HUẤN LUYỆN =================
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 30


class TrainingConfig:
    """
    Cấu hình chung cho quá trình huấn luyện
    """
    
    # ================= THIẾT BỊ =================
    # Tự động chọn GPU nếu có, ngược lại dùng CPU
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Sử dụng mixed precision training (FP16) để tăng tốc và tiết kiệm VRAM
    USE_AMP = True
    
    # ================= RANDOM SEED =================
    # Đặt seed để reproducibility
    SEED = 42
    
    # ================= EARLY STOPPING =================
    # Số epochs chờ trước khi dừng nếu không cải thiện
    PATIENCE = 10
    
    # ================= LOGGING =================
    # Số bước giữa các lần log
    LOG_INTERVAL = 100
    
    # Sử dụng TensorBoard
    USE_TENSORBOARD = True
    
    # Sử dụng Weights & Biases
    USE_WANDB = False
    WANDB_PROJECT = 'deepfake-detection'
    
    # ================= CHẾ ĐỘ HUẤN LUYỆN =================
    # 'unsupervised': Chỉ dùng ảnh thật (cho Diffusion, Transformer)
    # 'supervised': Dùng cả ảnh thật và giả (cho CNN/ViT, GCN, Fusion)
    # 'hybrid': Kết hợp cả hai
    TRAINING_MODE = 'hybrid'


# ================= HÀM TIỆN ÍCH =================

def create_directories():
    """
    Tạo các thư mục cần thiết nếu chưa tồn tại
    """
    directories = [
        DataConfig.CHECKPOINT_DIR,
        DataConfig.LOG_DIR,
        DataConfig.RESULTS_DIR,
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Đã tạo/kiểm tra thư mục: {directory}")


def print_config():
    """
    In ra toàn bộ cấu hình để kiểm tra
    """
    print("=" * 70)
    print("📋 CẤU HÌNH HỆ THỐNG NHẬN DIỆN DEEPFAKE")
    print("=" * 70)
    
    print("\n🗂️  Data Config:")
    print(f"   - Dataset root: {DataConfig.DATASET_ROOT}")
    print(f"   - Image size: {DataConfig.IMG_SIZE}x{DataConfig.IMG_SIZE}")
    print(f"   - Batch size: {DataConfig.BATCH_SIZE}")
    
    print("\n🎨 VQ-VAE Config:")
    print(f"   - Embedding dim: {VQVAEConfig.EMBEDDING_DIM}")
    print(f"   - Num embeddings: {VQVAEConfig.NUM_EMBEDDINGS}")
    
    print("\n🤖 Transformer Config:")
    print(f"   - D_model: {TransformerConfig.D_MODEL}")
    print(f"   - Num layers: {TransformerConfig.NUM_LAYERS}")
    print(f"   - Max seq len: {TransformerConfig.MAX_SEQ_LEN}")
    
    print("\n🌊 Diffusion Config:")
    print(f"   - Num timesteps: {DiffusionConfig.NUM_TIMESTEPS}")
    print(f"   - Beta schedule: {DiffusionConfig.BETA_SCHEDULE}")
    
    print("\n🖼️  CNN/ViT Config:")
    print(f"   - Backbone: {CNNViTConfig.BACKBONE_NAME}")
    print(f"   - Pretrained: {CNNViTConfig.PRETRAINED}")
    
    print("\n📊 GCN Config:")
    print(f"   - Num landmarks: {GCNConfig.NUM_LANDMARKS}")
    print(f"   - GCN type: {GCNConfig.GCN_TYPE}")
    
    print("\n🔗 Fusion Config:")
    print(f"   - Method: {FusionConfig.FUSION_METHOD}")
    print(f"   - Threshold: {FusionConfig.THRESHOLD}")
    
    print("\n⚙️  Training Config:")
    print(f"   - Device: {TrainingConfig.DEVICE}")
    print(f"   - Mode: {TrainingConfig.TRAINING_MODE}")
    print(f"   - Use AMP: {TrainingConfig.USE_AMP}")
    
    print("=" * 70)


if __name__ == "__main__":
    # Khi chạy file này trực tiếp, in ra cấu hình để kiểm tra
    print_config()
    create_directories()
