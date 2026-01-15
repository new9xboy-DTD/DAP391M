"""
=====================================================================
MÔ-ĐUN VQ-VAE (VECTOR QUANTIZED VARIATIONAL AUTOENCODER)
=====================================================================
Mô tả:
    VQ-VAE chuyển đổi ảnh thành các token rời rạc, tương tự như 
    tokenization trong NLP. Điều này cho phép:
    1. Nén ảnh thành biểu diễn compact
    2. Sử dụng các mô hình ngôn ngữ (như GPT) để học phân phối ảnh
    3. Phát hiện deepfake dựa trên xác suất của chuỗi token
    
    Kiến trúc:
    - Encoder: Nén ảnh thành latent space liên tục
    - Vector Quantization: Chuyển latent sang token rời rạc
    - Decoder: Tái tạo ảnh từ token
    
Tham khảo:
    - Van Den Oord et al., "Neural Discrete Representation Learning", NeurIPS 2017
    
Tác giả: DAP391M Team
Phiên bản: 1.0
=====================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Import config
from config import VQVAEConfig, DataConfig


# =====================================================================
# RESIDUAL BLOCK
# =====================================================================

class ResidualBlock(nn.Module):
    """
    Residual Block với skip connection
    
    Cấu trúc:
        input -> Conv -> ReLU -> Conv -> + input -> output
        
    Skip connection giúp gradient flow tốt hơn và
    cho phép model học các biến đổi residual thay vì mapping trực tiếp.
    
    Args:
        in_channels: Số kênh đầu vào
        out_channels: Số kênh đầu ra
    """
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # Block chính với 2 convolution layers
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # Skip connection với 1x1 conv nếu số kênh thay đổi
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x):
        """
        Forward pass với skip connection
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Output tensor (B, out_channels, H, W)
        """
        return F.relu(self.block(x) + self.skip(x))


# =====================================================================
# ENCODER
# =====================================================================

class Encoder(nn.Module):
    """
    Encoder của VQ-VAE
    
    Nén ảnh từ (B, 3, 256, 256) xuống (B, embedding_dim, H', W')
    với H', W' phụ thuộc vào số lần downsampling.
    
    Với hidden_dims=[64, 128, 256, 512], ta có 4 lần downsampling (stride=2)
    Ảnh 256x256 -> 128 -> 64 -> 32 -> 16 (factor = 16)
    
    Args:
        in_channels: Số kênh đầu vào (3 cho RGB)
        hidden_dims: List các hidden dimensions
        embedding_dim: Kích thước embedding cuối cùng
    """
    
    def __init__(self, in_channels=3, hidden_dims=None, embedding_dim=64):
        super().__init__()
        
        # Sử dụng default hidden dims nếu không được cung cấp
        if hidden_dims is None:
            hidden_dims = VQVAEConfig.HIDDEN_DIMS.copy()
        
        # Xây dựng encoder từ các blocks
        modules = []
        
        # Initial convolution
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, hidden_dims[0], kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(hidden_dims[0]),
                nn.ReLU(inplace=True)
            )
        )
        
        # Downsampling blocks
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    # Downsample với stride=2
                    nn.Conv2d(hidden_dims[i], hidden_dims[i+1], kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(hidden_dims[i+1]),
                    nn.ReLU(inplace=True),
                    # Residual block để tăng expressiveness
                    ResidualBlock(hidden_dims[i+1], hidden_dims[i+1])
                )
            )
        
        # Final projection to embedding dimension
        modules.append(
            nn.Sequential(
                nn.Conv2d(hidden_dims[-1], embedding_dim, kernel_size=1),
                nn.BatchNorm2d(embedding_dim)
            )
        )
        
        self.encoder = nn.Sequential(*modules)
        
        # Tính reduction factor để biết output size
        self.reduction_factor = 2 ** (len(hidden_dims) - 1)
        print(f"📐 Encoder reduction factor: {self.reduction_factor}x")
    
    def forward(self, x):
        """
        Encode ảnh thành latent representation
        
        Args:
            x: Input images (B, 3, H, W)
            
        Returns:
            Encoded features (B, embedding_dim, H/reduction, W/reduction)
        """
        return self.encoder(x)


# =====================================================================
# DECODER
# =====================================================================

class Decoder(nn.Module):
    """
    Decoder của VQ-VAE
    
    Tái tạo ảnh từ latent representation.
    Là quá trình ngược lại của Encoder (upsampling thay vì downsampling).
    
    Args:
        out_channels: Số kênh đầu ra (3 cho RGB)
        hidden_dims: List các hidden dimensions (ngược lại với encoder)
        embedding_dim: Kích thước embedding đầu vào
    """
    
    def __init__(self, out_channels=3, hidden_dims=None, embedding_dim=64):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = VQVAEConfig.HIDDEN_DIMS.copy()
        
        # Đảo ngược thứ tự hidden dims cho decoder
        hidden_dims = hidden_dims[::-1]
        
        # Xây dựng decoder từ các blocks
        modules = []
        
        # Initial projection from embedding
        modules.append(
            nn.Sequential(
                nn.Conv2d(embedding_dim, hidden_dims[0], kernel_size=1),
                nn.BatchNorm2d(hidden_dims[0]),
                nn.ReLU(inplace=True)
            )
        )
        
        # Upsampling blocks
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    # Residual block
                    ResidualBlock(hidden_dims[i], hidden_dims[i]),
                    # Upsample với transposed conv (stride=2)
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i+1], 
                                      kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(hidden_dims[i+1]),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Final convolution to output channels
        modules.append(
            nn.Sequential(
                nn.Conv2d(hidden_dims[-1], out_channels, kernel_size=3, padding=1),
                nn.Tanh()  # Output trong [-1, 1] (phù hợp với input normalization)
            )
        )
        
        self.decoder = nn.Sequential(*modules)
    
    def forward(self, x):
        """
        Decode latent representation thành ảnh
        
        Args:
            x: Latent features (B, embedding_dim, H', W')
            
        Returns:
            Reconstructed images (B, 3, H, W)
        """
        return self.decoder(x)


# =====================================================================
# VECTOR QUANTIZER
# =====================================================================

class VectorQuantizer(nn.Module):
    """
    Vector Quantization layer
    
    Chuyển đổi continuous latent vectors thành discrete tokens
    bằng cách tìm embedding gần nhất trong codebook.
    
    Codebook là một tập các embedding vectors (như vocabulary trong NLP).
    Mỗi position trong latent space được map sang một token ID.
    
    Args:
        num_embeddings: Số lượng embeddings trong codebook (vocabulary size)
        embedding_dim: Kích thước mỗi embedding vector
        commitment_cost: Hệ số commitment loss (β trong paper)
    """
    
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        
        self.num_embeddings = num_embeddings  # K trong paper
        self.embedding_dim = embedding_dim    # D trong paper
        self.commitment_cost = commitment_cost  # β
        
        # Codebook - bảng tra cứu embeddings
        # Shape: (num_embeddings, embedding_dim)
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        
        # Khởi tạo embeddings với uniform distribution
        self.embedding.weight.data.uniform_(
            -1.0 / num_embeddings,
            1.0 / num_embeddings
        )
        
        print(f"📚 Codebook size: {num_embeddings} x {embedding_dim}")
    
    def forward(self, z):
        """
        Quantize latent vectors
        
        Args:
            z: Continuous latent tensor (B, D, H, W)
            
        Returns:
            tuple: (quantized, vq_loss, perplexity, encoding_indices)
            - quantized: Quantized tensor (same shape as z)
            - vq_loss: Vector quantization loss
            - perplexity: Độ đa dạng của codebook sử dụng
            - encoding_indices: Token IDs (B, H*W)
        """
        # Chuyển từ (B, D, H, W) sang (B, H, W, D) rồi flatten thành (B*H*W, D)
        # để tính khoảng cách với codebook
        z = z.permute(0, 2, 3, 1).contiguous()  # (B, H, W, D)
        z_shape = z.shape
        z_flat = z.view(-1, self.embedding_dim)  # (B*H*W, D)
        
        # Tính khoảng cách L2 giữa z và tất cả embeddings trong codebook
        # distances[i, j] = ||z_i - e_j||^2
        # = ||z_i||^2 + ||e_j||^2 - 2 * z_i . e_j
        distances = (
            torch.sum(z_flat ** 2, dim=1, keepdim=True)  # ||z||^2
            + torch.sum(self.embedding.weight ** 2, dim=1)  # ||e||^2
            - 2 * torch.matmul(z_flat, self.embedding.weight.t())  # -2 * z.e
        )
        
        # Tìm embedding gần nhất cho mỗi position
        encoding_indices = torch.argmin(distances, dim=1)  # (B*H*W,)
        
        # Lấy quantized vectors từ codebook
        quantized_flat = self.embedding(encoding_indices)  # (B*H*W, D)
        
        # Reshape lại về (B, H, W, D) rồi (B, D, H, W)
        quantized = quantized_flat.view(z_shape)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()  # (B, D, H, W)
        
        # Tính VQ loss
        # 1. Codebook loss: ||sg[z] - e||^2 (cập nhật codebook)
        # 2. Commitment loss: ||z - sg[e]||^2 (giữ encoder output gần codebook)
        # sg = stop gradient
        z_original = z.permute(0, 3, 1, 2).contiguous()  # Quay lại (B, D, H, W)
        
        codebook_loss = F.mse_loss(quantized.detach(), z_original)
        commitment_loss = F.mse_loss(quantized, z_original.detach())
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss
        
        # Straight-through estimator
        # Forward: dùng quantized
        # Backward: gradient chảy thẳng qua như identity
        quantized = z_original + (quantized - z_original).detach()
        
        # Tính perplexity (đo độ đa dạng sử dụng codebook)
        # Perplexity cao = sử dụng nhiều embeddings khác nhau = tốt
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # Reshape indices về (B, H*W) để dùng cho Transformer
        batch_size = z_shape[0]
        spatial_size = z_shape[1] * z_shape[2]  # H * W
        encoding_indices = encoding_indices.view(batch_size, spatial_size)
        
        return quantized, vq_loss, perplexity, encoding_indices


class VectorQuantizerEMA(nn.Module):
    """
    Vector Quantization với Exponential Moving Average update
    
    Thay vì dùng gradient descent để cập nhật codebook,
    EMA update thường ổn định hơn và không cần tuning learning rate riêng.
    
    Codebook được cập nhật bằng:
        e_new = decay * e_old + (1 - decay) * mean(z_assigned)
        
    Args:
        num_embeddings: Số lượng embeddings
        embedding_dim: Kích thước embedding
        commitment_cost: Hệ số commitment loss
        decay: Hệ số decay cho EMA (0.99 là typical)
    """
    
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        
        # Codebook embeddings
        embedding = torch.randn(num_embeddings, embedding_dim)
        self.register_buffer('embedding', embedding)
        
        # EMA cluster size (số lượng vectors được assign cho mỗi embedding)
        self.register_buffer('cluster_size', torch.zeros(num_embeddings))
        
        # EMA embedding sum (tổng các vectors được assign)
        self.register_buffer('embedding_avg', embedding.clone())
        
        print(f"📚 EMA Codebook size: {num_embeddings} x {embedding_dim}")
    
    def forward(self, z):
        """
        Quantize với EMA update
        
        Args:
            z: Continuous latent tensor (B, D, H, W)
            
        Returns:
            tuple: (quantized, vq_loss, perplexity, encoding_indices)
        """
        z = z.permute(0, 2, 3, 1).contiguous()
        z_shape = z.shape
        z_flat = z.view(-1, self.embedding_dim)
        
        # Tính khoảng cách và tìm nearest embedding
        distances = (
            torch.sum(z_flat ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding ** 2, dim=1)
            - 2 * torch.matmul(z_flat, self.embedding.t())
        )
        
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
        
        # Lấy quantized vectors
        quantized_flat = F.embedding(encoding_indices, self.embedding)
        quantized = quantized_flat.view(z_shape)
        
        # EMA update (chỉ trong training)
        if self.training:
            # Cập nhật cluster size
            self.cluster_size.data = (
                self.decay * self.cluster_size
                + (1 - self.decay) * torch.sum(encodings, dim=0)
            )
            
            # Cập nhật embedding sum
            dw = torch.matmul(encodings.t(), z_flat)
            self.embedding_avg.data = (
                self.decay * self.embedding_avg
                + (1 - self.decay) * dw
            )
            
            # Normalize để có embeddings mới
            n = torch.sum(self.cluster_size)
            cluster_size = (
                (self.cluster_size + 1e-5)
                / (n + self.num_embeddings * 1e-5) * n
            )
            self.embedding.data = self.embedding_avg / cluster_size.unsqueeze(1)
        
        # Commitment loss only (codebook loss không cần vì dùng EMA)
        z_original = z.permute(0, 3, 1, 2).contiguous()
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        
        commitment_loss = self.commitment_cost * F.mse_loss(quantized.detach(), z_original)
        
        # Straight-through estimator
        quantized = z_original + (quantized - z_original).detach()
        
        # Perplexity
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # Reshape indices
        batch_size = z_shape[0]
        spatial_size = z_shape[1] * z_shape[2]
        encoding_indices = encoding_indices.view(batch_size, spatial_size)
        
        return quantized, commitment_loss, perplexity, encoding_indices


# =====================================================================
# VQ-VAE MODEL
# =====================================================================

class VQVAE(nn.Module):
    """
    Complete VQ-VAE Model
    
    Kết hợp Encoder, Vector Quantizer và Decoder thành một model hoàn chỉnh.
    
    Flow:
        input image -> Encoder -> continuous latent -> VQ -> discrete tokens -> Decoder -> reconstructed image
        
    Args:
        in_channels: Số kênh input (3 cho RGB)
        hidden_dims: List hidden dimensions cho Encoder/Decoder
        embedding_dim: Kích thước embedding
        num_embeddings: Số lượng embeddings trong codebook
        commitment_cost: Hệ số commitment loss
        use_ema: Sử dụng EMA update cho codebook
        decay: Decay rate cho EMA
    """
    
    def __init__(
        self,
        in_channels=DataConfig.IMG_CHANNELS,
        hidden_dims=None,
        embedding_dim=VQVAEConfig.EMBEDDING_DIM,
        num_embeddings=VQVAEConfig.NUM_EMBEDDINGS,
        commitment_cost=VQVAEConfig.COMMITMENT_COST,
        use_ema=True,
        decay=VQVAEConfig.DECAY
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = VQVAEConfig.HIDDEN_DIMS.copy()
        
        print("\n🏗️  Đang xây dựng VQ-VAE...")
        
        # Encoder
        self.encoder = Encoder(
            in_channels=in_channels,
            hidden_dims=hidden_dims,
            embedding_dim=embedding_dim
        )
        
        # Vector Quantizer
        if use_ema:
            self.vq = VectorQuantizerEMA(
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
                commitment_cost=commitment_cost,
                decay=decay
            )
        else:
            self.vq = VectorQuantizer(
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
                commitment_cost=commitment_cost
            )
        
        # Decoder
        self.decoder = Decoder(
            out_channels=in_channels,
            hidden_dims=hidden_dims,
            embedding_dim=embedding_dim
        )
        
        # Lưu config
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.reduction_factor = self.encoder.reduction_factor
        
        # In thông tin model
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"✅ VQ-VAE khởi tạo thành công!")
        print(f"📊 Tổng parameters: {total_params:,}")
        print(f"📊 Trainable parameters: {trainable_params:,}")
    
    def get_codebook(self):
        """
        Lấy codebook embeddings từ VQ module
        
        Helper method để tránh duplicate logic khi truy cập codebook.
        
        Returns:
            Tensor: Codebook embeddings (num_embeddings, embedding_dim)
        """
        if isinstance(self.vq, VectorQuantizerEMA):
            return self.vq.embedding
        else:
            return self.vq.embedding.weight
    
    def encode(self, x):
        """
        Encode ảnh thành discrete tokens
        
        Args:
            x: Input images (B, 3, H, W)
            
        Returns:
            tuple: (quantized, token_indices)
            - quantized: Quantized latent (B, D, H', W')
            - token_indices: Token IDs (B, H'*W')
        """
        z = self.encoder(x)
        quantized, _, _, indices = self.vq(z)
        return quantized, indices
    
    def decode(self, quantized):
        """
        Decode quantized latent thành ảnh
        
        Args:
            quantized: Quantized latent (B, D, H', W')
            
        Returns:
            Reconstructed images (B, 3, H, W)
        """
        return self.decoder(quantized)
    
    def decode_tokens(self, token_indices, spatial_shape):
        """
        Decode trực tiếp từ token indices
        
        Args:
            token_indices: Token IDs (B, seq_len)
            spatial_shape: (H', W') shape của latent
            
        Returns:
            Reconstructed images (B, 3, H, W)
        """
        batch_size = token_indices.shape[0]
        h, w = spatial_shape
        
        # Lấy embeddings từ codebook sử dụng helper method
        codebook = self.get_codebook()
        
        quantized = F.embedding(token_indices, codebook)  # (B, seq_len, D)
        
        # Reshape về spatial shape
        quantized = quantized.view(batch_size, h, w, self.embedding_dim)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()  # (B, D, H, W)
        
        return self.decoder(quantized)
    
    def forward(self, x):
        """
        Forward pass đầy đủ
        
        Args:
            x: Input images (B, 3, H, W)
            
        Returns:
            tuple: (recon, vq_loss, perplexity, indices)
            - recon: Reconstructed images
            - vq_loss: Vector quantization loss
            - perplexity: Codebook perplexity
            - indices: Token indices
        """
        # Encode
        z = self.encoder(x)
        
        # Quantize
        quantized, vq_loss, perplexity, indices = self.vq(z)
        
        # Decode
        recon = self.decoder(quantized)
        
        return recon, vq_loss, perplexity, indices
    
    def get_codebook_usage(self, indices):
        """
        Tính thống kê sử dụng codebook
        
        Args:
            indices: Token indices từ một batch (B, seq_len)
            
        Returns:
            dict: Thống kê bao gồm số tokens được sử dụng, histogram, etc.
        """
        indices_flat = indices.view(-1)
        unique_tokens = torch.unique(indices_flat).numel()
        
        # Histogram
        histogram = torch.bincount(indices_flat, minlength=self.num_embeddings)
        
        return {
            'unique_tokens': unique_tokens,
            'usage_ratio': unique_tokens / self.num_embeddings,
            'histogram': histogram.cpu().numpy()
        }


# =====================================================================
# LOSS FUNCTIONS
# =====================================================================

def vqvae_loss(recon, target, vq_loss, recon_weight=1.0, vq_weight=1.0):
    """
    Tính tổng loss cho VQ-VAE
    
    Loss = recon_loss + vq_loss
    
    Trong đó:
    - recon_loss: MSE hoặc L1 giữa ảnh gốc và ảnh tái tạo
    - vq_loss: Vector quantization loss (codebook + commitment)
    
    Args:
        recon: Reconstructed images (B, 3, H, W)
        target: Original images (B, 3, H, W)
        vq_loss: VQ loss từ model
        recon_weight: Trọng số cho reconstruction loss
        vq_weight: Trọng số cho VQ loss
        
    Returns:
        tuple: (total_loss, loss_dict)
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon, target)
    
    # Tổng loss
    total_loss = recon_weight * recon_loss + vq_weight * vq_loss
    
    return total_loss, {
        'total': total_loss.item(),
        'recon': recon_loss.item(),
        'vq': vq_loss.item()
    }


# =====================================================================
# MAIN (Test)
# =====================================================================

if __name__ == "__main__":
    """
    Test VQ-VAE model khi chạy file này trực tiếp
    """
    print("=" * 70)
    print("🧪 TEST VQ-VAE MODULE")
    print("=" * 70)
    
    # Tạo model
    print("\n1. Khởi tạo VQ-VAE model...")
    model = VQVAE()
    
    # Test với random input
    print("\n2. Test forward pass...")
    batch_size = 2
    x = torch.randn(batch_size, 3, DataConfig.IMG_SIZE, DataConfig.IMG_SIZE)
    print(f"   Input shape: {x.shape}")
    
    # Forward pass
    recon, vq_loss, perplexity, indices = model(x)
    
    print(f"   Output (recon) shape: {recon.shape}")
    print(f"   VQ loss: {vq_loss.item():.4f}")
    print(f"   Perplexity: {perplexity.item():.2f}")
    print(f"   Token indices shape: {indices.shape}")
    
    # Test encode/decode separately
    print("\n3. Test encode/decode riêng...")
    quantized, indices = model.encode(x)
    print(f"   Quantized shape: {quantized.shape}")
    print(f"   Indices shape: {indices.shape}")
    
    decoded = model.decode(quantized)
    print(f"   Decoded shape: {decoded.shape}")
    
    # Test codebook usage
    print("\n4. Kiểm tra codebook usage...")
    usage = model.get_codebook_usage(indices)
    print(f"   Unique tokens used: {usage['unique_tokens']}/{model.num_embeddings}")
    print(f"   Usage ratio: {usage['usage_ratio']:.2%}")
    
    print("\n" + "=" * 70)
    print("✅ HOÀN THÀNH TEST VQ-VAE!")
    print("=" * 70)
