"""
=====================================================================
MÔ-ĐUN DDPM (DENOISING DIFFUSION PROBABILISTIC MODEL)
=====================================================================
Mô tả:
    DDPM là model học cách khử nhiễu (denoise) ảnh.
    
    Phương pháp "Back-in-Time Diffusion" để phát hiện deepfake:
    1. Huấn luyện DDPM CHỈ trên tập ảnh khuôn mặt THẬT
    2. Model học phân phối của ảnh thật và cách khử nhiễu
    3. Khi kiểm tra ảnh nghi ngờ:
       - Thêm một ít nhiễu vào ảnh
       - Cho model khử nhiễu
       - Tính sai số (residual) giữa ảnh gốc và ảnh đã khử nhiễu
    4. Ảnh deepfake có "dấu vết nhân tạo" mà model không học được,
       nên sai số sẽ lớn hơn ảnh thật
    
    Ưu điểm:
    - Không cần nhãn giả/thật (unsupervised)
    - Chỉ cần ảnh thật để huấn luyện
    - Tổng quát tốt với các deepfake generators mới
    
Tham khảo:
    - Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020
    - Grabovski et al. (2024) - Back-in-Time Diffusion for Deepfake Detection
    
Tác giả: DAP391M Team
Phiên bản: 1.0
=====================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# Import config
from config import DiffusionConfig, DataConfig


# =====================================================================
# NOISE SCHEDULERS
# =====================================================================

def linear_beta_schedule(num_timesteps, beta_start=1e-4, beta_end=0.02):
    """
    Linear noise schedule
    
    β_t tăng tuyến tính từ beta_start đến beta_end
    
    Args:
        num_timesteps: Số bước diffusion T
        beta_start: β_1
        beta_end: β_T
        
    Returns:
        betas: Tensor (T,) chứa các giá trị beta
    """
    return torch.linspace(beta_start, beta_end, num_timesteps)


def cosine_beta_schedule(num_timesteps, s=0.008):
    """
    Cosine noise schedule (Nichol & Dhariwal, 2021)
    
    Schedule này smooth hơn và thường cho kết quả tốt hơn linear.
    Công thức:
        α_t = cos((t/T + s) / (1 + s) * π/2)^2
        β_t = 1 - α_t / α_{t-1}
    
    Args:
        num_timesteps: Số bước diffusion T
        s: Offset để tránh β quá nhỏ ở đầu
        
    Returns:
        betas: Tensor (T,) chứa các giá trị beta
    """
    steps = num_timesteps + 1
    t = torch.linspace(0, num_timesteps, steps) / num_timesteps
    
    # Compute alphas_cumprod
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    
    # Compute betas from alphas_cumprod
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    
    # Clip betas để tránh giá trị quá lớn hoặc nhỏ
    return torch.clip(betas, 0.0001, 0.9999)


def get_beta_schedule(schedule_type, num_timesteps, beta_start=1e-4, beta_end=0.02):
    """
    Lấy beta schedule theo loại
    
    Args:
        schedule_type: 'linear' hoặc 'cosine'
        num_timesteps: Số bước T
        beta_start, beta_end: Cho linear schedule
        
    Returns:
        betas: Tensor (T,)
    """
    if schedule_type == 'linear':
        return linear_beta_schedule(num_timesteps, beta_start, beta_end)
    elif schedule_type == 'cosine':
        return cosine_beta_schedule(num_timesteps)
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")


# =====================================================================
# U-NET COMPONENTS
# =====================================================================

class SinusoidalTimeEmbedding(nn.Module):
    """
    Time embedding cho diffusion models
    
    Chuyển timestep t thành vector embedding sử dụng sinusoidal encoding.
    Tương tự positional encoding trong Transformer.
    
    Args:
        dim: Kích thước embedding output
    """
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, t):
        """
        Args:
            t: Timestep tensor (B,)
            
        Returns:
            Time embedding (B, dim)
        """
        device = t.device
        half_dim = self.dim // 2
        
        # Compute frequencies
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        
        # Compute embeddings
        embeddings = t[:, None] * embeddings[None, :]  # (B, half_dim)
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        
        return embeddings  # (B, dim)


class ResnetBlock(nn.Module):
    """
    ResNet block với time embedding
    
    Cấu trúc:
        x -> GroupNorm -> SiLU -> Conv -> + time_emb -> GroupNorm -> SiLU -> Dropout -> Conv -> + x
    
    Args:
        in_channels: Số kênh đầu vào
        out_channels: Số kênh đầu ra
        time_emb_dim: Kích thước time embedding
        dropout: Dropout rate
    """
    
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        
        # First conv block
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        # Time embedding projection
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        # Second conv block
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x, time_emb):
        """
        Args:
            x: Input (B, C, H, W)
            time_emb: Time embedding (B, time_emb_dim)
            
        Returns:
            Output (B, out_channels, H, W)
        """
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        # Add time embedding
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, :, None, None]  # Broadcast to spatial dims
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """
    Self-attention block cho U-Net
    
    Sử dụng trong các resolution thấp để capture long-range dependencies.
    
    Args:
        channels: Số kênh
        num_heads: Số attention heads
    """
    
    def __init__(self, channels, num_heads=4):
        super().__init__()
        
        self.channels = channels
        self.num_heads = num_heads
        
        # Normalization
        self.norm = nn.GroupNorm(8, channels)
        
        # QKV projection
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        
        # Output projection
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)
    
    def forward(self, x):
        """
        Args:
            x: Input (B, C, H, W)
            
        Returns:
            Output (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # Normalize
        h = self.norm(x)
        
        # Get Q, K, V
        qkv = self.qkv(h)
        qkv = qkv.reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # Each: (B, heads, head_dim, H*W)
        
        # Transpose for attention
        q = q.transpose(-1, -2)  # (B, heads, H*W, head_dim)
        
        # Attention scores
        scale = (C // self.num_heads) ** -0.5
        attn = torch.matmul(q, k) * scale  # (B, heads, H*W, H*W)
        attn = F.softmax(attn, dim=-1)
        
        # Weighted sum
        out = torch.matmul(attn, v.transpose(-1, -2))  # (B, heads, H*W, head_dim)
        out = out.transpose(-1, -2).reshape(B, C, H, W)
        
        # Output projection with residual
        return x + self.proj_out(out)


class DownBlock(nn.Module):
    """
    Downsampling block trong U-Net
    
    Bao gồm ResNet blocks + optional attention + downsample
    
    Args:
        in_channels: Số kênh vào
        out_channels: Số kênh ra
        time_emb_dim: Kích thước time embedding
        num_blocks: Số ResNet blocks
        has_attention: Có dùng attention không
        dropout: Dropout rate
    """
    
    def __init__(self, in_channels, out_channels, time_emb_dim,
                 num_blocks=2, has_attention=False, dropout=0.1):
        super().__init__()
        
        # ResNet blocks
        self.blocks = nn.ModuleList([])
        for i in range(num_blocks):
            in_ch = in_channels if i == 0 else out_channels
            self.blocks.append(
                ResnetBlock(in_ch, out_channels, time_emb_dim, dropout)
            )
        
        # Optional attention
        self.attention = AttentionBlock(out_channels) if has_attention else None
        
        # Downsample (stride 2)
        self.downsample = nn.Conv2d(out_channels, out_channels, 
                                    kernel_size=3, stride=2, padding=1)
    
    def forward(self, x, time_emb):
        """
        Args:
            x: Input (B, in_channels, H, W)
            time_emb: Time embedding (B, time_emb_dim)
            
        Returns:
            tuple: (downsampled output, skip connection)
        """
        for block in self.blocks:
            x = block(x, time_emb)
        
        if self.attention is not None:
            x = self.attention(x)
        
        skip = x  # Skip connection trước downsample
        x = self.downsample(x)
        
        return x, skip


class UpBlock(nn.Module):
    """
    Upsampling block trong U-Net
    
    Bao gồm upsample + concat skip + ResNet blocks + optional attention
    
    Args:
        in_channels: Số kênh vào (bao gồm cả skip)
        out_channels: Số kênh ra
        time_emb_dim: Kích thước time embedding
        num_blocks: Số ResNet blocks
        has_attention: Có dùng attention không
        dropout: Dropout rate
    """
    
    def __init__(self, in_channels, out_channels, time_emb_dim,
                 num_blocks=2, has_attention=False, dropout=0.1):
        super().__init__()
        
        # Upsample
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels,
                                           kernel_size=4, stride=2, padding=1)
        
        # ResNet blocks (input includes skip connection)
        self.blocks = nn.ModuleList([])
        for i in range(num_blocks):
            # First block takes concatenated input
            in_ch = in_channels * 2 if i == 0 else out_channels
            self.blocks.append(
                ResnetBlock(in_ch, out_channels, time_emb_dim, dropout)
            )
        
        # Optional attention
        self.attention = AttentionBlock(out_channels) if has_attention else None
    
    def forward(self, x, skip, time_emb):
        """
        Args:
            x: Input (B, in_channels, H, W)
            skip: Skip connection từ encoder
            time_emb: Time embedding (B, time_emb_dim)
            
        Returns:
            Output (B, out_channels, H*2, W*2)
        """
        x = self.upsample(x)
        
        # Concat skip connection
        x = torch.cat([x, skip], dim=1)
        
        for block in self.blocks:
            x = block(x, time_emb)
        
        if self.attention is not None:
            x = self.attention(x)
        
        return x


class MiddleBlock(nn.Module):
    """
    Middle block của U-Net (bottleneck)
    
    Args:
        channels: Số kênh
        time_emb_dim: Kích thước time embedding
        dropout: Dropout rate
    """
    
    def __init__(self, channels, time_emb_dim, dropout=0.1):
        super().__init__()
        
        self.block1 = ResnetBlock(channels, channels, time_emb_dim, dropout)
        self.attention = AttentionBlock(channels)
        self.block2 = ResnetBlock(channels, channels, time_emb_dim, dropout)
    
    def forward(self, x, time_emb):
        x = self.block1(x, time_emb)
        x = self.attention(x)
        x = self.block2(x, time_emb)
        return x


# =====================================================================
# U-NET MODEL
# =====================================================================

class UNet(nn.Module):
    """
    U-Net cho DDPM
    
    Dự đoán noise ε từ ảnh nhiễu x_t và timestep t.
    
    Kiến trúc:
        - Encoder: Giảm dần resolution, tăng channels
        - Middle: Bottleneck với attention
        - Decoder: Tăng dần resolution, giảm channels với skip connections
    
    Args:
        in_channels: Số kênh input (3 cho RGB)
        base_channels: Số kênh cơ bản
        channel_mults: Bội số kênh cho mỗi level
        num_res_blocks: Số ResNet blocks mỗi level
        attention_levels: Các levels có attention
        time_emb_dim: Kích thước time embedding
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        in_channels=3,
        base_channels=DiffusionConfig.BASE_CHANNELS,
        channel_mults=None,
        num_res_blocks=DiffusionConfig.NUM_RES_BLOCKS,
        attention_levels=None,
        time_emb_dim=None,
        dropout=DiffusionConfig.DROPOUT
    ):
        super().__init__()
        
        if channel_mults is None:
            channel_mults = DiffusionConfig.CHANNEL_MULTS
        
        if attention_levels is None:
            # Attention ở các levels sau (resolution nhỏ)
            attention_levels = [len(channel_mults) - 2, len(channel_mults) - 1]
        
        if time_emb_dim is None:
            time_emb_dim = base_channels * 4
        
        print("\n🏗️  Đang xây dựng U-Net cho DDPM...")
        
        # Time embedding
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Initial convolution
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        
        # Encoder (Down path)
        self.downs = nn.ModuleList([])
        channels = [base_channels]
        in_ch = base_channels
        
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            has_attn = i in attention_levels
            
            self.downs.append(
                DownBlock(in_ch, out_ch, time_emb_dim,
                         num_res_blocks, has_attn, dropout)
            )
            channels.append(out_ch)
            in_ch = out_ch
        
        # Middle (Bottleneck)
        self.middle = MiddleBlock(in_ch, time_emb_dim, dropout)
        
        # Decoder (Up path)
        self.ups = nn.ModuleList([])
        
        for i, mult in enumerate(reversed(channel_mults)):
            out_ch = base_channels * mult
            has_attn = (len(channel_mults) - 1 - i) in attention_levels
            
            self.ups.append(
                UpBlock(in_ch, out_ch, time_emb_dim,
                       num_res_blocks, has_attn, dropout)
            )
            in_ch = out_ch
        
        # Final output
        self.final = nn.Sequential(
            nn.GroupNorm(8, base_channels * channel_mults[0]),
            nn.SiLU(),
            nn.Conv2d(base_channels * channel_mults[0], in_channels, kernel_size=3, padding=1)
        )
        
        # Save config
        self.channel_mults = channel_mults
        
        # Print model info
        total_params = sum(p.numel() for p in self.parameters())
        print(f"✅ U-Net khởi tạo thành công!")
        print(f"📊 Total parameters: {total_params:,}")
    
    def forward(self, x, t):
        """
        Dự đoán noise từ ảnh nhiễu
        
        Args:
            x: Noisy image (B, C, H, W)
            t: Timestep (B,)
            
        Returns:
            Predicted noise (B, C, H, W)
        """
        # Time embedding
        time_emb = self.time_embedding(t)
        
        # Initial conv
        x = self.init_conv(x)
        
        # Encoder với skip connections
        skips = []
        for down in self.downs:
            x, skip = down(x, time_emb)
            skips.append(skip)
        
        # Middle
        x = self.middle(x, time_emb)
        
        # Decoder với skip connections
        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip, time_emb)
        
        # Final output
        return self.final(x)


# =====================================================================
# DDPM MODEL
# =====================================================================

class DDPM(nn.Module):
    """
    Complete DDPM Model
    
    Bao gồm:
    - U-Net để dự đoán noise
    - Forward process (thêm noise)
    - Reverse process (khử noise)
    - Loss computation
    - Sampling
    - Anomaly score computation
    
    Args:
        unet: U-Net model
        num_timesteps: Số bước diffusion
        beta_schedule: Loại schedule ('linear' hoặc 'cosine')
    """
    
    def __init__(
        self,
        unet=None,
        num_timesteps=DiffusionConfig.NUM_TIMESTEPS,
        beta_schedule=DiffusionConfig.BETA_SCHEDULE
    ):
        super().__init__()
        
        print("\n🏗️  Đang xây dựng DDPM...")
        
        # U-Net
        if unet is None:
            self.unet = UNet()
        else:
            self.unet = unet
        
        self.num_timesteps = num_timesteps
        
        # Noise schedule
        betas = get_beta_schedule(beta_schedule, num_timesteps)
        
        # Register buffers (không phải parameters)
        self.register_buffer('betas', betas)
        
        # Pre-compute các giá trị cần thiết
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        
        # Các giá trị cho forward process q(x_t | x_0)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        
        # Các giá trị cho reverse process p(x_{t-1} | x_t)
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
        
        # Variance cho reverse process
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance', torch.log(
            torch.clamp(posterior_variance, min=1e-20)
        ))
        
        # Coefficients cho posterior mean
        self.register_buffer('posterior_mean_coef1',
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))
        
        print(f"✅ DDPM khởi tạo thành công!")
        print(f"📊 Num timesteps: {num_timesteps}")
        print(f"📊 Beta schedule: {beta_schedule}")
    
    def _extract(self, a, t, x_shape):
        """
        Extract giá trị từ tensor a theo timestep t
        
        Args:
            a: Tensor 1D (T,)
            t: Timestep tensor (B,)
            x_shape: Shape của x để broadcast
            
        Returns:
            Tensor (B, 1, 1, 1) để broadcast với x
        """
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    
    def q_sample(self, x_0, t, noise=None):
        """
        Forward process: Thêm noise vào ảnh
        
        q(x_t | x_0) = N(x_t; sqrt(α_t) * x_0, (1 - α_t) * I)
        
        x_t = sqrt(α_t) * x_0 + sqrt(1 - α_t) * ε
        
        Args:
            x_0: Original image (B, C, H, W)
            t: Timestep (B,)
            noise: Optional noise tensor
            
        Returns:
            x_t: Noisy image at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alpha = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        
        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
    
    def p_losses(self, x_0, t, noise=None):
        """
        Compute training loss
        
        L = E[||ε - ε_θ(x_t, t)||^2]
        
        Args:
            x_0: Original images (B, C, H, W)
            t: Random timesteps (B,)
            noise: Optional noise
            
        Returns:
            loss: Scalar loss value
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        # Forward process: x_0 -> x_t
        x_t = self.q_sample(x_0, t, noise)
        
        # Predict noise
        predicted_noise = self.unet(x_t, t)
        
        # MSE loss
        loss = F.mse_loss(predicted_noise, noise)
        
        return loss
    
    def forward(self, x_0):
        """
        Training forward pass
        
        Random timesteps được sample cho mỗi image trong batch.
        
        Args:
            x_0: Original images (B, C, H, W)
            
        Returns:
            loss: Training loss
        """
        batch_size = x_0.shape[0]
        device = x_0.device
        
        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device)
        
        return self.p_losses(x_0, t)
    
    @torch.no_grad()
    def p_sample(self, x_t, t):
        """
        Reverse process: Một bước khử noise
        
        p(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), σ_t^2 I)
        
        Args:
            x_t: Noisy image at timestep t (B, C, H, W)
            t: Current timestep (B,) hoặc scalar
            
        Returns:
            x_{t-1}: Denoised image
        """
        if isinstance(t, int):
            t = torch.tensor([t] * x_t.shape[0], device=x_t.device)
        
        # Predict noise
        predicted_noise = self.unet(x_t, t)
        
        # Compute mean
        sqrt_recip_alpha = self._extract(self.sqrt_recip_alphas, t, x_t.shape)
        beta = self._extract(self.betas, t, x_t.shape)
        sqrt_one_minus_alpha = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        
        model_mean = sqrt_recip_alpha * (x_t - beta * predicted_noise / sqrt_one_minus_alpha)
        
        # Add noise (không thêm ở t=0)
        if t[0] > 0:
            noise = torch.randn_like(x_t)
            posterior_var = self._extract(self.posterior_variance, t, x_t.shape)
            return model_mean + torch.sqrt(posterior_var) * noise
        else:
            return model_mean
    
    @torch.no_grad()
    def p_sample_loop(self, shape, device=None):
        """
        Full reverse process: Generate image từ noise
        
        Args:
            shape: Output shape (B, C, H, W)
            device: Device
            
        Returns:
            Generated images (B, C, H, W)
        """
        if device is None:
            device = next(self.parameters()).device
        
        # Start from pure noise
        img = torch.randn(shape, device=device)
        
        # Reverse process
        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.tensor([t] * shape[0], device=device)
            img = self.p_sample(img, t_batch)
        
        return img
    
    @torch.no_grad()
    def compute_anomaly_score(self, x, num_steps=None):
        """
        Tính anomaly score cho ảnh
        
        Phương pháp:
        1. Thêm một ít noise vào ảnh (forward process đến timestep t)
        2. Khử noise (reverse process từ t về 0)
        3. Tính sai số giữa ảnh gốc và ảnh đã khử nhiễu
        
        Ảnh deepfake có "dấu vết nhân tạo" mà model không học,
        nên sai số sẽ lớn hơn ảnh thật.
        
        Args:
            x: Input images (B, C, H, W)
            num_steps: Số bước diffusion (default: ANOMALY_TIMESTEPS)
            
        Returns:
            anomaly_score: Tensor (B,) score cho mỗi ảnh
        """
        self.eval()
        
        if num_steps is None:
            num_steps = DiffusionConfig.ANOMALY_TIMESTEPS
        
        batch_size = x.shape[0]
        device = x.device
        
        # Forward process: thêm noise đến timestep t
        t = torch.tensor([num_steps] * batch_size, device=device)
        noise = torch.randn_like(x)
        x_noisy = self.q_sample(x, t, noise)
        
        # Reverse process: khử noise
        x_denoised = x_noisy
        for i in reversed(range(num_steps)):
            t_batch = torch.tensor([i] * batch_size, device=device)
            x_denoised = self.p_sample(x_denoised, t_batch)
        
        # Compute residual/anomaly score
        if DiffusionConfig.ANOMALY_METRIC == 'mse':
            # Mean Squared Error per image
            residual = (x - x_denoised) ** 2
            anomaly_score = residual.mean(dim=[1, 2, 3])  # (B,)
        else:
            # L1 error
            residual = torch.abs(x - x_denoised)
            anomaly_score = residual.mean(dim=[1, 2, 3])  # (B,)
        
        return anomaly_score
    
    @torch.no_grad()
    def get_reconstruction(self, x, num_steps=None):
        """
        Lấy ảnh đã khử nhiễu để visualization
        
        Args:
            x: Input images (B, C, H, W)
            num_steps: Số bước diffusion
            
        Returns:
            tuple: (x_noisy, x_denoised, residual)
        """
        self.eval()
        
        if num_steps is None:
            num_steps = DiffusionConfig.ANOMALY_TIMESTEPS
        
        batch_size = x.shape[0]
        device = x.device
        
        # Forward
        t = torch.tensor([num_steps] * batch_size, device=device)
        noise = torch.randn_like(x)
        x_noisy = self.q_sample(x, t, noise)
        
        # Reverse
        x_denoised = x_noisy.clone()
        for i in reversed(range(num_steps)):
            t_batch = torch.tensor([i] * batch_size, device=device)
            x_denoised = self.p_sample(x_denoised, t_batch)
        
        # Residual
        residual = torch.abs(x - x_denoised)
        
        return x_noisy, x_denoised, residual


# =====================================================================
# MAIN (Test)
# =====================================================================

if __name__ == "__main__":
    """
    Test DDPM model khi chạy file này trực tiếp
    """
    print("=" * 70)
    print("🧪 TEST DDPM MODULE")
    print("=" * 70)
    
    # Tạo model với config nhỏ hơn để test
    print("\n1. Khởi tạo U-Net và DDPM...")
    unet = UNet(
        in_channels=3,
        base_channels=32,
        channel_mults=[1, 2, 4],
        num_res_blocks=1,
        dropout=0.1
    )
    
    ddpm = DDPM(
        unet=unet,
        num_timesteps=100,
        beta_schedule='cosine'
    )
    
    # Test với random input
    print("\n2. Test forward pass (training loss)...")
    batch_size = 2
    x = torch.randn(batch_size, 3, 64, 64)  # Smaller for testing
    print(f"   Input shape: {x.shape}")
    
    loss = ddpm(x)
    print(f"   Training loss: {loss.item():.4f}")
    
    # Test q_sample
    print("\n3. Test forward process (q_sample)...")
    t = torch.tensor([50, 50])
    x_noisy = ddpm.q_sample(x, t)
    print(f"   Noisy image shape: {x_noisy.shape}")
    
    # Test p_sample
    print("\n4. Test reverse process (p_sample)...")
    x_denoised = ddpm.p_sample(x_noisy, t)
    print(f"   Denoised image shape: {x_denoised.shape}")
    
    # Test anomaly score
    print("\n5. Test anomaly score computation...")
    anomaly_score = ddpm.compute_anomaly_score(x, num_steps=10)
    print(f"   Anomaly scores: {anomaly_score}")
    
    # Test get_reconstruction
    print("\n6. Test get_reconstruction...")
    x_noisy, x_recon, residual = ddpm.get_reconstruction(x, num_steps=10)
    print(f"   Noisy shape: {x_noisy.shape}")
    print(f"   Reconstructed shape: {x_recon.shape}")
    print(f"   Residual shape: {residual.shape}")
    
    print("\n" + "=" * 70)
    print("✅ HOÀN THÀNH TEST DDPM!")
    print("=" * 70)
