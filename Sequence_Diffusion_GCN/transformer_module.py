"""
=====================================================================
MÔ-ĐUN TRANSFORMER TỰ QUY HỒI (GPT-LIKE)
=====================================================================
Mô tả:
    Transformer này học phân phối xác suất của các token ảnh từ VQ-VAE.
    
    Ý tưởng chính:
    1. Ảnh được mã hóa thành chuỗi token bởi VQ-VAE
    2. Transformer học dự đoán token tiếp theo dựa trên các token trước
    3. Khi test, tính perplexity của chuỗi token
    4. Perplexity cao = ảnh không phù hợp phân phối thật = có thể là deepfake
    
    Đây là phương pháp "Sequence Modeling" - coi ảnh như một câu trong NLP
    và dùng language model để phát hiện bất thường.
    
Kiến trúc:
    - Token Embedding: Chuyển token ID thành vector
    - Positional Encoding: Thêm thông tin vị trí
    - Transformer Decoder: Stack các decoder layers với causal masking
    - Output Head: Dự đoán xác suất token tiếp theo
    
Tham khảo:
    - Radford et al., "Language Models are Unsupervised Multitask Learners" (GPT-2)
    - Esser et al., "Taming Transformers for High-Resolution Image Synthesis"
    
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
from config import TransformerConfig, VQVAEConfig


# =====================================================================
# POSITIONAL ENCODING
# =====================================================================

class SinusoidalPositionalEncoding(nn.Module):
    """
    Positional Encoding sử dụng hàm sin/cos
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    Positional encoding giúp model biết thứ tự của các tokens
    vì Transformer không có khái niệm về vị trí tự nhiên.
    
    Args:
        d_model: Kích thước embedding
        max_len: Độ dài tối đa của sequence
        dropout: Dropout rate
    """
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Tạo positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        # Áp dụng sin cho các vị trí chẵn, cos cho các vị trí lẻ
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Thêm batch dimension và register buffer (không train)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Thêm positional encoding vào input
        
        Args:
            x: Input tensor (B, seq_len, d_model)
            
        Returns:
            Output tensor với positional encoding (B, seq_len, d_model)
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """
    Positional Encoding học được (như GPT)
    
    Thay vì dùng hàm sin/cos cố định, ta học position embeddings
    như một tham số của model.
    
    Args:
        d_model: Kích thước embedding
        max_len: Độ dài tối đa của sequence
        dropout: Dropout rate
    """
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Learned position embeddings
        self.pos_embedding = nn.Embedding(max_len, d_model)
        
        # Khởi tạo với Xavier
        nn.init.xavier_uniform_(self.pos_embedding.weight)
    
    def forward(self, x):
        """
        Thêm learned positional encoding
        
        Args:
            x: Input tensor (B, seq_len, d_model)
            
        Returns:
            Output với positional encoding
        """
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        pos_embed = self.pos_embedding(positions)  # (seq_len, d_model)
        return self.dropout(x + pos_embed)


# =====================================================================
# TRANSFORMER COMPONENTS
# =====================================================================

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention với causal masking
    
    Causal masking đảm bảo token ở vị trí t chỉ attend được đến
    các tokens ở vị trí <= t (không nhìn vào tương lai).
    
    Args:
        d_model: Kích thước model
        nhead: Số attention heads
        dropout: Dropout rate
    """
    
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        
        assert d_model % nhead == 0, "d_model phải chia hết cho nhead"
        
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        # Linear projections cho Q, K, V
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # Scaling factor
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x, attn_mask=None):
        """
        Multi-head self-attention
        
        Args:
            x: Input tensor (B, seq_len, d_model)
            attn_mask: Optional attention mask (seq_len, seq_len)
            
        Returns:
            Output tensor (B, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape cho multi-head: (B, seq, d) -> (B, nhead, seq, head_dim)
        q = q.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        
        # Attention scores: (B, nhead, seq, seq)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Áp dụng causal mask nếu có
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == float('-inf'), float('-inf'))
        
        # Softmax và dropout
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Weighted sum của values
        out = torch.matmul(attn, v)  # (B, nhead, seq, head_dim)
        
        # Reshape lại: (B, nhead, seq, head_dim) -> (B, seq, d_model)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.out_proj(out)


class FeedForward(nn.Module):
    """
    Feed Forward Network (FFN) trong Transformer
    
    FFN(x) = max(0, xW1 + b1)W2 + b2
    
    Hoặc với GELU activation:
    FFN(x) = GELU(xW1 + b1)W2 + b2
    
    Args:
        d_model: Kích thước input/output
        dim_feedforward: Kích thước lớp ẩn (thường = 4 * d_model)
        dropout: Dropout rate
    """
    
    def __init__(self, d_model, dim_feedforward, dropout=0.1):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),  # GELU thường tốt hơn ReLU trong Transformers
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


class TransformerDecoderLayer(nn.Module):
    """
    Một layer của Transformer Decoder
    
    Cấu trúc:
        x -> LayerNorm -> Self-Attention -> + x -> LayerNorm -> FFN -> + x -> output
        
    Sử dụng Pre-LayerNorm (đặt LayerNorm trước attention/FFN)
    thay vì Post-LayerNorm gốc để training ổn định hơn.
    
    Args:
        d_model: Kích thước model
        nhead: Số attention heads
        dim_feedforward: Kích thước FFN
        dropout: Dropout rate
    """
    
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        
        # Pre-LayerNorm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Self-attention với causal masking
        self.self_attn = MultiHeadSelfAttention(d_model, nhead, dropout)
        
        # Feed-forward network
        self.ffn = FeedForward(d_model, dim_feedforward, dropout)
    
    def forward(self, x, attn_mask=None):
        """
        Forward pass qua một decoder layer
        
        Args:
            x: Input tensor (B, seq_len, d_model)
            attn_mask: Causal attention mask
            
        Returns:
            Output tensor (B, seq_len, d_model)
        """
        # Self-attention với residual connection
        x = x + self.self_attn(self.norm1(x), attn_mask=attn_mask)
        
        # FFN với residual connection
        x = x + self.ffn(self.norm2(x))
        
        return x


# =====================================================================
# GPT-LIKE TRANSFORMER
# =====================================================================

class ImageGPT(nn.Module):
    """
    GPT-like Transformer cho Image Tokens
    
    Model này học phân phối xác suất P(token_t | token_1, ..., token_{t-1})
    của các image tokens từ VQ-VAE.
    
    Khi suy luận, ta tính:
    - Log-likelihood: sum(log P(token_t | token_{<t}))
    - Perplexity: exp(-1/N * sum(log P(token_t)))
    
    Ảnh thật sẽ có perplexity thấp (phù hợp phân phối học được),
    trong khi deepfake có perplexity cao (không phù hợp).
    
    Args:
        vocab_size: Kích thước vocabulary (= num_embeddings của VQ-VAE)
        d_model: Kích thước embedding
        nhead: Số attention heads
        num_layers: Số decoder layers
        dim_feedforward: Kích thước FFN
        max_seq_len: Độ dài sequence tối đa
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        vocab_size=TransformerConfig.VOCAB_SIZE,
        d_model=TransformerConfig.D_MODEL,
        nhead=TransformerConfig.NHEAD,
        num_layers=TransformerConfig.NUM_LAYERS,
        dim_feedforward=TransformerConfig.DIM_FEEDFORWARD,
        max_seq_len=TransformerConfig.MAX_SEQ_LEN,
        dropout=TransformerConfig.DROPOUT
    ):
        super().__init__()
        
        print("\n🏗️  Đang xây dựng ImageGPT Transformer...")
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Token embedding - chuyển token ID thành vector
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding - learned positions như GPT
        self.pos_encoding = LearnedPositionalEncoding(
            d_model=d_model,
            max_len=max_seq_len,
            dropout=dropout
        )
        
        # Stack of Transformer decoder layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Final layer normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Output head - dự đoán token tiếp theo
        self.output_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights: output_head dùng cùng weights với token_embedding
        # Điều này giúp regularization và giảm số parameters
        self.output_head.weight = self.token_embedding.weight
        
        # Khởi tạo weights
        self._init_weights()
        
        # Tạo causal mask (cache để không tính lại mỗi forward)
        self._register_causal_mask(max_seq_len)
        
        # In thông tin model
        total_params = sum(p.numel() for p in self.parameters())
        print(f"✅ ImageGPT khởi tạo thành công!")
        print(f"📊 Vocab size: {vocab_size}")
        print(f"📊 D_model: {d_model}")
        print(f"📊 Num layers: {num_layers}")
        print(f"📊 Max seq len: {max_seq_len}")
        print(f"📊 Total parameters: {total_params:,}")
    
    def _init_weights(self):
        """
        Khởi tạo weights theo cách của GPT
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def _register_causal_mask(self, max_seq_len):
        """
        Tạo và register causal attention mask
        
        Causal mask là ma trận tam giác dưới:
        [[1, 0, 0, ...],
         [1, 1, 0, ...],
         [1, 1, 1, ...],
         ...]
        
        Các vị trí có 0 sẽ được mask (set attention = -inf trước softmax)
        """
        # Tạo mask: 1 ở vị trí được attend, 0 ở vị trí bị mask
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1)
        # Chuyển 1 -> -inf (bị mask), 0 -> 0 (được attend)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        self.register_buffer('causal_mask', mask)
    
    def get_causal_mask(self, seq_len):
        """
        Lấy causal mask cho sequence có độ dài nhất định
        
        Args:
            seq_len: Độ dài sequence
            
        Returns:
            Causal mask (seq_len, seq_len)
        """
        return self.causal_mask[:seq_len, :seq_len]
    
    def forward(self, token_indices, return_logits_only=False):
        """
        Forward pass
        
        Args:
            token_indices: Token IDs (B, seq_len)
            return_logits_only: Chỉ trả về logits (cho inference)
            
        Returns:
            nếu return_logits_only:
                logits: (B, seq_len, vocab_size)
            ngược lại:
                logits, loss nếu có labels
        """
        batch_size, seq_len = token_indices.shape
        
        # Giới hạn sequence length
        if seq_len > self.max_seq_len:
            token_indices = token_indices[:, :self.max_seq_len]
            seq_len = self.max_seq_len
        
        # Token embedding
        x = self.token_embedding(token_indices)  # (B, seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Get causal mask
        attn_mask = self.get_causal_mask(seq_len)
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)
        
        # Final normalization
        x = self.norm(x)
        
        # Output logits
        logits = self.output_head(x)  # (B, seq_len, vocab_size)
        
        return logits
    
    def compute_loss(self, token_indices, label_smoothing=0.0):
        """
        Tính cross-entropy loss cho next token prediction
        
        Args:
            token_indices: Token IDs (B, seq_len)
            label_smoothing: Label smoothing coefficient
            
        Returns:
            loss: Scalar loss value
        """
        # Forward pass
        logits = self.forward(token_indices)
        
        # Shift để align predictions với targets
        # logits[:, t] dự đoán token tại vị trí t+1
        # Nên target là token_indices shifted left by 1
        logits = logits[:, :-1].contiguous()  # (B, seq_len-1, vocab)
        targets = token_indices[:, 1:].contiguous()  # (B, seq_len-1)
        
        # Cross-entropy loss
        loss = F.cross_entropy(
            logits.view(-1, self.vocab_size),
            targets.view(-1),
            label_smoothing=label_smoothing
        )
        
        return loss
    
    @torch.no_grad()
    def compute_perplexity(self, token_indices):
        """
        Tính perplexity của một chuỗi token
        
        Perplexity = exp(-1/N * sum(log P(token_t | token_{<t})))
        
        Perplexity càng thấp = model càng "confident" về sequence
        = sequence càng phù hợp với phân phối học được
        
        Args:
            token_indices: Token IDs (B, seq_len)
            
        Returns:
            perplexity: Tensor (B,) perplexity cho mỗi sample trong batch
        """
        self.eval()
        
        # Forward pass
        logits = self.forward(token_indices)
        
        # Shift cho alignment
        logits = logits[:, :-1].contiguous()
        targets = token_indices[:, 1:].contiguous()
        
        # Log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Gather log prob của đúng token
        # log_probs: (B, seq_len-1, vocab)
        # targets: (B, seq_len-1)
        batch_size, seq_len, vocab_size = log_probs.shape
        
        # Expand targets để gather
        targets_expanded = targets.unsqueeze(-1)  # (B, seq_len, 1)
        token_log_probs = log_probs.gather(-1, targets_expanded).squeeze(-1)  # (B, seq_len)
        
        # Mean negative log likelihood
        mean_nll = -token_log_probs.mean(dim=-1)  # (B,)
        
        # Perplexity = exp(mean_nll)
        perplexity = torch.exp(mean_nll)
        
        return perplexity
    
    @torch.no_grad()
    def compute_anomaly_score(self, token_indices, normalize=True):
        """
        Tính anomaly score dựa trên perplexity
        
        Đây là metric chính để phát hiện deepfake:
        - Score cao = không phù hợp phân phối thật = có thể là deepfake
        - Score thấp = phù hợp phân phối thật = có thể là ảnh thật
        
        Args:
            token_indices: Token IDs (B, seq_len)
            normalize: Có normalize score không
            
        Returns:
            anomaly_score: Tensor (B,) score cho mỗi sample
        """
        perplexity = self.compute_perplexity(token_indices)
        
        if normalize:
            # Log transform để scale hợp lý hơn
            # Perplexity thường có range rất lớn
            anomaly_score = torch.log(perplexity + 1)
        else:
            anomaly_score = perplexity
        
        return anomaly_score
    
    @torch.no_grad()
    def generate(self, start_tokens, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate tokens tự quy hồi (autoregressive)
        
        Bắt đầu từ start_tokens, dự đoán và sample token tiếp theo,
        thêm vào sequence và lặp lại.
        
        Args:
            start_tokens: Token IDs bắt đầu (B, start_len)
            max_new_tokens: Số tokens mới cần generate
            temperature: Temperature cho sampling (cao = random hơn)
            top_k: Chỉ sample từ top-k tokens có xác suất cao nhất
            
        Returns:
            generated: Full sequence bao gồm start và generated tokens
        """
        self.eval()
        
        tokens = start_tokens.clone()
        
        for _ in range(max_new_tokens):
            # Crop nếu sequence quá dài
            if tokens.size(1) >= self.max_seq_len:
                context = tokens[:, -self.max_seq_len:]
            else:
                context = tokens
            
            # Forward pass
            logits = self.forward(context)
            
            # Lấy logits của token cuối cùng
            logits = logits[:, -1, :] / temperature  # (B, vocab)
            
            # Top-k filtering nếu cần
            if top_k is not None:
                values, _ = torch.topk(logits, top_k, dim=-1)
                min_value = values[:, -1].unsqueeze(-1)
                logits = torch.where(
                    logits < min_value,
                    torch.full_like(logits, float('-inf')),
                    logits
                )
            
            # Sample token tiếp theo
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            
            # Concat với sequence hiện tại
            tokens = torch.cat([tokens, next_token], dim=1)
        
        return tokens


# =====================================================================
# TRAINING UTILITIES
# =====================================================================

class TransformerTrainer:
    """
    Helper class để training ImageGPT
    
    Bao gồm:
    - Training loop
    - Learning rate scheduling với warmup
    - Logging và checkpointing
    
    Args:
        model: ImageGPT model
        optimizer: Optimizer
        device: Device để training
        warmup_steps: Số steps warmup cho LR scheduler
    """
    
    def __init__(self, model, optimizer, device, warmup_steps=4000):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.warmup_steps = warmup_steps
        
        # Learning rate scheduler với warmup
        self.global_step = 0
        self.base_lr = optimizer.param_groups[0]['lr']
    
    def get_lr(self):
        """
        Compute learning rate với warmup schedule
        
        LR = base_lr * min(step^{-0.5}, step * warmup^{-1.5})
        
        Tăng dần trong warmup, sau đó giảm dần theo sqrt(step)
        """
        step = max(self.global_step, 1)
        
        if step < self.warmup_steps:
            # Linear warmup
            return self.base_lr * step / self.warmup_steps
        else:
            # Decay
            return self.base_lr * (self.warmup_steps ** 0.5) * (step ** -0.5)
    
    def update_lr(self):
        """Update learning rate cho optimizer"""
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    
    def train_step(self, token_indices, label_smoothing=0.0):
        """
        Một bước training
        
        Args:
            token_indices: Token IDs (B, seq_len)
            label_smoothing: Label smoothing coefficient
            
        Returns:
            loss: Scalar loss value
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward và compute loss
        token_indices = token_indices.to(self.device)
        loss = self.model.compute_loss(token_indices, label_smoothing)
        
        # Backward
        loss.backward()
        
        # Gradient clipping để tránh exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Update
        self.optimizer.step()
        
        # Update learning rate
        self.global_step += 1
        self.update_lr()
        
        return loss.item()


# =====================================================================
# MAIN (Test)
# =====================================================================

if __name__ == "__main__":
    """
    Test ImageGPT Transformer khi chạy file này trực tiếp
    """
    print("=" * 70)
    print("🧪 TEST IMAGE GPT TRANSFORMER MODULE")
    print("=" * 70)
    
    # Tạo model với config nhỏ hơn để test
    print("\n1. Khởi tạo ImageGPT model...")
    model = ImageGPT(
        vocab_size=512,
        d_model=256,
        nhead=4,
        num_layers=4,
        dim_feedforward=512,
        max_seq_len=256,
        dropout=0.1
    )
    
    # Test với random tokens
    print("\n2. Test forward pass...")
    batch_size = 2
    seq_len = 100
    token_indices = torch.randint(0, 512, (batch_size, seq_len))
    print(f"   Input shape: {token_indices.shape}")
    
    # Forward pass
    logits = model(token_indices)
    print(f"   Output logits shape: {logits.shape}")
    
    # Test compute loss
    print("\n3. Test compute loss...")
    loss = model.compute_loss(token_indices)
    print(f"   Loss: {loss.item():.4f}")
    
    # Test compute perplexity
    print("\n4. Test compute perplexity...")
    perplexity = model.compute_perplexity(token_indices)
    print(f"   Perplexity: {perplexity}")
    
    # Test compute anomaly score
    print("\n5. Test compute anomaly score...")
    anomaly_score = model.compute_anomaly_score(token_indices)
    print(f"   Anomaly scores: {anomaly_score}")
    
    # Test generation
    print("\n6. Test generation...")
    start_tokens = torch.randint(0, 512, (1, 10))
    generated = model.generate(start_tokens, max_new_tokens=20, temperature=1.0)
    print(f"   Start tokens: {start_tokens.shape}")
    print(f"   Generated tokens: {generated.shape}")
    
    print("\n" + "=" * 70)
    print("✅ HOÀN THÀNH TEST IMAGE GPT TRANSFORMER!")
    print("=" * 70)
