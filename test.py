import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    """Cross-Attention: dùng đặc trưng ViT làm Query và đặc trưng CNN làm Key/Value.

    Ý tưởng:
    - ViT (q) đang có các token (patch token + có thể có CLS token) cần được "tái chú ý" (attend)
      lên thông tin không gian/chi tiết từ CNN (kv).
    - CNN (kv) cung cấp ngữ cảnh bổ sung làm Key/Value.

    Input/Output:
    - q:  (B, Nq, D)
    - kv: (B, Nk, D)
    - out:(B, Nq, D)
    """
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        # num_heads: số head trong multi-head attention
        self.num_heads = num_heads
        # head_dim: số chiều mỗi head. dim phải chia hết cho num_heads
        self.head_dim = dim // num_heads
        # scale = 1/sqrt(head_dim) để ổn định softmax (tránh giá trị quá lớn)
        self.scale = self.head_dim ** -0.5

        # Linear projection để tạo Q, K, V từ q và kv
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        # Chiếu ngược về dim ban đầu sau khi ghép multi-head
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, kv):
        # q: (B, Nq, D)   ← đặc trưng/token từ ViT
        # kv: (B, Nk, D)  ← đặc trưng/token từ CNN (đã reshape về dạng chuỗi token)
        B, Nq, D = q.shape
        Nk = kv.shape[1]

        # Tạo Q, K, V
        Q = self.q_proj(q)
        K = self.k_proj(kv)
        V = self.v_proj(kv)

        # Reshape sang multi-head:
        # (B, N, D) -> (B, N, num_heads, head_dim) -> transpose -> (B, num_heads, N, head_dim)
        Q = Q.view(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)

        # Tính attention score giữa mỗi query token (Nq) và mỗi key token (Nk)
        # attn shape: (B, num_heads, Nq, Nk)
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        # Chuẩn hoá theo Nk để ra trọng số attention
        attn = attn.softmax(dim=-1)
        # Dropout lên attention weights để regularize
        attn = self.dropout(attn)

        # Tổng có trọng số lên V để ra output từng head
        # out shape: (B, num_heads, Nq, head_dim)
        out = attn @ V
        # Ghép các head lại:
        # (B, num_heads, Nq, head_dim) -> (B, Nq, num_heads, head_dim) -> (B, Nq, D)
        out = out.transpose(1, 2).contiguous().view(B, Nq, D)

        # Chiếu ra output cuối cùng (vẫn giữ shape (B, Nq, D))
        return self.out_proj(out)
      
class CrossAttentionBlock(nn.Module):
    """Một block kiểu Transformer nhưng attention là cross-attention.

    Cấu trúc:
    1) LayerNorm + CrossAttention + residual
    2) LayerNorm + MLP (FFN) + residual

    Input:
    - vit_feat: (B, Nq, D)  token của ViT
    - cnn_feat: (B, Nk, D)  token của CNN
    Output:
    - (B, Nq, D)
    """
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        # Pre-Norm: chuẩn hoá trước attention
        self.norm1 = nn.LayerNorm(dim)
        self.attn = CrossAttention(dim, num_heads, dropout)

        # Pre-Norm: chuẩn hoá trước MLP/FFN
        self.norm2 = nn.LayerNorm(dim)
        # MLP/FFN: mở rộng dim lên dim*mlp_ratio rồi giảm lại dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )

    def forward(self, vit_feat, cnn_feat):
        # 1) Cross-attention (ViT query attend sang CNN key/value)
        # residual: giữ thông tin gốc + thêm thông tin từ attention
        x = vit_feat + self.attn(self.norm1(vit_feat), cnn_feat)
        # 2) MLP/FFN
        # residual: giúp học ổn định và sâu hơn
        x = x + self.mlp(self.norm2(x))
        return x
