"""
=====================================================================
MÔ-ĐUN GCN (GRAPH CONVOLUTIONAL NETWORK)
=====================================================================
Mô tả:
    GCN xử lý facial landmarks như một đồ thị để phát hiện bất thường
    trong cấu trúc khuôn mặt.
    
    Ý tưởng:
    - Nodes: Các điểm mốc (landmarks) trên khuôn mặt (68 điểm dlib)
    - Edges: Mối quan hệ giữa các landmarks (khoảng cách, góc)
    - GCN học biểu diễn cấu trúc khuôn mặt
    
    Deepfake thường có bất thường về cấu trúc:
    - Tỷ lệ khuôn mặt không tự nhiên
    - Vị trí mắt, mũi, miệng không đúng
    - Đường viền khuôn mặt không smooth
    
    GCN có thể phát hiện các bất thường này mà CNN đơn thuần
    có thể bỏ sót.
    
Tham khảo:
    - Kipf & Welling, "Semi-Supervised Classification with GCNs"
    - Samad & Bandhu (2025), "Deepfake Detection using CNN+GCN"
    
Tác giả: DAP391M Team
Phiên bản: 1.0
=====================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Import config
from config import GCNConfig


# =====================================================================
# GRAPH CONSTRUCTION
# =====================================================================

def create_facial_adjacency_matrix(num_landmarks=68, connection_type='full'):
    """
    Tạo adjacency matrix cho facial landmarks graph
    
    Mỗi landmark được kết nối với các landmarks lân cận theo cấu trúc
    của khuôn mặt (chin, eyebrow, eye, nose, mouth).
    
    Args:
        num_landmarks: Số landmarks (68 cho dlib standard)
        connection_type: Loại kết nối
            - 'sequential': Kết nối tuần tự trong mỗi facial part
            - 'full': Kết nối đầy đủ
            - 'spatial': Kết nối dựa trên khoảng cách không gian
            
    Returns:
        adj: Adjacency matrix (num_landmarks, num_landmarks)
    """
    adj = torch.zeros(num_landmarks, num_landmarks)
    
    if num_landmarks == 68:
        # Dlib 68 landmarks structure:
        # 0-16: Jaw/chin (17 points)
        # 17-21: Left eyebrow (5 points)
        # 22-26: Right eyebrow (5 points)
        # 27-35: Nose (9 points)
        # 36-41: Left eye (6 points)
        # 42-47: Right eye (6 points)
        # 48-59: Outer lip (12 points)
        # 60-67: Inner lip (8 points)
        
        parts = [
            list(range(0, 17)),     # Jaw
            list(range(17, 22)),    # Left eyebrow
            list(range(22, 27)),    # Right eyebrow
            list(range(27, 36)),    # Nose
            list(range(36, 42)),    # Left eye
            list(range(42, 48)),    # Right eye
            list(range(48, 60)),    # Outer lip
            list(range(60, 68)),    # Inner lip
        ]
        
        if connection_type == 'sequential':
            # Kết nối tuần tự trong mỗi part
            for part in parts:
                for i in range(len(part) - 1):
                    adj[part[i], part[i+1]] = 1
                    adj[part[i+1], part[i]] = 1
                # Đóng vòng cho mắt và môi
                if len(part) == 6 or len(part) >= 8:  # Eyes và lips
                    adj[part[0], part[-1]] = 1
                    adj[part[-1], part[0]] = 1
        
        elif connection_type == 'full':
            # Kết nối đầy đủ (fully connected)
            adj = torch.ones(num_landmarks, num_landmarks)
            # Loại bỏ self-loops
            adj.fill_diagonal_(0)
        
        else:  # 'spatial' - sẽ được tính từ tọa độ thực tế
            adj = torch.ones(num_landmarks, num_landmarks)
            adj.fill_diagonal_(0)
    
    else:
        # Default: fully connected
        adj = torch.ones(num_landmarks, num_landmarks)
        adj.fill_diagonal_(0)
    
    return adj


def compute_spatial_adjacency(landmarks, k=8, sigma=None):
    """
    Tạo adjacency matrix dựa trên khoảng cách không gian
    
    Mỗi node kết nối với k nearest neighbors.
    
    Args:
        landmarks: Tọa độ landmarks (N, 2) hoặc (N, 3)
        k: Số nearest neighbors
        sigma: Bandwidth cho Gaussian kernel (None = binary adjacency)
        
    Returns:
        adj: Adjacency matrix (N, N)
    """
    N = landmarks.shape[0]
    
    # Compute pairwise distances
    diff = landmarks.unsqueeze(0) - landmarks.unsqueeze(1)  # (N, N, D)
    dist = torch.norm(diff, dim=-1)  # (N, N)
    
    # Find k nearest neighbors
    _, indices = torch.topk(dist, k + 1, dim=-1, largest=False)
    
    # Create adjacency matrix
    adj = torch.zeros(N, N)
    for i in range(N):
        for j in indices[i, 1:]:  # Skip self (index 0)
            adj[i, j] = 1
            adj[j, i] = 1  # Symmetric
    
    if sigma is not None:
        # Gaussian kernel weighting
        adj = adj * torch.exp(-dist ** 2 / (2 * sigma ** 2))
    
    return adj


def normalize_adjacency(adj, add_self_loops=True):
    """
    Normalize adjacency matrix theo Kipf & Welling
    
    Ã = D^(-1/2) * (A + I) * D^(-1/2)
    
    Args:
        adj: Adjacency matrix (N, N)
        add_self_loops: Thêm self-loops (identity matrix)
        
    Returns:
        adj_norm: Normalized adjacency (N, N)
    """
    if add_self_loops:
        adj = adj + torch.eye(adj.size(0), device=adj.device)
    
    # Compute degree matrix
    degree = adj.sum(dim=-1)
    degree_inv_sqrt = torch.pow(degree, -0.5)
    degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0
    
    # D^(-1/2) * A * D^(-1/2)
    D_inv_sqrt = torch.diag(degree_inv_sqrt)
    adj_norm = torch.mm(torch.mm(D_inv_sqrt, adj), D_inv_sqrt)
    
    return adj_norm


# =====================================================================
# GCN LAYERS
# =====================================================================

class GCNLayer(nn.Module):
    """
    Graph Convolutional Layer (Kipf & Welling, 2017)
    
    H^{l+1} = σ(Ã * H^l * W)
    
    với Ã là normalized adjacency matrix.
    
    Args:
        in_features: Số features đầu vào cho mỗi node
        out_features: Số features đầu ra cho mỗi node
        bias: Có dùng bias không
    """
    
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # Learnable weights
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Khởi tạo parameters"""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x, adj):
        """
        Forward pass
        
        Args:
            x: Node features (B, N, in_features)
            adj: Normalized adjacency matrix (N, N)
            
        Returns:
            Output node features (B, N, out_features)
        """
        # Linear transformation: X * W
        support = torch.matmul(x, self.weight)  # (B, N, out_features)
        
        # Graph convolution: Ã * (X * W)
        # adj: (N, N), support: (B, N, out_features)
        output = torch.matmul(adj, support)  # (B, N, out_features)
        
        if self.bias is not None:
            output = output + self.bias
        
        return output


class GATLayer(nn.Module):
    """
    Graph Attention Layer (Veličković et al., 2018)
    
    Sử dụng attention mechanism để học trọng số cho các edges
    thay vì dùng fixed adjacency matrix.
    
    Args:
        in_features: Số features đầu vào
        out_features: Số features đầu ra
        num_heads: Số attention heads
        dropout: Dropout rate
        concat: Concat heads (True) hoặc average (False)
    """
    
    def __init__(self, in_features, out_features, num_heads=4, dropout=0.3, concat=True):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat
        
        # Linear transformation cho mỗi head
        self.W = nn.Parameter(torch.Tensor(num_heads, in_features, out_features))
        
        # Attention parameters
        self.a = nn.Parameter(torch.Tensor(num_heads, 2 * out_features, 1))
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)
    
    def forward(self, x, adj=None):
        """
        Forward pass
        
        Args:
            x: Node features (B, N, in_features)
            adj: Optional adjacency mask (N, N), 0 = masked
            
        Returns:
            Output node features (B, N, num_heads * out_features) if concat
                               (B, N, out_features) if not concat
        """
        B, N, _ = x.shape
        
        # Linear transformation: (B, N, H, out_features)
        h = torch.einsum('bni,hio->bnho', x, self.W)
        
        # Compute attention scores
        # Concatenate features cho mỗi pair of nodes
        h_repeat = h.unsqueeze(2).repeat(1, 1, N, 1, 1)  # (B, N, N, H, out)
        h_repeat_t = h.unsqueeze(1).repeat(1, N, 1, 1, 1)  # (B, N, N, H, out)
        concat_features = torch.cat([h_repeat, h_repeat_t], dim=-1)  # (B, N, N, H, 2*out)
        
        # Attention coefficients
        e = torch.einsum('bnnho,hol->bnnhl', concat_features, self.a).squeeze(-1)  # (B, N, N, H)
        e = self.leaky_relu(e)
        
        # Mask với adjacency matrix nếu có
        if adj is not None:
            mask = adj.unsqueeze(0).unsqueeze(-1)  # (1, N, N, 1)
            e = e.masked_fill(mask == 0, float('-inf'))
        
        # Softmax attention
        attention = F.softmax(e, dim=2)  # (B, N, N, H)
        attention = self.dropout(attention)
        
        # Weighted sum
        # attention: (B, N, N, H), h: (B, N, H, out)
        h_t = h.permute(0, 2, 1, 3)  # (B, H, N, out)
        attention_t = attention.permute(0, 3, 1, 2)  # (B, H, N, N)
        
        output = torch.matmul(attention_t, h_t)  # (B, H, N, out)
        output = output.permute(0, 2, 1, 3)  # (B, N, H, out)
        
        if self.concat:
            output = output.reshape(B, N, -1)  # (B, N, H * out)
        else:
            output = output.mean(dim=2)  # (B, N, out)
        
        return output


# =====================================================================
# GCN MODULE
# =====================================================================

class GCNModule(nn.Module):
    """
    Complete GCN Module cho facial landmark analysis
    
    Kiến trúc:
    1. Input: Tọa độ landmarks (N, 2)
    2. GCN layers: Học graph representation
    3. Pooling: Aggregate node features
    4. Output: Feature vector cho classification
    
    Args:
        num_landmarks: Số landmarks (default: 68)
        input_dim: Kích thước feature đầu vào cho mỗi node (2 cho x,y)
        hidden_dims: List các hidden dimensions
        output_dim: Kích thước output feature
        gcn_type: 'gcn' hoặc 'gat'
        num_heads: Số heads cho GAT
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        num_landmarks=GCNConfig.NUM_LANDMARKS,
        input_dim=GCNConfig.INPUT_DIM,
        hidden_dims=None,
        output_dim=GCNConfig.OUTPUT_DIM,
        gcn_type=GCNConfig.GCN_TYPE,
        num_heads=GCNConfig.NUM_HEADS,
        dropout=GCNConfig.DROPOUT
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = GCNConfig.HIDDEN_DIMS
        
        print(f"\n🏗️  Đang xây dựng GCN Module...")
        print(f"   Num landmarks: {num_landmarks}")
        print(f"   GCN type: {gcn_type}")
        
        self.num_landmarks = num_landmarks
        self.input_dim = input_dim
        self.gcn_type = gcn_type
        
        # Input projection (tọa độ raw -> initial features)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # GCN layers
        self.gcn_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        dims = [hidden_dims[0]] + hidden_dims
        
        for i in range(len(dims) - 1):
            if gcn_type == 'gat':
                # GAT layer
                layer = GATLayer(
                    in_features=dims[i],
                    out_features=dims[i+1] // num_heads if i < len(dims) - 2 else dims[i+1],
                    num_heads=num_heads,
                    dropout=dropout,
                    concat=(i < len(dims) - 2)  # Không concat ở layer cuối
                )
            else:
                # Standard GCN layer
                layer = GCNLayer(
                    in_features=dims[i],
                    out_features=dims[i+1]
                )
            
            self.gcn_layers.append(layer)
            self.norms.append(nn.LayerNorm(dims[i+1]))
        
        # Graph pooling
        # Sử dụng mean pooling + max pooling và concat
        self.pool = GraphPooling(pool_type='mean_max')
        
        # Output projection
        final_dim = dims[-1] * 2  # *2 vì dùng mean_max pooling
        self.output_proj = nn.Sequential(
            nn.Linear(final_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim)
        )
        
        # Default adjacency matrix (sẽ được normalize)
        adj = create_facial_adjacency_matrix(num_landmarks)
        adj = normalize_adjacency(adj)
        self.register_buffer('default_adj', adj)
        
        # Print model info
        total_params = sum(p.numel() for p in self.parameters())
        print(f"✅ GCN Module khởi tạo thành công!")
        print(f"📊 Total parameters: {total_params:,}")
    
    def forward(self, landmarks, adj=None):
        """
        Forward pass
        
        Args:
            landmarks: Landmark coordinates (B, N, 2) hoặc (B, N, 3)
            adj: Optional custom adjacency matrix (N, N)
            
        Returns:
            features: Graph-level features (B, output_dim)
        """
        B = landmarks.shape[0]
        
        # Normalize landmarks (center và scale)
        landmarks = self._normalize_landmarks(landmarks)
        
        # Input projection
        x = self.input_proj(landmarks)  # (B, N, hidden_dim)
        
        # Get adjacency matrix
        if adj is None:
            adj = self.default_adj
        
        # GCN layers
        for gcn, norm in zip(self.gcn_layers, self.norms):
            if self.gcn_type == 'gat':
                x = gcn(x, adj)
            else:
                x = gcn(x, adj)
            x = norm(x)
            x = F.relu(x)
        
        # Graph pooling
        x = self.pool(x)  # (B, final_dim)
        
        # Output projection
        features = self.output_proj(x)  # (B, output_dim)
        
        return features
    
    def _normalize_landmarks(self, landmarks):
        """
        Normalize landmarks để scale và position invariance
        
        Args:
            landmarks: (B, N, D)
            
        Returns:
            normalized landmarks: (B, N, D)
        """
        # Center around mean
        mean = landmarks.mean(dim=1, keepdim=True)
        landmarks = landmarks - mean
        
        # Scale to unit variance
        std = landmarks.std()
        if std > 0:
            landmarks = landmarks / std
        
        return landmarks


class GraphPooling(nn.Module):
    """
    Graph-level pooling để aggregate node features
    
    Args:
        pool_type: 'mean', 'max', 'sum', hoặc 'mean_max'
    """
    
    def __init__(self, pool_type='mean_max'):
        super().__init__()
        self.pool_type = pool_type
    
    def forward(self, x):
        """
        Args:
            x: Node features (B, N, D)
            
        Returns:
            Graph features (B, D) hoặc (B, 2*D) cho mean_max
        """
        if self.pool_type == 'mean':
            return x.mean(dim=1)
        elif self.pool_type == 'max':
            return x.max(dim=1)[0]
        elif self.pool_type == 'sum':
            return x.sum(dim=1)
        elif self.pool_type == 'mean_max':
            mean = x.mean(dim=1)
            max_pool = x.max(dim=1)[0]
            return torch.cat([mean, max_pool], dim=-1)
        else:
            raise ValueError(f"Unknown pool type: {self.pool_type}")


# =====================================================================
# GCN + CNN FUSION
# =====================================================================

class CNNGCNFusion(nn.Module):
    """
    Kết hợp CNN features và GCN features
    
    Như đề xuất trong Samad & Bandhu (2025):
    - CNN trích xuất pixel-level features
    - GCN trích xuất structural features từ landmarks
    - Fusion để có cả hai loại thông tin
    
    Args:
        cnn_feature_dim: Kích thước CNN features
        gcn_feature_dim: Kích thước GCN features
        fusion_dim: Kích thước sau fusion
        num_classes: Số classes output
    """
    
    def __init__(
        self,
        cnn_feature_dim=512,
        gcn_feature_dim=256,
        fusion_dim=256,
        num_classes=2
    ):
        super().__init__()
        
        # Projection layers
        self.cnn_proj = nn.Linear(cnn_feature_dim, fusion_dim)
        self.gcn_proj = nn.Linear(gcn_feature_dim, fusion_dim)
        
        # Attention-based fusion
        self.attention = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.Tanh(),
            nn.Linear(fusion_dim, 2),
            nn.Softmax(dim=-1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, num_classes)
        )
    
    def forward(self, cnn_features, gcn_features):
        """
        Args:
            cnn_features: CNN features (B, cnn_dim)
            gcn_features: GCN features (B, gcn_dim)
            
        Returns:
            logits: Classification logits (B, num_classes)
            fused_features: Fused features (B, fusion_dim)
        """
        # Project to same dimension
        cnn_proj = self.cnn_proj(cnn_features)  # (B, fusion_dim)
        gcn_proj = self.gcn_proj(gcn_features)  # (B, fusion_dim)
        
        # Attention weights
        concat = torch.cat([cnn_proj, gcn_proj], dim=-1)  # (B, 2*fusion_dim)
        weights = self.attention(concat)  # (B, 2)
        
        # Weighted sum
        fused = weights[:, 0:1] * cnn_proj + weights[:, 1:2] * gcn_proj  # (B, fusion_dim)
        
        # Classify
        logits = self.classifier(fused)
        
        return logits, fused


# =====================================================================
# MAIN (Test)
# =====================================================================

if __name__ == "__main__":
    """
    Test GCN module khi chạy file này trực tiếp
    """
    print("=" * 70)
    print("🧪 TEST GCN MODULE")
    print("=" * 70)
    
    # Test adjacency matrix creation
    print("\n1. Test tạo adjacency matrix...")
    adj = create_facial_adjacency_matrix(68, connection_type='sequential')
    print(f"   Adjacency matrix shape: {adj.shape}")
    print(f"   Số connections: {adj.sum().item()}")
    
    # Test normalize adjacency
    print("\n2. Test normalize adjacency...")
    adj_norm = normalize_adjacency(adj)
    print(f"   Normalized adjacency shape: {adj_norm.shape}")
    
    # Test GCN layer
    print("\n3. Test GCN Layer...")
    gcn_layer = GCNLayer(2, 64)
    x = torch.randn(2, 68, 2)  # (B, N, features)
    out = gcn_layer(x, adj_norm)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {out.shape}")
    
    # Test GAT layer
    print("\n4. Test GAT Layer...")
    gat_layer = GATLayer(2, 32, num_heads=4)
    out_gat = gat_layer(x, adj)
    print(f"   GAT output shape: {out_gat.shape}")
    
    # Test complete GCN module
    print("\n5. Test GCN Module...")
    gcn_module = GCNModule(
        num_landmarks=68,
        input_dim=2,
        hidden_dims=[64, 128],
        output_dim=128,
        gcn_type='gcn'
    )
    
    landmarks = torch.randn(2, 68, 2)
    features = gcn_module(landmarks)
    print(f"   Input landmarks shape: {landmarks.shape}")
    print(f"   Output features shape: {features.shape}")
    
    # Test GAT module
    print("\n6. Test GCN Module với GAT...")
    gcn_module_gat = GCNModule(
        num_landmarks=68,
        input_dim=2,
        hidden_dims=[64, 64],
        output_dim=128,
        gcn_type='gat',
        num_heads=4
    )
    
    features_gat = gcn_module_gat(landmarks)
    print(f"   GAT output features shape: {features_gat.shape}")
    
    # Test CNN-GCN fusion
    print("\n7. Test CNN-GCN Fusion...")
    fusion = CNNGCNFusion(
        cnn_feature_dim=512,
        gcn_feature_dim=128,
        fusion_dim=256,
        num_classes=2
    )
    
    cnn_feat = torch.randn(2, 512)
    gcn_feat = torch.randn(2, 128)
    logits, fused = fusion(cnn_feat, gcn_feat)
    print(f"   CNN features: {cnn_feat.shape}")
    print(f"   GCN features: {gcn_feat.shape}")
    print(f"   Fused features: {fused.shape}")
    print(f"   Logits: {logits.shape}")
    
    print("\n" + "=" * 70)
    print("✅ HOÀN THÀNH TEST GCN MODULE!")
    print("=" * 70)
