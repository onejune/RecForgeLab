# -*- coding: utf-8 -*-
"""
特征嵌入层 + 连续特征编码器
直接迁移自 autoresearch/continuous_features/feature_encoders.py，统一接口
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


# ============================================================================
# 连续特征编码器基类
# ============================================================================

class BaseEncoder(nn.Module):
    """所有连续特征编码器的基类"""

    @property
    def output_dim(self) -> int:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# ============================================================================
# 编码器实现（完整版，直接迁移自 continuous_features/feature_encoders.py）
# ============================================================================

class NoneEncoder(BaseEncoder):
    """不使用连续特征（ablation: 纯类别特征）"""

    def __init__(self, n_continuous: int = 8):
        super().__init__()
        self.n_continuous = n_continuous

    @property
    def output_dim(self) -> int:
        return 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :0]


class ScalarEncoder(BaseEncoder):
    """直接使用连续值，不做 embedding"""

    def __init__(self, n_continuous: int = 8):
        super().__init__()
        self.n_continuous = n_continuous

    @property
    def output_dim(self) -> int:
        return self.n_continuous

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class BucketEncoder(BaseEncoder):
    """等频分桶后做 embedding lookup"""

    def __init__(self, n_continuous: int = 8, n_buckets: int = 10, embedding_dim: int = 16):
        super().__init__()
        self.n_continuous = n_continuous
        self.n_buckets = n_buckets
        self.embedding_dim = embedding_dim

        self.embeddings = nn.ModuleList([
            nn.Embedding(n_buckets + 1, embedding_dim)
            for _ in range(n_continuous)
        ])
        self.register_buffer("boundaries", torch.zeros(n_continuous, n_buckets - 1))
        self._fitted = False

    @property
    def output_dim(self) -> int:
        return self.n_continuous * self.embedding_dim

    def fit(self, x: np.ndarray):
        """用训练数据计算等频分桶边界"""
        boundaries = []
        for i in range(self.n_continuous):
            col = x[:, i]
            quantiles = np.percentile(col, np.linspace(0, 100, self.n_buckets + 1)[1:-1])
            boundaries.append(quantiles)
        self.boundaries = torch.tensor(np.array(boundaries, dtype=np.float32), device=self.boundaries.device)
        self._fitted = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        bucket_ids = torch.zeros(batch, self.n_continuous, dtype=torch.long, device=x.device)
        for i in range(self.n_continuous):
            bucket_ids[:, i] = torch.bucketize(x[:, i].contiguous(), self.boundaries[i])
        embs = [self.embeddings[i](bucket_ids[:, i]) for i in range(self.n_continuous)]
        return torch.cat(embs, dim=-1)


class AutoDisEncoder(BaseEncoder):
    """AutoDis：软分桶，可学习的元 embedding 加权"""

    def __init__(self, n_continuous: int = 8, n_meta_embeddings: int = 16,
                 embedding_dim: int = 16, temperature: float = 1.0):
        super().__init__()
        self.n_continuous = n_continuous
        self.n_meta_embeddings = n_meta_embeddings
        self.embedding_dim = embedding_dim
        self.temperature = temperature

        self.meta_embeddings = nn.Parameter(
            torch.randn(n_continuous, n_meta_embeddings, embedding_dim) * 0.01
        )
        self.weight_nets = nn.ModuleList([
            nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, n_meta_embeddings))
            for _ in range(n_continuous)
        ])

    @property
    def output_dim(self) -> int:
        return self.n_continuous * self.embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for i in range(self.n_continuous):
            xi = x[:, i:i+1]
            logits = self.weight_nets[i](xi)
            weights = F.softmax(logits / self.temperature, dim=-1)
            emb = torch.matmul(weights, self.meta_embeddings[i])
            outputs.append(emb)
        return torch.cat(outputs, dim=-1)


class NumericEmbeddingEncoder(BaseEncoder):
    """每个特征独立小 MLP：scalar → embedding"""

    def __init__(self, n_continuous: int = 8, embedding_dim: int = 16, hidden_dim: int = 64):
        super().__init__()
        self.n_continuous = n_continuous
        self.embedding_dim = embedding_dim
        self.mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, embedding_dim))
            for _ in range(n_continuous)
        ])

    @property
    def output_dim(self) -> int:
        return self.n_continuous * self.embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [self.mlps[i](x[:, i:i+1]) for i in range(self.n_continuous)]
        return torch.cat(outputs, dim=-1)


class NumericEmbeddingDeepEncoder(BaseEncoder):
    """NumericEmbedding 加深版（3层MLP）"""

    def __init__(self, n_continuous: int = 8, embedding_dim: int = 16, hidden_dim: int = 64):
        super().__init__()
        self.n_continuous = n_continuous
        self.embedding_dim = embedding_dim
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, embedding_dim),
            ) for _ in range(n_continuous)
        ])

    @property
    def output_dim(self) -> int:
        return self.n_continuous * self.embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.mlps[i](x[:, i:i+1]) for i in range(self.n_continuous)], dim=-1)


class NumericEmbeddingSiLUEncoder(BaseEncoder):
    """NumericEmbedding SiLU 激活版"""

    def __init__(self, n_continuous: int = 8, embedding_dim: int = 16, hidden_dim: int = 64):
        super().__init__()
        self.n_continuous = n_continuous
        self.embedding_dim = embedding_dim
        self.mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, embedding_dim))
            for _ in range(n_continuous)
        ])

    @property
    def output_dim(self) -> int:
        return self.n_continuous * self.embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.mlps[i](x[:, i:i+1]) for i in range(self.n_continuous)], dim=-1)


class NumericEmbeddingLNEncoder(BaseEncoder):
    """NumericEmbedding + LayerNorm 版"""

    def __init__(self, n_continuous: int = 8, embedding_dim: int = 16, hidden_dim: int = 64):
        super().__init__()
        self.n_continuous = n_continuous
        self.embedding_dim = embedding_dim
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, embedding_dim),
            ) for _ in range(n_continuous)
        ])

    @property
    def output_dim(self) -> int:
        return self.n_continuous * self.embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.mlps[i](x[:, i:i+1]) for i in range(self.n_continuous)], dim=-1)


class NumericEmbeddingContextualEncoder(BaseEncoder):
    """Contextual NumericEmbedding：MLP + 域 embedding（FIVES 风格）"""

    def __init__(self, n_continuous: int = 8, embedding_dim: int = 16, hidden_dim: int = 64):
        super().__init__()
        self.n_continuous = n_continuous
        self.embedding_dim = embedding_dim
        self.mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, embedding_dim))
            for _ in range(n_continuous)
        ])
        self.field_bias = nn.Parameter(torch.randn(n_continuous, embedding_dim) * 0.01)

    @property
    def output_dim(self) -> int:
        return self.n_continuous * self.embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [self.mlps[i](x[:, i:i+1]) + self.field_bias[i].unsqueeze(0) for i in range(self.n_continuous)]
        return torch.cat(outputs, dim=-1)


class FTTransformerEncoder(BaseEncoder):
    """FT-Transformer：线性投影 + Transformer Encoder"""

    def __init__(self, n_continuous: int = 8, d_model: int = 32, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.n_continuous = n_continuous
        self.d_model = d_model

        self.W = nn.Parameter(torch.randn(n_continuous, d_model) * 0.01)
        self.b = nn.Parameter(torch.zeros(n_continuous, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, dropout=0.0, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    @property
    def output_dim(self) -> int:
        return self.n_continuous * self.d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.unsqueeze(-1) * self.W.unsqueeze(0) + self.b.unsqueeze(0)
        out = self.transformer(tokens)
        return out.reshape(out.shape[0], -1)


class PeriodicEncoder(BaseEncoder):
    """周期性激活函数编码（NeurIPS 2021）"""

    def __init__(self, n_continuous: int = 8, n_frequencies: int = 16, sigma: float = 1.0):
        super().__init__()
        self.n_continuous = n_continuous
        self.n_frequencies = n_frequencies
        self.w = nn.Parameter(torch.randn(n_continuous, n_frequencies) * sigma)
        self.b = nn.Parameter(torch.randn(n_continuous, n_frequencies) * sigma)

    @property
    def output_dim(self) -> int:
        return self.n_continuous * 2 * self.n_frequencies

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.unsqueeze(-1) * self.w.unsqueeze(0) + self.b.unsqueeze(0)
        out = torch.cat([torch.sin(z), torch.cos(z)], dim=-1)
        return out.reshape(out.shape[0], -1)


class FieldEmbeddingEncoder(BaseEncoder):
    """域嵌入：v_i * x_i（FM/DeepFM 经典做法）"""

    def __init__(self, n_continuous: int = 8, embedding_dim: int = 16):
        super().__init__()
        self.n_continuous = n_continuous
        self.embedding_dim = embedding_dim
        self.field_embeddings = nn.Parameter(torch.randn(n_continuous, embedding_dim) * 0.01)

    @property
    def output_dim(self) -> int:
        return self.n_continuous * self.embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.unsqueeze(-1) * self.field_embeddings.unsqueeze(0)
        return out.reshape(out.shape[0], -1)


class DLRMEncoder(BaseEncoder):
    """DLRM 风格：所有连续特征共享 MLP 压缩"""

    def __init__(self, n_continuous: int = 8, embedding_dim: int = 16, hidden_dim: int = 64):
        super().__init__()
        self.n_continuous = n_continuous
        self.embedding_dim = embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(n_continuous, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    @property
    def output_dim(self) -> int:
        return self.embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class PLREncoder(BaseEncoder):
    """PLR：分段线性表示（NeurIPS 2022）"""

    def __init__(self, n_continuous: int = 8, embedding_dim: int = 16, n_bins: int = 16):
        super().__init__()
        self.n_continuous = n_continuous
        self.embedding_dim = embedding_dim
        self.n_bins = n_bins
        self.w = nn.Parameter(torch.randn(n_continuous, n_bins) * 0.01)
        self.b = nn.Parameter(torch.zeros(n_continuous, n_bins))
        self.linears = nn.ModuleList([nn.Linear(n_bins, embedding_dim) for _ in range(n_continuous)])

    @property
    def output_dim(self) -> int:
        return self.n_continuous * self.embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.relu(x.unsqueeze(-1) * self.w.unsqueeze(0) + self.b.unsqueeze(0))
        outputs = [self.linears[i](z[:, i, :]) for i in range(self.n_continuous)]
        return torch.cat(outputs, dim=-1)


class MinMaxEncoder(BaseEncoder):
    """Min-Max 归一化"""

    def __init__(self, n_continuous: int = 8):
        super().__init__()
        self.n_continuous = n_continuous
        self.register_buffer("x_min", torch.zeros(n_continuous))
        self.register_buffer("x_max", torch.ones(n_continuous))

    @property
    def output_dim(self) -> int:
        return self.n_continuous

    def fit(self, x: np.ndarray):
        self.x_min = torch.tensor(x.min(axis=0), dtype=torch.float32)
        self.x_max = torch.tensor(x.max(axis=0), dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.x_min.to(x.device)) / (self.x_max.to(x.device) - self.x_min.to(x.device) + 1e-8)


class LogTransformEncoder(BaseEncoder):
    """对数变换：适用于长尾分布（如点击次数）"""

    def __init__(self, n_continuous: int = 8):
        super().__init__()
        self.n_continuous = n_continuous

    @property
    def output_dim(self) -> int:
        return self.n_continuous

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log1p(x.clamp(min=0.0))


# ============================================================================
# 编码器工厂
# ============================================================================

ENCODER_REGISTRY = {
    "none":          NoneEncoder,
    "scalar":        ScalarEncoder,
    "bucket":        BucketEncoder,
    "autodis":       AutoDisEncoder,
    "numeric":       NumericEmbeddingEncoder,
    "numeric_deep":  NumericEmbeddingDeepEncoder,
    "numeric_silu":  NumericEmbeddingSiLUEncoder,
    "numeric_ln":    NumericEmbeddingLNEncoder,
    "numeric_ctx":   NumericEmbeddingContextualEncoder,
    "fttransformer": FTTransformerEncoder,
    "periodic":      PeriodicEncoder,
    "field":         FieldEmbeddingEncoder,
    "dlrm":          DLRMEncoder,
    "plr":           PLREncoder,
    "minmax":        MinMaxEncoder,
    "log":           LogTransformEncoder,
}


def build_encoder(encoder_type: str, n_continuous: int, embedding_dim: int = 16, **kwargs) -> BaseEncoder:
    """构建连续特征编码器
    
    Args:
        encoder_type: 编码器类型名称
        n_continuous: 连续特征数量
        embedding_dim: 嵌入维度
        **kwargs: 额外参数
    """
    encoder_type = encoder_type.lower()
    if encoder_type not in ENCODER_REGISTRY:
        raise ValueError(f"未知编码器: {encoder_type}，可选: {list(ENCODER_REGISTRY.keys())}")
    
    cls = ENCODER_REGISTRY[encoder_type]
    
    # 根据编码器类型传入参数
    if encoder_type in ("none", "scalar", "minmax", "log"):
        return cls(n_continuous=n_continuous)
    elif encoder_type == "fttransformer":
        d_model = kwargs.get("d_model", embedding_dim)
        return cls(n_continuous=n_continuous, d_model=d_model,
                   n_heads=kwargs.get("n_heads", 4), n_layers=kwargs.get("n_layers", 2))
    elif encoder_type == "periodic":
        return cls(n_continuous=n_continuous, n_frequencies=kwargs.get("n_frequencies", 16))
    elif encoder_type == "plr":
        return cls(n_continuous=n_continuous, embedding_dim=embedding_dim, n_bins=kwargs.get("n_bins", 16))
    elif encoder_type == "dlrm":
        return cls(n_continuous=n_continuous, embedding_dim=embedding_dim)
    elif encoder_type == "field":
        return cls(n_continuous=n_continuous, embedding_dim=embedding_dim)
    elif encoder_type == "bucket":
        return cls(n_continuous=n_continuous, n_buckets=kwargs.get("n_buckets", 10), embedding_dim=embedding_dim)
    elif encoder_type == "autodis":
        return cls(n_continuous=n_continuous, n_meta_embeddings=kwargs.get("n_meta_embeddings", 16),
                   embedding_dim=embedding_dim)
    else:
        return cls(n_continuous=n_continuous, embedding_dim=embedding_dim)


# ============================================================================
# 统一特征嵌入层（类别 + 连续）
# ============================================================================

class FeatureEmbedding(nn.Module):
    """统一特征嵌入层
    
    类别特征：独立 Embedding 表
    连续特征：可配置的编码器
    """
    
    def __init__(
        self,
        sparse_vocab: Dict[str, int],   # {feat_name: vocab_size}
        sparse_feats: List[str],         # 特征名列表（顺序固定）
        dense_dim: int,                  # 连续特征数量
        embedding_dim: int = 16,
        encoder_type: str = "bucket",
        encoder_config: Optional[Dict] = None,
    ):
        super().__init__()
        
        self.sparse_feats = sparse_feats
        self.embedding_dim = embedding_dim
        self.dense_dim = dense_dim
        
        # 类别特征嵌入
        self.embeddings = nn.ModuleDict({
            feat: nn.Embedding(vocab_size + 1, embedding_dim, padding_idx=0)
            for feat, vocab_size in sparse_vocab.items()
            if feat in sparse_feats
        })
        
        # 连续特征编码器
        if dense_dim > 0:
            encoder_config = encoder_config or {}
            self.dense_encoder = build_encoder(encoder_type, dense_dim, embedding_dim, **encoder_config)
            self.dense_output_dim = self.dense_encoder.output_dim
        else:
            self.dense_encoder = None
            self.dense_output_dim = 0
        
        # 输出维度
        self.output_dim = len(sparse_feats) * embedding_dim + self.dense_output_dim
    
    def forward(
        self,
        x: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 特征字典，包含 sparse 特征（long）和 "dense"（float）
        
        Returns:
            [batch_size, output_dim]
        """
        ref_key = list(x.keys())[0]
        device = x[ref_key].device
        B = x[ref_key].size(0)
        
        # 类别特征嵌入
        emb_list = []
        for feat in self.sparse_feats:
            if feat in self.embeddings:
                emb_list.append(self.embeddings[feat](x[feat]))
            else:
                emb_list.append(torch.zeros(B, self.embedding_dim, device=device))
        
        # 连续特征编码
        parts = emb_list
        if self.dense_encoder is not None and "dense" in x:
            dense_out = self.dense_encoder(x["dense"])
            parts = emb_list + [dense_out]
        
        return torch.cat(parts, dim=-1)
    
    def get_3d_embeddings(self, x: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取 3D 嵌入（用于 FM 二阶交叉）
        
        Returns:
            emb_3d: [batch_size, n_sparse, embedding_dim]
            emb_flat: [batch_size, output_dim]
        """
        ref_key = list(x.keys())[0]
        device = x[ref_key].device
        B = x[ref_key].size(0)
        
        emb_list = []
        for feat in self.sparse_feats:
            if feat in self.embeddings:
                emb_list.append(self.embeddings[feat](x[feat]))
            else:
                emb_list.append(torch.zeros(B, self.embedding_dim, device=device))
        
        emb_3d = torch.stack(emb_list, dim=1)  # [B, n_sparse, emb_dim]
        
        parts = emb_list
        if self.dense_encoder is not None and "dense" in x:
            parts = emb_list + [self.dense_encoder(x["dense"])]
        
        emb_flat = torch.cat(parts, dim=-1)
        
        return emb_3d, emb_flat
