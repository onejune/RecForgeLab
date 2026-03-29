# -*- coding: utf-8 -*-
"""
AutoInt: Attention-based Feature Interaction Model
论文: AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks (CIKM 2019)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from ..base import CTRModel, register_model
from ..layers import FeatureEmbedding, MLPLayers


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力层（带残差 + LayerNorm）"""

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, num_fields, embed_dim]
        Returns:
            [batch, num_fields, embed_dim]
        """
        B, N, _ = x.shape
        residual = x

        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = self.dropout(F.softmax(scores, dim=-1))

        out = torch.matmul(attn, v)  # [B, heads, N, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, N, self.embed_dim)
        out = self.out_proj(out)

        return self.layer_norm(residual + self.dropout(out))


@register_model("autoint")
class AutoInt(CTRModel):
    """AutoInt: 自动特征交互学习

    通过多头自注意力机制自动学习高阶特征交互
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.num_heads = config.get("num_heads", 4)
        self.num_attention_layers = config.get("num_attention_layers", 3)
        self.dropout_prob = config.get("dropout_prob", 0.1)

        # 特征嵌入（复用框架统一接口）
        self.feature_embedding = FeatureEmbedding(
            sparse_vocab=dataset.sparse_vocab,
            sparse_feats=dataset.sparse_features,
            dense_dim=len(dataset.dense_features),
            embedding_dim=config["embedding_size"],
            encoder_type=config.get("encoder_type", "bucket"),
            encoder_config=config.get("encoder_config", {}),
        )
        self.embedding_size = config["embedding_size"]

        # 计算字段数（用于 attention）
        self.num_sparse = len(dataset.sparse_features)
        self.num_dense = len(dataset.dense_features)

        # 注意力层（作用在 3D embedding 上）
        self.attention_layers = nn.ModuleList([
            MultiHeadSelfAttention(
                embed_dim=self.embedding_size,
                num_heads=self.num_heads,
                dropout=self.dropout_prob,
            )
            for _ in range(self.num_attention_layers)
        ])

        # 输出层（attention 后展平 -> logit）
        # 注意：dense 编码器输出维度可能不是 embedding_size 的整数倍
        # 这里只对 sparse 部分做 attention，dense 直接拼接
        dense_output_dim = self.feature_embedding.dense_output_dim
        attn_out_dim = self.num_sparse * self.embedding_size

        # 可选 MLP
        mlp_hidden = config.get("mlp_hidden_size", [])
        if mlp_hidden:
            total_input = attn_out_dim + dense_output_dim
            self.mlp = nn.Sequential(
                MLPLayers(
                    layers=[total_input] + mlp_hidden,
                    dropout=self.dropout_prob,
                    bn=True,
                    last_activation=True,
                ),
                nn.Linear(mlp_hidden[-1], 1),
            )
        else:
            self.mlp = None
            self.output_layer = nn.Linear(attn_out_dim + dense_output_dim, 1)

    def _get_sparse_3d(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """获取稀疏特征的 3D embedding [B, num_sparse, emb_dim]"""
        B = x[list(x.keys())[0]].size(0)
        device = x[list(x.keys())[0]].device
        emb_list = []
        for feat in self.sparse_features:
            if feat in self.feature_embedding.embeddings:
                emb_list.append(self.feature_embedding.embeddings[feat](x[feat]))
            else:
                emb_list.append(torch.zeros(B, self.embedding_size, device=device))
        return torch.stack(emb_list, dim=1)  # [B, num_sparse, emb_dim]

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        B = x[list(x.keys())[0]].size(0)

        # Sparse: 3D attention
        if self.num_sparse > 0:
            sparse_3d = self._get_sparse_3d(x)  # [B, num_sparse, emb_dim]
            for attn in self.attention_layers:
                sparse_3d = attn(sparse_3d)
            attn_flat = sparse_3d.view(B, -1)  # [B, num_sparse * emb_dim]
        else:
            attn_flat = torch.zeros(B, 0, device=x[list(x.keys())[0]].device)

        # Dense: 直接编码
        parts = [attn_flat]
        if self.feature_embedding.dense_encoder is not None and "dense" in x:
            dense_out = self.feature_embedding.dense_encoder(x["dense"])
            parts.append(dense_out)

        combined = torch.cat(parts, dim=-1)

        if self.mlp is not None:
            return self.mlp(combined).squeeze(-1)
        else:
            return self.output_layer(combined).squeeze(-1)

    def predict(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))

    def calculate_loss(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(self.forward(x), x[self.label_field].float())


@register_model("autoint+")
class AutoIntPlus(AutoInt):
    """AutoInt+: AutoInt + FM 二阶交叉"""

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        # FM 二阶 bias
        self.fm_bias = nn.Parameter(torch.zeros(1))

    def _fm_second_order(self, emb_3d: torch.Tensor) -> torch.Tensor:
        """FM 二阶交叉"""
        sum_sq = emb_3d.sum(dim=1) ** 2  # [B, emb_dim]
        sq_sum = (emb_3d ** 2).sum(dim=1)  # [B, emb_dim]
        return 0.5 * (sum_sq - sq_sum).sum(dim=-1)  # [B]

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        B = x[list(x.keys())[0]].size(0)

        if self.num_sparse > 0:
            sparse_3d = self._get_sparse_3d(x)
            fm_out = self._fm_second_order(sparse_3d)

            for attn in self.attention_layers:
                sparse_3d = attn(sparse_3d)
            attn_flat = sparse_3d.view(B, -1)
        else:
            fm_out = torch.zeros(B, device=x[list(x.keys())[0]].device)
            attn_flat = torch.zeros(B, 0, device=fm_out.device)

        parts = [attn_flat]
        if self.feature_embedding.dense_encoder is not None and "dense" in x:
            parts.append(self.feature_embedding.dense_encoder(x["dense"]))

        combined = torch.cat(parts, dim=-1)

        if self.mlp is not None:
            logit = self.mlp(combined).squeeze(-1)
        else:
            logit = self.output_layer(combined).squeeze(-1)

        return logit + fm_out + self.fm_bias
