# -*- coding: utf-8 -*-
"""
xDeepFM: Compressed Interaction Network
论文: xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems (KDD 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List

from ..base import CTRModel, register_model
from ..layers import FeatureEmbedding, MLPLayers


class CompressedInteractionNetwork(nn.Module):
    """压缩交互网络 (CIN)

    向量级特征交叉，显式学习有界阶特征交互
    """

    def __init__(
        self,
        embed_dim: int,
        num_fields: int,
        cin_layer_sizes: List[int],
        split_half: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_fields = num_fields
        self.split_half = split_half

        self.conv_layers = nn.ModuleList()
        prev_size = num_fields

        for i, layer_size in enumerate(cin_layer_sizes):
            self.conv_layers.append(
                nn.Conv1d(
                    in_channels=prev_size * num_fields,
                    out_channels=layer_size,
                    kernel_size=1,
                )
            )
            if split_half and i < len(cin_layer_sizes) - 1:
                prev_size = layer_size // 2
            else:
                prev_size = layer_size

        # 输出维度（sum pooling 后）
        if split_half:
            self.output_dim = sum(s // 2 for s in cin_layer_sizes[:-1]) + cin_layer_sizes[-1]
        else:
            self.output_dim = sum(cin_layer_sizes)

    def forward(self, embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embed: [batch, num_fields, embed_dim]
        Returns:
            [batch, output_dim]  (sum pooled over embed_dim)
        """
        B = embed.shape[0]
        X = embed  # [B, H_k, D]
        X0 = embed  # 原始 embedding，固定不变

        cross_outputs = []

        for i, conv in enumerate(self.conv_layers):
            # 外积: [B, H_{k-1}, D] x [B, H_0, D] -> [B, H_{k-1}*H_0, D]
            outer = torch.einsum('bmd,bnd->bmnd', X, X0)  # [B, H_{k-1}, H_0, D]
            outer = outer.view(B, -1, self.embed_dim)      # [B, H_{k-1}*H_0, D]

            # Conv1d: [B, in_channels, D] -> [B, layer_size, D]
            cross = F.relu(conv(outer))

            if self.split_half and i < len(self.conv_layers) - 1:
                X, cross_out = torch.split(cross, cross.shape[1] // 2, dim=1)
            else:
                cross_out = cross
                X = cross

            # sum pooling over embed_dim
            cross_outputs.append(cross_out.sum(dim=-1))  # [B, half_size]

        return torch.cat(cross_outputs, dim=1)  # [B, output_dim]


@register_model("xdeepfm")
class xDeepFM(CTRModel):
    """xDeepFM: CIN + DNN + Linear

    - CIN: 显式学习有界阶向量级特征交互
    - DNN: 隐式学习高阶特征交互
    - Linear: 一阶特征
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.cin_layer_sizes = config.get("cin_layer_sizes", [128, 128, 64])
        self.split_half = config.get("split_half", True)
        self.dropout_prob = config.get("dropout_prob", 0.1)

        # 特征嵌入
        self.feature_embedding = FeatureEmbedding(
            sparse_vocab=dataset.sparse_vocab,
            sparse_feats=dataset.sparse_features,
            dense_dim=len(dataset.dense_features),
            embedding_dim=config["embedding_size"],
            encoder_type=config.get("encoder_type", "bucket"),
            encoder_config=config.get("encoder_config", {}),
        )
        input_dim = self.feature_embedding.output_dim
        num_sparse = len(dataset.sparse_features)

        # Linear 层（一阶）
        self.linear = nn.Linear(input_dim, 1, bias=True)

        # CIN（只作用在稀疏特征的 3D embedding 上）
        self.cin = CompressedInteractionNetwork(
            embed_dim=config["embedding_size"],
            num_fields=num_sparse,
            cin_layer_sizes=self.cin_layer_sizes,
            split_half=self.split_half,
        )
        self.cin_output_layer = nn.Linear(self.cin.output_dim, 1, bias=False)

        # DNN
        mlp_hidden = config.get("mlp_hidden_size", [256, 128, 64])
        self.dnn = nn.Sequential(
            MLPLayers(
                layers=[input_dim] + mlp_hidden,
                dropout=self.dropout_prob,
                bn=True,
                last_activation=True,
            ),
            nn.Linear(mlp_hidden[-1], 1),
        )

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        emb_3d, emb_flat = self.feature_embedding.get_3d_embeddings(x)

        # 1. Linear
        linear_out = self.linear(emb_flat).squeeze(-1)

        # 2. CIN（仅稀疏特征 3D）
        if emb_3d.shape[1] > 0:
            cin_out = self.cin(emb_3d)
            cin_score = self.cin_output_layer(cin_out).squeeze(-1)
        else:
            cin_score = torch.zeros_like(linear_out)

        # 3. DNN
        dnn_out = self.dnn(emb_flat).squeeze(-1)

        return linear_out + cin_score + dnn_out

    def predict(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))

    def calculate_loss(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(self.forward(x), x[self.label_field].float())
