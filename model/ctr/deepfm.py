# -*- coding: utf-8 -*-
"""
DeepFM: A Factorization-Machine based Neural Network for CTR Prediction (IJCAI 2017)
直接迁移自 autoresearch/continuous_features/models.py，保留完整 FM 实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from ..base import CTRModel, register_model
from ..layers import MLPLayers, FeatureEmbedding


@register_model("deepfm")
class DeepFM(CTRModel):
    """DeepFM 模型
    
    FM 部分（一阶 + 二阶）+ Deep 部分，共享特征嵌入
    """
    
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        
        self.mlp_hidden_size = config["mlp_hidden_size"]
        self.dropout_prob = config["dropout_prob"]
        
        # 特征嵌入（支持可配置的连续特征编码器）
        self.feature_embedding = FeatureEmbedding(
            sparse_vocab=dataset.sparse_vocab,
            sparse_feats=dataset.sparse_features,
            dense_dim=len(dataset.dense_features),
            embedding_dim=config["embedding_size"],
            encoder_type=config["encoder_type"],
            encoder_config=config.get("encoder_config", {}),
        )
        input_dim = self.feature_embedding.output_dim
        
        # FM 一阶：每个稀疏特征独立 1-D lookup + dense 线性
        self.fm_linear_sparse = nn.ModuleDict({
            feat: nn.Embedding(vocab_size + 1, 1, padding_idx=0)
            for feat, vocab_size in dataset.sparse_vocab.items()
            if feat in dataset.sparse_features
        })
        if dataset.dense_features:
            self.fm_linear_dense = nn.Linear(len(dataset.dense_features), 1, bias=False)
        else:
            self.fm_linear_dense = None
        self.fm_bias = nn.Parameter(torch.zeros(1))
        
        # Deep 部分
        self.deep = nn.Sequential(
            MLPLayers(
                layers=[input_dim] + self.mlp_hidden_size,
                dropout=self.dropout_prob,
                bn=True,
                last_activation=True,
            ),
            nn.Linear(self.mlp_hidden_size[-1], 1),
        )
    
    def _fm_first_order(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """FM 一阶线性部分"""
        ref_key = list(x.keys())[0]
        device = x[ref_key].device
        B = x[ref_key].size(0)
        
        first = torch.zeros(B, device=device)
        for feat, emb in self.fm_linear_sparse.items():
            if feat in x:
                first = first + emb(x[feat]).squeeze(-1)
        
        if self.fm_linear_dense is not None and "dense" in x:
            first = first + self.fm_linear_dense(x["dense"]).squeeze(-1)
        
        return first + self.fm_bias
    
    def _fm_second_order(self, emb_3d: torch.Tensor) -> torch.Tensor:
        """FM 二阶交叉：0.5 * (||sum(Vi*xi)||^2 - sum(||Vi*xi||^2))"""
        sum_emb = emb_3d.sum(dim=1)
        sum_sq = (sum_emb ** 2).sum(dim=-1)
        sq_sum = (emb_3d ** 2).sum(dim=1).sum(dim=-1)
        return 0.5 * (sum_sq - sq_sum)
    
    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        emb_3d, emb_flat = self.feature_embedding.get_3d_embeddings(x)
        
        fm_first = self._fm_first_order(x)
        fm_second = self._fm_second_order(emb_3d)
        deep_out = self.deep(emb_flat).squeeze(-1)
        
        return fm_first + fm_second + deep_out
    
    def predict(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))
    
    def calculate_loss(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        label = x[self.label_field].float()
        logits = self.forward(x)
        return F.binary_cross_entropy_with_logits(logits, label)
