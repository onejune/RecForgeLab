# -*- coding: utf-8 -*-
"""
PPNet: Personalized Parallel Network

论文参考: Scenario-Wise-Rec

核心思想：
- 将特征分为 ID 特征和行为特征
- 每个域有独立的 Tower
- GateNU 门控基于 ID 特征调制行为特征

代码参考: Scenario-Wise-Rec
"""

import torch
import torch.nn as nn

from ..base import MultiDomainModel, register_model
from ..layers import FeatureEmbedding, MLPLayers


class GateNU(nn.Module):
    """Gate Network Unit"""
    
    def __init__(self, input_dim, output_dim, hidden_dim=None, gemma=2.0):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = output_dim
        self.gemma = gemma
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x) * self.gemma


class PPTowerBlock(nn.Module):
    """PPNet 的域特定 Tower
    
    每层都有独立的 GateNU 门控
    """
    
    def __init__(self, input_dim, mlp_dims, dropout=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.dims = [input_dim] + mlp_dims
        
        self.mlp_layers = nn.ModuleList()
        self.gate_layers = nn.ModuleList()
        
        for i in range(len(self.dims) - 1):
            self.mlp_layers.append(
                MLPLayers(
                    input_dim=self.dims[i],
                    dims=[self.dims[i + 1]],
                    dropout=dropout,
                    activation="relu",
                )
            )
            self.gate_layers.append(GateNU(self.dims[0], self.dims[i + 1]))
        
        self.final_layer = nn.Linear(self.dims[-1], 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, agn_emb, gate_input_emb):
        """前向传播
        
        Args:
            agn_emb: 行为特征嵌入 [B, input_dim]
            gate_input_emb: 门控输入（ID 特征）[B, input_dim]
        
        Returns:
            [B, 1] logits
        """
        hidden = agn_emb
        for i in range(len(self.mlp_layers)):
            gate_out = self.gate_layers[i](gate_input_emb)
            hidden = self.mlp_layers[i](hidden)
            hidden = hidden * gate_out
        logits = self.final_layer(hidden)
        return self.sigmoid(logits)


@register_model("ppnet")
class PPNet(MultiDomainModel):
    """PPNet: Personalized Parallel Network

    多域 CTR 预估模型：
    - ID Features: 用户/物品 ID 特征（用于门控）
    - Agnostic Features: 行为特征（被门控调制）
    - 每个域独立的 Tower

    配置示例:
    ```yaml
    model: ppnet
    num_domains: 3
    mlp_hidden_size: [256, 128, 64]
    domain_field: domain_indicator
    id_features: [user_id, item_id]
    agnostic_features: [behavior_1, behavior_2, ...]
    ```
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # 模型参数
        self.mlp_hidden_size = config.get("mlp_hidden_size", [256, 128, 64])
        self.dropout = config.get("dropout_prob", 0.0)

        # 区分 ID 特征和行为特征
        self.id_features = config.get("id_features", self.sparse_features[:2])
        self.agnostic_features = config.get(
            "agnostic_features",
            self.sparse_features + self.dense_features
        )

        # 特征嵌入
        self.id_embedding = FeatureEmbedding(config, dataset, feature_list=self.id_features)
        self.agnostic_embedding = FeatureEmbedding(
            config, dataset, feature_list=self.agnostic_features
        )

        self.id_dim = self.id_embedding.output_dim
        self.agnostic_dim = self.agnostic_embedding.output_dim
        self.input_dim = self.id_dim + self.agnostic_dim

        # 每个域独立的 Tower
        self.domain_towers = nn.ModuleList([
            PPTowerBlock(self.input_dim, self.mlp_hidden_size, self.dropout)
            for _ in range(self.num_domains)
        ])

    def forward(self, batch):
        """前向传播

        Args:
            batch: 包含特征和 domain_indicator 的字典

        Returns:
            [B, 1] logits
        """
        domain_id = self.get_domain_id(batch)

        # ID 特征嵌入
        id_x = self.id_embedding(batch, feature_list=self.id_features)
        id_x = id_x.view(id_x.size(0), -1)  # [B, id_dim]

        # 行为特征嵌入
        agn_x = self.agnostic_embedding(batch, feature_list=self.agnostic_features)
        agn_x = agn_x.view(agn_x.size(0), -1)  # [B, agnostic_dim]

        # 门控输入：ID 特征 + 行为特征（detach）
        gate_input = torch.cat([id_x, agn_x.detach()], dim=1)
        agn_input = torch.cat([id_x, agn_x], dim=1)

        # 每个域独立处理
        final = torch.zeros(batch[self.label_field].size(0), 1, device=id_x.device)
        
        for d in range(self.num_domains):
            mask = (domain_id == d)
            if mask.sum() > 0:
                domain_out = self.domain_towers[d](
                    agn_input[mask],
                    gate_input[mask]
                )  # [num_domain_samples, 1]
                final[mask] = domain_out

        # 转换为 logits（因为 tower 输出是 sigmoid 后的概率）
        logits = torch.log(final + 1e-7) - torch.log(1 - final + 1e-7)

        return logits

    def calculate_loss(self, batch):
        """计算损失"""
        logits = self.forward(batch)
        label = batch[self.label_field].float().unsqueeze(-1)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, label)
        return loss

    def predict(self, batch):
        """推理预测"""
        logits = self.forward(batch)
        return torch.sigmoid(logits).squeeze(-1)
