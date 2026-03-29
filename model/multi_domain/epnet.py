# -*- coding: utf-8 -*-
"""
EPNet: Embedding Personalization Network

论文参考: Scenario-Wise-Rec

核心思想：
- 将特征分为场景特征（Scenario）和 统一特征（Agnostic）
- 通过 GateNU 门控单元，基于场景特征调制统一特征
- 简单高效的多域适应方法

代码参考: Scenario-Wise-Rec
"""

import torch
import torch.nn as nn

from ..base import MultiDomainModel, register_model
from ..layers import FeatureEmbedding, MLPLayers


class GateNU(nn.Module):
    """Gate Network Unit
    
    通过门控机制调制特征表示
    output = sigmoid(MLP(input)) * gemma
    """
    
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


@register_model("epnet")
class EPNet(MultiDomainModel):
    """EPNet: Embedding Personalization Network

    通过门控机制实现多域适应：
    - Scenario Features: 场景/域特定特征
    - Agnostic Features: 统一特征
    - GateNU: 基于场景特征调制统一特征

    配置示例:
    ```yaml
    model: epnet
    num_domains: 3
    mlp_hidden_size: [256, 128, 64]
    domain_field: domain_indicator
    scenario_features: [domain_id, campaign_type]  # 场景特征
    agnostic_features: [user_id, item_id, ...]     # 统一特征
    ```
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # 模型参数
        self.mlp_hidden_size = config.get("mlp_hidden_size", [256, 128, 64])
        self.dropout = config.get("dropout_prob", 0.0)

        # 区分场景特征和统一特征
        # 默认：domain_indicator 作为场景特征，其余作为统一特征
        self.scenario_features = config.get(
            "scenario_features", 
            [self.domain_field]
        )
        self.agnostic_features = config.get(
            "agnostic_features",
            self.sparse_features + self.dense_features
        )

        # 特征嵌入
        self.scenario_embedding = FeatureEmbedding(
            config, dataset, feature_list=self.scenario_features
        )
        self.agnostic_embedding = FeatureEmbedding(
            config, dataset, feature_list=self.agnostic_features
        )

        self.scenario_dim = self.scenario_embedding.output_dim
        self.agnostic_dim = self.agnostic_embedding.output_dim
        self.input_dim = self.scenario_dim + self.agnostic_dim

        # GateNU: 基于场景特征调制统一特征
        self.gate = GateNU(self.input_dim, self.agnostic_dim)

        # MLP Tower
        self.mlp = MLPLayers(
            input_dim=self.agnostic_dim,
            dims=self.mlp_hidden_size + [1],
            activation="relu",
            dropout=self.dropout,
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, batch):
        """前向传播

        Args:
            batch: 包含特征的字典

        Returns:
            [B, 1] logits
        """
        # 场景特征嵌入
        scenario_x = self.scenario_embedding(batch, feature_list=self.scenario_features)
        scenario_x = scenario_x.view(scenario_x.size(0), -1)  # [B, scenario_dim]

        # 统一特征嵌入
        agnostic_x = self.agnostic_embedding(batch, feature_list=self.agnostic_features)
        agnostic_x = agnostic_x.view(agnostic_x.size(0), -1)  # [B, agnostic_dim]

        # 门控调制
        gate_input = torch.cat([scenario_x, agnostic_x.detach()], dim=1)
        gate_output = self.gate(gate_input)  # [B, agnostic_dim]
        agnostic_x = agnostic_x * gate_output

        # MLP 预测
        logits = self.mlp(agnostic_x)  # [B, 1]

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
