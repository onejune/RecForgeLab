# -*- coding: utf-8 -*-
"""
M2M: Meta Learning for Multi-Domain CTR Prediction

论文: https://dl.acm.org/doi/abs/10.1145/3488560.3498479 (WSDM 2022)

核心思想：
- Meta-Learning：学习如何为每个域生成模型参数
- Transformer：提取特征表示
- Meta-Attention：动态融合多个专家
- Meta-Tower：域特定的预测塔

代码参考: Scenario-Wise-Rec
"""

import torch
import torch.nn as nn

from ..base import MultiDomainModel, register_model
from ..layers import FeatureEmbedding, MLPLayers


@register_model("m2m")
class M2M(MultiDomainModel):
    """M2M: Meta Learning for Multi-Domain CTR Prediction

    通过元学习机制实现多域 CTR 预估：
    - Transformer 提取特征表示
    - 多专家网络提供多样化特征
    - Meta-Attention 动态加权专家输出
    - Meta-Tower 为每个域生成专属预测塔

    配置示例:
    ```yaml
    model: m2m
    num_domains: 3
    num_experts: 4
    expert_output_size: 16
    transformer_layers: 2
    transformer_heads: 4
    domain_field: domain_indicator
    ```
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # 模型参数
        self.num_experts = config.get("num_experts", 4)
        self.expert_output_size = config.get("expert_output_size", 16)
        self.transformer_layers = config.get("transformer_layers", 2)
        self.transformer_heads = config.get("transformer_heads", 4)

        # 特征嵌入
        self.embedding = FeatureEmbedding(config, dataset)
        self.input_dim = self.embedding.output_dim

        # Domain Embedding
        self.domain_embedding = nn.Embedding(self.num_domains, self.expert_output_size)

        # Transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.input_dim,
                nhead=self.transformer_heads,
                dim_feedforward=self.input_dim * 4,
                dropout=0.1,
                batch_first=True,
            ),
            num_layers=self.transformer_layers,
        )

        # 专家网络
        self.experts = nn.ModuleList([
            MLPLayers(
                input_dim=self.input_dim,
                dims=[self.expert_output_size],
                activation="leakyrelu",
                dropout=0.0,
            )
            for _ in range(self.num_experts)
        ])

        # Meta-Attention 模块
        self.task_mlp = MLPLayers(
            input_dim=self.expert_output_size,
            dims=[self.expert_output_size],
            activation="leakyrelu",
        )
        self.scenario_mlp = MLPLayers(
            input_dim=self.expert_output_size,
            dims=[self.expert_output_size],
            activation="leakyrelu",
        )
        self.vw_mlp = MLPLayers(
            input_dim=self.expert_output_size,
            dims=[2 * self.expert_output_size * self.expert_output_size],
            activation="leakyrelu",
        )
        self.vb_mlp = MLPLayers(
            input_dim=self.expert_output_size,
            dims=[2 * self.expert_output_size],
            activation="leakyrelu",
        )
        self.v = nn.Parameter(torch.ones(2 * self.expert_output_size, 1))

        # Meta-Tower 模块
        self.tower_w_mlp = MLPLayers(
            input_dim=self.expert_output_size,
            dims=[self.expert_output_size * self.expert_output_size],
            activation="leakyrelu",
        )
        self.tower_b_mlp = MLPLayers(
            input_dim=self.expert_output_size,
            dims=[self.expert_output_size],
            activation="leakyrelu",
        )
        self.output_mlp = MLPLayers(
            input_dim=self.expert_output_size,
            dims=[64, 32, 1],
        )

        self.relu = nn.LeakyReLU(0.1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, batch):
        """前向传播

        Args:
            batch: 包含特征和 domain_indicator 的字典

        Returns:
            [B, 1] logits
        """
        domain_id = self.get_domain_id(batch)

        # 特征嵌入
        embed = self.embedding(batch)  # [B, num_fields, D]
        x = embed.view(embed.size(0), -1)  # [B, input_dim]

        # Domain Embedding
        domain_emb = self.domain_embedding(domain_id)  # [B, expert_output_size]

        # Transformer 编码
        transformer_out = self.transformer(x.unsqueeze(1)).squeeze(1)  # [B, input_dim]

        # 专家网络
        expert_outputs = []
        for expert in self.experts:
            expert_out = expert(transformer_out)  # [B, expert_output_size]
            expert_outputs.append(expert_out)
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [B, num_experts, expert_output_size]

        # Meta-Attention
        scenario_out = self.scenario_mlp(domain_emb)  # [B, expert_output_size]
        task_output = self.task_mlp(domain_emb)  # [B, expert_output_size]

        meta_input = torch.cat([
            expert_outputs,
            task_output.unsqueeze(1).expand(-1, self.num_experts, -1)
        ], dim=-1)  # [B, num_experts, 2 * expert_output_size]

        meta_weight = self.vw_mlp(scenario_out).reshape(
            -1, 2 * self.expert_output_size, 2 * self.expert_output_size
        )  # [B, 2*E, 2*E]
        meta_bias = self.vb_mlp(scenario_out).unsqueeze(1)  # [B, 1, 2*E]
        meta_output = self.relu(
            torch.matmul(meta_input, meta_weight).squeeze(2) + meta_bias
        )  # [B, num_experts, 2*E]
        meta_output = torch.matmul(meta_output, self.v.unsqueeze(0)).squeeze(-1)  # [B, num_experts]
        alpha = torch.softmax(meta_output, dim=1).unsqueeze(-1)  # [B, num_experts, 1]

        # 加权融合专家输出
        rt = torch.sum(alpha * expert_outputs, dim=1)  # [B, expert_output_size]

        # Meta-Tower
        tower_weight = self.tower_w_mlp(scenario_out).reshape(
            -1, self.expert_output_size, self.expert_output_size
        )  # [B, E, E]
        tower_bias = self.tower_b_mlp(scenario_out)  # [B, E]
        output = self.relu(
            torch.matmul(rt.unsqueeze(1), tower_weight).squeeze(1) + tower_bias + rt
        )  # [B, expert_output_size]

        logits = self.output_mlp(output)  # [B, 1]

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
