# -*- coding: utf-8 -*-
"""
M3oE: Multi-Domain Multi-Task Mixture of Experts

论文: https://arxiv.org/abs/2404.18465

核心思想：
- 多域 + 多任务联合建模
- 三维 MoE：Domain × Task × Expert
- 自适应路由机制

代码参考: Scenario-Wise-Rec（简化版）
"""

import torch
import torch.nn as nn

from ..base import MultiDomainModel, register_model
from ..layers import FeatureEmbedding, MLPLayers


@register_model("m3oe")
class M3oE(MultiDomainModel):
    """M3oE: Multi-Domain Multi-Task Mixture of Experts

    多域多任务联合建模：
    - 多个专家网络提供多样化特征
    - 域门控：选择适合当前域的专家
    - 任务门控：选择适合当前任务的专家
    - 三维融合：Domain × Task × Expert

    配置示例:
    ```yaml
    model: m3oe
    num_domains: 3
    num_tasks: 2
    num_experts: 4
    expert_hidden_size: [256, 128]
    tower_hidden_size: [64, 32]
    domain_field: domain_indicator
    ```
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # 模型参数
        self.num_experts = config.get("num_experts", 4)
        self.num_tasks = config.get("num_tasks", 1)  # 单任务 CTR 时为 1
        self.expert_hidden_size = config.get("expert_hidden_size", [256, 128])
        self.tower_hidden_size = config.get("tower_hidden_size", [64, 32])
        self.dropout = config.get("dropout_prob", 0.0)

        # 特征嵌入
        self.embedding = FeatureEmbedding(config, dataset)
        self.input_dim = self.embedding.output_dim

        # ===== 专家网络 =====
        self.experts = nn.ModuleList([
            MLPLayers(
                input_dim=self.input_dim,
                dims=self.expert_hidden_size,
                dropout=self.dropout,
                activation="relu",
            )
            for _ in range(self.num_experts)
        ])
        self.expert_output_dim = self.expert_hidden_size[-1] if self.expert_hidden_size else self.input_dim

        # ===== 域门控网络 =====
        # 每个域一个门控，选择专家组合
        self.domain_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.input_dim, self.num_experts),
                nn.Softmax(dim=-1)
            )
            for _ in range(self.num_domains)
        ])

        # ===== 任务塔 =====
        # 每个任务一个塔
        self.task_towers = nn.ModuleList([
            MLPLayers(
                input_dim=self.expert_output_dim,
                dims=self.tower_hidden_size + [1],
                dropout=self.dropout,
                activation="relu",
            )
            for _ in range(self.num_tasks)
        ])

        self.sigmoid = nn.Sigmoid()

    def forward(self, batch):
        """前向传播

        Args:
            batch: 包含特征和 domain_indicator 的字典

        Returns:
            [B, 1] logits（单任务）或 Dict[task_name, [B, 1]]（多任务）
        """
        domain_id = self.get_domain_id(batch)

        # 特征嵌入
        embed = self.embedding(batch)  # [B, num_fields, D]
        x = embed.view(embed.size(0), -1)  # [B, input_dim]

        # 专家输出
        expert_outputs = []
        for expert in self.experts:
            expert_out = expert(x)  # [B, expert_output_dim]
            expert_outputs.append(expert_out)
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [B, num_experts, expert_output_dim]

        # 域门控加权
        final_expert_output = torch.zeros(
            x.size(0), self.expert_output_dim, device=x.device
        )

        for d in range(self.num_domains):
            mask = (domain_id == d)
            if mask.sum() > 0:
                # 域门控权重
                gate_weights = self.domain_gates[d](x[mask])  # [num_domain_samples, num_experts]
                
                # 加权融合专家输出
                weighted_output = torch.matmul(
                    gate_weights.unsqueeze(1),  # [N, 1, num_experts]
                    expert_outputs[mask]  # [N, num_experts, expert_output_dim]
                ).squeeze(1)  # [N, expert_output_dim]
                
                final_expert_output[mask] = weighted_output

        # 任务塔预测
        if self.num_tasks == 1:
            logits = self.task_towers[0](final_expert_output)  # [B, 1]
            return logits
        else:
            # 多任务输出
            outputs = {}
            for i, tower in enumerate(self.task_towers):
                outputs[f"task_{i}"] = tower(final_expert_output)
            return outputs

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
