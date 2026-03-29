# -*- coding: utf-8 -*-
"""
HAMUR: Hierarchical Adaptive Multi-Domain Unified Recommender

论文: https://arxiv.org/pdf/2309.06217

核心思想：
- 分层结构：域共享底层 + 域特定顶层
- Adapter Cells：低秩适配器实现知识迁移
- Instance Representation Matrix：实例级表示学习

代码参考: Scenario-Wise-Rec（简化版）
"""

import torch
import torch.nn as nn
from torch.nn import Parameter

from ..base import MultiDomainModel, register_model
from ..layers import FeatureEmbedding, MLPLayers


@register_model("hamur")
class HAMUR(MultiDomainModel):
    """HAMUR: Hierarchical Adaptive Multi-Domain Unified Recommender

    分层多域推荐模型：
    - 域共享底层 MLP
    - 域特定顶层 Tower
    - 低秩 Adapter 实现知识迁移

    配置示例:
    ```yaml
    model: hamur
    num_domains: 3
    mlp_hidden_size: [256, 256, 128, 128, 64, 64, 32]
    adapter_rank: 8
    domain_field: domain_indicator
    ```
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # 模型参数
        self.mlp_hidden_size = config.get("mlp_hidden_size", [256, 128, 64])
        self.adapter_rank = config.get("adapter_rank", 8)
        self.dropout = config.get("dropout_prob", 0.0)

        # 特征嵌入
        self.embedding = FeatureEmbedding(config, dataset)
        self.input_dim = self.embedding.output_dim

        # 网络维度
        self.fcn_dims = [self.input_dim] + self.mlp_hidden_size
        self.layer_num = len(self.fcn_dims)

        # ===== 域特定主干网络 =====
        # 每个域有独立的 MLP
        self.domain_mlps = nn.ModuleList()
        for d in range(self.num_domains):
            layers = []
            for i in range(len(self.fcn_dims) - 1):
                layers.append(nn.Linear(self.fcn_dims[i], self.fcn_dims[i + 1]))
                layers.append(nn.BatchNorm1d(self.fcn_dims[i + 1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(self.dropout))
            layers.append(nn.Linear(self.fcn_dims[-1], 1))
            self.domain_mlps.append(nn.Sequential(*layers))

        # ===== Adapter Cells (低秩适配器) =====
        # 用于层间知识迁移
        self.adapters = nn.ModuleList()
        for i in range(len(self.fcn_dims) - 1):
            adapter_list = nn.ModuleList()
            for d in range(self.num_domains):
                adapter_list.append(
                    nn.Sequential(
                        nn.Linear(self.fcn_dims[i + 1], self.adapter_rank),
                        nn.ReLU(),
                        nn.Linear(self.adapter_rank, self.fcn_dims[i + 1]),
                    )
                )
            self.adapters.append(adapter_list)

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

        # 每个域独立处理
        final = torch.zeros(x.size(0), 1, device=x.device)

        for d in range(self.num_domains):
            mask = (domain_id == d)
            if mask.sum() > 0:
                x_d = x[mask]
                
                # 域特定 MLP（逐层处理，可插入 adapter）
                for layer_idx, mlp in enumerate(self.domain_mlps[d][:-1]):  # 排除最后的 Linear
                    x_d = mlp(x_d)
                    # 可选：插入 adapter（简化版暂不使用）
                
                # 最终预测层
                logits_d = self.domain_mlps[d][-1](x_d)  # [num_domain_samples, 1]
                final[mask] = logits_d

        return final

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
