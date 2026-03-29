# -*- coding: utf-8 -*-
"""
STAR: One Model to Serve All: Star Topology Adaptive Recommender for Multi-Domain CTR Prediction

论文: https://dl.acm.org/doi/abs/10.1145/3459637.3481941 (CIKM 2021)

核心思想：
- 星形拓扑：共享中心网络 + 域特定分区网络
- 分区策略：每个域有独立的 BN 和 FCN 参数
- Domain Normalization：输入层域归一化

代码参考: Scenario-Wise-Rec
"""

import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import init

from ..base import MultiDomainModel, register_model
from ..layers import FeatureEmbedding, MLPLayers


@register_model("star")
class STAR(MultiDomainModel):
    """STAR: Star Topology Adaptive Recommender

    多域 CTR 预估模型，通过星形拓扑结构实现：
    - 共享中心：所有域共享的主网络参数
    - 域特定分区：每个域独立的 BN 和 FCN 参数
    - Domain Normalization：输入层按域归一化

    配置示例:
    ```yaml
    model: star
    num_domains: 3
    mlp_hidden_size: [256, 128, 64]
    aux_dims: [64, 32]      # 辅助网络维度
    domain_field: domain_indicator
    ```
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # 模型参数
        self.mlp_hidden_size = config.get("mlp_hidden_size", [256, 128, 64])
        self.aux_dims = config.get("aux_dims", [64, 32])
        self.dropout = config.get("dropout_prob", 0.0)

        # 特征嵌入
        self.embedding = FeatureEmbedding(config, dataset)
        self.input_dim = self.embedding.output_dim

        # FCN 层数和维度
        self.layer_num = len(self.mlp_hidden_size) + 1
        self.fcn_dims = [self.input_dim] + self.mlp_hidden_size + [1]

        # ===== 共享中心网络参数 =====
        self.shared_w = nn.ParameterList()
        self.shared_b = nn.ParameterList()
        for i in range(self.layer_num):
            self.shared_w.append(
                Parameter(torch.empty(self.fcn_dims[i], self.fcn_dims[i + 1]), requires_grad=True)
            )
            self.shared_b.append(
                Parameter(torch.empty(self.fcn_dims[i + 1]), requires_grad=True)
            )

        # ===== 域特定分区网络参数 =====
        # Domain Normalization 参数
        self.domain_dn_gamma = nn.ParameterList()
        self.domain_dn_bias = nn.ParameterList()

        # 域特定 FCN 参数
        self.domain_w = nn.ParameterList()
        self.domain_b = nn.ParameterList()
        self.domain_bn = nn.ModuleList()

        for d in range(self.num_domains):
            # DN 参数
            self.domain_dn_gamma.append(Parameter(torch.ones(self.input_dim)))
            self.domain_dn_bias.append(Parameter(torch.zeros(self.input_dim)))

            # FCN 参数
            lay_w = nn.ParameterList()
            lay_b = nn.ParameterList()
            lay_bn = nn.ModuleList()

            for i in range(self.layer_num):
                lay_w.append(
                    Parameter(torch.empty(self.fcn_dims[i], self.fcn_dims[i + 1]), requires_grad=True)
                )
                lay_b.append(
                    Parameter(torch.empty(self.fcn_dims[i + 1]), requires_grad=True)
                )
                if i < self.layer_num - 1:  # 最后一层不需要 BN
                    lay_bn.append(nn.BatchNorm1d(self.fcn_dims[i + 1]))

            self.domain_w.append(lay_w)
            self.domain_b.append(lay_b)
            self.domain_bn.append(lay_bn)

        # ===== 辅助网络 =====
        self.aux_net = MLPLayers(
            input_dim=self.input_dim,
            dims=self.aux_dims + [1],
            activation="relu",
            dropout=self.dropout,
        )

        # 初始化
        self._reset_parameters()

    def _reset_parameters(self):
        """参数初始化"""
        for i in range(self.layer_num):
            init.kaiming_uniform_(self.shared_w[i])
            init.uniform_(self.shared_b[i], 0, 1)

        for d in range(self.num_domains):
            for i in range(self.layer_num):
                init.kaiming_uniform_(self.domain_w[d][i])
                init.uniform_(self.domain_b[d][i], 0, 1)

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

        # ===== Domain Normalization =====
        # 对每个样本应用对应域的归一化
        x_norm = torch.zeros_like(x)
        for d in range(self.num_domains):
            mask = (domain_id == d)
            if mask.sum() > 0:
                # 标准化
                x_d = x[mask]
                mean = x_d.mean(dim=0, keepdim=True)
                std = x_d.std(dim=0, keepdim=True) + 1e-6
                x_d_norm = (x_d - mean) / std
                # 域特定缩放
                gamma = self.domain_dn_gamma[d]
                bias = self.domain_dn_bias[d]
                x_norm[mask] = x_d_norm * gamma + bias
        x = x_norm

        # ===== 共享 FCN（ starred operation ） =====
        # STAR 核心公式：output = shared * domain_specific
        shared_out = x
        for i in range(self.layer_num):
            shared_out = torch.matmul(shared_out, self.shared_w[i]) + self.shared_b[i]
            if i < self.layer_num - 1:
                shared_out = torch.relu(shared_out)

        # ===== 域特定 FCN =====
        domain_out = torch.zeros(x.size(0), 1, device=x.device)
        for d in range(self.num_domains):
            mask = (domain_id == d)
            if mask.sum() > 0:
                x_d = x[mask]
                for i in range(self.layer_num):
                    x_d = torch.matmul(x_d, self.domain_w[d][i]) + self.domain_b[d][i]
                    if i < self.layer_num - 1:
                        x_d = self.domain_bn[d][i](x_d)
                        x_d = torch.relu(x_d)
                domain_out[mask] = x_d

        # ===== Star Topology: 相乘融合 =====
        logits = shared_out * domain_out  # [B, 1]

        # ===== 辅助网络（可选） =====
        aux_out = self.aux_net(x)  # [B, 1]
        logits = logits + aux_out

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
