# -*- coding: utf-8 -*-
"""
ZILN: Zero-Inflated LogNormal Model

论文: A Deep Probabilistic Model for Customer Lifetime Value Prediction (Google, 2018)

核心思想：
- LTV 分布是零膨胀的：P(LTV=0) > 0
- 非零值服从 LogNormal 分布
- 混合模型：分类（是否付费）+ 回归（条件价值）

损失函数：
L = -log(π * I(y=0) + (1-π) * LN(y|μ,σ) * I(y>0))

其中：
- π: 付费概率
- μ, σ: LogNormal 分布参数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple

from .base import LTVModel, register_ltv_model, LTVRegressionBase
from ...utils.enum import ModelType


@register_ltv_model("ziln")
class ZILN(LTVRegressionBase):
    """Zero-Inflated LogNormal Model
    
    输出：
    - prob: 付费概率 π ∈ [0, 1]
    - mu: LogNormal 均值参数
    - log_sigma: LogNormal 标准差参数（log 空间）
    
    预测：
    - LTV = (1 - π) * exp(mu + σ²/2)  # LogNormal 期望
    """
    
    model_type = ModelType.LTV
    
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        
        hidden_size = config.get("hidden_sizes", [256, 128, 64])[-1]
        
        # 输出头
        self.prob_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        
        self.mu_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        
        self.log_sigma_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            # 保证 σ > 0，使用 softplus 或直接输出 log(σ)
        )
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播
        
        Returns:
            prob: 付费概率，shape (batch_size,)
            mu: LogNormal μ，shape (batch_size,)
            log_sigma: LogNormal log(σ)，shape (batch_size,)
        """
        x = self._embed_features(batch)
        h = self.mlp[:-1](x)  # 不经过最后一层 dropout
        
        prob = self.prob_head(h).squeeze(-1)
        mu = self.mu_head(h).squeeze(-1)
        log_sigma = self.log_sigma_head(h).squeeze(-1)
        
        return prob, mu, log_sigma
    
    def calculate_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算 ZILN Loss
        
        L = -log p(y) where:
        - p(0) = π
        - p(y>0) = (1-π) * LN(y; μ, σ)
        """
        prob, mu, log_sigma = self.forward(batch)
        labels = batch[self.label_field].float()
        
        # LogNormal 似然：ln N(y; μ, σ) = -ln(σ√2π) - (ln y - μ)² / 2σ²
        # 为了数值稳定，使用 clamp
        eps = 1e-6
        y = torch.clamp(labels, min=eps)
        log_y = torch.log(y)
        
        sigma = torch.exp(log_sigma)
        sigma = torch.clamp(sigma, min=eps, max=10.0)
        
        # LogNormal log-likelihood
        log_likelihood = -log_sigma - 0.5 * np.log(2 * np.pi) - 0.5 * ((log_y - mu) / sigma) ** 2
        
        # 混合概率
        # p(y|y=0) = π
        # p(y|y>0) = (1-π) * exp(log_likelihood)
        zero_mask = (labels < 1e-6).float()
        
        # log p(y)
        # y=0: log(π)
        # y>0: log(1-π) + log_likelihood
        prob = torch.clamp(prob, min=eps, max=1-eps)
        log_p_zero = torch.log(prob)
        log_p_nonzero = torch.log(1 - prob) + log_likelihood
        
        # 选择对应项
        log_prob = zero_mask * log_p_zero + (1 - zero_mask) * log_p_nonzero
        
        # 负对数似然
        loss = -log_prob.mean()
        
        return loss
    
    def predict(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """预测 LTV
        
        E[LTV] = (1-π) * E[exp(X)] where X ~ N(μ, σ²)
        E[exp(X)] = exp(μ + σ²/2)
        """
        prob, mu, log_sigma = self.forward(batch)
        
        sigma = torch.exp(log_sigma)
        sigma = torch.clamp(sigma, min=1e-6, max=10.0)
        
        # LogNormal 期望
        conditional_ltv = torch.exp(mu + 0.5 * sigma ** 2)
        
        # 期望 LTV
        expected_ltv = (1 - prob) * conditional_ltv
        
        return expected_ltv
    
    def predict_distribution(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """预测分布参数"""
        prob, mu, log_sigma = self.forward(batch)
        
        return {
            "prob": prob,
            "mu": mu,
            "sigma": torch.exp(log_sigma),
            "expected_ltv": self.predict(batch),
        }
