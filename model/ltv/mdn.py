# -*- coding: utf-8 -*-
"""
Mixture Density Network (MDN) for LTV

混合密度网络：预测混合分布参数

LTV 分布可以用多个分布的混合来建模：
- 组件 1：零点质量（未付费用户）
- 组件 2-：LogNormal/Gamma 分布（付费用户的不同价值层级）

输出：
- 混合权重 π_1, ..., π_K
- 每个组件的分布参数

预测：E[LTV] = Σ π_k * E[component_k]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple
import math

from .base import LTVModel, register_ltv_model, LTVRegressionBase
from ...utils.enum import ModelType


@register_ltv_model("mdn")
class MDNLTV(LTVRegressionBase):
    """Mixture Density Network for LTV
    
    混合 K 个 LogNormal 分布：
    p(y) = Σ π_k * LogNormal(y; μ_k, σ_k)
    
    配置：
    - num_components: 混合组件数量，默认 3
    - component_type: "lognormal" | "gamma"
    """
    
    model_type = ModelType.LTV
    
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        
        self.num_components = config.get("num_components", 3)
        self.component_type = config.get("component_type", "lognormal")
        
        hidden_size = config.get("hidden_sizes", [256, 128, 64])[-1]
        
        # 混合权重
        self.pi_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_components),
            nn.Softmax(dim=-1),
        )
        
        # 组件参数
        if self.component_type == "lognormal":
            # μ, log(σ) for each component
            self.mu_head = nn.Linear(hidden_size, self.num_components)
            self.log_sigma_head = nn.Linear(hidden_size, self.num_components)
        else:  # gamma
            # shape α, rate β
            self.alpha_head = nn.Sequential(
                nn.Linear(hidden_size, self.num_components),
                nn.Softplus(),  # α > 0
            )
            self.beta_head = nn.Sequential(
                nn.Linear(hidden_size, self.num_components),
                nn.Softplus(),  # β > 0
            )
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        """前向传播
        
        Returns:
            pi: 混合权重，shape (batch_size, K)
            params: 分布参数
        """
        x = self._embed_features(batch)
        h = self.mlp[:-1](x)
        
        pi = self.pi_head(h)
        
        if self.component_type == "lognormal":
            mu = self.mu_head(h)
            log_sigma = self.log_sigma_head(h)
            sigma = torch.exp(torch.clamp(log_sigma, min=-5, max=5))
            return pi, mu, sigma
        else:
            alpha = self.alpha_head(h)
            beta = self.beta_head(h)
            return pi, alpha, beta
    
    def calculate_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算 MDN Loss = -log p(y)"""
        labels = batch[self.label_field].float()
        
        if self.component_type == "lognormal":
            pi, mu, sigma = self.forward(batch)
            
            # LogNormal 似然
            y = torch.clamp(labels.unsqueeze(-1), min=1e-6)  # (batch, 1)
            log_y = torch.log(y)
            
            # log p(y | component_k)
            log_prob_k = -torch.log(sigma) - 0.5 * math.log(2 * math.pi) - 0.5 * ((log_y - mu) / sigma) ** 2
            
            # log p(y) = log Σ π_k * p(y | k)
            log_prob = torch.logsumexp(torch.log(pi) + log_prob_k, dim=-1)
            
        else:  # gamma
            pi, alpha, beta = self.forward(batch)
            
            y = torch.clamp(labels.unsqueeze(-1), min=1e-6)
            
            # Gamma 似然: p(y; α, β) = β^α / Γ(α) * y^(α-1) * exp(-βy)
            log_prob_k = alpha * torch.log(beta) - torch.lgamma(alpha) + (alpha - 1) * torch.log(y) - beta * y
            
            log_prob = torch.logsumexp(torch.log(pi) + log_prob_k, dim=-1)
        
        # 负对数似然
        loss = -log_prob.mean()
        
        return loss
    
    def predict(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """预测 LTV = E[y] = Σ π_k * E[component_k]"""
        if self.component_type == "lognormal":
            pi, mu, sigma = self.forward(batch)
            
            # LogNormal 期望: E[X] = exp(μ + σ²/2)
            component_mean = torch.exp(mu + 0.5 * sigma ** 2)
            
        else:  # gamma
            pi, alpha, beta = self.forward(batch)
            
            # Gamma 期望: E[X] = α / β
            component_mean = alpha / beta
        
        # 混合期望
        ltv = torch.sum(pi * component_mean, dim=-1)
        
        return ltv
    
    def predict_distribution(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """预测分布参数"""
        if self.component_type == "lognormal":
            pi, mu, sigma = self.forward(batch)
            return {
                "pi": pi,
                "mu": mu,
                "sigma": sigma,
                "expected_ltv": self.predict(batch),
            }
        else:
            pi, alpha, beta = self.forward(batch)
            return {
                "pi": pi,
                "alpha": alpha,
                "beta": beta,
                "expected_ltv": self.predict(batch),
            }
