# -*- coding: utf-8 -*-
"""
Paid-Only LTV Model

只在付费用户上建模，预测条件价值 E[LTV | paid]

场景：
- LTV=0 样本已过滤
- 预测付费用户的价值
- 分布特征：右偏，无零膨胀

方法：
- Log-Normal 回归：log(LTV) ~ N(μ, σ²)
- Gamma 回归：LTV ~ Gamma(α, β)
- Shifted Log-Normal：log(LTV - shift) ~ N(μ, σ²)
- Box-Cox 变换回归
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
import math

from .base import LTVModel, register_ltv_model, LTVRegressionBase
from ...utils.enum import ModelType


# ============================================================
# Log-Normal 回归
# ============================================================

@register_ltv_model("lognormal")
class LogNormalLTV(LTVRegressionBase):
    """Log-Normal 回归
    
    假设 log(LTV) ~ N(μ, σ²)
    
    优点：
    - 适合右偏分布
    - 预测值保证 > 0
    - 可输出不确定性（σ）
    
    输出：
    - mu: log(LTV) 的均值
    - sigma: log(LTV) 的标准差
    
    预测：
    - E[LTV] = exp(μ + σ²/2)
    """
    
    model_type = ModelType.LTV
    
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        
        hidden_size = config.get("hidden_sizes", [256, 128, 64])[-1]
        
        # 输出 mu 和 log(sigma)
        self.mu_head = nn.Linear(hidden_size, 1)
        self.log_sigma_head = nn.Linear(hidden_size, 1)
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        x = self._embed_features(batch)
        h = self.mlp[:-1](x)
        
        mu = self.mu_head(h).squeeze(-1)
        log_sigma = self.log_sigma_head(h).squeeze(-1)
        
        return mu, log_sigma
    
    def calculate_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Log-Normal 负对数似然
        
        -log p(y) = log(σ) + 0.5 * log(2π) + 0.5 * ((log(y) - μ) / σ)²
        """
        mu, log_sigma = self.forward(batch)
        labels = batch[self.label_field].float()
        
        # Log-Normal 似然
        eps = 1e-6
        y = torch.clamp(labels, min=eps)
        log_y = torch.log(y)
        
        sigma = torch.exp(torch.clamp(log_sigma, min=-5, max=5))
        
        # NLL
        nll = log_sigma + 0.5 * math.log(2 * math.pi) + 0.5 * ((log_y - mu) / sigma) ** 2
        
        return nll.mean()
    
    def predict(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """预测 E[LTV] = exp(μ + σ²/2)"""
        mu, log_sigma = self.forward(batch)
        sigma = torch.exp(torch.clamp(log_sigma, min=-5, max=5))
        
        return torch.exp(mu + 0.5 * sigma ** 2)
    
    def predict_distribution(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """预测分布参数"""
        mu, log_sigma = self.forward(batch)
        sigma = torch.exp(torch.clamp(log_sigma, min=-5, max=5))
        
        return {
            "mu": mu,
            "sigma": sigma,
            "median": torch.exp(mu),
            "mean": torch.exp(mu + 0.5 * sigma ** 2),
        }


# ============================================================
# Gamma 回归
# ============================================================

@register_ltv_model("gamma")
class GammaLTV(LTVRegressionBase):
    """Gamma 回归
    
    假设 LTV ~ Gamma(α, β)
    
    优点：
    - 适合右偏分布
    - 灵活的形状参数
    
    参数化：
    - shape α (concentration)
    - rate β (inverse scale)
    
    E[LTV] = α / β
    Var[LTV] = α / β²
    
    损失：Gamma NLL
    -log p(y; α, β) = α*log(β) - log(Γ(α)) + (α-1)*log(y) - β*y
    """
    
    model_type = ModelType.LTV
    
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        
        hidden_size = config.get("hidden_sizes", [256, 128, 64])[-1]
        
        # 输出 log(α) 和 log(β)，保证 > 0
        self.log_alpha_head = nn.Sequential(
            nn.Linear(hidden_size, 1),
            # α > 0
        )
        self.log_beta_head = nn.Linear(hidden_size, 1)
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        x = self._embed_features(batch)
        h = self.mlp[:-1](x)
        
        log_alpha = self.log_alpha_head(h).squeeze(-1)
        log_beta = self.log_beta_head(h).squeeze(-1)
        
        # 限制范围
        log_alpha = torch.clamp(log_alpha, min=-5, max=10)
        log_beta = torch.clamp(log_beta, min=-5, max=10)
        
        return log_alpha, log_beta
    
    def calculate_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Gamma 负对数似然"""
        log_alpha, log_beta = self.forward(batch)
        labels = batch[self.label_field].float()
        
        alpha = torch.exp(log_alpha)
        beta = torch.exp(log_beta)
        
        eps = 1e-6
        y = torch.clamp(labels, min=eps)
        
        # Gamma NLL: α*log(β) - lgamma(α) + (α-1)*log(y) - β*y
        nll = -log_alpha * log_beta + torch.lgamma(alpha) - (alpha - 1) * torch.log(y) + beta * y
        
        return nll.mean()
    
    def predict(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """预测 E[LTV] = α / β"""
        log_alpha, log_beta = self.forward(batch)
        alpha = torch.exp(log_alpha)
        beta = torch.exp(log_beta)
        
        return alpha / beta
    
    def predict_distribution(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """预测分布参数"""
        log_alpha, log_beta = self.forward(batch)
        alpha = torch.exp(log_alpha)
        beta = torch.exp(log_beta)
        
        return {
            "alpha": alpha,
            "beta": beta,
            "mean": alpha / beta,
            "variance": alpha / (beta ** 2),
        }


# ============================================================
# Shifted Log-Normal 回归
# ============================================================

@register_ltv_model("shifted_lognormal")
class ShiftedLogNormalLTV(LTVRegressionBase):
    """Shifted Log-Normal 回归
    
    假设 log(LTV - shift) ~ N(μ, σ²)
    
    优点：
    - 比标准 Log-Normal 更灵活
    - shift 参数可学习或固定
    
    适用：
    - LTV 有最小值阈值（如最低付费金额）
    """
    
    model_type = ModelType.LTV
    
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        
        hidden_size = config.get("hidden_sizes", [256, 128, 64])[-1]
        
        # shift 参数（可学习或固定）
        self.learn_shift = config.get("learn_shift", True)
        if self.learn_shift:
            self.shift = nn.Parameter(torch.tensor(config.get("shift_init", 1.0)))
        else:
            self.shift = config.get("shift", 1.0)
        
        self.mu_head = nn.Linear(hidden_size, 1)
        self.log_sigma_head = nn.Linear(hidden_size, 1)
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播"""
        x = self._embed_features(batch)
        h = self.mlp[:-1](x)
        
        mu = self.mu_head(h).squeeze(-1)
        log_sigma = self.log_sigma_head(h).squeeze(-1)
        
        if self.learn_shift:
            shift = F.softplus(self.shift)  # 保证 shift > 0
        else:
            shift = torch.tensor(self.shift, device=h.device)
        
        return mu, log_sigma, shift
    
    def calculate_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Shifted Log-Normal NLL"""
        mu, log_sigma, shift = self.forward(batch)
        labels = batch[self.label_field].float()
        
        # y - shift
        y_shifted = torch.clamp(labels - shift, min=1e-6)
        log_y = torch.log(y_shifted)
        
        sigma = torch.exp(torch.clamp(log_sigma, min=-5, max=5))
        
        # NLL（加上 Jacobian 校正）
        nll = log_sigma + 0.5 * math.log(2 * math.pi) + 0.5 * ((log_y - mu) / sigma) ** 2
        nll = nll - torch.log(y_shifted / labels.clamp(min=1e-6))  # Jacobian
        
        return nll.mean()
    
    def predict(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """预测 E[LTV] = shift + exp(μ + σ²/2)"""
        mu, log_sigma, shift = self.forward(batch)
        sigma = torch.exp(torch.clamp(log_sigma, min=-5, max=5))
        
        return shift + torch.exp(mu + 0.5 * sigma ** 2)


# ============================================================
# Box-Cox 变换回归
# ============================================================

@register_ltv_model("boxcox")
class BoxCoxLTV(LTVRegressionBase):
    """Box-Cox 变换回归
    
    Box-Cox 变换：
    y(λ) = (y^λ - 1) / λ   if λ ≠ 0
         = log(y)          if λ = 0
    
    假设 y(λ) ~ N(μ, σ²)
    
    优点：
    - 自动寻找最优变换
    - λ 可学习
    
    λ 的含义：
    - λ = 0: Log 变换
    - λ = 1: 不变换
    - λ = 0.5: 平方根变换
    """
    
    model_type = ModelType.LTV
    
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        
        hidden_size = config.get("hidden_sizes", [256, 128, 64])[-1]
        
        # λ 参数（可学习）
        self.learn_lambda = config.get("learn_lambda", True)
        if self.learn_lambda:
            # 初始化接近 0（log 变换）
            self.lambda_param = nn.Parameter(torch.tensor(config.get("lambda_init", 0.1)))
        else:
            self.lambda_param = config.get("lambda", 0.0)
        
        self.mu_head = nn.Linear(hidden_size, 1)
        self.log_sigma_head = nn.Linear(hidden_size, 1)
    
    def _boxcox_transform(self, y: torch.Tensor, lam: float) -> torch.Tensor:
        """Box-Cox 变换"""
        eps = 1e-6
        y = torch.clamp(y, min=eps)
        
        if abs(lam) < 1e-6:
            return torch.log(y)
        else:
            return (y ** lam - 1) / lam
    
    def _boxcox_inverse(self, z: torch.Tensor, lam: float) -> torch.Tensor:
        """Box-Cox 逆变换"""
        if abs(lam) < 1e-6:
            return torch.exp(z)
        else:
            return (lam * z + 1) ** (1 / lam)
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """前向传播"""
        x = self._embed_features(batch)
        h = self.mlp[:-1](x)
        
        mu = self.mu_head(h).squeeze(-1)
        log_sigma = self.log_sigma_head(h).squeeze(-1)
        
        if self.learn_lambda:
            lam = torch.tanh(self.lambda_param) * 2  # λ ∈ (-2, 2)
        else:
            lam = self.lambda_param
        
        return mu, log_sigma, lam
    
    def calculate_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Box-Cox 变换后的 NLL"""
        mu, log_sigma, lam = self.forward(batch)
        labels = batch[self.label_field].float()
        
        # Box-Cox 变换
        if isinstance(lam, torch.Tensor):
            lam_val = lam.item() if lam.numel() == 1 else lam
        else:
            lam_val = lam
        
        z = self._boxcox_transform(labels, lam_val if not isinstance(lam_val, torch.Tensor) else lam_val.item())
        
        sigma = torch.exp(torch.clamp(log_sigma, min=-5, max=5))
        
        # NLL
        nll = log_sigma + 0.5 * math.log(2 * math.pi) + 0.5 * ((z - mu) / sigma) ** 2
        
        # Jacobian 校正
        eps = 1e-6
        y = torch.clamp(labels, min=eps)
        jacobian = (lam_val - 1) * torch.log(y) if abs(lam_val) > 1e-6 else torch.zeros_like(y)
        nll = nll - jacobian
        
        return nll.mean()
    
    def predict(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """预测 LTV"""
        mu, log_sigma, lam = self.forward(batch)
        
        sigma = torch.exp(torch.clamp(log_sigma, min=-5, max=5))
        
        # 逆变换
        z_mean = mu + 0.5 * sigma ** 2
        
        if isinstance(lam, torch.Tensor):
            lam_val = lam.item() if lam.numel() == 1 else lam
        else:
            lam_val = lam
        
        return self._boxcox_inverse(z_mean, lam_val if not isinstance(lam_val, torch.Tensor) else lam_val.item())


# ============================================================
# 注册别名：Simple Log Regression
# ============================================================

@register_ltv_model("log_regression")
class LogRegressionLTV(LTVRegressionBase):
    """简单 Log 回归
    
    预测 log(LTV)，使用 MSE Loss
    
    简化版，易于训练
    """
    
    model_type = ModelType.LTV
    
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        
        hidden_size = config.get("hidden_sizes", [256, 128, 64])[-1]
        
        # 直接输出 log(LTV)
        self.output_layer = nn.Linear(hidden_size, 1)
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """前向传播"""
        x = self._embed_features(batch)
        h = self.mlp[:-1](x)
        return self.output_layer(h).squeeze(-1)
    
    def calculate_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """MSE on log(LTV)"""
        pred = self.forward(batch)
        labels = batch[self.label_field].float()
        
        log_target = torch.log(torch.clamp(labels, min=1.0))
        
        return F.mse_loss(pred, log_target)
    
    def predict(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """预测 LTV = exp(pred)"""
        pred = self.forward(batch)
        return torch.exp(pred)
