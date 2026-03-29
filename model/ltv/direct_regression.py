# -*- coding: utf-8 -*-
"""
Direct Regression LTV Model

直接回归建模，支持多种损失函数：
- MSE (L2 Loss)
- MAE (L1 Loss)
- Huber Loss
- Log-Cosh Loss
- Quantile Loss
- WCE (Weighted Cross Entropy)
- Log-MSE (预测 log(LTV+1))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple

from .base import LTVModel, register_ltv_model, LTVRegressionBase
from ...utils.enum import ModelType


# ============================================================
# 损失函数定义
# ============================================================

def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE (L2 Loss)"""
    return F.mse_loss(pred, target)


def mae_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MAE (L1 Loss)"""
    return F.l1_loss(pred, target)


def huber_loss(pred: torch.Tensor, target: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    """Huber Loss
    
    对小误差用 MSE，大误差用 MAE，平滑过渡
    
    L = 0.5 * (y - ŷ)²          if |y - ŷ| <= δ
        δ * (|y - ŷ| - 0.5*δ)   otherwise
    """
    diff = torch.abs(pred - target)
    quadratic = torch.min(diff, torch.tensor(delta, device=diff.device))
    linear = diff - quadratic
    return torch.mean(0.5 * quadratic ** 2 + delta * linear)


def log_cosh_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Log-Cosh Loss
    
    L = log(cosh(y - ŷ))
    
    优点：
    - 处处可微
    - 小误差类似 x²/2（MSE）
    - 大误差类似 |x| - log(2)（MAE）
    """
    diff = pred - target
    return torch.mean(torch.log(torch.cosh(diff + 1e-12)))


def quantile_loss(pred: torch.Tensor, target: torch.Tensor, 
                  quantiles: list = [0.1, 0.5, 0.9]) -> torch.Tensor:
    """Quantile Loss (Pinball Loss)
    
    预测多个分位数，捕获不确定性
    
    L_q = max(q*(y - ŷ), (1-q)*(ŷ - y))
    
    Args:
        pred: shape (batch_size, num_quantiles)
        target: shape (batch_size,)
        quantiles: 分位数列表
    """
    losses = []
    for i, q in enumerate(quantiles):
        error = target - pred[:, i]
        losses.append(torch.max(q * error, (1 - q) * -error))
    
    return torch.mean(torch.stack(losses))


def weighted_mse_loss(pred: torch.Tensor, target: torch.Tensor, 
                      weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """加权 MSE
    
    Args:
        weights: 样本权重，如付费样本权重更高
    """
    if weights is None:
        return F.mse_loss(pred, target)
    
    return torch.mean(weights * (pred - target) ** 2)


def log_mse_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Log-MSE Loss
    
    预测 log(LTV + 1)，缓解右偏分布
    
    Args:
        pred: 预测的 log(LTV + 1)
        target: 真实的 LTV
    """
    log_target = torch.log(target + 1)
    return F.mse_loss(pred, log_target)


def wce_loss(pred: torch.Tensor, target: torch.Tensor, 
             zero_weight: float = 0.1, nonzero_weight: float = 1.0) -> torch.Tensor:
    """Weighted Cross Entropy for LTV
    
    将 LTV 预估转为加权二分类问题：
    - 零值 vs 非零
    - 对非零样本给予更高权重
    
    Args:
        pred: 预测的付费概率（0-1）
        target: 真实 LTV
        zero_weight: 零值样本权重
        nonzero_weight: 非零样本权重
    """
    # 二值标签
    binary_target = (target > 0).float()
    
    # 权重
    weights = torch.where(target > 0, 
                          torch.tensor(nonzero_weight, device=target.device),
                          torch.tensor(zero_weight, device=target.device))
    
    # 加权 BCE
    bce = F.binary_cross_entropy(pred, binary_target, reduction='none')
    return torch.mean(weights * bce)


# ============================================================
# Direct Regression Model
# ============================================================

@register_ltv_model("direct_regression")
class DirectRegressionLTV(LTVRegressionBase):
    """直接回归 LTV 模型
    
    支持多种损失函数：
    - "mse": 基础 MSE
    - "mae": MAE（鲁棒）
    - "huber": Huber Loss
    - "log_cosh": Log-Cosh Loss
    - "quantile": 分位数损失
    - "log_mse": 预测 log(LTV+1)
    - "weighted_mse": 加权 MSE
    - "wce": 加权交叉熵
    
    配置：
    - loss_type: 损失函数类型
    - huber_delta: Huber delta 参数
    - quantiles: 分位数列表（quantile loss）
    - zero_weight / nonzero_weight: WCE 权重
    - log_transform: 是否 log 变换（预测 log LTV）
    """
    
    model_type = ModelType.LTV
    
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        
        # 损失函数配置
        self.loss_type = config.get("loss_type", "mse")
        self.huber_delta = config.get("huber_delta", 1.0)
        self.quantiles = config.get("quantiles", [0.1, 0.5, 0.9])
        self.zero_weight = config.get("zero_weight", 0.1)
        self.nonzero_weight = config.get("nonzero_weight", 1.0)
        self.log_transform = config.get("log_transform", False)
        
        hidden_size = config.get("hidden_sizes", [256, 128, 64])[-1]
        
        # 输出层
        if self.loss_type == "quantile":
            # 多头输出：每个分位数一个
            self.output_layer = nn.Linear(hidden_size, len(self.quantiles))
        elif self.loss_type == "wce":
            # 输出概率
            self.output_layer = nn.Sequential(
                nn.Linear(hidden_size, 1),
                nn.Sigmoid(),
            )
        else:
            # 单输出
            if self.log_transform:
                # 预测 log(LTV + 1)，不需要激活
                self.output_layer = nn.Linear(hidden_size, 1)
            else:
                # 预测 LTV，保证非负
                self.output_layer = nn.Sequential(
                    nn.Linear(hidden_size, 1),
                    nn.Softplus(),  # 保证 > 0
                )
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """前向传播"""
        x = self._embed_features(batch)
        h = self.mlp[:-1](x)
        output = self.output_layer(h)
        
        if self.loss_type == "quantile":
            return output  # (batch_size, num_quantiles)
        else:
            return output.squeeze(-1)  # (batch_size,)
    
    def calculate_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算损失"""
        pred = self.forward(batch)
        target = batch[self.label_field].float()
        
        if self.loss_type == "mse":
            return mse_loss(pred, target)
        
        elif self.loss_type == "mae":
            return mae_loss(pred, target)
        
        elif self.loss_type == "huber":
            return huber_loss(pred, target, delta=self.huber_delta)
        
        elif self.loss_type == "log_cosh":
            return log_cosh_loss(pred, target)
        
        elif self.loss_type == "quantile":
            return quantile_loss(pred, target, quantiles=self.quantiles)
        
        elif self.loss_type == "log_mse":
            return log_mse_loss(pred, target)
        
        elif self.loss_type == "weighted_mse":
            # 非零样本权重更高
            weights = torch.where(target > 0,
                                  torch.tensor(self.nonzero_weight, device=target.device),
                                  torch.tensor(self.zero_weight, device=target.device))
            return weighted_mse_loss(pred, target, weights)
        
        elif self.loss_type == "wce":
            return wce_loss(pred, target, 
                           zero_weight=self.zero_weight,
                           nonzero_weight=self.nonzero_weight)
        
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
    
    def predict(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """预测 LTV"""
        pred = self.forward(batch)
        
        if self.loss_type == "quantile":
            # 返回中位数预测（第 2 个分位数）
            median_idx = len(self.quantiles) // 2
            return pred[:, median_idx]
        
        elif self.loss_type == "wce":
            # WCE 只预测付费概率，乘以平均 LTV
            # 这里简化：假设平均付费 LTV 已知
            avg_paid_ltv = self.config.get("avg_paid_ltv", 300.0)
            return pred * avg_paid_ltv
        
        elif self.log_transform:
            # 从 log 空间转换回来
            return torch.exp(pred) - 1
        
        else:
            return pred
    
    def predict_distribution(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """预测分布参数"""
        if self.loss_type == "quantile":
            pred = self.forward(batch)
            return {
                "quantiles": {f"q{int(q*100)}": pred[:, i] for i, q in enumerate(self.quantiles)},
                "median": pred[:, len(self.quantiles) // 2],
            }
        else:
            return {"value": self.predict(batch)}


# ============================================================
# 注册别名
# ============================================================

@register_ltv_model("mse_ltv")
class MSELTV(DirectRegressionLTV):
    """MSE 回归 LTV"""
    def __init__(self, config, dataset):
        config["loss_type"] = "mse"
        super().__init__(config, dataset)


@register_ltv_model("mae_ltv")
class MAELTV(DirectRegressionLTV):
    """MAE 回归 LTV"""
    def __init__(self, config, dataset):
        config["loss_type"] = "mae"
        super().__init__(config, dataset)


@register_ltv_model("huber_ltv")
class HuberLTV(DirectRegressionLTV):
    """Huber 回归 LTV"""
    def __init__(self, config, dataset):
        config["loss_type"] = "huber"
        super().__init__(config, dataset)


@register_ltv_model("log_cosh_ltv")
class LogCoshLTV(DirectRegressionLTV):
    """Log-Cosh 回归 LTV"""
    def __init__(self, config, dataset):
        config["loss_type"] = "log_cosh"
        super().__init__(config, dataset)


@register_ltv_model("quantile_ltv")
class QuantileLTV(DirectRegressionLTV):
    """分位数回归 LTV"""
    def __init__(self, config, dataset):
        config["loss_type"] = "quantile"
        super().__init__(config, dataset)


@register_ltv_model("log_mse_ltv")
class LogMSELTV(DirectRegressionLTV):
    """Log-MSE 回归 LTV（预测 log(LTV+1)）"""
    def __init__(self, config, dataset):
        config["loss_type"] = "log_mse"
        super().__init__(config, dataset)
