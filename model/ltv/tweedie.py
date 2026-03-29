# -*- coding: utf-8 -*-
"""
Tweedie Loss Model

Tweedie 分布：
- 属于指数分布族
- 方差 ∝ mean^p，p ∈ (1, 2) 时为零膨胀连续分布
- 特别适合 LTV 这种零膨胀 + 右偏分布

损失函数：
L = -log p(y; μ, φ, p)
  = (y * μ^(1-p) / (1-p) - μ^(2-p) / (2-p)) / φ + ...

当 p → 1: Poisson
当 p → 2: Gamma
当 p ∈ (1, 2): Compound Poisson-Gamma（零膨胀）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict

from .base import LTVModel, register_ltv_model, LTVRegressionBase
from ...utils.enum import ModelType


def tweedie_loss(pred: torch.Tensor, target: torch.Tensor, 
                 p: float = 1.5, eps: float = 1e-6) -> torch.Tensor:
    """Tweedie Negative Log-Likelihood
    
    Args:
        pred: 预测值 μ（必须 > 0）
        target: 真实值 y
        p: Tweedie power 参数，(1, 2)
        eps: 数值稳定性
    
    Returns:
        loss: Tweedie NLL
    """
    pred = torch.clamp(pred, min=eps)
    target = torch.clamp(target, min=0)
    
    # Tweedie NLL:
    # -log p(y; μ, φ, p) ∝ y * μ^(1-p) / (1-p) - μ^(2-p) / (2-p)
    # 当 y = 0: -μ^(2-p) / (2-p)
    
    a = pred ** (1 - p) / (1 - p)
    b = pred ** (2 - p) / (2 - p)
    
    loss = -target * a + b
    
    return loss.mean()


@register_ltv_model("tweedie")
class TweedieLTV(LTVRegressionBase):
    """Tweedie Loss LTV 模型
    
    直接预测 LTV 值，使用 Tweedie Loss 处理零膨胀分布。
    
    配置：
    - tweedie_p: Tweedie power 参数，默认 1.5
    """
    
    model_type = ModelType.LTV
    
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        
        # Tweedie power 参数
        self.tweedie_p = config.get("tweedie_p", 1.5)
        
        hidden_size = config.get("hidden_sizes", [256, 128, 64])[-1]
        
        # 输出层：直接预测 LTV（保证 > 0）
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Softplus(),  # 保证输出 > 0
        )
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """前向传播，预测 LTV"""
        x = self._embed_features(batch)
        h = self.mlp[:-1](x)
        ltv = self.output_layer(h).squeeze(-1)
        return ltv
    
    def calculate_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Tweedie Loss"""
        pred = self.forward(batch)
        labels = batch[self.label_field].float()
        
        return tweedie_loss(pred, labels, p=self.tweedie_p)
    
    def predict(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """预测 LTV"""
        return self.forward(batch)
