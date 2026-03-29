# -*- coding: utf-8 -*-
"""
Two-Stage LTV Model

两阶段模型：
- Stage 1: 预测付费概率 P(paid)
- Stage 2: 预测条件价值 E[LTV | paid]

预测：LTV = P(paid) * E[LTV | paid]

优点：
- 解耦两个任务，可独立优化
- 付费率通常较低，分开建模更稳定
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple

from .base import LTVModel, register_ltv_model, LTVRegressionBase
from ...utils.enum import ModelType
from ..layers import MLPLayers


@register_ltv_model("two_stage")
class TwoStageLTV(LTVRegressionBase):
    """两阶段 LTV 模型
    
    架构：
    - 共享 Embedding
    - 付费预测塔：二分类
    - 价值预测塔：回归（只在付费样本上训练）
    """
    
    model_type = ModelType.LTV
    
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        
        hidden_size = config.get("hidden_sizes", [256, 128, 64])[-1]
        
        # Stage 1: 付费预测（二分类）
        self.paid_tower = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(config.get("dropout", 0.2)),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        
        # Stage 2: 条件价值预测（回归）
        # 输出 log(LTV|paid)，使用 MSE Loss
        self.value_tower = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(config.get("dropout", 0.2)),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            # 不加激活，输出 log space
        )
        
        # 是否使用 log 变换
        self.log_transform = config.get("log_transform", True)
        
        # 两阶段权重
        self.paid_weight = config.get("paid_weight", 1.0)
        self.value_weight = config.get("value_weight", 1.0)
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播
        
        Returns:
            paid_prob: 付费概率，shape (batch_size,)
            conditional_value: 条件价值预测，shape (batch_size,)
        """
        x = self._embed_features(batch)
        h = self.mlp[:-1](x)  # 去掉最后一层
        
        paid_prob = self.paid_tower(h).squeeze(-1)
        conditional_value = self.value_tower(h).squeeze(-1)
        
        return paid_prob, conditional_value
    
    def calculate_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算两阶段损失"""
        paid_prob, conditional_value = self.forward(batch)
        labels = batch[self.label_field].float()
        
        # Stage 1: 付费预测（二分类 BCE）
        paid_label = (labels > 0).float()
        paid_loss = F.binary_cross_entropy(paid_prob, paid_label)
        
        # Stage 2: 条件价值预测（只在付费样本上计算）
        paid_mask = paid_label > 0.5
        
        if paid_mask.sum() > 0:
            if self.log_transform:
                # 预测 log(LTV)
                target = torch.log(torch.clamp(labels[paid_mask], min=1.0))
                pred = conditional_value[paid_mask]
                value_loss = F.mse_loss(pred, target)
            else:
                # 直接预测 LTV
                value_loss = F.mse_loss(conditional_value[paid_mask], labels[paid_mask])
        else:
            value_loss = torch.tensor(0.0, device=labels.device)
        
        # 总损失
        loss = self.paid_weight * paid_loss + self.value_weight * value_loss
        
        return loss
    
    def predict(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """预测 LTV = P(paid) * E[LTV | paid]"""
        paid_prob, conditional_value = self.forward(batch)
        
        if self.log_transform:
            # conditional_value 是 log space
            conditional_ltv = torch.exp(conditional_value)
        else:
            conditional_ltv = F.relu(conditional_value)  # 保证非负
        
        ltv = paid_prob * conditional_ltv
        
        return ltv
    
    def predict_distribution(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """预测分布参数"""
        paid_prob, conditional_value = self.forward(batch)
        
        if self.log_transform:
            conditional_ltv = torch.exp(conditional_value)
        else:
            conditional_ltv = F.relu(conditional_value)
        
        return {
            "paid_prob": paid_prob,
            "conditional_ltv": conditional_ltv,
            "expected_ltv": paid_prob * conditional_ltv,
        }
