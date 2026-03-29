# -*- coding: utf-8 -*-
"""
Ordinal Regression LTV Model

将 LTV 分桶，转为有序多分类问题。

优点：
- 对异常值鲁棒
- 保留桶的顺序信息
- 可建模复杂的非线性关系

方法：
1. 将 LTV 分成 K 个有序桶
2. 预测每个桶的概率
3. 期望值 = Σ prob_k * bucket_mid_k
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple

from .base import LTVModel, register_ltv_model, LTVRegressionBase
from ...utils.enum import ModelType


@register_ltv_model("ordinal")
class OrdinalLTV(LTVRegressionBase):
    """Ordinal Regression LTV 模型
    
    配置：
    - num_bins: 分桶数量，默认 10
    - bin_boundaries: 桶边界，如 [0, 1, 5, 10, 50, 100, 500, 1000, 5000, 10000]
    - ordinal_method: "softmax" | "corn" (cumulative ordinal regression)
    """
    
    model_type = ModelType.LTV
    
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        
        # 分桶配置
        self.num_bins = config.get("num_bins", 10)
        self.bin_boundaries = config.get("bin_boundaries", None)
        self.ordinal_method = config.get("ordinal_method", "softmax")
        
        # 默认分桶（对数尺度）
        if self.bin_boundaries is None:
            self.bin_boundaries = [0] + [10 ** (i * 0.5) for i in range(self.num_bins)]
        
        self.num_bins = len(self.bin_boundaries) - 1
        self.bin_midpoints = self._compute_bin_midpoints()
        
        hidden_size = config.get("hidden_sizes", [256, 128, 64])[-1]
        
        if self.ordinal_method == "softmax":
            # Softmax 多分类
            self.output_layer = nn.Linear(hidden_size, self.num_bins)
        elif self.ordinal_method == "corn":
            # Cumulative Ordinal Regression Network
            # 预测 P(y > k | x) for k = 1, ..., K-1
            self.output_layer = nn.Linear(hidden_size, self.num_bins - 1)
        else:
            raise ValueError(f"Unknown ordinal method: {self.ordinal_method}")
        
        # 注册 bin_midpoints 为 buffer（会随模型移动到设备）
        self.register_buffer(
            "bin_midpoints_tensor",
            torch.tensor(self.bin_midpoints, dtype=torch.float32)
        )
    
    def _compute_bin_midpoints(self) -> List[float]:
        """计算每个桶的中点值"""
        midpoints = []
        for i in range(self.num_bins):
            low = self.bin_boundaries[i]
            high = self.bin_boundaries[i + 1]
            midpoints.append((low + high) / 2)
        return midpoints
    
    def _ltv_to_bin(self, ltv: torch.Tensor) -> torch.Tensor:
        """将 LTV 值映射到桶索引"""
        bins = torch.zeros_like(ltv, dtype=torch.long)
        
        for i in range(self.num_bins):
            mask = ltv >= self.bin_boundaries[i + 1]
            bins[mask] = i + 1
        
        bins = torch.clamp(bins, 0, self.num_bins - 1)
        return bins
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """前向传播，返回桶概率"""
        x = self._embed_features(batch)
        h = self.mlp[:-1](x)
        
        if self.ordinal_method == "softmax":
            logits = self.output_layer(h)
            probs = F.softmax(logits, dim=-1)
        else:  # corn
            # P(y > k | x)
            cum_probs = torch.sigmoid(self.output_layer(h))
            # 转换为 P(y = k | x)
            # P(y = 0) = 1 - P(y > 0)
            # P(y = k) = P(y > k-1) - P(y > k)
            # P(y = K-1) = P(y > K-2)
            # 这里简化处理，直接用 cum_probs
            probs = torch.zeros(h.size(0), self.num_bins, device=h.device)
            probs[:, 0] = 1 - cum_probs[:, 0]
            for k in range(1, self.num_bins - 1):
                probs[:, k] = cum_probs[:, k-1] - cum_probs[:, k]
            probs[:, -1] = cum_probs[:, -1]
        
        return probs
    
    def calculate_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算交叉熵损失"""
        probs = self.forward(batch)
        labels = batch[self.label_field].float()
        
        # LTV → 桶索引
        bin_labels = self._ltv_to_bin(labels)
        
        if self.ordinal_method == "softmax":
            # 标准交叉熵
            loss = F.cross_entropy(probs, bin_labels)
        else:  # corn
            # Corn 损失
            # 需要将 label 转换为 cumulative label
            cum_labels = (bin_labels.unsqueeze(1) > torch.arange(self.num_bins - 1, device=labels.device)).float()
            cum_probs = torch.sigmoid(self.output_layer(self.mlp[:-1](self._embed_features(batch))))
            loss = F.binary_cross_entropy(cum_probs, cum_labels)
        
        return loss
    
    def predict(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """预测 LTV = E[y] = Σ prob_k * midpoint_k"""
        probs = self.forward(batch)
        
        # 加权求和
        ltv = torch.matmul(probs, self.bin_midpoints_tensor)
        
        return ltv
    
    def predict_distribution(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """预测分布参数"""
        probs = self.forward(batch)
        
        return {
            "bin_probs": probs,
            "expected_ltv": torch.matmul(probs, self.bin_midpoints_tensor),
        }
