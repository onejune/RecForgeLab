# -*- coding: utf-8 -*-
"""
LTV 预估模型基类

LTV 预估的特殊性：
1. 零膨胀：大部分用户 LTV=0（未付费）
2. 右偏分布：少数高价值用户贡献大部分收益
3. 延迟反馈：真实 LTV 需要时间积累

评估指标：
- MAE, MSE, RMSE：绝对误差
- MAPE：相对误差
- Gini, AUC：排序能力
- Decile Lift：分段提升度
"""

import torch
import torch.nn as nn
import numpy as np
from abc import abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr

from ...utils.enum import ModelType
from ..base import BaseModel, register_model
from ..layers import FeatureEmbedding, MLPLayers


def register_ltv_model(name: str):
    """LTV 模型注册装饰器"""
    return register_model(name)


class LTVModel(BaseModel):
    """LTV 预估模型基类
    
    子类需要实现：
    - forward(): 前向传播
    - calculate_loss(): 损失计算
    - predict(): 预测 LTV 值
    
    可选覆盖：
    - predict_distribution(): 预测分布参数（概率模型）
    """
    
    model_type = ModelType.LTV
    label_field = "ltv"  # LTV 标签字段
    
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        
        # LTV 特有配置
        self.ltv_max = config.get("ltv_max", 10000.0)  # LTV 上限（截断）
        self.log_transform = config.get("log_transform", True)  # 是否 log 变换
        self.zero_threshold = config.get("zero_threshold", 0.5)  # 零值判断阈值
    
    @abstractmethod
    def predict(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """预测 LTV 值
        
        Args:
            batch: 输入 batch
        
        Returns:
            LTV 预测值，shape (batch_size,)
        """
        raise NotImplementedError
    
    def predict_distribution(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """预测分布参数（概率模型覆盖此方法）
        
        Returns:
            分布参数字典，如 {"mean": ..., "std": ...}
        """
        return {"value": self.predict(batch)}
    
    def evaluate(self, preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """评估 LTV 预测结果
        
        Args:
            preds: 预测值
            labels: 真实值
        
        Returns:
            评估指标字典
        """
        # 基础指标
        mae = mean_absolute_error(labels, preds)
        mse = mean_squared_error(labels, preds)
        rmse = np.sqrt(mse)
        
        # MAPE（避免除零）
        mask = labels > 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((labels[mask] - preds[mask]) / labels[mask]))
        else:
            mape = 0.0
        
        # 排序指标
        gini = self._compute_gini(labels, preds)
        spearman_corr, _ = spearmanr(labels, preds)
        
        # Decile 分析
        decile_metrics = self._compute_decile_metrics(labels, preds)
        
        # 零值预测准确率
        pred_zero = (preds < self.zero_threshold).astype(int)
        label_zero = (labels < 1e-6).astype(int)
        zero_acc = np.mean(pred_zero == label_zero)
        
        return {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "MAPE": mape,
            "Gini": gini,
            "Spearman": spearman_corr,
            "Zero_Acc": zero_acc,
            **decile_metrics,
        }
    
    def _compute_gini(self, labels: np.ndarray, preds: np.ndarray) -> float:
        """计算 Gini 系数
        
        Gini = 2 * AUC - 1
        """
        from sklearn.metrics import roc_auc_score
        
        # 将 LTV 视为正样本权重
        try:
            # 二值化：LTV > 0 视为正样本
            binary_labels = (labels > 0).astype(int)
            if binary_labels.sum() > 0 and binary_labels.sum() < len(binary_labels):
                auc = roc_auc_score(binary_labels, preds)
                return 2 * auc - 1
        except:
            pass
        
        # 备选：排序计算
        n = len(labels)
        sorted_idx = np.argsort(preds)[::-1]  # 降序
        sorted_labels = labels[sorted_idx]
        cumsum = np.cumsum(sorted_labels)
        gini = (2 * np.sum(cumsum) / (n * np.sum(labels)) - (n + 1) / n) if np.sum(labels) > 0 else 0
        return max(0, gini)
    
    def _compute_decile_metrics(self, labels: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
        """计算 Decile 提升度
        
        将用户按预测 LTV 分成 10 组，计算每组的实际 LTV 均值
        """
        n = len(preds)
        decile_size = n // 10
        
        sorted_idx = np.argsort(preds)[::-1]  # 降序
        
        metrics = {}
        for i in range(10):
            start = i * decile_size
            end = start + decile_size if i < 9 else n
            decile_labels = labels[sorted_idx[start:end]]
            metrics[f"Decile_{i+1}_Mean"] = np.mean(decile_labels)
        
        # Top 10% Lift: Top 10% / Bottom 10%
        top_mean = metrics["Decile_1_Mean"]
        bottom_mean = metrics.get("Decile_10_Mean", 1e-6)
        metrics["Top_Lift"] = top_mean / max(bottom_mean, 1e-6)
        
        return metrics


class LTVRegressionBase(LTVModel):
    """回归类 LTV 模型基类
    
    包含通用的 Embedding + MLP 结构
    """
    
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        
        # 特征嵌入
        self.feature_embedding = FeatureEmbedding(config, dataset)
        
        # MLP
        input_dim = self.feature_embedding.output_dim
        hidden_sizes = config.get("hidden_sizes", [256, 128, 64])
        
        self.mlp = MLPLayers(
            input_dim=input_dim,
            hidden_sizes=hidden_sizes,
            dropout=config.get("dropout", 0.2),
            activation=config.get("activation", "relu"),
        )
        
        # 输出层
        self.output_layer = nn.Linear(hidden_sizes[-1], 1)
    
    def _embed_features(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """特征嵌入"""
        return self.feature_embedding(batch)
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """前向传播"""
        x = self._embed_features(batch)
        x = self.mlp(x)
        output = self.output_layer(x)
        return output.squeeze(-1)
