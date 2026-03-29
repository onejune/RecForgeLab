# -*- coding: utf-8 -*-
"""
评估指标
支持 AUC, LogLoss, GAUC, PCOC, ECE
"""

import numpy as np
from typing import Optional, Tuple
from sklearn.metrics import roc_auc_score, log_loss


class BaseMetric:
    """指标基类"""
    
    def __init__(self, config=None):
        self.config = config or {}
    
    def calculate(self, labels: np.ndarray, preds: np.ndarray, **kwargs) -> float:
        raise NotImplementedError


class AUC(BaseMetric):
    """AUC 指标"""
    
    def calculate(self, labels: np.ndarray, preds: np.ndarray, **kwargs) -> float:
        """计算 AUC
        
        Args:
            labels: 真实标签 [N]
            preds: 预测分数 [N]
        
        Returns:
            AUC 值
        """
        # 检查是否只有一个类别
        if len(np.unique(labels)) < 2:
            return 0.5
        return roc_auc_score(labels, preds)


class LogLossMetric(BaseMetric):
    """LogLoss 指标"""
    
    def calculate(self, labels: np.ndarray, preds: np.ndarray, **kwargs) -> float:
        """计算 LogLoss"""
        # 裁剪预测值避免 log(0)
        preds = np.clip(preds, 1e-7, 1 - 1e-7)
        return log_loss(labels, preds)


class GAUC(BaseMetric):
    """Group AUC - 按 user_id 分组计算 AUC 的加权平均"""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.min_samples = config.get("gauc_min_samples", 2) if config else 2
    
    def calculate(
        self, 
        labels: np.ndarray, 
        preds: np.ndarray, 
        groups: Optional[np.ndarray] = None,
        **kwargs
    ) -> float:
        """计算 GAUC
        
        Args:
            labels: 真实标签 [N]
            preds: 预测分数 [N]
            groups: 分组 ID（如 user_id）[N]
        
        Returns:
            GAUC 值（按组样本数加权平均）
        """
        if groups is None:
            # 没有 group 信息，退化为普通 AUC
            return AUC(self.config).calculate(labels, preds)
        
        # 按 group 分组计算
        unique_groups = np.unique(groups)
        auc_scores = []
        group_weights = []
        
        for g in unique_groups:
            mask = groups == g
            g_labels = labels[mask]
            g_preds = preds[mask]
            
            # 跳过样本太少或只有一个类别的组
            if len(g_labels) < self.min_samples or len(np.unique(g_labels)) < 2:
                continue
            
            try:
                g_auc = roc_auc_score(g_labels, g_preds)
                auc_scores.append(g_auc)
                group_weights.append(len(g_labels))
            except Exception:
                continue
        
        if len(auc_scores) == 0:
            return 0.5
        
        # 按样本数加权平均
        weights = np.array(group_weights, dtype=np.float32)
        weights = weights / weights.sum()
        return float(np.average(auc_scores, weights=weights))


class PCOC(BaseMetric):
    """Prediction Click-Over-Click - 预测校准度
    
    PCOC = sum(preds) / sum(labels)
    PCOC = 1.0 表示完美校准
    """
    
    def calculate(self, labels: np.ndarray, preds: np.ndarray, **kwargs) -> float:
        """计算 PCOC
        
        Args:
            labels: 真实标签 [N]
            preds: 预测分数 [N]
        
        Returns:
            PCOC 值（理想值 = 1.0）
        """
        pred_sum = preds.sum()
        label_sum = labels.sum()
        
        if label_sum == 0:
            return float('inf') if pred_sum > 0 else 1.0
        
        return float(pred_sum / label_sum)


class ECE(BaseMetric):
    """Expected Calibration Error - 期望校准误差
    
    衡量预测概率与实际准确率之间的差距
    """
    
    def __init__(self, config=None):
        super().__init__(config)
        self.n_bins = config.get("ece_n_bins", 10) if config else 10
    
    def calculate(self, labels: np.ndarray, preds: np.ndarray, **kwargs) -> float:
        """计算 ECE
        
        Args:
            labels: 真实标签 [N]
            preds: 预测分数 [N]
            n_bins: 分桶数量
        
        Returns:
            ECE 值（越小越好，理想值 = 0）
        """
        n_bins = self.n_bins
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        
        ece = 0.0
        total_samples = len(labels)
        
        for i in range(n_bins):
            # 找到落在当前桶内的样本
            in_bin = (preds > bin_boundaries[i]) & (preds <= bin_boundaries[i + 1])
            bin_size = in_bin.sum()
            
            if bin_size > 0:
                # 桶内平均预测值
                avg_confidence = preds[in_bin].mean()
                # 桶内实际正例率
                avg_accuracy = labels[in_bin].mean()
                # 加权误差
                ece += (bin_size / total_samples) * abs(avg_confidence - avg_accuracy)
        
        return float(ece)


class MSE(BaseMetric):
    """均方误差"""
    
    def calculate(self, labels: np.ndarray, preds: np.ndarray, **kwargs) -> float:
        return float(np.mean((preds - labels) ** 2))


class MAE(BaseMetric):
    """平均绝对误差"""
    
    def calculate(self, labels: np.ndarray, preds: np.ndarray, **kwargs) -> float:
        return float(np.mean(np.abs(preds - labels)))


# 指标注册表
METRIC_REGISTRY = {
    "auc": AUC,
    "logloss": LogLossMetric,
    "gauc": GAUC,
    "pcoc": PCOC,
    "ece": ECE,
    "mse": MSE,
    "mae": MAE,
}


def get_metric(name: str, config=None):
    """获取指标实例
    
    Args:
        name: 指标名称（不区分大小写）
        config: 配置
    
    Returns:
        指标实例
    """
    name = name.lower()
    if name not in METRIC_REGISTRY:
        raise ValueError(f"Unknown metric: {name}. Available: {list(METRIC_REGISTRY.keys())}")
    return METRIC_REGISTRY[name](config)
