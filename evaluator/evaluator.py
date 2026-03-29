# -*- coding: utf-8 -*-
"""
评估器
支持单任务和多任务评估
"""

import numpy as np
import torch
from typing import Dict, List, Optional, OrderedDict as OrderedDictType
from collections import OrderedDict

from .metrics import get_metric
from ..utils import get_logger


class Evaluator:
    """评估器
    
    统一管理多个评估指标
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = get_logger()
        
        # 加载指标
        self.metric_names = [m.lower() for m in config["metrics"]]
        self.metrics = {name: get_metric(name, config) for name in self.metric_names}
        
        # 验证指标
        self.valid_metric = config.get("valid_metric", "auc").lower()
        if self.valid_metric not in self.metric_names:
            self.valid_metric = self.metric_names[0]
    
    def evaluate(
        self,
        labels: np.ndarray,
        preds: np.ndarray,
        groups: Optional[np.ndarray] = None,
    ) -> OrderedDict:
        """评估
        
        Args:
            labels: 真实标签 [N]
            preds: 预测分数 [N]
            groups: 分组 ID（用于 GAUC）
        
        Returns:
            {metric_name: value}
        """
        results = OrderedDict()
        
        for name in self.metric_names:
            metric = self.metrics[name]
            
            # GAUC 需要分组信息
            if name == "gauc":
                value = metric.calculate(labels, preds, groups=groups)
            else:
                value = metric.calculate(labels, preds)
            
            results[name] = value
        
        return results
    
    def evaluate_multitask(
        self,
        labels: Dict[str, np.ndarray],
        preds: Dict[str, np.ndarray],
        groups: Optional[np.ndarray] = None,
    ) -> OrderedDict:
        """多任务评估
        
        Args:
            labels: {task_name: labels}
            preds: {task_name: preds}
            groups: 分组 ID
        
        Returns:
            {task_name: {metric_name: value}}
        """
        results = OrderedDict()
        
        for task, task_labels in labels.items():
            task_preds = preds.get(task)
            if task_preds is None:
                continue
            
            task_results = self.evaluate(task_labels, task_preds, groups)
            for metric_name, value in task_results.items():
                results[f"{task}_{metric_name}"] = value
        
        return results
    
    def get_valid_score(self, results: OrderedDict) -> float:
        """获取验证指标分数"""
        return results.get(self.valid_metric, 0.0)
    
    def better(self, a: float, b: float) -> bool:
        """判断 a 是否比 b 好
        
        对于 AUC/GAUC: 越大越好
        对于 LogLoss/ECE: 越小越好
        对于 PCOC: 越接近 1 越好
        """
        if self.valid_metric in ["auc", "gauc"]:
            return a > b
        elif self.valid_metric in ["logloss", "ece", "mse", "mae"]:
            return a < b
        elif self.valid_metric == "pcoc":
            # PCOC 越接近 1 越好
            return abs(a - 1.0) < abs(b - 1.0)
        else:
            return a > b
    
    def format_results(self, results: OrderedDict) -> str:
        """格式化输出结果"""
        parts = []
        for name, value in results.items():
            if name == "pcoc":
                parts.append(f"{name.upper()}: {value:.4f}")
            elif name in ["auc", "gauc"]:
                parts.append(f"{name.upper()}: {value:.6f}")
            elif name in ["logloss", "ece"]:
                parts.append(f"{name.upper()}: {value:.6f}")
            else:
                parts.append(f"{name.upper()}: {value:.6f}")
        return " | ".join(parts)
