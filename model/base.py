# -*- coding: utf-8 -*-
"""
模型基类体系
支持：
- 模型注册机制 @register_model("deepfm")
- 统一的 calculate_loss / predict 接口
- 多任务 task weight 配置
- model_type 属性（CTR/MULTITASK/SSL）
- extra_metrics 钩子
"""

import torch
import torch.nn as nn
from abc import abstractmethod
from typing import Dict, List, Optional, Any, Tuple

from ..utils.enum import ModelType
from ..utils.logger import get_logger, set_color


# ============================================================
# 模型注册表
# ============================================================

MODEL_REGISTRY: Dict[str, type] = {}


def register_model(name: str):
    """模型注册装饰器
    
    用法::
    
        @register_model("deepfm")
        class DeepFM(CTRModel):
            ...
    """
    def decorator(cls):
        key = name.lower()
        if key in MODEL_REGISTRY:
            get_logger().warning(f"Model '{key}' already registered, overwriting.")
        MODEL_REGISTRY[key] = cls
        cls._registered_name = key
        return cls
    return decorator


def get_model(name: str) -> type:
    """根据名称获取模型类
    
    Args:
        name: 模型名称（大小写不敏感）
    
    Returns:
        模型类
    
    Raises:
        KeyError: 模型未注册
    """
    key = name.lower()
    if key not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise KeyError(
            f"Model '{name}' not found. Available models: {available}\n"
            f"Make sure to import the model module before calling get_model()."
        )
    return MODEL_REGISTRY[key]


# ============================================================
# 基类
# ============================================================

class BaseModel(nn.Module):
    """所有模型的基类
    
    子类必须实现：
    - calculate_loss(batch) -> Tensor
    - predict(batch) -> Tensor or Dict[str, Tensor]
    
    可选覆盖：
    - extra_metrics(batch, outputs) -> Dict[str, float]
    """

    # 子类可覆盖
    model_type: ModelType = ModelType.CTR

    def __init__(self, config, dataset):
        super().__init__()
        self.config = config
        self.device = config["device"]
        self.model_name = config.get("model", self.__class__.__name__)
        self.label_field = config.get("label_field", "label")
        self.embedding_size = config["embedding_size"]

        # 从 dataset 获取特征信息
        self.sparse_features: List[str] = dataset.sparse_features
        self.dense_features: List[str] = dataset.dense_features
        self.num_sparse_features: int = len(self.sparse_features)
        self.num_dense_features: int = len(self.dense_features)
        self.num_features: int = self.num_sparse_features + self.num_dense_features

    def _print_param_count(self):
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        get_logger().info(set_color("Trainable parameters", "blue") + f": {n:,}")

    @abstractmethod
    def calculate_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算训练损失
        
        Args:
            batch: 来自 DataLoader 的一个 batch，包含特征和标签
        
        Returns:
            scalar loss tensor
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, batch: Dict[str, torch.Tensor]):
        """推理预测
        
        Args:
            batch: 特征 batch
        
        Returns:
            CTR 模型: [B] float tensor (概率)
            多任务模型: Dict[task_name, [B] tensor]
        """
        raise NotImplementedError

    def extra_metrics(
        self,
        batch: Dict[str, torch.Tensor],
        outputs: Any,
    ) -> Dict[str, float]:
        """模型自定义指标钩子（可选覆盖）
        
        在每个 eval step 调用，模型可返回自定义指标（如专家利用率、gate 熵等）
        
        Args:
            batch: 当前 batch
            outputs: predict() 的输出
        
        Returns:
            Dict[metric_name, value]，空字典表示无自定义指标
        """
        return {}

    def __repr__(self) -> str:
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (
            f"{self.__class__.__name__}("
            f"type={self.model_type.value}, "
            f"params={n:,})"
        )


# ============================================================
# CTR 单任务基类
# ============================================================

class CTRModel(BaseModel):
    """CTR 单任务模型基类
    
    model_type = ModelType.CTR
    predict() 返回 [B] 概率 tensor
    """

    model_type = ModelType.CTR

    def __init__(self, config, dataset):
        super().__init__(config, dataset)


# ============================================================
# 多任务基类
# ============================================================

class MultiTaskModel(BaseModel):
    """多任务模型基类
    
    model_type = ModelType.MULTITASK
    predict() 返回 Dict[task_name, [B] tensor]
    
    内置 task_weights 支持，calculate_loss 时自动加权。
    """

    model_type = ModelType.MULTITASK

    # 子类可覆盖的任务标签映射
    LABEL_MAP: Dict[str, str] = {
        "ctr": "click_label",
        "cvr": "label",
        "ctcvr": "label",
    }

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.tasks: List[str] = config.get("tasks", ["ctr", "cvr"])
        self.task_weights: List[float] = config.get("task_weights", [1.0] * len(self.tasks))

        if len(self.task_weights) != len(self.tasks):
            raise ValueError(
                f"task_weights length ({len(self.task_weights)}) "
                f"must match tasks length ({len(self.tasks)})"
            )

    def get_task_label(self, task: str, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """获取任务对应的标签 tensor"""
        label_key = self.LABEL_MAP.get(task, task + "_label")
        if label_key not in batch:
            # 回退到通用 label
            label_key = self.label_field
        return batch[label_key].float()

    def extra_metrics(
        self,
        batch: Dict[str, torch.Tensor],
        outputs: Any,
    ) -> Dict[str, float]:
        """默认返回各任务权重（方便监控）"""
        return {f"weight_{t}": w for t, w in zip(self.tasks, self.task_weights)}


# ============================================================
# SSL 基类
# ============================================================

class SSLModel(BaseModel):
    """自监督学习模型基类
    
    model_type = ModelType.SSL
    支持两阶段训练：
    - phase="pretrain": 返回对比损失
    - phase="finetune": 返回下游任务损失
    """

    model_type = ModelType.SSL

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.ssl_weight: float = config.get("ssl_weight", 1.0)
        self.training_phase: str = config.get("training_phase", "pretrain")  # pretrain / finetune / joint

    def set_phase(self, phase: str):
        """切换训练阶段"""
        assert phase in ("pretrain", "finetune", "joint"), f"Unknown phase: {phase}"
        self.training_phase = phase

    def ssl_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """对比/自监督损失（子类实现）"""
        raise NotImplementedError

    def downstream_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """下游任务损失（子类实现）"""
        raise NotImplementedError

    def calculate_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.training_phase == "pretrain":
            return self.ssl_loss(batch)
        elif self.training_phase == "finetune":
            return self.downstream_loss(batch)
        else:  # joint
            return self.ssl_weight * self.ssl_loss(batch) + self.downstream_loss(batch)


# ============================================================
# Multi-Domain 基类
# ============================================================

class MultiDomainModel(BaseModel):
    """多域/多场景模型基类
    
    model_type = ModelType.MULTIDOMAIN
    核心特点：
    - 输入包含 domain_indicator 字段，标识样本所属域
    - predict() 返回 [B] 概率 tensor
    - 支持分域评估（各域单独计算指标）
    """

    model_type = ModelType.MULTIDOMAIN

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        
        # 域配置
        self.domain_field: str = config.get("domain_field", "domain_indicator")
        self.num_domains: int = config.get("num_domains", 1)
        
        # 域特征（用于 domain embedding，可选）
        self.domain_features: List[str] = config.get("domain_features", [])
        
        # 从 dataset 获取域数量（如果未配置）
        if hasattr(dataset, "num_domains"):
            self.num_domains = max(self.num_domains, dataset.num_domains)

    def get_domain_id(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """获取域标识 tensor [B]"""
        domain_id = batch.get(self.domain_field)
        if domain_id is None:
            raise KeyError(
                f"'{self.domain_field}' not found in batch. "
                f"Multi-domain models require a domain indicator field."
            )
        return domain_id.long()

    def extra_metrics(
        self,
        batch: Dict[str, torch.Tensor],
        outputs: Any,
    ) -> Dict[str, float]:
        """默认返回域数量"""
        return {"num_domains": float(self.num_domains)}
