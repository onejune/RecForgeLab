# -*- coding: utf-8 -*-
"""
RecForgeLab: DSP 广告 CTR/CVR 预估实验框架

核心特性:
- 插件式注册机制
- 18 个内置模型 (CTR + 多任务 + Multi-Domain)
- 16 种连续特征编码器
- 配置驱动实验
"""

__version__ = "1.1.0"

from .utils import Config, get_logger
from .data import DSPDataset, DATASET_REGISTRY, register_dataset
from .model import (
    BaseModel, CTRModel, MultiTaskModel, MultiDomainModel, SSLModel,
    MODEL_REGISTRY, register_model, get_model,
    DeepFM, DCN, DCNv2, AutoInt, xDeepFM,
    ESMM, MMoE, PLE, SharedBottom, DirectCTCVR,
    STAR, M2M, EPNet, PPNet, HAMUR, M3oE,
)
from .trainer import Trainer, MultiTaskTrainer
from .evaluator import Evaluator


__all__ = [
    "__version__",
    # 配置
    "Config",
    "get_logger",
    # 数据
    "DSPDataset",
    "DATASET_REGISTRY",
    "register_dataset",
    # 模型基类
    "BaseModel",
    "CTRModel",
    "MultiTaskModel",
    "MultiDomainModel",
    "SSLModel",
    # 模型注册
    "MODEL_REGISTRY",
    "register_model",
    "get_model",
    # CTR 模型
    "DeepFM",
    "DCN",
    "DCNv2",
    "AutoInt",
    "xDeepFM",
    # 多任务模型
    "ESMM",
    "MMoE",
    "PLE",
    "SharedBottom",
    "DirectCTCVR",
    # Multi-Domain 模型
    "STAR",
    "M2M",
    "EPNet",
    "PPNet",
    "HAMUR",
    "M3oE",
    # 训练/评估
    "Trainer",
    "MultiTaskTrainer",
    "Evaluator",
]
