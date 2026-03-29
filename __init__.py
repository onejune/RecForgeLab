# -*- coding: utf-8 -*-
"""
RecForgeLab: DSP CTR/CVR Prediction Framework
"""

from .utils import Config, get_logger
from .data import DSPDataset, DATASET_REGISTRY, register_dataset
from .model import (
    BaseModel, CTRModel, MultiTaskModel, SSLModel,
    MODEL_REGISTRY, register_model, get_model,
    DeepFM, DCN, DCNv2, AutoInt, xDeepFM,
    ESMM, MMoE, PLE, SharedBottom, DirectCTCVR,
)
from .trainer import Trainer, MultiTaskTrainer
from .evaluator import Evaluator


__all__ = [
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
    # 训练/评估
    "Trainer",
    "MultiTaskTrainer",
    "Evaluator",
]
