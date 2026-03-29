# -*- coding: utf-8 -*-
"""
RecForgeLab 模型模块
"""

from .base import (
    BaseModel, CTRModel, MultiTaskModel, MultiDomainModel, SSLModel, LTVModel,
    MODEL_REGISTRY, register_model, get_model,
)

# 导入所有模型，触发注册
from .ctr import DeepFM, DCN, DCNv2, AutoInt, xDeepFM
from .multitask import ESMM, MMoE, PLE, SharedBottom, DirectCTCVR
from .multi_domain import STAR, M2M, EPNet, PPNet, HAMUR, M3oE
from .ltv import ZILN, TwoStageLTV, TweedieLTV, OrdinalLTV, MDNLTV

__all__ = [
    # 基类
    "BaseModel",
    "CTRModel",
    "MultiTaskModel",
    "MultiDomainModel",
    "SSLModel",
    "LTVModel",
    # 注册机制
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
    # LTV 模型
    "ZILN",
    "TwoStageLTV",
    "TweedieLTV",
    "OrdinalLTV",
    "MDNLTV",
]
