# -*- coding: utf-8 -*-
"""
LTV (Life Time Value) 预估模型

包含多种 LTV 预估方法：
- ZILN: Zero-Inflated LogNormal
- TwoStage: 两阶段模型（分类 + 回归）
- Tweedie: Tweedie Loss
- Ordinal: Ordinal Regression
- MDN: Mixture Density Network
- DeepAR: 自回归概率预测
"""

from .base import LTVModel, register_ltv_model
from .ziln import ZILN
from .two_stage import TwoStageLTV
from .tweedie import TweedieLTV
from .ordinal import OrdinalLTV
from .mdn import MDNLTV
from .direct_regression import (
    DirectRegressionLTV, MSELTV, MAELTV, HuberLTV,
    LogCoshLTV, QuantileLTV, LogMSELTV,
)

__all__ = [
    "LTVModel",
    "register_ltv_model",
    # 概率模型
    "ZILN",
    "TwoStageLTV",
    "TweedieLTV",
    "OrdinalLTV",
    "MDNLTV",
    # 直接回归
    "DirectRegressionLTV",
    "MSELTV",
    "MAELTV",
    "HuberLTV",
    "LogCoshLTV",
    "QuantileLTV",
    "LogMSELTV",
]
