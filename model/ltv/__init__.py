# -*- coding: utf-8 -*-
"""
LTV (Life Time Value) 预估模型

包含多种 LTV 预估方法：

**处理零膨胀的方法**（LTV ≥ 0）：
- ZILN: Zero-Inflated LogNormal
- TwoStage: 两阶段模型
- Tweedie: Tweedie Loss
- Ordinal: Ordinal Regression
- MDN: Mixture Density Network

**直接回归**（LTV ≥ 0）：
- MSE, MAE, Huber, Log-Cosh, Quantile, Log-MSE

**付费用户建模**（LTV > 0）：
- LogNormal: Log-Normal 回归
- Gamma: Gamma 回归
- ShiftedLogNormal: Shifted Log-Normal
- BoxCox: Box-Cox 变换回归
- LogRegression: 简单 log 回归
"""

from .base import LTVModel, register_ltv_model

# 概率模型（零膨胀）
from .ziln import ZILN
from .two_stage import TwoStageLTV
from .tweedie import TweedieLTV
from .ordinal import OrdinalLTV
from .mdn import MDNLTV

# 直接回归
from .direct_regression import (
    DirectRegressionLTV, MSELTV, MAELTV, HuberLTV,
    LogCoshLTV, QuantileLTV, LogMSELTV,
)

# 付费用户建模
from .paid_only import (
    LogNormalLTV, GammaLTV, ShiftedLogNormalLTV, BoxCoxLTV, LogRegressionLTV,
)

__all__ = [
    "LTVModel",
    "register_ltv_model",
    # 零膨胀模型
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
    # 付费用户建模
    "LogNormalLTV",
    "GammaLTV",
    "ShiftedLogNormalLTV",
    "BoxCoxLTV",
    "LogRegressionLTV",
]
