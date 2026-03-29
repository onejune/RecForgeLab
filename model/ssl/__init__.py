# -*- coding: utf-8 -*-
"""
SSL 模块
"""

from .contrastive import (
    SSLContrastive,
    SSLMomentumContrastive,
    SSLUserBehaviorContrastive,
    InfoNCELoss,
    SupConLoss,
)

__all__ = [
    "SSLContrastive",
    "SSLMomentumContrastive",
    "SSLUserBehaviorContrastive",
    "InfoNCELoss",
    "SupConLoss",
]
