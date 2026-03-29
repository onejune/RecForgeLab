# -*- coding: utf-8 -*-
"""
实验模块

提供可复用的实验运行框架：
- ExperimentRunner: 统一入口，策略模式选择实验模式
- BaseExperiment: 实验基类，可继承定制
- 预置模式: Single/Compare/GridSearch/SSL
"""

from .runner import ExperimentRunner
from .modes import (
    BaseExperiment,
    SingleExperiment,
    CompareExperiment,
    GridSearchExperiment,
    SSLExperiment,
)

__all__ = [
    "ExperimentRunner",
    "BaseExperiment",
    "SingleExperiment",
    "CompareExperiment",
    "GridSearchExperiment",
    "SSLExperiment",
]
