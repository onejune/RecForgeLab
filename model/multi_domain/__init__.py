# -*- coding: utf-8 -*-
"""
Multi-Domain 模型

多场景/多域推荐模型，支持在一个模型中同时学习多个业务场景（domain）。
核心特点：
- 输入包含 domain_indicator 字段，标识样本所属域
- 模型结构包含共享部分 + 域特定部分
- 适合多业务场景联合训练，共享知识同时保留域差异

已实现模型：
- SharedBottom: 共享底层，各域独立 Tower
- MMOE: 多专家多门控（多域版本）
- PLE: 渐进式分层提取（多域版本）
- STAR: 星形拓扑，共享中心 + 域特定分区
- M2M: Meta-learning for Multi-domain
- EPNet: Embedding Personalization Network
- PPNet: Personalized Parallel Network
- HAMUR: Hierarchical Adaptive Multi-Domain
- M3oE: Multi-Domain Multi-Task Mixture of Experts
- AdaSparse: 自适应稀疏化
- AdaptDHM: 自适应域感知混合
- SARNet: Scenario-Aware Routing Network
"""

from .star import STAR
from .m2m import M2M
from .epnet import EPNet
from .ppnet import PPNet
from .hamur import HAMUR
from .m3oe import M3oE

__all__ = [
    "STAR",
    "M2M",
    "EPNet",
    "PPNet",
    "HAMUR",
    "M3oE",
]
