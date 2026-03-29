# -*- coding: utf-8 -*-
"""
模型层统一导出

包含：
- 特征嵌入层（FeatureEmbedding）
- 连续特征编码器（16 种）
- MLP 层（MLPLayers）
- FM 层（FM, FMLayer）
- 交叉网络（CrossNetwork, CrossNetworkV2）
- 注意力层（MultiHeadAttention）
"""

from .embedding import (
    FeatureEmbedding,
    BaseEncoder,
    ScalarEncoder,
    BucketEncoder,
    AutoDisEncoder,
    NumericEmbeddingEncoder,
    NumericEmbeddingDeepEncoder,
    NumericEmbeddingSiLUEncoder,
    NumericEmbeddingLNEncoder,
    NumericEmbeddingContextualEncoder,
    FTTransformerEncoder,
    PeriodicEncoder,
    FieldEmbeddingEncoder,
    DLRMEncoder,
    PLREncoder,
    MinMaxEncoder,
    LogTransformEncoder,
    NoneEncoder,
    build_encoder,
    ENCODER_REGISTRY,
)

from .mlp import (
    MLPLayers,
    FM,
    CrossNetwork,
    CrossNetworkV2,
    MultiHeadAttention,
)

# fm.py 中的 FMLayer（如果存在）
try:
    from .fm import FMLayer
    _has_fm_layer = True
except ImportError:
    _has_fm_layer = False
    FMLayer = None

__all__ = [
    # 特征嵌入
    "FeatureEmbedding",
    # 编码器基类
    "BaseEncoder",
    # 16 种连续特征编码器
    "ScalarEncoder",
    "BucketEncoder",
    "AutoDisEncoder",
    "NumericEmbeddingEncoder",
    "NumericEmbeddingDeepEncoder",
    "NumericEmbeddingSiLUEncoder",
    "NumericEmbeddingLNEncoder",
    "NumericEmbeddingContextualEncoder",
    "FTTransformerEncoder",
    "PeriodicEncoder",
    "FieldEmbeddingEncoder",
    "DLRMEncoder",
    "PLREncoder",
    "MinMaxEncoder",
    "LogTransformEncoder",
    "NoneEncoder",
    # 编码器工厂
    "build_encoder",
    "ENCODER_REGISTRY",
    # MLP 及相关层
    "MLPLayers",
    "FM",
    "CrossNetwork",
    "CrossNetworkV2",
    "MultiHeadAttention",
    # FM 层（来自 fm.py）
    "FMLayer",
]
