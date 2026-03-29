# -*- coding: utf-8 -*-
"""
枚举类型定义
借鉴 RecBole 的 enum_type 设计
"""

from enum import Enum


class ModelType(Enum):
    """模型类型"""
    CTR = "ctr"               # CTR 单任务
    CVR = "cvr"               # CVR 单任务
    MULTITASK = "multitask"   # 多任务（CTR+CVR）
    MULTIDOMAIN = "multi_domain"  # 多域/多场景
    SEQUENTIAL = "sequential" # 序列模型
    SSL = "ssl"               # 自监督学习
    LTV = "ltv"               # LTV 预估


class InputType(Enum):
    """输入类型"""
    POINTWISE = "pointwise"   # 点式（CTR/CVR）
    PAIRWISE = "pairwise"     # 对式（排序）


class FeatureType(Enum):
    """特征类型"""
    TOKEN = "token"           # 类别特征（离散）
    FLOAT = "float"           # 数值特征（连续）
    TOKEN_SEQ = "token_seq"   # 类别序列特征
    FLOAT_SEQ = "float_seq"   # 数值序列特征


class FeatureSource(Enum):
    """特征来源"""
    INTERACTION = "interaction"  # 交互特征
    USER = "user"                # 用户特征
    ITEM = "item"                # 物品/广告特征
    CONTEXT = "context"          # 上下文特征


class EncoderType(Enum):
    """连续特征编码器类型"""
    SCALAR = "scalar"                # 直接归一化
    BUCKET = "bucket"                # 分桶
    NUMERIC_EMBEDDING = "numeric"    # NumericEmbedding
    AUTODIS = "autodis"              # AutoDis
    FTTRANSFORMER = "fttransformer"  # FTTransformer
    PERIODIC = "periodic"            # PeriodicEncoder


class TaskType(Enum):
    """任务类型（多任务场景）"""
    CTR = "ctr"      # 点击率
    CVR = "cvr"      # 转化率
    CTCVR = "ctcvr"  # 点击转化率


class LossType(Enum):
    """损失函数类型"""
    BCE = "bce"           # Binary Cross Entropy
    BPR = "bpr"           # Bayesian Personalized Ranking
    CROSS_ENTROPY = "ce"  # Cross Entropy
