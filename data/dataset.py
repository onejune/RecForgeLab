# -*- coding: utf-8 -*-
"""
数据集体系
支持：
- 数据集注册机制 @register_dataset("ivr_v16")
- 多种数据格式：parquet 单文件、Spark 分区目录、CSV
- 特征类型自动推断
- 负采样（neg_sample_ratio）
- 特征缓存（处理后的特征缓存到磁盘，下次直接加载）
- 数据集分割（按日期/随机）
- train_date_range 按日期目录过滤
"""

import os
import glob
import hashlib
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

from ..utils.logger import get_logger, set_color


# ============================================================
# 数据集注册表
# ============================================================

DATASET_REGISTRY: Dict[str, type] = {}


def register_dataset(name: str):
    """数据集注册装饰器

    用法::

        @register_dataset("ivr_v16")
        class IVRv16Dataset(DSPDataset):
            ...
    """
    def decorator(cls):
        key = name.lower()
        if key in DATASET_REGISTRY:
            get_logger().warning(f"Dataset '{key}' already registered, overwriting.")
        DATASET_REGISTRY[key] = cls
        cls._registered_name = key
        return cls
    return decorator


def get_dataset_class(name: str) -> type:
    """根据名称获取数据集类"""
    key = name.lower()
    if key in DATASET_REGISTRY:
        return DATASET_REGISTRY[key]
    # 未注册时返回通用 DSPDataset
    return DSPDataset


# ============================================================
# 核心数据集类
# ============================================================

class DSPDataset(TorchDataset):
    """DSP 广告数据集（通用实现）

    支持：
    - 单 parquet/csv 文件
    - Spark 输出的分区目录（part-*.parquet）
    - 按日期组织的多目录（train/YYYY-MM-DD/part-*.parquet）
    - 特征缓存（避免重复预处理）
    - 负采样
    """

    def __init__(
        self,
        config,
        phase: str = "train",
        encoders: Optional[Dict] = None,  # 传入已 fit 的编码器（用于 valid/test）
    ):
        self.config = config
        self.phase = phase
        self.logger = get_logger()

        # 特征配置
        self.sparse_features: List[str] = config.get("sparse_features", [])
        self.dense_features: List[str] = config.get("dense_features", [])
        self.label_field: str = config.get("label_field", "label")
        self.ctr_label_field: str = config.get("ctr_label_field", "click_label")
        self.cvr_label_field: str = config.get("cvr_label_field", "label")

        # 编码器
        self.sparse_vocab: Dict[str, int] = {}
        self.feature_encoders: Dict = {}

        # 尝试从缓存加载
        cache_path = self._get_cache_path()
        if cache_path and cache_path.exists() and encoders is None:
            self.logger.info(set_color(f"[{phase}] Loading from cache: ", "yellow") + str(cache_path))
            self._load_from_cache(cache_path)
        else:
            # 加载原始数据
            df = self._load_data(config, phase)
            self.logger.info(f"[{phase}] loaded {len(df):,} rows")

            # 负采样（仅训练集）
            if phase == "train":
                df = self._neg_sample(df, config)

            # 特征编码
            if encoders is not None:
                self.feature_encoders = encoders
                df = self._apply_encoders(df)
            else:
                df, self.feature_encoders = self._fit_and_transform(df)

            # 构建 tensor
            self._build_tensors(df)

            # 保存缓存
            if cache_path and encoders is None:
                self._save_to_cache(cache_path)

        self.logger.info(
            set_color(f"[{phase}]", "green") +
            f" {len(self):,} samples | "
            f"{len(self.sparse_features)} sparse + {len(self.dense_features)} dense features"
        )

    # ----------------------------------------------------------------
    # 缓存
    # ----------------------------------------------------------------

    def _get_cache_path(self) -> Optional[Path]:
        """生成缓存路径（基于配置 hash）"""
        if not self.config.get("use_cache", False):
            return None

        cache_dir = Path(self.config.get("cache_dir", "/tmp/recforgelab_cache"))
        cache_dir.mkdir(parents=True, exist_ok=True)

        # 用关键配置生成 hash
        key_items = {
            "data_path": str(self.config.get("data_path", "")),
            "dataset": str(self.config.get("dataset", "")),
            "sparse_features": sorted(self.sparse_features),
            "dense_features": sorted(self.dense_features),
            "phase": self.phase,
            "train_date_range": str(self.config.get("train_date_range")),
            "neg_sample_ratio": str(self.config.get("neg_sample_ratio")),
        }
        key_str = str(sorted(key_items.items()))
        key_hash = hashlib.md5(key_str.encode()).hexdigest()[:12]

        dataset_name = self.config.get("dataset", "unknown")
        return cache_dir / f"{dataset_name}_{self.phase}_{key_hash}.pkl"

    def _save_to_cache(self, cache_path: Path):
        """保存到缓存"""
        data = {
            "sparse_tensor": self.sparse_tensor,
            "dense_tensor": self.dense_tensor,
            "label_tensors": {
                k: getattr(self, k)
                for k in ["label_tensor", "ctr_label_tensor", "cvr_label_tensor"]
                if hasattr(self, k)
            },
            "sparse_vocab": self.sparse_vocab,
            "feature_encoders": self.feature_encoders,
        }
        with open(cache_path, "wb") as f:
            pickle.dump(data, f)
        self.logger.info(f"[{self.phase}] Cache saved to {cache_path}")

    def _load_from_cache(self, cache_path: Path):
        """从缓存加载"""
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        self.sparse_tensor = data["sparse_tensor"]
        self.dense_tensor = data["dense_tensor"]
        for k, v in data["label_tensors"].items():
            setattr(self, k, v)
        self.sparse_vocab = data["sparse_vocab"]
        self.feature_encoders = data["feature_encoders"]

    # ----------------------------------------------------------------
    # 数据加载
    # ----------------------------------------------------------------

    def _load_data(self, config, phase: str) -> pd.DataFrame:
        """加载数据，支持多种格式"""
        data_path = Path(config["data_path"])
        data_format = config.get("data_format", "auto")

        if data_format == "spark_dir":
            return self._load_spark_dir(data_path, phase, config)
        elif data_format == "single_file":
            return self._load_single_file(data_path, phase, config)
        else:
            # 自动检测
            phase_dir = data_path / phase
            if phase_dir.exists() and phase_dir.is_dir():
                parts = list(phase_dir.glob("part-*.parquet"))
                if parts:
                    return self._load_parquet_dir(phase_dir)
                date_dirs = [d for d in phase_dir.iterdir() if d.is_dir()]
                if date_dirs:
                    return self._load_spark_dir(data_path, phase, config)
            # 单文件
            return self._load_single_file(data_path, phase, config)

    def _load_parquet_dir(self, dir_path: Path) -> pd.DataFrame:
        """加载目录下所有 part-*.parquet"""
        files = sorted(dir_path.glob("part-*.parquet"))
        if not files:
            files = sorted(dir_path.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files found in {dir_path}")
        dfs = [pd.read_parquet(f) for f in files]
        return pd.concat(dfs, ignore_index=True)

    def _load_spark_dir(self, data_path: Path, phase: str, config) -> pd.DataFrame:
        """加载 Spark 输出的按日期分区目录

        结构：data_path/train/YYYY-MM-DD/ 或 data_path/train/YYYY-MM-DD.parquet/
        """
        phase_dir = data_path / phase

        if phase == "train":
            date_range = config.get("train_date_range")
        else:
            date_range = config.get("test_date_range")

        all_dirs = sorted([d for d in phase_dir.iterdir() if d.is_dir()])

        if date_range:
            start, end = date_range
            all_dirs = [
                d for d in all_dirs
                if start <= d.name.rstrip(".parquet") <= end
            ]

        if not all_dirs:
            raise FileNotFoundError(f"No data found in {phase_dir} with date_range={date_range}")

        self.logger.info(
            f"[{phase}] loading {len(all_dirs)} date dirs: "
            f"{all_dirs[0].name} → {all_dirs[-1].name}"
        )

        dfs = [self._load_parquet_dir(d) for d in all_dirs]
        return pd.concat(dfs, ignore_index=True)

    def _load_single_file(self, data_path: Path, phase: str, config) -> pd.DataFrame:
        """加载单文件"""
        if phase == "train":
            fname = config.get("train_file", "train.parquet")
        elif phase == "valid":
            fname = config.get("valid_file", "valid.parquet")
        else:
            fname = config.get("test_file", "test.parquet")

        fpath = data_path / fname
        if not fpath.exists():
            # 尝试 CSV
            csv_path = fpath.with_suffix(".csv")
            if csv_path.exists():
                return pd.read_csv(csv_path)
            raise FileNotFoundError(f"Data file not found: {fpath}")

        if fpath.suffix == ".parquet":
            return pd.read_parquet(fpath)
        elif fpath.suffix == ".csv":
            return pd.read_csv(fpath)
        else:
            raise ValueError(f"Unsupported file format: {fpath.suffix}")

    # ----------------------------------------------------------------
    # 负采样
    # ----------------------------------------------------------------

    def _neg_sample(self, df: pd.DataFrame, config) -> pd.DataFrame:
        """负采样（仅对训练集）

        neg_sample_ratio: 正样本:负样本 = 1:neg_sample_ratio
        """
        ratio = config.get("neg_sample_ratio")
        if ratio is None or ratio <= 0:
            return df

        label_col = config.get("ctr_label_field", config.get("label_field", "label"))
        if label_col not in df.columns:
            return df

        pos_df = df[df[label_col] == 1]
        neg_df = df[df[label_col] == 0]

        n_neg_target = int(len(pos_df) * ratio)
        if len(neg_df) > n_neg_target:
            neg_df = neg_df.sample(n=n_neg_target, random_state=42)

        result = pd.concat([pos_df, neg_df], ignore_index=True).sample(frac=1, random_state=42)
        self.logger.info(
            f"[train] neg_sample: {len(df):,} -> {len(result):,} "
            f"(pos={len(pos_df):,}, neg={len(neg_df):,})"
        )
        return result

    # ----------------------------------------------------------------
    # 特征编码
    # ----------------------------------------------------------------

    def _infer_feature_types(self, df: pd.DataFrame):
        """自动推断特征类型（如果未配置）"""
        if not self.sparse_features and not self.dense_features:
            for col in df.columns:
                if col in (self.label_field, self.ctr_label_field, self.cvr_label_field):
                    continue
                if df[col].dtype in (object, "category") or df[col].nunique() < 100:
                    self.sparse_features.append(col)
                else:
                    self.dense_features.append(col)
            self.logger.info(
                f"Auto-inferred: {len(self.sparse_features)} sparse, "
                f"{len(self.dense_features)} dense features"
            )

    def _fit_and_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """训练集：fit 编码器并 transform"""
        self._infer_feature_types(df)
        encoders = {}

        for col in self.sparse_features:
            if col not in df.columns:
                continue
            le = LabelEncoder()
            df[col] = df[col].fillna("__NaN__").astype(str)
            df[col] = le.fit_transform(df[col])
            self.sparse_vocab[col] = len(le.classes_)
            encoders[col] = le

        for col in self.dense_features:
            if col not in df.columns:
                continue
            df[col] = df[col].fillna(0).astype(np.float32)

        return df, encoders

    def _apply_encoders(self, df: pd.DataFrame) -> pd.DataFrame:
        """valid/test：使用已 fit 的编码器"""
        for col in self.sparse_features:
            if col not in df.columns:
                continue
            le = self.feature_encoders.get(col)
            if le is None:
                continue
            df[col] = df[col].fillna("__NaN__").astype(str)
            known = set(le.classes_)
            df[col] = df[col].apply(lambda v: v if v in known else "__NaN__")
            df[col] = le.transform(df[col])
            self.sparse_vocab[col] = len(le.classes_)

        for col in self.dense_features:
            if col not in df.columns:
                continue
            df[col] = df[col].fillna(0).astype(np.float32)

        return df

    # ----------------------------------------------------------------
    # Tensor 构建
    # ----------------------------------------------------------------

    def _build_tensors(self, df: pd.DataFrame):
        """构建 tensor 缓存"""
        sparse_data = [
            df[col].values.astype(np.int64)
            for col in self.sparse_features
            if col in df.columns
        ]
        dense_data = [
            df[col].values.astype(np.float32)
            for col in self.dense_features
            if col in df.columns
        ]

        self.sparse_tensor = (
            torch.from_numpy(np.column_stack(sparse_data)) if sparse_data else None
        )
        self.dense_tensor = (
            torch.from_numpy(np.column_stack(dense_data)) if dense_data else None
        )

        # 标签
        if self.ctr_label_field in df.columns:
            self.ctr_label_tensor = torch.from_numpy(
                df[self.ctr_label_field].values.astype(np.float32)
            )
        if self.cvr_label_field in df.columns:
            self.cvr_label_tensor = torch.from_numpy(
                df[self.cvr_label_field].values.astype(np.float32)
            )
        if self.label_field in df.columns:
            self.label_tensor = torch.from_numpy(
                df[self.label_field].values.astype(np.float32)
            )

    # ----------------------------------------------------------------
    # Dataset 接口
    # ----------------------------------------------------------------

    def __len__(self) -> int:
        if self.sparse_tensor is not None:
            return len(self.sparse_tensor)
        if self.dense_tensor is not None:
            return len(self.dense_tensor)
        if hasattr(self, "label_tensor"):
            return len(self.label_tensor)
        return 0

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        item = {}

        if self.sparse_tensor is not None:
            for i, feat in enumerate(self.sparse_features):
                item[feat] = self.sparse_tensor[idx, i]

        if self.dense_tensor is not None:
            item["dense"] = self.dense_tensor[idx]

        if hasattr(self, "ctr_label_tensor"):
            item[self.ctr_label_field] = self.ctr_label_tensor[idx]
        if hasattr(self, "cvr_label_tensor") and self.cvr_label_field != self.ctr_label_field:
            item[self.cvr_label_field] = self.cvr_label_tensor[idx]
        if hasattr(self, "label_tensor"):
            item[self.label_field] = self.label_tensor[idx]

        return item


# ============================================================
# 工厂函数
# ============================================================

def create_dataset(config, phase: str = "train", encoders=None) -> DSPDataset:
    """创建数据集

    根据 config["dataset"] 查找注册的数据集类，未注册则使用 DSPDataset。
    """
    dataset_name = config.get("dataset", "")
    cls = get_dataset_class(dataset_name)
    return cls(config, phase=phase, encoders=encoders)


def create_dataloader(dataset: DSPDataset, config, shuffle: bool = True):
    """创建 DataLoader"""
    batch_size = config["train_batch_size"] if shuffle else config.get("eval_batch_size", config["train_batch_size"])
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.get("num_workers", 4),
        pin_memory=config.get("pin_memory", True),
    )
