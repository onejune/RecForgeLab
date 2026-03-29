# -*- coding: utf-8 -*-
"""
配置管理模块（完整版）
支持：
- 配置继承（base_config: path/to/base.yaml）
- 参数网格搜索定义（grid_search: {lr: [0.001, 0.0001], ...}）
- 实验组定义（experiments: [{name: ..., config: {...}}, ...]）
- 配置校验（必填字段检查）
- 配置打印（格式化输出）
- 命令行覆盖（--key=value）

优先级（从高到低）：
  命令行参数 > config_dict > 配置文件 > base_config > 默认值
"""

import os
import sys
import yaml
import argparse
from typing import Dict, List, Optional, Any, Iterator, Tuple
from pathlib import Path
from copy import deepcopy
from itertools import product


# ============================================================
# 默认配置
# ============================================================

DEFAULT_CONFIG: Dict[str, Any] = {
    # 模型相关
    "model": None,
    "embedding_size": 16,
    "mlp_hidden_size": [256, 128, 64],
    "dropout_prob": 0.2,
    "encoder_type": "bucket",
    "encoder_config": {},

    # 多任务相关
    "tasks": ["ctr", "cvr"],
    "task_weights": [1.0, 1.0],

    # 训练相关
    "learning_rate": 0.001,
    "train_batch_size": 4096,
    "eval_batch_size": 8192,
    "epochs": 10,
    "early_stop_patience": 3,
    "optimizer": "adam",
    "weight_decay": 0.0,
    "grad_clip": None,
    "scheduler": None,
    "scheduler_config": {},
    "use_amp": False,

    # 数据相关
    "dataset": None,
    "data_path": None,
    "data_format": "auto",
    "label_field": "label",
    "ctr_label_field": "click_label",
    "cvr_label_field": "label",
    "sparse_features": [],
    "dense_features": [],
    "neg_sample_ratio": None,
    "use_cache": False,
    "cache_dir": "/tmp/recforgelab_cache",
    "num_workers": 4,
    "pin_memory": True,

    # 评估相关
    "metrics": ["AUC", "LogLoss"],
    "valid_metric": "AUC",
    "eval_step": 1,

    # 系统相关
    "device": "cuda",
    "seed": 2024,
    "checkpoint_dir": "./saved",
    "log_dir": "./logs",
    "experiment_name": None,

    # 实验管理
    "save_model": True,
    "load_model": None,
    "use_tensorboard": False,
    "wandb": False,
    "wandb_project": "recforgelab",
}

# 必填字段（至少在运行前要有值）
REQUIRED_FIELDS: List[str] = []  # 可根据需要添加，如 ["model", "dataset"]


# ============================================================
# Config 类
# ============================================================

class Config:
    """配置管理器

    支持四种配置来源，优先级从高到低：
    1. 命令行参数
    2. 参数字典 (config_dict)
    3. 配置文件列表 (config_file_list)
    4. 默认值

    特殊字段：
    - base_config: 继承的基础配置文件路径
    - grid_search: 网格搜索参数定义
    - experiments: 多组实验定义
    """

    def __init__(
        self,
        model: Optional[str] = None,
        dataset: Optional[str] = None,
        config_file_list: Optional[List[str]] = None,
        config_dict: Optional[Dict[str, Any]] = None,
        parse_cmd: bool = True,
    ):
        # 1. 加载默认配置
        self._config: Dict[str, Any] = deepcopy(DEFAULT_CONFIG)

        # 2. 加载配置文件（支持继承）
        if config_file_list:
            for file_path in config_file_list:
                self._load_yaml_with_inheritance(file_path)

        # 3. 加载参数字典
        if config_dict:
            self._config.update(config_dict)

        # 4. 加载命令行参数
        if parse_cmd:
            cmd_config = self._parse_cmd_line()
            if cmd_config:
                self._config.update(cmd_config)

        # 5. 显式参数覆盖
        if model:
            self._config["model"] = model
        if dataset:
            self._config["dataset"] = dataset

        # 6. 加载模型特定默认配置（优先级低于用户配置）
        self._load_model_default_config()

        # 7. 加载数据集特定默认配置
        self._load_dataset_default_config()

        # 8. 生成实验名称
        if self._config.get("experiment_name") is None:
            model_name = self._config.get("model", "unknown")
            dataset_name = self._config.get("dataset", "unknown")
            self._config["experiment_name"] = f"{model_name}_{dataset_name}"

        # 9. 设置随机种子
        self._set_seed()

    # ----------------------------------------------------------------
    # YAML 加载（支持继承）
    # ----------------------------------------------------------------

    def _load_yaml_with_inheritance(self, file_path: str):
        """加载 YAML，支持 base_config 继承"""
        file_path = str(file_path)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Config file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            yaml_config = yaml.safe_load(f) or {}

        # 先加载 base_config
        base_config_path = yaml_config.pop("base_config", None)
        if base_config_path:
            # 相对路径解析
            base_path = Path(file_path).parent / base_config_path
            self._load_yaml_with_inheritance(str(base_path))

        # 再用当前文件覆盖（当前文件优先级更高）
        # 过滤掉框架保留字段
        reserved = {"grid_search", "experiments"}
        for key, value in yaml_config.items():
            if key not in reserved:
                self._config[key] = value
            else:
                self._config[key] = value  # 也保存，供后续使用

    def _load_yaml(self, file_path: str):
        """简单加载 YAML（不处理继承）"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Config file not found: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            yaml_config = yaml.safe_load(f)
            if yaml_config:
                self._config.update(yaml_config)

    # ----------------------------------------------------------------
    # 命令行解析
    # ----------------------------------------------------------------

    def _parse_cmd_line(self) -> Dict[str, Any]:
        """解析命令行参数（宽松模式，不报错未知参数）"""
        # 如果是 pytest 或 ipython 环境，跳过命令行解析
        if "pytest" in sys.modules or "IPython" in sys.modules:
            return {}

        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--model", type=str, default=None)
        parser.add_argument("--dataset", type=str, default=None)
        parser.add_argument("--config", type=str, default=None)
        parser.add_argument("--mode", type=str, default=None)
        parser.add_argument("--learning_rate", "--lr", type=float, default=None, dest="learning_rate")
        parser.add_argument("--epochs", type=int, default=None)
        parser.add_argument("--batch_size", type=int, default=None)
        parser.add_argument("--device", type=str, default=None)
        parser.add_argument("--seed", type=int, default=None)
        parser.add_argument("--experiment_name", type=str, default=None)

        try:
            args, unknown = parser.parse_known_args()
        except SystemExit:
            return {}

        config: Dict[str, Any] = {}
        for key, value in vars(args).items():
            if value is not None:
                if key == "batch_size":
                    config["train_batch_size"] = value
                elif key == "config":
                    self._load_yaml_with_inheritance(value)
                else:
                    config[key] = value

        # 解析 --key=value 或 --key value 格式的未知参数
        i = 0
        while i < len(unknown):
            arg = unknown[i]
            if arg.startswith("--"):
                if "=" in arg:
                    key, val = arg[2:].split("=", 1)
                    config[key] = self._cast_value(val)
                elif i + 1 < len(unknown) and not unknown[i + 1].startswith("--"):
                    key = arg[2:]
                    config[key] = self._cast_value(unknown[i + 1])
                    i += 1
            i += 1

        return config

    @staticmethod
    def _cast_value(value: str) -> Any:
        """尝试将字符串转换为合适的类型"""
        # bool
        if value.lower() in ("true", "yes"):
            return True
        if value.lower() in ("false", "no"):
            return False
        # int
        try:
            return int(value)
        except ValueError:
            pass
        # float
        try:
            return float(value)
        except ValueError:
            pass
        # list (简单支持 [a,b,c])
        if value.startswith("[") and value.endswith("]"):
            try:
                import ast
                return ast.literal_eval(value)
            except Exception:
                pass
        return value

    # ----------------------------------------------------------------
    # 模型/数据集默认配置
    # ----------------------------------------------------------------

    def _load_model_default_config(self):
        """加载模型特定的默认配置（优先级低于用户配置）"""
        model = self._config.get("model")
        if not model:
            return
        config_dir = Path(__file__).parent.parent / "config" / "model"
        model_config_path = config_dir / f"{model.lower()}.yaml"
        if model_config_path.exists():
            model_defaults = yaml.safe_load(model_config_path.read_text()) or {}
            for key, value in model_defaults.items():
                # 只填充用户未设置的字段（等于默认值的字段）
                if self._config.get(key) == DEFAULT_CONFIG.get(key):
                    self._config[key] = value

    def _load_dataset_default_config(self):
        """加载数据集特定的默认配置"""
        dataset = self._config.get("dataset")
        if not dataset:
            return
        config_dir = Path(__file__).parent.parent / "config" / "dataset"
        dataset_config_path = config_dir / f"{dataset.lower()}.yaml"
        if dataset_config_path.exists():
            dataset_defaults = yaml.safe_load(dataset_config_path.read_text()) or {}
            for key, value in dataset_defaults.items():
                if self._config.get(key) == DEFAULT_CONFIG.get(key):
                    self._config[key] = value

    # ----------------------------------------------------------------
    # 随机种子
    # ----------------------------------------------------------------

    def _set_seed(self):
        """设置随机种子"""
        import random
        import numpy as np
        import torch

        seed = self._config.get("seed", 2024)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # ----------------------------------------------------------------
    # 配置校验
    # ----------------------------------------------------------------

    def validate(self, required_fields: Optional[List[str]] = None):
        """校验必填字段

        Args:
            required_fields: 必填字段列表，None 则使用 REQUIRED_FIELDS

        Raises:
            ValueError: 缺少必填字段
        """
        fields = required_fields or REQUIRED_FIELDS
        missing = [f for f in fields if not self._config.get(f)]
        if missing:
            raise ValueError(f"Missing required config fields: {missing}")

    # ----------------------------------------------------------------
    # 网格搜索
    # ----------------------------------------------------------------

    def get_grid_search_configs(self) -> List["Config"]:
        """生成网格搜索的所有配置组合

        配置文件格式::

            grid_search:
              learning_rate: [0.001, 0.0001]
              batch_size: [2048, 4096]
              dropout_prob: [0.1, 0.2]

        Returns:
            所有参数组合的 Config 列表
        """
        grid = self._config.get("grid_search", {})
        if not grid:
            return [self]

        keys = list(grid.keys())
        values = list(grid.values())

        configs = []
        for combo in product(*values):
            new_config = deepcopy(self)
            for k, v in zip(keys, combo):
                new_config[k] = v
            # 更新实验名
            suffix = "_".join(f"{k}={v}" for k, v in zip(keys, combo))
            base_name = self._config.get("experiment_name", "exp")
            new_config["experiment_name"] = f"{base_name}_{suffix}"
            configs.append(new_config)

        return configs

    # ----------------------------------------------------------------
    # 多实验组
    # ----------------------------------------------------------------

    def get_experiment_configs(self) -> List[Tuple[str, "Config"]]:
        """获取多组实验配置

        配置文件格式::

            experiments:
              - name: deepfm_baseline
                config:
                  model: deepfm
                  learning_rate: 0.001
              - name: dcn_baseline
                config:
                  model: dcn
                  learning_rate: 0.001

        Returns:
            [(exp_name, Config), ...]
        """
        experiments = self._config.get("experiments", [])
        if not experiments:
            return [(self._config.get("experiment_name", "default"), self)]

        result = []
        for exp in experiments:
            new_config = deepcopy(self)
            exp_name = exp.get("name", "unnamed")
            exp_cfg = exp.get("config", {})
            new_config._config.update(exp_cfg)
            new_config["experiment_name"] = exp_name
            result.append((exp_name, new_config))

        return result

    # ----------------------------------------------------------------
    # 配置打印
    # ----------------------------------------------------------------

    def print_config(self, title: str = "Configuration"):
        """格式化打印配置"""
        from ..utils.logger import get_logger, set_color
        logger = get_logger()

        logger.info(set_color("=" * 60, "cyan"))
        logger.info(set_color(f"  {title}", "cyan"))
        logger.info(set_color("=" * 60, "cyan"))

        # 分组打印
        groups = {
            "Model": ["model", "embedding_size", "mlp_hidden_size", "dropout_prob",
                      "encoder_type", "encoder_config"],
            "Training": ["learning_rate", "train_batch_size", "eval_batch_size",
                         "epochs", "early_stop_patience", "optimizer", "weight_decay",
                         "grad_clip", "scheduler", "use_amp"],
            "Data": ["dataset", "data_path", "data_format", "sparse_features",
                     "dense_features", "label_field", "neg_sample_ratio", "use_cache"],
            "Evaluation": ["metrics", "valid_metric", "eval_step"],
            "System": ["device", "seed", "checkpoint_dir", "log_dir", "experiment_name"],
        }

        printed_keys = set()
        for group_name, keys in groups.items():
            logger.info(set_color(f"\n  [{group_name}]", "yellow"))
            for key in keys:
                if key in self._config:
                    value = self._config[key]
                    logger.info(f"    {key:<30} = {value}")
                    printed_keys.add(key)

        # 打印未分组的用户自定义字段
        extra = {k: v for k, v in self._config.items()
                 if k not in printed_keys
                 and k not in ("grid_search", "experiments")}
        if extra:
            logger.info(set_color("\n  [Other]", "yellow"))
            for key, value in extra.items():
                logger.info(f"    {key:<30} = {value}")

        logger.info(set_color("=" * 60, "cyan"))

    # ----------------------------------------------------------------
    # 标准接口
    # ----------------------------------------------------------------

    def __getitem__(self, key: str) -> Any:
        return self._config[key]

    def __setitem__(self, key: str, value: Any):
        self._config[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._config

    def get(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)

    def update(self, config_dict: Dict[str, Any]):
        self._config.update(config_dict)

    def to_dict(self) -> Dict[str, Any]:
        return deepcopy(self._config)

    def save(self, path: str):
        """保存配置到 YAML 文件"""
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self._config, f, default_flow_style=False, allow_unicode=True)

    def __repr__(self) -> str:
        model = self._config.get("model", "?")
        dataset = self._config.get("dataset", "?")
        return f"Config(model={model}, dataset={dataset})"

    def __deepcopy__(self, memo):
        new = Config.__new__(Config)
        new._config = deepcopy(self._config)
        return new
