# -*- coding: utf-8 -*-
"""
实验运行器

统一入口，策略模式选择实验模式。
"""

from typing import Dict, Any, Optional
from pathlib import Path

from ..utils.config import Config
from ..utils.logger import get_logger, set_color, init_logger
from ..model import MODEL_REGISTRY
from .modes import (
    BaseExperiment,
    SingleExperiment,
    CompareExperiment,
    GridSearchExperiment,
    SSLExperiment,
)


# 策略映射
MODE_MAP = {
    "single": SingleExperiment,
    "compare": CompareExperiment,
    "grid_search": GridSearchExperiment,
    "ssl": SSLExperiment,
}


class ExperimentRunner:
    """实验运行器
    
    用法:
        runner = ExperimentRunner(config_file="config.yaml", mode="compare")
        results = runner.run()
        
        # 或直接传入 Config
        runner = ExperimentRunner(config=my_config)
        results = runner.run()
    """
    
    def __init__(
        self,
        config_file: Optional[str] = None,
        config: Optional[Config] = None,
        mode: str = "single",
        model: Optional[str] = None,
        dataset: Optional[str] = None,
    ):
        """
        Args:
            config_file: 配置文件路径
            config: 配置对象（优先于 config_file）
            mode: 实验模式 (single/compare/grid_search/ssl)
            model: 模型名称
            dataset: 数据集名称
        """
        # 构建配置
        if config is not None:
            self.config = config
        else:
            config_file_list = [config_file] if config_file else None
            self.config = Config(
                model=model,
                dataset=dataset,
                config_file_list=config_file_list,
                parse_cmd=False,
            )
        
        # 确定模式
        self.mode = mode or self.config.get("mode", "single")
        
        # 初始化日志
        init_logger(self.config)
        self.logger = get_logger()
    
    def run(self) -> Dict[str, Any]:
        """运行实验"""
        if self.mode not in MODE_MAP:
            raise ValueError(f"Unknown mode: {self.mode}. Available: {list(MODE_MAP.keys())}")
        
        experiment_class = MODE_MAP[self.mode]
        experiment = experiment_class(self.config)
        
        return experiment.run()


def list_models():
    """列出所有注册模型"""
    print("Registered models:")
    print("-" * 60)
    
    # 按类型分组
    groups = {
        "CTR": [],
        "Multitask": [],
        "Multi-Domain": [],
        "SSL": [],
        "Other": [],
    }
    
    for name in sorted(MODEL_REGISTRY.keys()):
        cls = MODEL_REGISTRY[name]
        model_type = getattr(cls, 'model_type', None)
        type_name = model_type.value if model_type else 'other'
        
        if type_name == 'ctr':
            groups["CTR"].append((name, cls))
        elif type_name == 'multitask':
            groups["Multitask"].append((name, cls))
        elif type_name == 'multi_domain':
            groups["Multi-Domain"].append((name, cls))
        elif type_name == 'ssl':
            groups["SSL"].append((name, cls))
        else:
            groups["Other"].append((name, cls))
    
    for group_name, models in groups.items():
        if models:
            print(f"\n[{group_name} Models]")
            for name, cls in models:
                print(f"  {name:<20} -> {cls.__name__}")
    
    print("-" * 60)
    print(f"Total: {len(MODEL_REGISTRY)} models")
