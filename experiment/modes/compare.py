# -*- coding: utf-8 -*-
"""
多模型对比实验
"""

from typing import Dict, Any, List, Tuple
from copy import deepcopy

from .base import BaseExperiment
from .single import SingleExperiment
from ...utils.logger import get_logger, set_color


class CompareExperiment(BaseExperiment):
    """多模型对比实验
    
    对比多个模型/配置在同一数据集上的表现。
    
    用法:
        # 通过配置文件
        # config.yaml:
        #   experiments:
        #     - name: deepfm
        #       config: {model: deepfm}
        #     - name: star
        #       config: {model: star}
        
        experiment = CompareExperiment(config)
        results = experiment.run()
    """
    
    MODE_NAME = "compare"
    
    def __init__(self, config):
        super().__init__(config)
        self.experiment_configs = []
        self.all_results = {}
    
    def setup(self):
        """解析实验配置"""
        super().setup()
        self.experiment_configs = self.config.get_experiment_configs()
        self.logger.info(set_color(f"Comparing {len(self.experiment_configs)} experiments", "yellow"))
    
    def run(self) -> Dict[str, Dict[str, Any]]:
        """运行所有对比实验"""
        self.setup()
        
        for exp_name, exp_config in self.experiment_configs:
            self.logger.info(set_color(f"\n[Experiment: {exp_name}]", "cyan"))
            
            try:
                # 复用 SingleExperiment
                single_exp = SingleExperiment(exp_config)
                results = single_exp.run()
                self.all_results[exp_name] = results
            except Exception as e:
                self.logger.error(f"Experiment {exp_name} failed: {e}")
                self.all_results[exp_name] = {"error": str(e)}
        
        self._print_summary()
        return self.all_results
    
    def _print_summary(self):
        """打印对比汇总"""
        self.logger.info(set_color("\n[Comparison Summary]", "green"))
        
        valid_metric = self.config.get("valid_metric", "AUC").lower()
        
        sorted_items = sorted(
            [(k, v) for k, v in self.all_results.items() if "error" not in v],
            key=lambda x: x[1].get(valid_metric, 0),
            reverse=True,
        )
        
        for name, results in sorted_items:
            score = results.get(valid_metric, 0)
            self.logger.info(f"  {name:<40} {valid_metric}={score:.6f}")
