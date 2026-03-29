# -*- coding: utf-8 -*-
"""
单次实验
"""

from typing import Dict, Any

from .base import BaseExperiment


class SingleExperiment(BaseExperiment):
    """单次实验
    
    标准流程: setup → load_data → build_model → train → evaluate → save_results
    
    用法:
        experiment = SingleExperiment(config)
        results = experiment.run()
        
        # 自定义
        class MyExperiment(SingleExperiment):
            def evaluate(self):
                results = super().evaluate()
                # 添加自定义指标
                results['custom'] = ...
                return results
    """
    
    MODE_NAME = "single"
    
    def run(self) -> Dict[str, Any]:
        """运行单次实验"""
        self.setup()
        self.load_data()
        self.build_model()
        self.train()
        results = self.evaluate()
        self.save_results(results)
        return results
