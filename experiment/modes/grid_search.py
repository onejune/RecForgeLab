# -*- coding: utf-8 -*-
"""
网格搜索实验
"""

from typing import Dict, Any, List

from .base import BaseExperiment
from .single import SingleExperiment
from ...utils.logger import get_logger, set_color


class GridSearchExperiment(BaseExperiment):
    """网格搜索实验
    
    遍历参数组合，找出最优配置。
    
    用法:
        # config.yaml:
        #   grid_search:
        #     learning_rate: [0.001, 0.0001]
        #     embedding_size: [16, 32]
        
        experiment = GridSearchExperiment(config)
        results = experiment.run()
    """
    
    MODE_NAME = "grid_search"
    
    def __init__(self, config):
        super().__init__(config)
        self.grid_configs = []
        self.all_results = []
    
    def setup(self):
        """解析网格配置"""
        super().setup()
        self.grid_configs = list(self.config.get_grid_search_configs())
        self.logger.info(set_color(f"Grid search: {len(self.grid_configs)} combinations", "yellow"))
    
    def run(self) -> List[Dict[str, Any]]:
        """运行所有网格实验"""
        self.setup()
        
        for i, grid_config in enumerate(self.grid_configs):
            exp_name = grid_config.get("experiment_name", f"grid_{i}")
            self.logger.info(set_color(f"\n[Grid {i+1}/{len(self.grid_configs)}] {exp_name}", "cyan"))
            
            try:
                # 复用 SingleExperiment
                single_exp = SingleExperiment(grid_config)
                results = single_exp.run()
                
                self.all_results.append({
                    "experiment_name": exp_name,
                    "config": grid_config.to_dict(),
                    "results": results,
                })
            except Exception as e:
                self.logger.error(f"Grid search failed for {exp_name}: {e}")
                self.all_results.append({
                    "experiment_name": exp_name,
                    "config": grid_config.to_dict(),
                    "error": str(e),
                })
        
        self._print_summary()
        return self.all_results
    
    def _print_summary(self):
        """打印网格搜索汇总"""
        self.logger.info(set_color("\n[Grid Search Summary]", "green"))
        
        valid_metric = self.config.get("valid_metric", "AUC").lower()
        
        sorted_results = sorted(
            [r for r in self.all_results if "results" in r],
            key=lambda x: x["results"].get(valid_metric, 0),
            reverse=True,
        )
        
        for r in sorted_results:
            score = r["results"].get(valid_metric, 0)
            self.logger.info(f"  {r['experiment_name']:<50} {valid_metric}={score:.6f}")
