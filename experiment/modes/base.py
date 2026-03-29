# -*- coding: utf-8 -*-
"""
实验基类

提供标准实验流程，子类可覆盖任意步骤实现定制。

流程:
    1. setup()        - 初始化日志、设备等
    2. load_data()    - 加载数据集
    3. build_model()  - 构建模型
    4. train()        - 训练模型
    5. evaluate()     - 评估模型
    6. save_results() - 保存结果

用法:
    # 直接使用
    experiment = SingleExperiment(config)
    experiment.run()

    # 自定义
    class MyExperiment(SingleExperiment):
        def evaluate(self):
            # 自定义评估逻辑
            results = super().evaluate()
            results['custom_metric'] = ...
            return results
"""

import torch
from abc import abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path

from ...utils.config import Config
from ...utils.logger import get_logger, set_color, init_logger
from ...data import create_dataset, create_dataloader
from ...model import get_model, MultiDomainModel
from ...trainer import Trainer


class BaseExperiment:
    """实验基类
    
    子类必须实现:
        - run(): 实验主流程
    
    子类可选覆盖:
        - setup(): 初始化
        - load_data(): 数据加载
        - build_model(): 模型构建
        - train(): 训练逻辑
        - evaluate(): 评估逻辑
        - save_results(): 结果保存
    """
    
    # 模式名称
    MODE_NAME: str = "base"
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = None
        self.device = None
        
        # 数据
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None
        
        # 模型
        self.model = None
        self.trainer = None
        
        # 结果
        self.results = {}
    
    def setup(self):
        """初始化：日志、设备等"""
        self.logger = get_logger()
        self.device = torch.device(
            self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        self.config["device"] = str(self.device)
        
        self.logger.info(set_color("=" * 60, "green"))
        self.logger.info(set_color(f"  RecForgeLab Experiment [{self.MODE_NAME}]", "green"))
        self.logger.info(set_color("=" * 60, "green"))
    
    def load_data(self):
        """加载数据集"""
        self.logger.info(set_color("\n[Loading Data]", "cyan"))
        
        self.train_dataset = create_dataset(self.config, phase="train")
        self.valid_dataset = create_dataset(
            self.config, phase="valid",
            encoders=self.train_dataset.feature_encoders
        )
        self.test_dataset = create_dataset(
            self.config, phase="test",
            encoders=self.train_dataset.feature_encoders
        )
        
        self.train_loader = create_dataloader(self.train_dataset, self.config, shuffle=True)
        self.valid_loader = create_dataloader(self.valid_dataset, self.config, shuffle=False)
        self.test_loader = create_dataloader(self.test_dataset, self.config, shuffle=False)
        
        self.logger.info(f"  Train: {len(self.train_dataset)} samples")
        self.logger.info(f"  Valid: {len(self.valid_dataset)} samples")
        self.logger.info(f"  Test:  {len(self.test_dataset)} samples")
    
    def build_model(self):
        """构建模型"""
        self.logger.info(set_color("\n[Building Model]", "cyan"))
        
        model_class = get_model(self.config["model"])
        self.model = model_class(self.config, self.train_dataset)
        self.model = self.model.to(self.device)
        self.model._print_param_count()
        
        # 检测 Multi-Domain
        if isinstance(self.model, MultiDomainModel):
            self.logger.info(set_color(f"  [Multi-Domain Model: {self.model.num_domains} domains]", "yellow"))
    
    def train(self) -> Dict[str, Any]:
        """训练模型"""
        self.logger.info(set_color("\n[Training]", "cyan"))
        
        self.trainer = Trainer(self.config, self.model)
        train_result = self.trainer.train(self.train_loader, self.valid_loader)
        
        self.logger.info(f"  Best score: {train_result.get('best_score', 0):.6f} "
                        f"@ epoch {train_result.get('best_epoch', 0) + 1}")
        
        return train_result
    
    def evaluate(self) -> Dict[str, float]:
        """评估模型"""
        self.logger.info(set_color("\n[Evaluating]", "cyan"))
        
        # Multi-Domain 分域评估
        if isinstance(self.model, MultiDomainModel):
            return self._evaluate_multi_domain()
        else:
            return self.trainer._evaluate_epoch(self.test_loader, epoch=-1)
    
    def _evaluate_multi_domain(self) -> Dict[str, Any]:
        """Multi-Domain 模型分域评估"""
        import numpy as np
        from sklearn.metrics import roc_auc_score, log_loss
        
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_domains = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                batch = {k: v.to(self.device) if hasattr(v, 'to') else v 
                        for k, v in batch.items()}
                preds = self.model.predict(batch)
                
                all_preds.append(preds.cpu().numpy())
                all_labels.append(batch[self.config.get("label_field", "label")].cpu().numpy())
                all_domains.append(batch[self.config.get("domain_field", "domain_indicator")].cpu().numpy())
        
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        all_domains = np.concatenate(all_domains)
        
        # 整体指标
        overall_auc = roc_auc_score(all_labels, all_preds)
        overall_logloss = log_loss(all_labels, all_preds)
        
        # 分域指标
        domain_metrics = {}
        num_domains = self.model.num_domains
        
        for d in range(num_domains):
            mask = (all_domains == d)
            if mask.sum() > 0:
                d_preds = all_preds[mask]
                d_labels = all_labels[mask]
                domain_metrics[f"domain_{d}"] = {
                    "auc": roc_auc_score(d_labels, d_preds),
                    "logloss": log_loss(d_labels, d_preds),
                    "samples": int(mask.sum()),
                }
                self.logger.info(f"  Domain_{d}: AUC={domain_metrics[f'domain_{d}']['auc']:.4f}, "
                               f"LogLoss={domain_metrics[f'domain_{d}']['logloss']:.4f}, "
                               f"samples={mask.sum()}")
        
        return {
            "AUC": overall_auc,
            "LogLoss": overall_logloss,
            "domains": domain_metrics,
        }
    
    def save_results(self, results: Dict[str, Any]):
        """保存结果"""
        self.results = results
        
        self.logger.info(set_color("\n[Results]", "green"))
        for k, v in results.items():
            if isinstance(v, dict):
                self.logger.info(f"  {k}:")
                for kk, vv in v.items():
                    self.logger.info(f"    {kk:<18} = {vv}")
            else:
                self.logger.info(f"  {k:<20} = {v:.6f}")
    
    @abstractmethod
    def run(self) -> Dict[str, Any]:
        """运行实验（子类实现）"""
        raise NotImplementedError
