# -*- coding: utf-8 -*-
"""
SSL 两阶段训练实验
"""

from typing import Dict, Any
from copy import deepcopy

from .base import BaseExperiment
from .single import SingleExperiment
from ...utils.config import Config
from ...utils.logger import get_logger, set_color
from ...data import create_dataset, create_dataloader
from ...model import get_model
from ...trainer import Trainer


class SSLExperiment(BaseExperiment):
    """SSL 两阶段训练实验
    
    Phase 1: SSL 预训练（对比学习）
    Phase 2: 下游任务微调
    
    用法:
        # config.yaml:
        #   model: ssl_contrastive
        #   pretrain_epochs: 10
        #   finetune_epochs: 5
        
        experiment = SSLExperiment(config)
        results = experiment.run()
    """
    
    MODE_NAME = "ssl"
    
    def __init__(self, config):
        super().__init__(config)
        self.pretrain_config = None
        self.finetune_config = None
    
    def setup(self):
        """准备两阶段配置"""
        super().setup()
        
        # Phase 1: 预训练配置
        self.pretrain_config = Config(config_dict=self.config.to_dict(), parse_cmd=False)
        self.pretrain_config["training_phase"] = "pretrain"
        self.pretrain_config["epochs"] = self.config.get("pretrain_epochs", 5)
        self.pretrain_config["experiment_name"] = self.config.get("experiment_name", "ssl") + "_pretrain"
        
        # Phase 2: 微调配置
        self.finetune_config = Config(config_dict=self.config.to_dict(), parse_cmd=False)
        self.finetune_config["training_phase"] = "finetune"
        self.finetune_config["epochs"] = self.config.get("finetune_epochs", self.config.get("epochs", 10))
        self.finetune_config["experiment_name"] = self.config.get("experiment_name", "ssl") + "_finetune"
    
    def run(self) -> Dict[str, Any]:
        """运行两阶段训练"""
        self.setup()
        
        # ===== Phase 1: SSL 预训练 =====
        self.logger.info(set_color("\n[Phase 1] SSL Pretraining", "yellow"))
        
        self.load_data()
        self.build_model()
        self.train()
        
        # ===== Phase 2: 微调 =====
        self.logger.info(set_color("\n[Phase 2] Finetuning", "yellow"))
        
        # 切换模型到微调模式
        if hasattr(self.model, "set_phase"):
            self.model.set_phase("finetune")
        
        # 更新 trainer 配置
        self.trainer = Trainer(self.finetune_config, self.model)
        self.trainer.train(self.train_loader, self.valid_loader)
        
        # 评估
        results = self.evaluate()
        self.save_results(results)
        
        return results
