# -*- coding: utf-8 -*-
"""
MMoE: Multi-gate Mixture-of-Experts (KDD 2018)
PLE: Progressive Layered Extraction (RecSys 2020)
SharedBottom: 共享底层 MLP

直接迁移自 autoresearch/multitask/src/models.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List

from ..base import MultiTaskModel, register_model
from ..layers import MLPLayers, FeatureEmbedding


@register_model("shared_bottom")
class SharedBottom(MultiTaskModel):
    """共享底层 MLP + 各任务独立 Tower"""
    
    def __init__(self, config, dataset):
        config["tasks"] = config.get("tasks", ["ctr", "cvr"])
        config["task_weights"] = config.get("task_weights", [1.0, 1.0])
        super().__init__(config, dataset)
        
        self.feature_embedding = FeatureEmbedding(
            sparse_vocab=dataset.sparse_vocab,
            sparse_feats=dataset.sparse_features,
            dense_dim=len(dataset.dense_features),
            embedding_dim=config["embedding_size"],
            encoder_type=config["encoder_type"],
        )
        input_dim = self.feature_embedding.output_dim
        
        self.shared_mlp = MLPLayers(
            layers=[input_dim] + config["mlp_hidden_size"],
            dropout=config["dropout_prob"],
            last_activation=True,
        )
        tower_in = config["mlp_hidden_size"][-1]
        
        self.towers = nn.ModuleDict({
            task: nn.Sequential(
                nn.Linear(tower_in, 64), nn.ReLU(),
                nn.Linear(64, 1),
            ) for task in self.tasks
        })
    
    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        emb = self.feature_embedding(x)
        shared = self.shared_mlp(emb)
        return {task: torch.sigmoid(self.towers[task](shared).squeeze(-1)) for task in self.tasks}
    
    def calculate_loss(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        preds = self.forward(x)
        label_map = {"ctr": "click_label", "cvr": "label", "ctcvr": "label"}
        total = sum(
            F.binary_cross_entropy(preds[t], x[label_map.get(t, "label")].float())
            for t in self.tasks
        )
        return total
    
    def predict(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.forward(x)


@register_model("mmoe")
class MMoE(MultiTaskModel):
    """Multi-gate Mixture-of-Experts
    
    共享专家 + 每任务独立 Gate + 独立 Tower
    """
    
    def __init__(self, config, dataset):
        config["tasks"] = config.get("tasks", ["ctr", "cvr"])
        config["task_weights"] = config.get("task_weights", [1.0, 1.0])
        super().__init__(config, dataset)
        
        self.num_experts = config.get("num_experts", 8)
        self.expert_hidden_size = config.get("expert_hidden_size", config["mlp_hidden_size"])
        self.tower_hidden_size = config.get("tower_hidden_size", [64])
        
        self.feature_embedding = FeatureEmbedding(
            sparse_vocab=dataset.sparse_vocab,
            sparse_feats=dataset.sparse_features,
            dense_dim=len(dataset.dense_features),
            embedding_dim=config["embedding_size"],
            encoder_type=config["encoder_type"],
        )
        input_dim = self.feature_embedding.output_dim
        
        # 共享专家
        self.experts = nn.ModuleList([
            MLPLayers(
                layers=[input_dim] + self.expert_hidden_size,
                dropout=config["dropout_prob"],
                last_activation=True,
            ) for _ in range(self.num_experts)
        ])
        
        expert_out_dim = self.expert_hidden_size[-1]
        
        # 每任务独立 Gate
        self.gates = nn.ModuleDict({
            task: nn.Sequential(
                nn.Linear(input_dim, self.num_experts),
                nn.Softmax(dim=-1),
            ) for task in self.tasks
        })
        
        # 每任务独立 Tower
        self.towers = nn.ModuleDict({
            task: nn.Sequential(
                MLPLayers(
                    layers=[expert_out_dim] + self.tower_hidden_size,
                    dropout=config["dropout_prob"],
                    last_activation=True,
                ),
                nn.Linear(self.tower_hidden_size[-1], 1),
            ) for task in self.tasks
        })
    
    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        emb = self.feature_embedding(x)
        
        # Expert outputs: [B, num_experts, expert_out_dim]
        expert_outputs = torch.stack([e(emb) for e in self.experts], dim=1)
        
        preds = {}
        for task in self.tasks:
            gate = self.gates[task](emb)  # [B, num_experts]
            task_input = torch.einsum("be,bed->bd", gate, expert_outputs)
            preds[task] = torch.sigmoid(self.towers[task](task_input).squeeze(-1))
        
        return preds
    
    def calculate_loss(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        preds = self.forward(x)
        label_map = {"ctr": "click_label", "cvr": "label", "ctcvr": "label"}
        
        total = torch.tensor(0.0, device=list(x.values())[0].device)
        for i, task in enumerate(self.tasks):
            label = x[label_map.get(task, "label")].float()
            
            # CVR 只在点击样本上计算
            if task == "cvr":
                click_mask = x["click_label"] == 1
                if click_mask.sum() > 0:
                    loss = F.binary_cross_entropy(preds[task][click_mask], label[click_mask])
                else:
                    loss = torch.tensor(0.0, device=total.device)
            else:
                loss = F.binary_cross_entropy(preds[task], label)
            
            total = total + self.task_weights[i] * loss
        
        return total
    
    def predict(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.forward(x)


@register_model("ple")
class PLE(MultiTaskModel):
    """Progressive Layered Extraction (RecSys 2020)
    
    共享专家 + 任务特定专家 + 每任务独立 Gate + Tower
    """
    
    def __init__(self, config, dataset):
        config["tasks"] = config.get("tasks", ["ctr", "cvr"])
        config["task_weights"] = config.get("task_weights", [1.0, 1.0])
        super().__init__(config, dataset)
        
        self.num_shared_experts = config.get("num_experts", 4)
        self.num_task_experts = config.get("num_task_experts", 4)
        self.expert_hidden_size = config.get("expert_hidden_size", config["mlp_hidden_size"])
        self.tower_hidden_size = config.get("tower_hidden_size", [64])
        
        self.feature_embedding = FeatureEmbedding(
            sparse_vocab=dataset.sparse_vocab,
            sparse_feats=dataset.sparse_features,
            dense_dim=len(dataset.dense_features),
            embedding_dim=config["embedding_size"],
            encoder_type=config["encoder_type"],
        )
        input_dim = self.feature_embedding.output_dim
        
        # 共享专家
        self.shared_experts = nn.ModuleList([
            MLPLayers(
                layers=[input_dim] + self.expert_hidden_size,
                dropout=config["dropout_prob"],
                last_activation=True,
            ) for _ in range(self.num_shared_experts)
        ])
        
        # 任务特定专家
        self.task_experts = nn.ModuleDict({
            task: nn.ModuleList([
                MLPLayers(
                    layers=[input_dim] + self.expert_hidden_size,
                    dropout=config["dropout_prob"],
                    last_activation=True,
                ) for _ in range(self.num_task_experts)
            ]) for task in self.tasks
        })
        
        expert_out_dim = self.expert_hidden_size[-1]
        total_experts = self.num_shared_experts + self.num_task_experts
        
        # 每任务 Gate（考虑共享+任务特定专家）
        self.gates = nn.ModuleDict({
            task: nn.Sequential(
                nn.Linear(input_dim, total_experts),
                nn.Softmax(dim=-1),
            ) for task in self.tasks
        })
        
        # Tower
        self.towers = nn.ModuleDict({
            task: nn.Sequential(
                MLPLayers(
                    layers=[expert_out_dim] + self.tower_hidden_size,
                    dropout=config["dropout_prob"],
                    last_activation=True,
                ),
                nn.Linear(self.tower_hidden_size[-1], 1),
            ) for task in self.tasks
        })
    
    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        emb = self.feature_embedding(x)
        
        # 共享专家输出
        shared_out = torch.stack([e(emb) for e in self.shared_experts], dim=1)
        
        preds = {}
        for task in self.tasks:
            # 任务特定专家输出
            task_out = torch.stack([e(emb) for e in self.task_experts[task]], dim=1)
            
            # 合并所有专家
            all_experts = torch.cat([shared_out, task_out], dim=1)
            
            # Gate 加权
            gate = self.gates[task](emb)
            task_input = torch.einsum("be,bed->bd", gate, all_experts)
            
            preds[task] = torch.sigmoid(self.towers[task](task_input).squeeze(-1))
        
        return preds
    
    def calculate_loss(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        preds = self.forward(x)
        label_map = {"ctr": "click_label", "cvr": "label", "ctcvr": "label"}
        
        total = torch.tensor(0.0, device=list(x.values())[0].device)
        for i, task in enumerate(self.tasks):
            label = x[label_map.get(task, "label")].float()
            
            if task == "cvr":
                click_mask = x["click_label"] == 1
                if click_mask.sum() > 0:
                    loss = F.binary_cross_entropy(preds[task][click_mask], label[click_mask])
                else:
                    loss = torch.tensor(0.0, device=total.device)
            else:
                loss = F.binary_cross_entropy(preds[task], label)
            
            total = total + self.task_weights[i] * loss
        
        return total
    
    def predict(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.forward(x)


@register_model("direct_ctcvr")
class DirectCTCVR(MultiTaskModel):
    """直接建模 P(buy|imp) 的单任务 DNN，用于 baseline 对比"""
    
    def __init__(self, config, dataset):
        config["tasks"] = ["ctcvr"]
        config["task_weights"] = [1.0]
        super().__init__(config, dataset)
        
        self.feature_embedding = FeatureEmbedding(
            sparse_vocab=dataset.sparse_vocab,
            sparse_feats=dataset.sparse_features,
            dense_dim=len(dataset.dense_features),
            embedding_dim=config["embedding_size"],
            encoder_type=config["encoder_type"],
        )
        input_dim = self.feature_embedding.output_dim
        
        self.mlp = MLPLayers(
            layers=[input_dim] + config["mlp_hidden_size"] + [1],
            dropout=config["dropout_prob"],
            last_activation=False,
        )
    
    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        emb = self.feature_embedding(x)
        p_ctcvr = torch.sigmoid(self.mlp(emb).squeeze(-1))
        return {"ctr": torch.zeros_like(p_ctcvr), "cvr": torch.zeros_like(p_ctcvr), "ctcvr": p_ctcvr}
    
    def calculate_loss(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        preds = self.forward(x)
        return F.binary_cross_entropy(preds["ctcvr"], x["label"].float())
    
    def predict(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.forward(x)
