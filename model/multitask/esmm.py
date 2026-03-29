# -*- coding: utf-8 -*-
"""
ESMM: Entire Space Multi-Task Model (SIGIR 2018)
ESCM2: ESMM + Counterfactual Regularization

直接迁移自 autoresearch/multitask/src/models.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from ..base import MultiTaskModel, register_model
from ..layers import MLPLayers, FeatureEmbedding


@register_model("esmm")
class ESMM(MultiTaskModel):
    """Entire Space Multi-Task Model
    
    pCTR = sigmoid(f_ctr(x))
    pCVR = sigmoid(f_cvr(x))
    pCTCVR = pCTR × pCVR
    
    Loss = BCE(pCTR, click) + BCE(pCTCVR, conversion)
    """
    
    def __init__(self, config, dataset):
        config["tasks"] = ["ctr", "ctcvr"]
        config["task_weights"] = [1.0, 1.0]
        super().__init__(config, dataset)
        
        self.mlp_hidden_size = config["mlp_hidden_size"]
        self.dropout_prob = config["dropout_prob"]
        
        self.feature_embedding = FeatureEmbedding(
            sparse_vocab=dataset.sparse_vocab,
            sparse_feats=dataset.sparse_features,
            dense_dim=len(dataset.dense_features),
            embedding_dim=config["embedding_size"],
            encoder_type=config["encoder_type"],
            encoder_config=config.get("encoder_config", {}),
        )
        input_dim = self.feature_embedding.output_dim
        
        # CTR Tower
        self.ctr_mlp = MLPLayers(
            layers=[input_dim] + self.mlp_hidden_size,
            dropout=self.dropout_prob,
            last_activation=False,
        )
        self.ctr_output = nn.Linear(self.mlp_hidden_size[-1], 1)
        
        # CVR Tower
        self.cvr_mlp = MLPLayers(
            layers=[input_dim] + self.mlp_hidden_size,
            dropout=self.dropout_prob,
            last_activation=False,
        )
        self.cvr_output = nn.Linear(self.mlp_hidden_size[-1], 1)
    
    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        emb = self.feature_embedding(x)
        
        p_ctr = torch.sigmoid(self.ctr_output(self.ctr_mlp(emb)).squeeze(-1))
        p_cvr = torch.sigmoid(self.cvr_output(self.cvr_mlp(emb)).squeeze(-1))
        p_ctcvr = p_ctr * p_cvr
        
        return {"ctr": p_ctr, "cvr": p_cvr, "ctcvr": p_ctcvr}
    
    def calculate_loss(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        preds = self.forward(x)
        
        ctr_label = x["click_label"].float()
        ctcvr_label = x["label"].float()
        
        loss_ctr = F.binary_cross_entropy(preds["ctr"], ctr_label)
        loss_ctcvr = F.binary_cross_entropy(preds["ctcvr"], ctcvr_label)
        
        return loss_ctr + loss_ctcvr
    
    def predict(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.forward(x)


@register_model("escm2")
class ESCM2(MultiTaskModel):
    """ESMM + 反事实正则化（Counterfactual Regularization）
    
    额外惩罚未点击样本上的高 CVR 预测：
      L_CR = mean(p_cvr[y_ctr==0]^2)
      Total = BCE(p_ctr, y_ctr) + BCE(p_ctcvr, y_ctcvr) + λ * L_CR
    """
    
    def __init__(self, config, dataset):
        config["tasks"] = ["ctr", "ctcvr"]
        config["task_weights"] = [1.0, 1.0]
        super().__init__(config, dataset)
        
        self.lam = config.get("escm2_lambda", 0.1)
        self.mlp_hidden_size = config["mlp_hidden_size"]
        self.dropout_prob = config["dropout_prob"]
        
        self.feature_embedding = FeatureEmbedding(
            sparse_vocab=dataset.sparse_vocab,
            sparse_feats=dataset.sparse_features,
            dense_dim=len(dataset.dense_features),
            embedding_dim=config["embedding_size"],
            encoder_type=config["encoder_type"],
            encoder_config=config.get("encoder_config", {}),
        )
        input_dim = self.feature_embedding.output_dim
        
        self.ctr_tower = MLPLayers(
            layers=[input_dim] + self.mlp_hidden_size + [1],
            dropout=self.dropout_prob,
            last_activation=False,
        )
        self.ctcvr_tower = MLPLayers(
            layers=[input_dim] + self.mlp_hidden_size + [1],
            dropout=self.dropout_prob,
            last_activation=False,
        )
    
    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        emb = self.feature_embedding(x)
        p_ctr = torch.sigmoid(self.ctr_tower(emb).squeeze(-1))
        p_ctcvr = torch.sigmoid(self.ctcvr_tower(emb).squeeze(-1))
        p_cvr = torch.clamp(p_ctcvr / (p_ctr + 1e-8), 0.0, 1.0)
        return {"ctr": p_ctr, "cvr": p_cvr, "ctcvr": p_ctcvr}
    
    def calculate_loss(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        preds = self.forward(x)
        ctr_label = x["click_label"].float()
        ctcvr_label = x["label"].float()
        
        loss_ctr = F.binary_cross_entropy(preds["ctr"], ctr_label)
        loss_ctcvr = F.binary_cross_entropy(preds["ctcvr"], ctcvr_label)
        
        # 反事实正则化
        non_click_mask = (ctr_label == 0)
        if non_click_mask.sum() > 0:
            loss_cr = (preds["cvr"][non_click_mask] ** 2).mean()
        else:
            loss_cr = torch.tensor(0.0, device=preds["cvr"].device)
        
        return loss_ctr + loss_ctcvr + self.lam * loss_cr
    
    def predict(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.forward(x)
