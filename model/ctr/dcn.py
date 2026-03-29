# -*- coding: utf-8 -*-
"""
DCN: Deep & Cross Network (ADKDD 2017)
DCNv2: Improved Deep & Cross Network (WWW 2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from ..base import CTRModel, register_model
from ..layers import MLPLayers, CrossNetwork, CrossNetworkV2, FeatureEmbedding


@register_model("dcn")
class DCN(CTRModel):
    """Deep & Cross Network"""
    
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        
        self.feature_embedding = FeatureEmbedding(
            sparse_vocab=dataset.sparse_vocab,
            sparse_feats=dataset.sparse_features,
            dense_dim=len(dataset.dense_features),
            embedding_dim=config["embedding_size"],
            encoder_type=config["encoder_type"],
        )
        input_dim = self.feature_embedding.output_dim
        
        self.cross_net = CrossNetwork(input_dim, config.get("cross_layer_num", 3))
        self.mlp = MLPLayers(
            layers=[input_dim] + config["mlp_hidden_size"],
            dropout=config["dropout_prob"], bn=True,
        )
        self.predict_layer = nn.Linear(input_dim + config["mlp_hidden_size"][-1], 1)
    
    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        emb = self.feature_embedding(x)
        cross_out = self.cross_net(emb)
        deep_out = self.mlp(emb)
        return self.predict_layer(torch.cat([cross_out, deep_out], dim=1)).squeeze(-1)
    
    def predict(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))
    
    def calculate_loss(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(self.forward(x), x[self.label_field].float())


@register_model("dcnv2")
class DCNv2(CTRModel):
    """DCNv2: Improved Deep & Cross Network"""
    
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        
        self.feature_embedding = FeatureEmbedding(
            sparse_vocab=dataset.sparse_vocab,
            sparse_feats=dataset.sparse_features,
            dense_dim=len(dataset.dense_features),
            embedding_dim=config["embedding_size"],
            encoder_type=config["encoder_type"],
        )
        input_dim = self.feature_embedding.output_dim
        
        self.cross_net = CrossNetworkV2(input_dim, config.get("cross_layer_num", 3),
                                        config.get("cross_low_rank", None))
        self.mlp = MLPLayers(
            layers=[input_dim] + config["mlp_hidden_size"],
            dropout=config["dropout_prob"], bn=True,
        )
        self.predict_layer = nn.Linear(input_dim + config["mlp_hidden_size"][-1], 1)
    
    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        emb = self.feature_embedding(x)
        cross_out = self.cross_net(emb)
        deep_out = self.mlp(emb)
        return self.predict_layer(torch.cat([cross_out, deep_out], dim=1)).squeeze(-1)
    
    def predict(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))
    
    def calculate_loss(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(self.forward(x), x[self.label_field].float())
