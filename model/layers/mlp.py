# -*- coding: utf-8 -*-
"""
MLP 层
借鉴 RecBole 的 MLPLayers 设计
"""

import torch
import torch.nn as nn
from typing import List, Optional, Union


class MLPLayers(nn.Module):
    """多层感知机
    
    支持可配置的隐藏层、激活函数、BatchNorm、Dropout
    """
    
    def __init__(
        self,
        layers: List[int],
        dropout: float = 0.0,
        activation: str = "relu",
        bn: bool = False,
        last_activation: bool = True,
        last_bn: bool = False,
    ):
        """初始化 MLP
        
        Args:
            layers: 各层维度，如 [256, 128, 64] 表示 3 层 MLP
            dropout: Dropout 概率
            activation: 激活函数，支持 relu, sigmoid, tanh, leakyrelu
            bn: 是否使用 BatchNorm
            last_activation: 最后一层是否使用激活函数
            last_bn: 最后一层是否使用 BatchNorm
        """
        super().__init__()
        
        self.layers = layers
        self.dropout = dropout
        self.activation = activation
        self.use_bn = bn
        
        mlp_modules = []
        for idx, (input_size, output_size) in enumerate(zip(layers[:-1], layers[1:])):
            # Dropout
            if dropout > 0:
                mlp_modules.append(nn.Dropout(p=dropout))
            
            # Linear
            mlp_modules.append(nn.Linear(input_size, output_size))
            
            # BatchNorm
            if bn and (idx < len(layers) - 2 or last_bn):
                mlp_modules.append(nn.BatchNorm1d(output_size))
            
            # Activation
            if idx < len(layers) - 2 or last_activation:
                mlp_modules.append(self._get_activation(activation))
        
        self.mlp = nn.Sequential(*mlp_modules)
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _get_activation(self, activation: str) -> nn.Module:
        """获取激活函数"""
        if activation.lower() == "relu":
            return nn.ReLU()
        elif activation.lower() == "sigmoid":
            return nn.Sigmoid()
        elif activation.lower() == "tanh":
            return nn.Tanh()
        elif activation.lower() == "leakyrelu":
            return nn.LeakyReLU()
        elif activation.lower() == "none":
            return nn.Identity()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight.data)
            if module.bias is not None:
                nn.init.zeros_(module.bias.data)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: [batch_size, input_dim]
        
        Returns:
            [batch_size, output_dim]
        """
        return self.mlp(x)


class FM(nn.Module):
    """Factorization Machine
    
    二阶特征交叉
    """
    
    def __init__(self, reduce_sum: bool = True):
        super().__init__()
        self.reduce_sum = reduce_sum
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: [batch_size, num_fields, embed_dim]
        
        Returns:
            [batch_size, 1] if reduce_sum else [batch_size, embed_dim]
        """
        # FM 公式: 0.5 * (sum(x)^2 - sum(x^2))
        square_of_sum = torch.sum(x, dim=1) ** 2        # [batch_size, embed_dim]
        sum_of_square = torch.sum(x ** 2, dim=1)        # [batch_size, embed_dim]
        
        fm_out = 0.5 * (square_of_sum - sum_of_square)  # [batch_size, embed_dim]
        
        if self.reduce_sum:
            fm_out = torch.sum(fm_out, dim=1, keepdim=True)  # [batch_size, 1]
        
        return fm_out


class CrossNetwork(nn.Module):
    """Cross Network (DCN)
    
    显式高阶特征交叉
    """
    
    def __init__(self, input_dim: int, num_layers: int = 3):
        super().__init__()
        
        self.num_layers = num_layers
        
        # 每层一个权重矩阵和偏置
        self.w = nn.ModuleList([
            nn.Linear(input_dim, 1, bias=False) for _ in range(num_layers)
        ])
        self.b = nn.ParameterList([
            nn.Parameter(torch.zeros(input_dim)) for _ in range(num_layers)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: [batch_size, input_dim]
        
        Returns:
            [batch_size, input_dim]
        """
        x0 = x
        for i in range(self.num_layers):
            # x_{l+1} = x_0 * (w_l^T * x_l) + b_l + x_l
            x = x0 * self.w[i](x) + self.b[i] + x
        
        return x


class CrossNetworkV2(nn.Module):
    """Cross Network V2 (DCNv2)
    
    使用矩阵版本，更强表达能力
    """
    
    def __init__(self, input_dim: int, num_layers: int = 3, low_rank: Optional[int] = None):
        super().__init__()
        
        self.num_layers = num_layers
        self.low_rank = low_rank
        
        if low_rank is not None:
            # 低秩版本
            self.w = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(input_dim, low_rank, bias=False),
                    nn.Linear(low_rank, input_dim, bias=False),
                ) for _ in range(num_layers)
            ])
        else:
            # 完整矩阵版本
            self.w = nn.ModuleList([
                nn.Linear(input_dim, input_dim, bias=False) for _ in range(num_layers)
            ])
        
        self.b = nn.ParameterList([
            nn.Parameter(torch.zeros(input_dim)) for _ in range(num_layers)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: [batch_size, input_dim]
        
        Returns:
            [batch_size, input_dim]
        """
        x0 = x
        for i in range(self.num_layers):
            # x_{l+1} = x_0 * (W_l * x_l) + b_l + x_l
            x = x0 * self.w[i](x) + self.b[i] + x
        
        return x


class MultiHeadAttention(nn.Module):
    """多头自注意力"""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播
        
        Args:
            x: [batch_size, num_fields, embed_dim]
            mask: [batch_size, num_fields]
        
        Returns:
            [batch_size, num_fields, embed_dim]
        """
        attn_out, _ = self.mha(x, x, x, key_padding_mask=mask)
        out = self.layer_norm(x + attn_out)
        return out
