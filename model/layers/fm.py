# -*- coding: utf-8 -*-
"""
FMLayer: Factorization Machine Layer
用于 DeepFM 和 xDeepFM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FMLayer(nn.Module):
    """FM 层
    
    一阶特征 + 二阶特征交互
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, embed: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            embed: [batch, num_fields, embed_dim]
        
        Returns:
            [batch, 1] FM 输出
        """
        # 一阶特征（求和）
        fm_first = embed.sum(dim=1)  # [batch, embed_dim]
        
        # 二阶特征交互
        # (sum(x_i))^2 - sum(x_i^2)
        sum_square = torch.sum(embed, dim=1) ** 2  # [batch, embed_dim]
        square_sum = torch.sum(embed ** 2, dim=1)  # [batch, embed_dim]
        fm_second = 0.5 * (sum_square - square_sum)  # [batch, embed_dim]
        
        # 组合
        fm_out = fm_first + fm_second  # [batch, embed_dim]
        
        return fm_out.sum(dim=-1, keepdim=True)  # [batch, 1]


class CrossNetwork(nn.Module):
    """Cross Network (DCN)
    
    显式特征交叉
    """
    
    def __init__(self, input_dim: int, num_layers: int = 3):
        super().__init__()
        self.num_layers = num_layers
        
        # 每层的权重和偏置
        self.weight_list = nn.ParameterList([
            nn.Parameter(torch.zeros(input_dim, 1))
            for _ in range(num_layers)
        ])
        self.bias_list = nn.ParameterList([
            nn.Parameter(torch.zeros(input_dim))
            for _ in range(num_layers)
        ])
        
        self._init_weights()
    
    def _init_weights(self):
        for w in self.weight_list:
            nn.init.xavier_uniform_(w)
        for b in self.bias_list:
            nn.init.zeros_(b)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, input_dim]
        
        Returns:
            [batch, input_dim]
        """
        x0 = x
        
        for i in range(self.num_layers):
            # x_{l+1} = x0 * (x_l^T * w_l) + b_l + x_l
            cross = torch.matmul(x, self.weight_list[i])  # [batch, 1]
            x = x0 * cross + self.bias_list[i] + x
        
        return x


class CrossNetworkV2(nn.Module):
    """Cross Network V2 (DCNv2)
    
    使用低秩分解减少参数
    """
    
    def __init__(
        self,
        input_dim: int,
        num_layers: int = 3,
        low_rank: int = 64,
    ):
        super().__init__()
        self.num_layers = num_layers
        
        # 低秩分解
        self.U_list = nn.ParameterList([
            nn.Parameter(torch.zeros(input_dim, low_rank))
            for _ in range(num_layers)
        ])
        self.V_list = nn.ParameterList([
            nn.Parameter(torch.zeros(low_rank, input_dim))
            for _ in range(num_layers)
        ])
        self.bias_list = nn.ParameterList([
            nn.Parameter(torch.zeros(input_dim))
            for _ in range(num_layers)
        ])
        
        self._init_weights()
    
    def _init_weights(self):
        for U, V in zip(self.U_list, self.V_list):
            nn.init.xavier_uniform_(U)
            nn.init.xavier_uniform_(V)
        for b in self.bias_list:
            nn.init.zeros_(b)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, input_dim]
        
        Returns:
            [batch, input_dim]
        """
        x0 = x
        
        for i in range(self.num_layers):
            # W = U @ V (低秩分解)
            # x_{l+1} = x0 * (x_l^T * W) + b_l + x_l
            proj = torch.matmul(x, self.U_list[i])  # [batch, low_rank]
            proj = torch.matmul(proj, self.V_list[i])  # [batch, input_dim]
            x = x0 * proj + self.bias_list[i] + x
        
        return x
