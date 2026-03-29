#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LTV 预估模型对比实验

对比方法：
- ZILN: Zero-Inflated LogNormal
- TwoStage: 两阶段模型
- Tweedie: Tweedie Loss
- Ordinal: Ordinal Regression
- MDN: Mixture Density Network

评估指标：
- MAE, MSE, RMSE
- MAPE
- Gini, Spearman
- Zero Accuracy
- Decile Lift
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Any
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# 直接使用本地导入
try:
    from recforgelab.utils.config import Config
    from recforgelab.utils.logger import get_logger, set_color, init_logger
except ImportError:
    # 如果 recforgelab 未安装，跳过这些导入
    pass


# ============================================================
# 模拟 LTV 数据集
# ============================================================

class SyntheticLTVDataset(Dataset):
    """合成 LTV 数据集（模拟零膨胀 + 右偏分布）
    
    特点：
    - 零膨胀：~95% 用户 LTV=0
    - 右偏：非零值服从 LogNormal 分布
    - 异质性：不同用户群体价值差异大
    """
    
    def __init__(self, n_samples=10000, n_features=20, zero_ratio=0.95, seed=42):
        np.random.seed(seed)
        
        self.n_samples = n_samples
        
        # 生成特征
        self.features = np.random.randn(n_samples, n_features).astype(np.float32)
        
        # 生成 LTV
        # Step 1: 付费概率（与特征相关）
        paid_prob = 1 / (1 + np.exp(-(self.features[:, 0] + 0.5 * self.features[:, 1])))
        paid_prob = paid_prob * (1 - zero_ratio) / paid_prob.mean()  # 归一化
        
        # Step 2: 付费标记
        is_paid = np.random.rand(n_samples) < paid_prob
        
        # Step 3: 条件价值（LogNormal）
        # LTV | paid ~ LogNormal(μ, σ²)
        mu = 5.0 + 0.5 * self.features[:, 2]  # 基础价值
        sigma = 1.0
        ltv = np.zeros(n_samples)
        ltv[is_paid] = np.exp(np.random.normal(mu[is_paid], sigma))
        
        # 截断
        ltv = np.clip(ltv, 0, 10000)
        
        self.labels = ltv.astype(np.float32)
        self.is_paid = is_paid
        
        # 统计
        self.zero_ratio = (ltv == 0).mean()
        self.paid_mean = ltv[is_paid].mean() if is_paid.sum() > 0 else 0
        self.ltv_mean = ltv.mean()
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'ltv': self.labels[idx],
        }
    
    def get_stats(self):
        return {
            'n_samples': self.n_samples,
            'zero_ratio': self.zero_ratio,
            'paid_mean': self.paid_mean,
            'ltv_mean': self.ltv_mean,
            'paid_count': self.is_paid.sum(),
        }


def collate_fn(batch):
    """整理 batch"""
    features = torch.tensor(np.array([b['features'] for b in batch]))
    ltv = torch.tensor(np.array([b['ltv'] for b in batch]))
    return {
        'features': features,
        'sparse_features': torch.zeros(features.size(0), 1, dtype=torch.long),  # dummy
        'dense_features': features,
        'ltv': ltv,
        'label': ltv,  # 兼容基类
    }


class PaidOnlyLTVDataset(Dataset):
    """付费用户数据集（只包含 LTV > 0 的样本）
    
    用于测试付费用户建模方法
    """
    
    def __init__(self, n_samples=10000, n_features=20, seed=42):
        np.random.seed(seed)
        
        self.n_samples = n_samples
        
        # 生成特征
        self.features = np.random.randn(n_samples, n_features).astype(np.float32)
        
        # 生成 LTV（只生成正值，LogNormal 分布）
        mu = 5.0 + 0.5 * self.features[:, 2]
        sigma = 1.0
        self.labels = np.exp(np.random.normal(mu, sigma)).astype(np.float32)
        
        # 截断
        self.labels = np.clip(self.labels, 1, 10000)
        
        self.ltv_mean = self.labels.mean()
        self.ltv_std = self.labels.std()
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'ltv': self.labels[idx],
        }
    
    def get_stats(self):
        return {
            'n_samples': self.n_samples,
            'ltv_mean': self.ltv_mean,
            'ltv_std': self.ltv_std,
            'ltv_median': np.median(self.labels),
        }


# ============================================================
# 简化版 LTV 模型（适配合成数据）
# ============================================================

class SimpleLTVBase(torch.nn.Module):
    """简化版 LTV 模型基类"""
    
    def __init__(self, n_features=20, hidden_sizes=[128, 64]):
        super().__init__()
        
        layers = []
        prev_dim = n_features
        for h in hidden_sizes:
            layers.append(torch.nn.Linear(prev_dim, h))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(0.2))
            prev_dim = h
        
        self.mlp = torch.nn.Sequential(*layers)
    
    def _print_param_count(self):
        n = sum(p.numel() for p in self.parameters())
        print(f"  Parameters: {n:,}")


class SimpleZILN(SimpleLTVBase):
    """简化版 ZILN"""
    
    def __init__(self, n_features=20, hidden_sizes=[128, 64]):
        super().__init__(n_features, hidden_sizes)
        
        h = hidden_sizes[-1]
        self.prob_head = torch.nn.Sequential(
            torch.nn.Linear(h, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
            torch.nn.Sigmoid(),
        )
        self.mu_head = torch.nn.Sequential(
            torch.nn.Linear(h, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
        )
        self.log_sigma_head = torch.nn.Sequential(
            torch.nn.Linear(h, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
        )
    
    def forward(self, batch):
        h = self.mlp(batch['features'])
        prob = self.prob_head(h).squeeze(-1)
        mu = self.mu_head(h).squeeze(-1)
        log_sigma = self.log_sigma_head(h).squeeze(-1)
        return prob, mu, log_sigma
    
    def calculate_loss(self, batch):
        prob, mu, log_sigma = self.forward(batch)
        labels = batch['ltv']
        
        eps = 1e-6
        y = torch.clamp(labels, min=eps)
        log_y = torch.log(y)
        
        sigma = torch.exp(torch.clamp(log_sigma, min=-5, max=5))
        
        log_likelihood = -log_sigma - 0.5 * np.log(2 * np.pi) - 0.5 * ((log_y - mu) / sigma) ** 2
        
        zero_mask = (labels < 1e-6).float()
        prob = torch.clamp(prob, min=eps, max=1-eps)
        
        log_p_zero = torch.log(prob)
        log_p_nonzero = torch.log(1 - prob) + log_likelihood
        
        log_prob = zero_mask * log_p_zero + (1 - zero_mask) * log_p_nonzero
        
        return -log_prob.mean()
    
    def predict(self, batch):
        prob, mu, log_sigma = self.forward(batch)
        sigma = torch.exp(torch.clamp(log_sigma, min=-5, max=5))
        conditional_ltv = torch.exp(mu + 0.5 * sigma ** 2)
        return (1 - prob) * conditional_ltv


class SimpleTwoStage(SimpleLTVBase):
    """简化版 TwoStage"""
    
    def __init__(self, n_features=20, hidden_sizes=[128, 64]):
        super().__init__(n_features, hidden_sizes)
        
        h = hidden_sizes[-1]
        self.paid_tower = torch.nn.Sequential(
            torch.nn.Linear(h, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
            torch.nn.Sigmoid(),
        )
        self.value_tower = torch.nn.Sequential(
            torch.nn.Linear(h, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )
    
    def forward(self, batch):
        h = self.mlp(batch['features'])
        paid_prob = self.paid_tower(h).squeeze(-1)
        conditional_value = self.value_tower(h).squeeze(-1)
        return paid_prob, conditional_value
    
    def calculate_loss(self, batch):
        paid_prob, conditional_value = self.forward(batch)
        labels = batch['ltv']
        
        paid_label = (labels > 0).float()
        paid_loss = torch.nn.functional.binary_cross_entropy(paid_prob, paid_label)
        
        paid_mask = paid_label > 0.5
        if paid_mask.sum() > 0:
            target = torch.log(torch.clamp(labels[paid_mask], min=1.0))
            pred = conditional_value[paid_mask]
            value_loss = torch.nn.functional.mse_loss(pred, target)
        else:
            value_loss = torch.tensor(0.0, device=labels.device)
        
        return paid_loss + value_loss
    
    def predict(self, batch):
        paid_prob, conditional_value = self.forward(batch)
        conditional_ltv = torch.exp(conditional_value)
        return paid_prob * conditional_ltv


class SimpleTweedie(SimpleLTVBase):
    """简化版 Tweedie"""
    
    def __init__(self, n_features=20, hidden_sizes=[128, 64], p=1.5):
        super().__init__(n_features, hidden_sizes)
        
        h = hidden_sizes[-1]
        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(h, 1),
            torch.nn.Softplus(),
        )
        self.p = p
    
    def forward(self, batch):
        h = self.mlp(batch['features'])
        return self.output_layer(h).squeeze(-1)
    
    def calculate_loss(self, batch):
        pred = self.forward(batch)
        labels = batch['ltv']
        
        pred = torch.clamp(pred, min=1e-6)
        
        a = pred ** (1 - self.p) / (1 - self.p)
        b = pred ** (2 - self.p) / (2 - self.p)
        
        loss = -labels * a + b
        return loss.mean()
    
    def predict(self, batch):
        return self.forward(batch)


class SimpleOrdinal(SimpleLTVBase):
    """简化版 Ordinal"""
    
    def __init__(self, n_features=20, hidden_sizes=[128, 64], num_bins=10):
        super().__init__(n_features, hidden_sizes)
        
        self.num_bins = num_bins
        self.bin_boundaries = [0] + [10 ** (i * 0.5) for i in range(num_bins)]
        self.bin_midpoints = [(self.bin_boundaries[i] + self.bin_boundaries[i+1]) / 2 
                              for i in range(num_bins)]
        
        h = hidden_sizes[-1]
        self.output_layer = torch.nn.Linear(h, num_bins)
        
        self.register_buffer(
            "bin_midpoints_tensor",
            torch.tensor(self.bin_midpoints, dtype=torch.float32)
        )
    
    def _ltv_to_bin(self, ltv):
        bins = torch.zeros_like(ltv, dtype=torch.long)
        for i in range(self.num_bins):
            mask = ltv >= self.bin_boundaries[i + 1]
            bins[mask] = i + 1
        return torch.clamp(bins, 0, self.num_bins - 1)
    
    def forward(self, batch):
        h = self.mlp(batch['features'])
        logits = self.output_layer(h)
        return torch.nn.functional.softmax(logits, dim=-1)
    
    def calculate_loss(self, batch):
        probs = self.forward(batch)
        labels = batch['ltv']
        bin_labels = self._ltv_to_bin(labels)
        return torch.nn.functional.cross_entropy(probs, bin_labels)
    
    def predict(self, batch):
        probs = self.forward(batch)
        return torch.matmul(probs, self.bin_midpoints_tensor)


class SimpleMDN(SimpleLTVBase):
    """简化版 MDN"""
    
    def __init__(self, n_features=20, hidden_sizes=[128, 64], num_components=3):
        super().__init__(n_features, hidden_sizes)
        
        self.num_components = num_components
        h = hidden_sizes[-1]
        
        self.pi_head = torch.nn.Sequential(
            torch.nn.Linear(h, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, num_components),
            torch.nn.Softmax(dim=-1),
        )
        self.mu_head = torch.nn.Linear(h, num_components)
        self.log_sigma_head = torch.nn.Linear(h, num_components)
    
    def forward(self, batch):
        h = self.mlp(batch['features'])
        pi = self.pi_head(h)
        mu = self.mu_head(h)
        log_sigma = self.log_sigma_head(h)
        sigma = torch.exp(torch.clamp(log_sigma, min=-5, max=5))
        return pi, mu, sigma
    
    def calculate_loss(self, batch):
        pi, mu, sigma = self.forward(batch)
        labels = batch['ltv']
        
        y = torch.clamp(labels.unsqueeze(-1), min=1e-6)
        log_y = torch.log(y)
        
        log_prob_k = -torch.log(sigma) - 0.5 * np.log(2 * np.pi) - 0.5 * ((log_y - mu) / sigma) ** 2
        
        log_prob = torch.logsumexp(torch.log(pi) + log_prob_k, dim=-1)
        
        return -log_prob.mean()
    
    def predict(self, batch):
        pi, mu, sigma = self.forward(batch)
        # 数值稳定性：限制 sigma
        sigma = torch.clamp(sigma, min=1e-3, max=10.0)
        component_mean = torch.exp(torch.clamp(mu + 0.5 * sigma ** 2, max=20))
        return torch.sum(pi * component_mean, dim=-1)


# ============================================================
# 直接回归模型（新增）
# ============================================================

class SimpleMSE(SimpleLTVBase):
    """MSE 直接回归"""
    
    def __init__(self, n_features=20, hidden_sizes=[128, 64]):
        super().__init__(n_features, hidden_sizes)
        h = hidden_sizes[-1]
        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(h, 1),
            torch.nn.Softplus(),
        )
    
    def forward(self, batch):
        h = self.mlp(batch['features'])
        return self.output_layer(h).squeeze(-1)
    
    def calculate_loss(self, batch):
        pred = self.forward(batch)
        labels = batch['ltv']
        return torch.nn.functional.mse_loss(pred, labels)
    
    def predict(self, batch):
        return self.forward(batch)


class SimpleMAE(SimpleLTVBase):
    """MAE 直接回归"""
    
    def __init__(self, n_features=20, hidden_sizes=[128, 64]):
        super().__init__(n_features, hidden_sizes)
        h = hidden_sizes[-1]
        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(h, 1),
            torch.nn.Softplus(),
        )
    
    def forward(self, batch):
        h = self.mlp(batch['features'])
        return self.output_layer(h).squeeze(-1)
    
    def calculate_loss(self, batch):
        pred = self.forward(batch)
        labels = batch['ltv']
        return torch.nn.functional.l1_loss(pred, labels)
    
    def predict(self, batch):
        return self.forward(batch)


class SimpleHuber(SimpleLTVBase):
    """Huber 直接回归"""
    
    def __init__(self, n_features=20, hidden_sizes=[128, 64], delta=10.0):
        super().__init__(n_features, hidden_sizes)
        h = hidden_sizes[-1]
        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(h, 1),
            torch.nn.Softplus(),
        )
        self.delta = delta
    
    def forward(self, batch):
        h = self.mlp(batch['features'])
        return self.output_layer(h).squeeze(-1)
    
    def calculate_loss(self, batch):
        pred = self.forward(batch)
        labels = batch['ltv']
        
        diff = torch.abs(pred - labels)
        quadratic = torch.min(diff, torch.tensor(self.delta, device=diff.device))
        linear = diff - quadratic
        
        return torch.mean(0.5 * quadratic ** 2 + self.delta * linear)
    
    def predict(self, batch):
        return self.forward(batch)


class SimpleLogMSE(SimpleLTVBase):
    """Log-MSE 直接回归（预测 log(LTV+1)）"""
    
    def __init__(self, n_features=20, hidden_sizes=[128, 64]):
        super().__init__(n_features, hidden_sizes)
        h = hidden_sizes[-1]
        # 输出 log(LTV + 1)，不需要激活
        self.output_layer = torch.nn.Linear(h, 1)
    
    def forward(self, batch):
        h = self.mlp(batch['features'])
        return self.output_layer(h).squeeze(-1)
    
    def calculate_loss(self, batch):
        pred = self.forward(batch)
        labels = batch['ltv']
        log_target = torch.log(labels + 1)
        return torch.nn.functional.mse_loss(pred, log_target)
    
    def predict(self, batch):
        pred = self.forward(batch)
        return torch.exp(pred) - 1


class SimpleWeightedMSE(SimpleLTVBase):
    """加权 MSE（非零样本权重更高）"""
    
    def __init__(self, n_features=20, hidden_sizes=[128, 64], 
                 zero_weight=0.1, nonzero_weight=1.0):
        super().__init__(n_features, hidden_sizes)
        h = hidden_sizes[-1]
        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(h, 1),
            torch.nn.Softplus(),
        )
        self.zero_weight = zero_weight
        self.nonzero_weight = nonzero_weight
    
    def forward(self, batch):
        h = self.mlp(batch['features'])
        return self.output_layer(h).squeeze(-1)
    
    def calculate_loss(self, batch):
        pred = self.forward(batch)
        labels = batch['ltv']
        
        weights = torch.where(labels > 0,
                              torch.tensor(self.nonzero_weight, device=labels.device),
                              torch.tensor(self.zero_weight, device=labels.device))
        
        return torch.mean(weights * (pred - labels) ** 2)
    
    def predict(self, batch):
        return self.forward(batch)


class SimpleQuantile(SimpleLTVBase):
    """分位数回归"""
    
    def __init__(self, n_features=20, hidden_sizes=[128, 64], 
                 quantiles=[0.1, 0.5, 0.9]):
        super().__init__(n_features, hidden_sizes)
        self.quantiles = quantiles
        h = hidden_sizes[-1]
        self.output_layer = torch.nn.Linear(h, len(quantiles))
    
    def forward(self, batch):
        h = self.mlp(batch['features'])
        return self.output_layer(h)  # (batch, num_quantiles)
    
    def calculate_loss(self, batch):
        pred = self.forward(batch)
        labels = batch['ltv']
        
        losses = []
        for i, q in enumerate(self.quantiles):
            error = labels - pred[:, i]
            losses.append(torch.max(q * error, (1 - q) * -error))
        
        return torch.mean(torch.stack(losses))
    
    def predict(self, batch):
        pred = self.forward(batch)
        # 返回中位数
        median_idx = len(self.quantiles) // 2
        return pred[:, median_idx]


# ============================================================
# 付费用户建模（LTV > 0）
# ============================================================

class SimpleLogNormal(SimpleLTVBase):
    """Log-Normal 回归（假设 log(LTV) ~ N(μ, σ²)）"""
    
    def __init__(self, n_features=20, hidden_sizes=[128, 64]):
        super().__init__(n_features, hidden_sizes)
        h = hidden_sizes[-1]
        self.mu_head = torch.nn.Linear(h, 1)
        self.log_sigma_head = torch.nn.Linear(h, 1)
    
    def forward(self, batch):
        h = self.mlp(batch['features'])
        mu = self.mu_head(h).squeeze(-1)
        log_sigma = self.log_sigma_head(h).squeeze(-1)
        return mu, log_sigma
    
    def calculate_loss(self, batch):
        mu, log_sigma = self.forward(batch)
        labels = batch['ltv']
        
        eps = 1e-6
        y = torch.clamp(labels, min=eps)
        log_y = torch.log(y)
        
        sigma = torch.exp(torch.clamp(log_sigma, min=-5, max=5))
        
        nll = log_sigma + 0.5 * np.log(2 * np.pi) + 0.5 * ((log_y - mu) / sigma) ** 2
        
        return nll.mean()
    
    def predict(self, batch):
        mu, log_sigma = self.forward(batch)
        sigma = torch.exp(torch.clamp(log_sigma, min=-5, max=5))
        return torch.exp(mu + 0.5 * sigma ** 2)


class SimpleGamma(SimpleLTVBase):
    """Gamma 回归（假设 LTV ~ Gamma(α, β)）"""
    
    def __init__(self, n_features=20, hidden_sizes=[128, 64]):
        super().__init__(n_features, hidden_sizes)
        h = hidden_sizes[-1]
        self.log_alpha_head = torch.nn.Linear(h, 1)
        self.log_beta_head = torch.nn.Linear(h, 1)
    
    def forward(self, batch):
        h = self.mlp(batch['features'])
        log_alpha = torch.clamp(self.log_alpha_head(h).squeeze(-1), min=-5, max=10)
        log_beta = torch.clamp(self.log_beta_head(h).squeeze(-1), min=-5, max=10)
        return log_alpha, log_beta
    
    def calculate_loss(self, batch):
        log_alpha, log_beta = self.forward(batch)
        labels = batch['ltv']
        
        alpha = torch.exp(log_alpha)
        beta = torch.exp(log_beta)
        
        eps = 1e-6
        y = torch.clamp(labels, min=eps)
        
        nll = -log_alpha * log_beta + torch.lgamma(alpha) - (alpha - 1) * torch.log(y) + beta * y
        
        return nll.mean()
    
    def predict(self, batch):
        log_alpha, log_beta = self.forward(batch)
        alpha = torch.exp(log_alpha)
        beta = torch.exp(log_beta)
        return alpha / beta


class SimpleLogRegression(SimpleLTVBase):
    """简单 Log 回归（预测 log(LTV)，MSE Loss）"""
    
    def __init__(self, n_features=20, hidden_sizes=[128, 64]):
        super().__init__(n_features, hidden_sizes)
        h = hidden_sizes[-1]
        self.output_layer = torch.nn.Linear(h, 1)
    
    def forward(self, batch):
        h = self.mlp(batch['features'])
        return self.output_layer(h).squeeze(-1)
    
    def calculate_loss(self, batch):
        pred = self.forward(batch)
        labels = batch['ltv']
        log_target = torch.log(torch.clamp(labels, min=1.0))
        return torch.nn.functional.mse_loss(pred, log_target)
    
    def predict(self, batch):
        pred = self.forward(batch)
        return torch.exp(pred)


# ============================================================
# 评估函数
# ============================================================

def evaluate_ltv(preds, labels):
    """评估 LTV 预测结果"""
    preds = np.array(preds)
    labels = np.array(labels)
    
    # 基础指标
    mae = mean_absolute_error(labels, preds)
    mse = mean_squared_error(labels, preds)
    rmse = np.sqrt(mse)
    
    # MAPE
    mask = labels > 0
    mape = np.mean(np.abs((labels[mask] - preds[mask]) / labels[mask])) if mask.sum() > 0 else 0
    
    # Gini
    try:
        from sklearn.metrics import roc_auc_score
        binary_labels = (labels > 0).astype(int)
        if binary_labels.sum() > 0 and binary_labels.sum() < len(binary_labels):
            auc = roc_auc_score(binary_labels, preds)
            gini = 2 * auc - 1
        else:
            gini = 0
    except:
        gini = 0
    
    # Spearman
    spearman, _ = spearmanr(labels, preds)
    
    # Zero Accuracy
    pred_zero = (preds < 0.5).astype(int)
    label_zero = (labels < 1e-6).astype(int)
    zero_acc = np.mean(pred_zero == label_zero)
    
    # Decile Lift
    n = len(preds)
    sorted_idx = np.argsort(preds)[::-1]
    decile_size = n // 10
    top_mean = labels[sorted_idx[:decile_size]].mean()
    bottom_mean = labels[sorted_idx[-decile_size:]].mean()
    lift = top_mean / max(bottom_mean, 1e-6)
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'Gini': gini,
        'Spearman': spearman,
        'Zero_Acc': zero_acc,
        'Top_Lift': lift,
    }


# ============================================================
# 训练函数
# ============================================================

def train_model(model, train_loader, test_loader, epochs=10, lr=0.001):
    """训练模型"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            loss = model.calculate_loss(batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"  Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
    
    # 评估
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            preds = model.predict(batch)
            all_preds.append(preds.numpy())
            all_labels.append(batch['ltv'].numpy())
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    return evaluate_ltv(all_preds, all_labels)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='LTV Model Comparison')
    parser.add_argument('--n_samples', type=int, default=50000)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--model', type=str, default=None, help='Run specific model')
    parser.add_argument('--paid_only', action='store_true', help='Only use paid samples (LTV > 0)')
    args = parser.parse_args()
    
    print("=" * 80)
    print("LTV 预估模型对比实验")
    if args.paid_only:
        print("【付费用户模式】只使用 LTV > 0 的样本")
    print("=" * 80)
    
    # 生成数据
    print("\n[生成数据]")
    
    if args.paid_only:
        # 付费用户数据集
        train_data = PaidOnlyLTVDataset(n_samples=args.n_samples, seed=42)
        test_data = PaidOnlyLTVDataset(n_samples=args.n_samples // 5, seed=123)
        
        stats = train_data.get_stats()
        print(f"  训练样本: {stats['n_samples']:,}")
        print(f"  LTV 均值: {stats['ltv_mean']:.2f}")
        print(f"  LTV 中位数: {stats['ltv_median']:.2f}")
        print(f"  LTV 标准差: {stats['ltv_std']:.2f}")
    else:
        # 完整数据集（含零值）
        train_data = SyntheticLTVDataset(n_samples=args.n_samples, seed=42)
        test_data = SyntheticLTVDataset(n_samples=args.n_samples // 5, seed=123)
        
        stats = train_data.get_stats()
        print(f"  训练样本: {stats['n_samples']:,}")
        print(f"  零值比例: {stats['zero_ratio']:.2%}")
        print(f"  付费用户均值: {stats['paid_mean']:.2f}")
        print(f"  整体均值: {stats['ltv_mean']:.2f}")
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=args.batch_size * 2, shuffle=False, collate_fn=collate_fn)
    
    # 模型列表
    if args.paid_only:
        # 付费用户建模专用
        models = {
            'lognormal': SimpleLogNormal,
            'log_regression': SimpleLogRegression,
            'mse': SimpleMSE,
            'mae': SimpleMAE,
            'huber': SimpleHuber,
            'log_mse': SimpleLogMSE,
            'quantile': SimpleQuantile,
        }
    else:
        # 完整模型列表
        models = {
            # 概率模型
            'ziln': SimpleZILN,
            'two_stage': SimpleTwoStage,
            'tweedie': SimpleTweedie,
            'ordinal': SimpleOrdinal,
            'mdn': SimpleMDN,
            # 直接回归
            'mse': SimpleMSE,
            'mae': SimpleMAE,
            'huber': SimpleHuber,
            'log_mse': SimpleLogMSE,
            'weighted_mse': SimpleWeightedMSE,
            'quantile': SimpleQuantile,
            # 付费用户建模
            'lognormal': SimpleLogNormal,
            'log_regression': SimpleLogRegression,
        }
    
    if args.model:
        models = {args.model: models[args.model]}
    
    # 训练和评估
    results = {}
    
    for name, ModelClass in models.items():
        print(f"\n{'='*60}")
        print(f"模型: {name}")
        print(f"{'='*60}")
        
        model = ModelClass(n_features=20, hidden_sizes=[128, 64])
        model._print_param_count()
        
        metrics = train_model(model, train_loader, test_loader, epochs=args.epochs, lr=args.lr)
        results[name] = metrics
        
        print(f"\n[评估结果]")
        for k, v in metrics.items():
            print(f"  {k:<12} = {v:.6f}")
    
    # 汇总表
    print("\n" + "=" * 100)
    print("对比汇总")
    print("=" * 100)
    
    header = f"{'Model':<12} {'MAE':<10} {'MSE':<12} {'RMSE':<10} {'MAPE':<10} {'Gini':<10} {'Spearman':<10} {'Zero_Acc':<10} {'Top_Lift':<10}"
    print(header)
    print("-" * 100)
    
    for name, metrics in results.items():
        row = f"{name:<12} {metrics['MAE']:<10.4f} {metrics['MSE']:<12.4f} {metrics['RMSE']:<10.4f} "
        row += f"{metrics['MAPE']:<10.4f} {metrics['Gini']:<10.4f} {metrics['Spearman']:<10.4f} "
        row += f"{metrics['Zero_Acc']:<10.4f} {metrics['Top_Lift']:<10.4f}"
        print(row)
    
    print("=" * 100)


if __name__ == '__main__':
    main()
