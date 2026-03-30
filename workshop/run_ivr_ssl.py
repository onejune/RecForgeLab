#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IVR SSL 对比学习实验

对比方法：
1. Baseline (DeepFM 无 SSL)
2. SimGCL (嵌入扰动对比学习)
3. SupCon (标签监督对比学习)
4. Domain-CL (域对比学习)
5. Feature-Mask (特征 mask 对比学习)

训练策略：
- Joint: SSL loss + CVR loss 联合训练

用法:
    python workshop/run_ivr_ssl.py --device cuda:1
    python workshop/run_ivr_ssl.py --device cuda:1 --model simgcl
"""

import time
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, log_loss


# ============================================================
# 数据集
# ============================================================

class IVRSSLDataset(Dataset):
    """IVR SSL 数据集"""
    
    def __init__(self, data_path, max_samples=None):
        print(f"Loading {data_path}...")
        start = time.time()
        
        df = pd.read_parquet(data_path)
        if max_samples:
            df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
        
        # 标签
        self.labels = df['ctcvr_label'].values.astype(np.float32)
        
        # 域标识
        self.domain_ids = df['business_type'].values.astype(np.int64)
        
        # 特征列
        self.feature_cols = [c for c in df.columns if c not in ['click_label', 'ctcvr_label', 'business_type']]
        self.features = df[self.feature_cols].values.astype(np.int64)
        
        load_time = time.time() - start
        print(f"  Loaded {len(df):,} samples, {len(self.feature_cols)} features in {load_time:.2f}s")
        print(f"  CVR: {self.labels.mean():.4f}")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'label': self.labels[idx],
            'domain_indicator': self.domain_ids[idx],
            'features': self.features[idx],
        }


def collate_fn(batch):
    labels = torch.tensor([b['label'] for b in batch])
    domain_ids = torch.tensor([b['domain_indicator'] for b in batch])
    features = torch.tensor(np.array([b['features'] for b in batch]))
    return {
        'label': labels,
        'domain_indicator': domain_ids,
        'features': features,
    }


# ============================================================
# 模型定义
# ============================================================

class DeepFMBase(nn.Module):
    """DeepFM 基础模型"""
    
    def __init__(self, vocab_sizes: List[int], embedding_size: int = 16, 
                 hidden_sizes: List[int] = [256, 128, 64]):
        super().__init__()
        self.n_fields = len(vocab_sizes)
        self.embedding_size = embedding_size
        
        # Embeddings
        self.embeddings = nn.ModuleList([
            nn.Embedding(vs, embedding_size) for vs in vocab_sizes
        ])
        
        # Deep
        layers = []
        prev = self.n_fields * embedding_size
        for h in hidden_sizes:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.2)])
            prev = h
        self.deep = nn.Sequential(*layers)
        
        # Output
        self.output = nn.Linear(hidden_sizes[-1], 1)
        
        # Init
        for emb in self.embeddings:
            nn.init.xavier_uniform_(emb.weight)
    
    def get_embeddings(self, batch) -> torch.Tensor:
        """获取特征嵌入 [B, n_fields, embed_dim]"""
        embs = [emb(batch['features'][:, i]) for i, emb in enumerate(self.embeddings)]
        return torch.stack(embs, dim=1)
    
    def get_embed_vector(self, batch) -> torch.Tensor:
        """获取展平的嵌入向量 [B, n_fields * embed_dim]"""
        embs = self.get_embeddings(batch)
        return embs.view(embs.size(0), -1)
    
    def forward(self, batch):
        x = self.get_embed_vector(batch)
        deep_out = self.deep(x)
        logits = self.output(deep_out).squeeze(-1)
        return logits
    
    def calculate_loss(self, batch):
        logits = self.forward(batch)
        return F.binary_cross_entropy_with_logits(logits, batch['label'])
    
    def predict(self, batch):
        return torch.sigmoid(self.forward(batch))


class ProjectionHead(nn.Module):
    """对比学习投影头"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


# ============================================================
# SSL 方法实现
# ============================================================

class SimGCL(DeepFMBase):
    """SimGCL: 简单有效的对比学习
    
    核心思想：在 embedding 上加扰动，同一样本的两个扰动视图为正例
    """
    
    def __init__(self, vocab_sizes: List[int], embedding_size: int = 16,
                 hidden_sizes: List[int] = [256, 128, 64],
                 proj_dim: int = 64, temperature: float = 0.1, eps: float = 0.1,
                 cl_weight: float = 0.1):
        super().__init__(vocab_sizes, embedding_size, hidden_sizes)
        
        input_dim = self.n_fields * embedding_size
        self.projection = ProjectionHead(input_dim, hidden_sizes[0], proj_dim)
        self.temperature = temperature
        self.eps = eps
        self.cl_weight = cl_weight
    
    def perturb_embeddings(self, embeds: torch.Tensor) -> torch.Tensor:
        """对嵌入添加扰动"""
        noise = F.normalize(torch.randn_like(embeds), p=2, dim=-1) * self.eps
        return embeds + noise
    
    def contrastive_loss(self, batch) -> torch.Tensor:
        """SimGCL 对比损失"""
        # 获取嵌入
        embeds = self.get_embed_vector(batch)  # [B, D]
        
        # 两个扰动视图
        z1 = self.projection(self.perturb_embeddings(embeds))
        z2 = self.projection(self.perturb_embeddings(embeds))
        
        # InfoNCE loss
        batch_size = z1.shape[0]
        
        # 相似度矩阵
        sim = torch.matmul(z1, z2.T) / self.temperature  # [B, B]
        
        # 正例：对角线元素
        pos_sim = torch.diag(sim)  # [B]
        
        # 负例：非对角线元素
        # logits: [B, B], 正例在对角线位置
        labels = torch.arange(batch_size, device=z1.device)
        
        loss = F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)
        return loss / 2
    
    def calculate_loss(self, batch):
        # CVR loss
        logits = self.forward(batch)
        cvr_loss = F.binary_cross_entropy_with_logits(logits, batch['label'])
        
        # SSL loss
        cl_loss = self.contrastive_loss(batch)
        
        return cvr_loss + self.cl_weight * cl_loss


class SupCon(DeepFMBase):
    """Supervised Contrastive Learning
    
    核心思想：同标签样本为正例，不同标签为负例
    """
    
    def __init__(self, vocab_sizes: List[int], embedding_size: int = 16,
                 hidden_sizes: List[int] = [256, 128, 64],
                 proj_dim: int = 64, temperature: float = 0.1,
                 cl_weight: float = 0.1):
        super().__init__(vocab_sizes, embedding_size, hidden_sizes)
        
        input_dim = self.n_fields * embedding_size
        self.projection = ProjectionHead(input_dim, hidden_sizes[0], proj_dim)
        self.temperature = temperature
        self.cl_weight = cl_weight
    
    def contrastive_loss(self, batch) -> torch.Tensor:
        """监督对比损失"""
        embeds = self.get_embed_vector(batch)
        z = self.projection(embeds)
        
        labels = batch['label']
        batch_size = z.shape[0]
        
        # 相似度矩阵
        sim = torch.matmul(z, z.T) / self.temperature  # [B, B]
        
        # 正例 mask：同标签
        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.T).float()
        
        # 移除对角线（自身）
        logits_mask = 1 - torch.eye(batch_size, device=z.device)
        pos_mask = pos_mask * logits_mask
        
        # 如果没有正例，返回 0
        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=z.device)
        
        # 计算损失
        exp_sim = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_sim.sum(1, keepdim=True) + 1e-8)
        
        # 只在有正例的样本上计算
        mask_sum = pos_mask.sum(1)
        valid_mask = mask_sum > 0
        
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=z.device)
        
        mean_log_prob = (pos_mask * log_prob).sum(1) / (mask_sum + 1e-8)
        loss = -mean_log_prob[valid_mask].mean()
        
        return loss
    
    def calculate_loss(self, batch):
        logits = self.forward(batch)
        cvr_loss = F.binary_cross_entropy_with_logits(logits, batch['label'])
        cl_loss = self.contrastive_loss(batch)
        return cvr_loss + self.cl_weight * cl_loss


class DomainCL(DeepFMBase):
    """Domain Contrastive Learning
    
    核心思想：同 domain 样本为正例，不同 domain 为负例
    """
    
    def __init__(self, vocab_sizes: List[int], embedding_size: int = 16,
                 hidden_sizes: List[int] = [256, 128, 64],
                 proj_dim: int = 64, temperature: float = 0.1,
                 cl_weight: float = 0.1):
        super().__init__(vocab_sizes, embedding_size, hidden_sizes)
        
        input_dim = self.n_fields * embedding_size
        self.projection = ProjectionHead(input_dim, hidden_sizes[0], proj_dim)
        self.temperature = temperature
        self.cl_weight = cl_weight
    
    def contrastive_loss(self, batch) -> torch.Tensor:
        """域对比损失"""
        embeds = self.get_embed_vector(batch)
        z = self.projection(embeds)
        
        domain_ids = batch['domain_indicator']
        batch_size = z.shape[0]
        
        # 相似度矩阵
        sim = torch.matmul(z, z.T) / self.temperature
        
        # 正例 mask：同 domain
        domain_ids = domain_ids.view(-1, 1)
        pos_mask = (domain_ids == domain_ids.T).float()
        
        # 移除对角线
        logits_mask = 1 - torch.eye(batch_size, device=z.device)
        pos_mask = pos_mask * logits_mask
        
        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=z.device)
        
        exp_sim = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_sim.sum(1, keepdim=True) + 1e-8)
        
        mask_sum = pos_mask.sum(1)
        valid_mask = mask_sum > 0
        
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=z.device)
        
        mean_log_prob = (pos_mask * log_prob).sum(1) / (mask_sum + 1e-8)
        loss = -mean_log_prob[valid_mask].mean()
        
        return loss
    
    def calculate_loss(self, batch):
        logits = self.forward(batch)
        cvr_loss = F.binary_cross_entropy_with_logits(logits, batch['label'])
        cl_loss = self.contrastive_loss(batch)
        return cvr_loss + self.cl_weight * cl_loss


class FeatureMaskCL(DeepFMBase):
    """Feature Mask Contrastive Learning
    
    核心思想：随机 mask 部分特征，同一样本的不同 mask 视图为正例
    """
    
    def __init__(self, vocab_sizes: List[int], embedding_size: int = 16,
                 hidden_sizes: List[int] = [256, 128, 64],
                 proj_dim: int = 64, temperature: float = 0.1,
                 mask_ratio: float = 0.2, cl_weight: float = 0.1):
        super().__init__(vocab_sizes, embedding_size, hidden_sizes)
        
        input_dim = self.n_fields * embedding_size
        self.projection = ProjectionHead(input_dim, hidden_sizes[0], proj_dim)
        self.temperature = temperature
        self.mask_ratio = mask_ratio
        self.cl_weight = cl_weight
    
    def mask_features(self, embeds: torch.Tensor) -> torch.Tensor:
        """随机 mask 部分特征"""
        # embeds: [B, n_fields, embed_dim]
        mask = (torch.rand(embeds.shape[1], device=embeds.device) > self.mask_ratio).float()
        mask = mask.unsqueeze(0).unsqueeze(-1)  # [1, n_fields, 1]
        return embeds * mask
    
    def contrastive_loss(self, batch) -> torch.Tensor:
        """特征 mask 对比损失"""
        embeds = self.get_embeddings(batch)  # [B, n_fields, embed_dim]
        
        # 两个不同的 mask 视图
        masked1 = self.mask_features(embeds).view(embeds.size(0), -1)
        masked2 = self.mask_features(embeds).view(embeds.size(0), -1)
        
        z1 = self.projection(masked1)
        z2 = self.projection(masked2)
        
        # InfoNCE
        batch_size = z1.shape[0]
        sim = torch.matmul(z1, z2.T) / self.temperature
        labels = torch.arange(batch_size, device=z1.device)
        
        loss = F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)
        return loss / 2
    
    def calculate_loss(self, batch):
        logits = self.forward(batch)
        cvr_loss = F.binary_cross_entropy_with_logits(logits, batch['label'])
        cl_loss = self.contrastive_loss(batch)
        return cvr_loss + self.cl_weight * cl_loss


class DirectAU(DeepFMBase):
    """DirectAU: Alignment + Uniformity
    
    核心思想：
    - Alignment：拉近正样本对
    - Uniformity：使表征在超球面上均匀分布
    """
    
    def __init__(self, vocab_sizes: List[int], embedding_size: int = 16,
                 hidden_sizes: List[int] = [256, 128, 64],
                 proj_dim: int = 64, gamma: float = 1.0,
                 au_weight: float = 0.1):
        super().__init__(vocab_sizes, embedding_size, hidden_sizes)
        
        input_dim = self.n_fields * embedding_size
        self.projection = ProjectionHead(input_dim, hidden_sizes[0], proj_dim)
        self.gamma = gamma
        self.au_weight = au_weight
    
    def alignment_loss(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Alignment loss：正样本对距离最小化（向量化版本）"""
        batch_size = z.shape[0]
        
        # 采样避免 OOM
        sample_size = min(512, batch_size)
        if batch_size > sample_size:
            indices = torch.randperm(batch_size, device=z.device)[:sample_size]
            z = z[indices]
            labels = labels[indices]
            batch_size = sample_size
        
        # 正样本对：同标签
        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.T).float()
        pos_mask.fill_diagonal_(0)  # 移除自身
        
        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=z.device)
        
        # L2 距离矩阵
        dist = torch.cdist(z, z, p=2).pow(2)
        
        # 加权平均
        align_loss = (dist * pos_mask).sum() / (pos_mask.sum() + 1e-8)
        
        return align_loss
    
    def uniformity_loss(self, z: torch.Tensor) -> torch.Tensor:
        """Uniformity loss：表征均匀分布（采样版本避免 OOM）"""
        # 采样部分样本计算
        batch_size = z.shape[0]
        sample_size = min(256, batch_size)  # 最多采样 256 个
        
        if batch_size <= sample_size:
            sampled = z
        else:
            indices = torch.randperm(batch_size, device=z.device)[:sample_size]
            sampled = z[indices]
        
        # Gaussian potential
        return torch.pdist(sampled, p=2).pow(2).mul(-2).exp().mean().log()
    
    def calculate_loss(self, batch):
        logits = self.forward(batch)
        cvr_loss = F.binary_cross_entropy_with_logits(logits, batch['label'])
        
        embeds = self.get_embed_vector(batch)
        z = self.projection(embeds)
        
        align_loss = self.alignment_loss(z, batch['label'])
        uniform_loss = self.uniformity_loss(z)
        
        au_loss = align_loss + self.gamma * uniform_loss
        
        return cvr_loss + self.au_weight * au_loss


# ============================================================
# 训练与评估
# ============================================================

def train_model(model, train_loader, valid_loader, config, device, model_name):
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    best_auc = 0
    best_epoch = 0
    
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        total_cvr_loss = 0
        total_cl_loss = 0
        
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            loss = model.calculate_loss(batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Evaluate
        metrics = evaluate(model, valid_loader, device)
        print(f"[{model_name}] Epoch {epoch+1}/{config['epochs']} - "
              f"Loss: {total_loss/len(train_loader):.4f} - AUC: {metrics['AUC']:.4f}")
        
        if metrics['AUC'] > best_auc:
            best_auc = metrics['AUC']
            best_epoch = epoch + 1
    
    print(f"[{model_name}] Best AUC: {best_auc:.4f} @ epoch {best_epoch}")
    return best_auc


def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels, all_domains = [], [], []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            preds = model.predict(batch)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch['label'].cpu().numpy())
            all_domains.append(batch['domain_indicator'].cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_domains = np.concatenate(all_domains)
    
    results = {
        'AUC': roc_auc_score(all_labels, all_preds),
        'LogLoss': log_loss(all_labels, all_preds),
    }
    
    # Per-domain AUC
    unique_domains = np.unique(all_domains)
    for d in unique_domains:
        mask = (all_domains == d)
        if mask.sum() > 100 and all_labels[mask].var() > 0:
            results[f'domain_{d}_auc'] = roc_auc_score(all_labels[mask], all_preds[mask])
    
    return results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=8192)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--model', type=str, default=None,
                       choices=['deepfm', 'simgcl', 'supcon', 'domain_cl', 'feature_mask', 'directau'])
    parser.add_argument('--compare_all', action='store_true')
    parser.add_argument('--cl_weight', type=float, default=0.1, help='Contrastive loss weight')
    args = parser.parse_args()
    
    print("=" * 70)
    print("IVR SSL 对比学习实验")
    print("=" * 70)
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"CL weight: {args.cl_weight}")
    
    data_root = Path('/mnt/workspace/dataset/ivr_sample_v16_ctcvr_sample')
    
    # 加载词表大小
    with open(data_root / 'meta.json') as f:
        meta = json.load(f)
    with open(data_root / 'vocab_sizes.json') as f:
        vocab_sizes_dict = json.load(f)
    
    feature_cols = [c for c in meta['feature_cols'] if c != 'business_type']
    vocab_sizes = [vocab_sizes_dict[c] for c in feature_cols]
    
    # 加载数据
    print("\n[加载数据]")
    total_start = time.time()
    
    train_ds = IVRSSLDataset(data_root / 'train', max_samples=args.max_samples)
    test_ds = IVRSSLDataset(data_root / 'test', max_samples=args.max_samples // 5 if args.max_samples else None)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False, collate_fn=collate_fn, num_workers=4)
    
    load_time = time.time() - total_start
    print(f"\n数据加载总耗时: {load_time:.2f}s")
    
    # 模型列表
    if args.compare_all:
        models_to_run = ['deepfm', 'simgcl', 'supcon', 'domain_cl', 'feature_mask', 'directau']
    elif args.model:
        models_to_run = [args.model]
    else:
        models_to_run = ['deepfm', 'simgcl', 'supcon', 'domain_cl', 'feature_mask', 'directau']
    
    config = {
        'epochs': args.epochs, 
        'lr': args.lr,
        'cl_weight': args.cl_weight,
    }
    results = {}
    train_times = {}
    
    device = torch.device(args.device)
    
    for model_name in models_to_run:
        print(f"\n{'='*70}")
        print(f"模型: {model_name.upper()}")
        print(f"{'='*70}")
        
        # 构建模型
        if model_name == 'deepfm':
            model = DeepFMBase(vocab_sizes)
        elif model_name == 'simgcl':
            model = SimGCL(vocab_sizes, cl_weight=args.cl_weight)
        elif model_name == 'supcon':
            model = SupCon(vocab_sizes, cl_weight=args.cl_weight)
        elif model_name == 'domain_cl':
            model = DomainCL(vocab_sizes, cl_weight=args.cl_weight)
        elif model_name == 'feature_mask':
            model = FeatureMaskCL(vocab_sizes, cl_weight=args.cl_weight)
        elif model_name == 'directau':
            model = DirectAU(vocab_sizes, au_weight=args.cl_weight)
        
        model = model.to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"参数量: {n_params:,}")
        
        # 训练
        train_start = time.time()
        best_auc = train_model(model, train_loader, test_loader, config, device, model_name.upper())
        train_times[model_name] = time.time() - train_start
        
        # 评估
        print("\n[评估]")
        metrics = evaluate(model, test_loader, device)
        results[model_name] = metrics
        
        print(f"整体 AUC: {metrics['AUC']:.4f}, LogLoss: {metrics['LogLoss']:.4f}")
        
        # 分域 AUC
        print("\n分域 AUC (Top 5):")
        domain_aucs = [(k, v) for k, v in metrics.items() if k.startswith('domain_')]
        domain_aucs.sort(key=lambda x: x[1], reverse=True)
        for k, v in domain_aucs[:5]:
            print(f"  {k}: {v:.4f}")
    
    # 汇总
    total_time = time.time() - total_start
    print("\n" + "=" * 70)
    print("实验汇总")
    print("=" * 70)
    print(f"{'Model':<15} {'AUC':<10} {'LogLoss':<10} {'Train Time':<12}")
    print("-" * 50)
    for model_name in models_to_run:
        m = results[model_name]
        print(f"{model_name.upper():<15} {m['AUC']:<10.4f} {m['LogLoss']:<10.4f} {train_times[model_name]:<12.1f}s")
    print("-" * 50)
    print(f"总耗时: {total_time:.1f}s")
    print("=" * 70)
    
    # 保存报告
    report_dir = Path('/mnt/workspace/git_project/RecForgeLab/reports')
    report_dir.mkdir(exist_ok=True)
    
    report = {
        'experiment': 'ivr_ssl_comparison',
        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'device': args.device,
            'max_samples': args.max_samples,
            'cl_weight': args.cl_weight,
        },
        'dataset': {
            'train_samples': len(train_ds),
            'test_samples': len(test_ds),
            'num_features': len(train_ds.feature_cols),
        },
        'results': results,
        'train_times': train_times,
        'total_time': total_time,
    }
    
    report_file = report_dir / f'ivr_ssl_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n报告已保存: {report_file}")


if __name__ == '__main__':
    main()
