#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IVR Multi-Domain 模型对比实验

使用框架的 layers 组件，workshop 脚本定义实验专用的模型类。

重构说明：
- 模型定义仍保留在 workshop，因为框架模型的 config/dataset 接口过于复杂
- 使用框架的 layers（FeatureEmbedding, MLPLayers）复用代码
- 避免重复造轮子，同时保持实验脚本的简洁性

用法:
    python workshop/run_ivr_multi_domain.py --device cuda:1
    python workshop/run_ivr_multi_domain.py --device cuda:1 --compare_all
"""

import time
import json
import argparse
from pathlib import Path
from typing import List, Dict

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, log_loss


# ============================================================
# 数据集
# ============================================================

class IVRMultiDomainDataset(Dataset):
    """IVR Multi-Domain 数据集"""
    
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
        
        # 特征列（排除标签和域标识）
        self.feature_cols = [c for c in df.columns if c not in ['click_label', 'ctcvr_label', 'business_type']]
        self.features = df[self.feature_cols].values.astype(np.int64)
        
        # 域数量
        self.num_domains = int(df['business_type'].max() + 1)
        
        load_time = time.time() - start
        print(f"  Loaded {len(df):,} samples, {len(self.feature_cols)} features, {self.num_domains} domains in {load_time:.2f}s")
    
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
# 模型定义（使用框架设计模式）
# ============================================================

class DeepFM(nn.Module):
    """DeepFM 基线模型"""
    
    def __init__(self, vocab_sizes: List[int], embedding_size: int = 16, 
                 hidden_sizes: List[int] = [256, 128, 64]):
        super().__init__()
        self.n_fields = len(vocab_sizes)
        
        # Embeddings
        self.embeddings = nn.ModuleList([
            nn.Embedding(vs, embedding_size) for vs in vocab_sizes
        ])
        
        # FM linear
        self.fm_linear = nn.Linear(self.n_fields * embedding_size, 1)
        
        # Deep
        layers = []
        prev = self.n_fields * embedding_size
        for h in hidden_sizes:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.2)])
            prev = h
        self.deep = nn.Sequential(*layers)
        
        # Output
        self.output = nn.Linear(hidden_sizes[-1] + 1, 1)
        
        # Init
        for emb in self.embeddings:
            nn.init.xavier_uniform_(emb.weight)
    
    def forward(self, batch):
        embs = [emb(batch['features'][:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat(embs, dim=1)
        
        fm_out = self.fm_linear(x)
        deep_out = self.deep(x)
        logits = self.output(torch.cat([fm_out, deep_out], dim=1)).squeeze(-1)
        return logits
    
    def calculate_loss(self, batch):
        logits = self.forward(batch)
        return nn.functional.binary_cross_entropy_with_logits(logits, batch['label'])
    
    def predict(self, batch):
        return torch.sigmoid(self.forward(batch))


class STAR(nn.Module):
    """STAR: Star Topology Adaptive Recommender
    
    参考: model/multi_domain/star.py
    核心思想：共享中心网络 * 域特定分区网络
    """
    
    def __init__(self, vocab_sizes: List[int], num_domains: int, 
                 embedding_size: int = 16, hidden_sizes: List[int] = [256, 128, 64]):
        super().__init__()
        self.n_fields = len(vocab_sizes)
        self.num_domains = num_domains
        
        # Embeddings
        self.embeddings = nn.ModuleList([
            nn.Embedding(vs, embedding_size) for vs in vocab_sizes
        ])
        
        input_dim = self.n_fields * embedding_size
        
        # 共享 FCN
        self.shared_fcn = nn.Sequential(
            nn.Linear(input_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], 1),
        )
        
        # 域特定 BN + MLP
        self.domain_bn = nn.ModuleList([
            nn.BatchNorm1d(input_dim) for _ in range(num_domains)
        ])
        self.domain_mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            ) for _ in range(num_domains)
        ])
        
        # 辅助网络
        self.aux_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        
        # Init
        for emb in self.embeddings:
            nn.init.xavier_uniform_(emb.weight)
    
    def forward(self, batch):
        # Embeddings
        embs = [emb(batch['features'][:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat(embs, dim=1)
        
        # 共享输出
        shared_out = self.shared_fcn(x)
        
        # 域特定输出
        domain_id = batch['domain_indicator']
        domain_out = torch.zeros(x.size(0), 1, device=x.device)
        
        for d in range(self.num_domains):
            mask = (domain_id == d)
            if mask.sum() > 1:
                x_d = self.domain_bn[d](x[mask])
                domain_out[mask] = self.domain_mlp[d](x_d)
            elif mask.sum() == 1:
                domain_out[mask] = self.domain_mlp[d](x[mask])
        
        # Star topology: 相乘
        logits = (shared_out * domain_out).squeeze(-1)
        
        # 辅助网络
        aux_out = self.aux_net(x).squeeze(-1)
        logits = logits + aux_out
        
        return logits
    
    def calculate_loss(self, batch):
        logits = self.forward(batch)
        return nn.functional.binary_cross_entropy_with_logits(logits, batch['label'])
    
    def predict(self, batch):
        return torch.sigmoid(self.forward(batch))


class M3oE(nn.Module):
    """M3oE: Multi-Domain Mixture of Experts
    
    参考: model/multi_domain/m3oe.py
    核心思想：多个专家 + 域门控路由
    """
    
    def __init__(self, vocab_sizes: List[int], num_domains: int,
                 num_experts: int = 4, embedding_size: int = 16,
                 expert_hidden: List[int] = [128, 64]):
        super().__init__()
        self.n_fields = len(vocab_sizes)
        self.num_domains = num_domains
        self.num_experts = num_experts
        
        # Embeddings
        self.embeddings = nn.ModuleList([
            nn.Embedding(vs, embedding_size) for vs in vocab_sizes
        ])
        
        input_dim = self.n_fields * embedding_size
        
        # 专家网络
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_hidden[0]),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(expert_hidden[0], expert_hidden[1]),
                nn.ReLU(),
            ) for _ in range(num_experts)
        ])
        
        # 域门控
        self.domain_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, num_experts),
                nn.Softmax(dim=-1)
            ) for _ in range(num_domains)
        ])
        
        # 输出层
        self.output = nn.Linear(expert_hidden[-1], 1)
        
        # Init
        for emb in self.embeddings:
            nn.init.xavier_uniform_(emb.weight)
    
    def forward(self, batch):
        # Embeddings
        embs = [emb(batch['features'][:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat(embs, dim=1)
        
        # 专家输出
        expert_outs = [expert(x) for expert in self.experts]
        expert_outs = torch.stack(expert_outs, dim=1)  # [B, num_experts, expert_dim]
        
        # 域门控加权
        domain_id = batch['domain_indicator']
        final_out = torch.zeros(x.size(0), expert_outs.size(-1), device=x.device)
        
        for d in range(self.num_domains):
            mask = (domain_id == d)
            if mask.sum() > 0:
                gate = self.domain_gates[d](x[mask])  # [mask_sum, num_experts]
                weighted = torch.bmm(gate.unsqueeze(1), expert_outs[mask]).squeeze(1)
                final_out[mask] = weighted
        
        logits = self.output(final_out).squeeze(-1)
        return logits
    
    def calculate_loss(self, batch):
        logits = self.forward(batch)
        return nn.functional.binary_cross_entropy_with_logits(logits, batch['label'])
    
    def predict(self, batch):
        return torch.sigmoid(self.forward(batch))


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
    
    # Per-domain metrics
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
    parser.add_argument('--model', type=str, default=None, choices=['deepfm', 'star', 'm3oe'])
    parser.add_argument('--compare_all', action='store_true')
    args = parser.parse_args()
    
    print("=" * 70)
    print("IVR Multi-Domain 模型对比实验")
    print("=" * 70)
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    
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
    
    train_ds = IVRMultiDomainDataset(data_root / 'train', max_samples=args.max_samples)
    test_ds = IVRMultiDomainDataset(data_root / 'test', max_samples=args.max_samples // 5 if args.max_samples else None)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False, collate_fn=collate_fn, num_workers=4)
    
    num_domains = train_ds.num_domains
    load_time = time.time() - total_start
    print(f"\n数据加载总耗时: {load_time:.2f}s")
    
    # 模型列表
    if args.compare_all:
        models_to_run = ['deepfm', 'star', 'm3oe']
    elif args.model:
        models_to_run = [args.model]
    else:
        models_to_run = ['deepfm', 'star', 'm3oe']
    
    config = {'epochs': args.epochs, 'lr': args.lr}
    results = {}
    train_times = {}
    
    device = torch.device(args.device)
    
    for model_name in models_to_run:
        print(f"\n{'='*70}")
        print(f"模型: {model_name.upper()}")
        print(f"{'='*70}")
        
        # 构建模型
        if model_name == 'deepfm':
            model = DeepFM(vocab_sizes)
        elif model_name == 'star':
            model = STAR(vocab_sizes, num_domains)
        elif model_name == 'm3oe':
            model = M3oE(vocab_sizes, num_domains)
        
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
        
        # 打印主要域的 AUC
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
    print(f"{'Model':<10} {'AUC':<10} {'LogLoss':<10} {'Train Time':<12}")
    print("-" * 50)
    for model_name in models_to_run:
        m = results[model_name]
        print(f"{model_name.upper():<10} {m['AUC']:<10.4f} {m['LogLoss']:<10.4f} {train_times[model_name]:<12.1f}s")
    print("-" * 50)
    print(f"总耗时: {total_time:.1f}s")
    print("=" * 70)
    
    # 保存报告
    report_dir = Path('/mnt/workspace/git_project/RecForgeLab/reports')
    report_dir.mkdir(exist_ok=True)
    
    report = {
        'experiment': 'ivr_multi_domain_comparison',
        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'device': args.device,
            'max_samples': args.max_samples,
        },
        'dataset': {
            'train_samples': len(train_ds),
            'test_samples': len(test_ds),
            'num_features': len(train_ds.feature_cols),
            'num_domains': train_ds.num_domains,
        },
        'results': results,
        'train_times': train_times,
        'total_time': total_time,
    }
    
    report_file = report_dir / f'ivr_multi_domain_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n报告已保存: {report_file}")


if __name__ == '__main__':
    main()
