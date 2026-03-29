#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Domain 模型对比实验：Ali-CCP 数据集

用法:
    python experiments/run_multi_domain_ali_ccp.py --model star
    python experiments/run_multi_domain_ali_ccp.py --compare_all
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================
# 数据集
# ============================================================

class AliCCPMultiDomainDataset(Dataset):
    """Ali-CCP Multi-Domain 数据集
    
    domain_indicator 来自列 '301'，映射为 {1:0, 2:1, 3:2}
    """
    
    def __init__(self, csv_path, max_samples=None):
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        if max_samples:
            df = df.head(max_samples)
        
        # 标签
        self.labels = df['click'].values.astype(np.float32)
        
        # 域标识: 301 列 -> domain_indicator
        domain_map = {1: 0, 2: 1, 3: 2}
        self.domain_ids = df['301'].map(domain_map).values.astype(np.int64)
        
        # 稀疏特征（整数）
        sparse_cols = ['101', '121', '122', '124', '125', '126', '127', '128', '129',
                       '205', '206', '207', '210', '216', '508', '509', '702', '853']
        self.sparse_data = df[sparse_cols].values.astype(np.int64)
        
        # 稠密特征（浮点）
        dense_cols = ['D109_14', 'D110_14', 'D127_14', 'D150_14', 'D508', 'D509', 'D702', 'D853']
        self.dense_data = df[dense_cols].values.astype(np.float32)
        
        # 统计信息
        self.num_samples = len(df)
        self.num_sparse = len(sparse_cols)
        self.num_dense = len(dense_cols)
        self.num_domains = 3
        
        # 词汇表大小
        self.sparse_vocab_sizes = {}
        for col in sparse_cols:
            self.sparse_vocab_sizes[col] = int(df[col].max()) + 1
        
        # 归一化稠密特征
        self.dense_means = self.dense_data.mean(axis=0)
        self.dense_stds = self.dense_data.std(axis=0) + 1e-8
        self.dense_data = (self.dense_data - self.dense_means) / self.dense_stds
        
        print(f"  Samples: {self.num_samples}")
        print(f"  Sparse features: {self.num_sparse}, Dense features: {self.num_dense}")
        print(f"  Domain distribution: {np.bincount(self.domain_ids)}")
        print(f"  Label distribution: click={self.labels.mean():.4f}")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return {
            'label': self.labels[idx],
            'domain_indicator': self.domain_ids[idx],
            'sparse_features': self.sparse_data[idx],
            'dense_features': self.dense_data[idx],
        }


def collate_fn(batch):
    """整理 batch"""
    labels = torch.tensor([b['label'] for b in batch])
    domain_ids = torch.tensor([b['domain_indicator'] for b in batch])
    sparse = torch.tensor([b['sparse_features'] for b in batch])
    dense = torch.tensor([b['dense_features'] for b in batch])
    
    return {
        'label': labels,
        'click_label': labels,  # 兼容多任务命名
        'domain_indicator': domain_ids,
        'sparse_features': sparse,
        'dense_features': dense,
    }


# ============================================================
# 简化版模型（适配数据格式）
# ============================================================

class SimpleMultiDomainModel(torch.nn.Module):
    """简化版 Multi-Domain 模型基类"""
    
    def __init__(self, num_sparse, num_dense, sparse_vocab_sizes, 
                 num_domains, embedding_size=16, hidden_sizes=[256, 128, 64]):
        super().__init__()
        self.num_sparse = num_sparse
        self.num_dense = num_dense
        self.num_domains = num_domains
        self.embedding_size = embedding_size
        
        # 稀疏特征嵌入
        self.embeddings = torch.nn.ModuleList()
        for i in range(num_sparse):
            vocab_size = list(sparse_vocab_sizes.values())[i]
            self.embeddings.append(
                torch.nn.Embedding(vocab_size, embedding_size)
            )
        
        # 稠密特征处理
        self.dense_proj = torch.nn.Linear(num_dense, embedding_size * num_dense)
        
        # 输入维度
        self.input_dim = embedding_size * (num_sparse + num_dense)
        
        # 共享 MLP
        layers = []
        prev_dim = self.input_dim
        for h in hidden_sizes:
            layers.append(torch.nn.Linear(prev_dim, h))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(0.2))
            prev_dim = h
        layers.append(torch.nn.Linear(prev_dim, 1))
        self.shared_mlp = torch.nn.Sequential(*layers)
        
        # 域特定输出
        self.domain_towers = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(1, 1),
            )
            for _ in range(num_domains)
        ])
    
    def forward(self, batch):
        # 稀疏嵌入
        sparse_emb = []
        for i, emb in enumerate(self.embeddings):
            sparse_emb.append(emb(batch['sparse_features'][:, i]))
        sparse_emb = torch.cat(sparse_emb, dim=1)  # [B, num_sparse * emb]
        
        # 稠密特征
        dense_emb = self.dense_proj(batch['dense_features'])  # [B, num_dense * emb]
        
        # 拼接
        x = torch.cat([sparse_emb, dense_emb], dim=1)
        
        # 共享 MLP
        logits = self.shared_mlp(x)
        
        return logits
    
    def calculate_loss(self, batch):
        logits = self.forward(batch)
        labels = batch['label'].unsqueeze(-1)
        return torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    
    def predict(self, batch):
        logits = self.forward(batch)
        return torch.sigmoid(logits).squeeze(-1)


class SimpleSTAR(SimpleMultiDomainModel):
    """简化版 STAR 模型"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 域特定 BN
        self.domain_bn = torch.nn.ModuleList([
            torch.nn.BatchNorm1d(self.input_dim)
            for _ in range(self.num_domains)
        ])
        
        # 域特定 MLP
        self.domain_mlps = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(self.input_dim, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 1),
            )
            for _ in range(self.num_domains)
        ])
    
    def forward(self, batch):
        # 稀疏嵌入
        sparse_emb = []
        for i, emb in enumerate(self.embeddings):
            sparse_emb.append(emb(batch['sparse_features'][:, i]))
        sparse_emb = torch.cat(sparse_emb, dim=1)
        
        # 稠密特征
        dense_emb = self.dense_proj(batch['dense_features'])
        
        # 拼接
        x = torch.cat([sparse_emb, dense_emb], dim=1)
        
        # 共享 MLP（所有域共享）
        shared_logits = self.shared_mlp(x)
        
        # 域特定处理
        domain_logits = torch.zeros(x.size(0), 1, device=x.device)
        domain_id = batch['domain_indicator']
        
        for d in range(self.num_domains):
            mask = (domain_id == d)
            if mask.sum() > 0:
                x_d = self.domain_bn[d](x[mask])
                x_d = self.domain_mlps[d](x_d)
                domain_logits[mask] = x_d
        
        # STAR: 共享 * 域特定
        logits = shared_logits * domain_logits
        
        return logits


class SimpleM3oE(SimpleMultiDomainModel):
    """简化版 M3oE 模型"""
    
    def __init__(self, *args, num_experts=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_experts = num_experts
        
        # 专家网络
        expert_hidden = [128, 64]
        self.experts = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(self.input_dim, expert_hidden[0]),
                torch.nn.ReLU(),
                torch.nn.Linear(expert_hidden[0], expert_hidden[1]),
                torch.nn.ReLU(),
            )
            for _ in range(num_experts)
        ])
        
        # 域门控
        self.domain_gates = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(self.input_dim, num_experts),
                torch.nn.Softmax(dim=-1)
            )
            for _ in range(self.num_domains)
        ])
        
        # 输出层
        self.output_layer = torch.nn.Linear(expert_hidden[-1], 1)
    
    def forward(self, batch):
        # 稀疏嵌入
        sparse_emb = []
        for i, emb in enumerate(self.embeddings):
            sparse_emb.append(emb(batch['sparse_features'][:, i]))
        sparse_emb = torch.cat(sparse_emb, dim=1)
        
        # 稠密特征
        dense_emb = self.dense_proj(batch['dense_features'])
        
        # 拼接
        x = torch.cat([sparse_emb, dense_emb], dim=1)
        
        # 专家输出
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [B, num_experts, dim]
        
        # 域门控加权
        domain_id = batch['domain_indicator']
        final_output = torch.zeros(x.size(0), expert_outputs.size(-1), device=x.device)
        
        for d in range(self.num_domains):
            mask = (domain_id == d)
            if mask.sum() > 0:
                gate = self.domain_gates[d](x[mask])  # [N, num_experts]
                weighted = torch.matmul(gate.unsqueeze(1), expert_outputs[mask]).squeeze(1)
                final_output[mask] = weighted
        
        logits = self.output_layer(final_output)
        
        return logits


# ============================================================
# 训练和评估
# ============================================================

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        
        optimizer.zero_grad()
        loss = model.calculate_loss(batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_domains = []
    
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
    
    return all_preds, all_labels, all_domains


def compute_metrics(preds, labels, domains):
    """计算指标"""
    from sklearn.metrics import roc_auc_score, log_loss
    
    # 整体指标
    auc = roc_auc_score(labels, preds)
    logloss = log_loss(labels, preds, eps=1e-7)
    
    # 分域指标
    domain_metrics = {}
    for d in range(3):
        mask = (domains == d)
        if mask.sum() > 0:
            d_auc = roc_auc_score(labels[mask], preds[mask])
            d_logloss = log_loss(labels[mask], preds[mask], eps=1e-7)
            domain_metrics[f'domain_{d}'] = {'auc': d_auc, 'logloss': d_logloss}
    
    return {
        'AUC': auc,
        'LogLoss': logloss,
        'domains': domain_metrics,
    }


def run_experiment(model_name, train_loader, valid_loader, test_loader, 
                   dataset_info, device, epochs=5, lr=0.001):
    """运行单个实验"""
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")
    
    # 创建模型
    if model_name == 'deepfm':
        model = SimpleMultiDomainModel(
            dataset_info['num_sparse'], dataset_info['num_dense'],
            dataset_info['sparse_vocab_sizes'], dataset_info['num_domains']
        )
    elif model_name == 'star':
        model = SimpleSTAR(
            dataset_info['num_sparse'], dataset_info['num_dense'],
            dataset_info['sparse_vocab_sizes'], dataset_info['num_domains']
        )
    elif model_name == 'm3oe':
        model = SimpleM3oE(
            dataset_info['num_sparse'], dataset_info['num_dense'],
            dataset_info['sparse_vocab_sizes'], dataset_info['num_domains']
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # 训练
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        
        # 验证
        preds, labels, domains = evaluate(model, valid_loader, device)
        valid_metrics = compute_metrics(preds, labels, domains)
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f} - "
              f"Valid AUC: {valid_metrics['AUC']:.4f}")
    
    # 测试
    preds, labels, domains = evaluate(model, test_loader, device)
    test_metrics = compute_metrics(preds, labels, domains)
    
    print(f"\nTest Results:")
    print(f"  Overall: AUC={test_metrics['AUC']:.4f}, LogLoss={test_metrics['LogLoss']:.4f}")
    for d, m in test_metrics['domains'].items():
        print(f"  {d}: AUC={m['auc']:.4f}, LogLoss={m['logloss']:.4f}")
    
    return test_metrics


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Multi-Domain Experiments on Ali-CCP')
    parser.add_argument('--model', type=str, default=None, help='Model name')
    parser.add_argument('--compare_all', action='store_true', help='Compare all models')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--max_samples', type=int, default=None, help='Max samples for quick test')
    args = parser.parse_args()
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 数据路径
    data_path = Path('/mnt/data/oss_wanjun/pai_work/open_research/dataset/ali_ccp')
    
    # 加载数据
    print("\nLoading Ali-CCP dataset...")
    train_ds = AliCCPMultiDomainDataset(data_path / 'ali_ccp_train.csv', args.max_samples)
    valid_ds = AliCCPMultiDomainDataset(data_path / 'ali_ccp_val.csv', args.max_samples)
    test_ds = AliCCPMultiDomainDataset(data_path / 'ali_ccp_test.csv', args.max_samples)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size * 2, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False, collate_fn=collate_fn)
    
    dataset_info = {
        'num_sparse': train_ds.num_sparse,
        'num_dense': train_ds.num_dense,
        'sparse_vocab_sizes': train_ds.sparse_vocab_sizes,
        'num_domains': train_ds.num_domains,
    }
    
    # 运行实验
    results = {}
    
    if args.compare_all:
        models = ['deepfm', 'star', 'm3oe']
    elif args.model:
        models = [args.model]
    else:
        models = ['deepfm', 'star', 'm3oe']
    
    for model_name in models:
        results[model_name] = run_experiment(
            model_name, train_loader, valid_loader, test_loader,
            dataset_info, device, epochs=args.epochs, lr=args.lr
        )
    
    # 汇总结果
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print(f"{'Model':<15} {'AUC':<10} {'LogLoss':<10} {'Domain_0_AUC':<15} {'Domain_1_AUC':<15} {'Domain_2_AUC':<15}")
    print("-"*80)
    for model_name, metrics in results.items():
        d0 = metrics['domains'].get('domain_0', {}).get('auc', 0)
        d1 = metrics['domains'].get('domain_1', {}).get('auc', 0)
        d2 = metrics['domains'].get('domain_2', {}).get('auc', 0)
        print(f"{model_name:<15} {metrics['AUC']:<10.4f} {metrics['LogLoss']:<10.4f} "
              f"{d0:<15.4f} {d1:<15.4f} {d2:<15.4f}")
    print("="*80)


if __name__ == '__main__':
    main()
