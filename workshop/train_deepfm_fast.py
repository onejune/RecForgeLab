#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用预处理数据集快速训练 DeepFM
"""

import time
import json
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, log_loss


class FastIVRDataset(Dataset):
    """快速加载的 IVR 数据集"""
    
    def __init__(self, data_path, max_samples=None):
        print(f"Loading {data_path}...")
        start = time.time()
        
        df = pd.read_parquet(data_path)
        if max_samples:
            df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
        
        self.labels = df['ctcvr_label'].values.astype(np.float32)
        
        # 特征列（排除标签）
        feature_cols = [c for c in df.columns if c not in ['click_label', 'ctcvr_label']]
        self.features = df[feature_cols].values.astype(np.int64)
        self.n_features = len(feature_cols)
        
        load_time = time.time() - start
        print(f"  Loaded {len(df):,} samples, {self.n_features} features in {load_time:.2f}s")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'label': self.labels[idx],
        }


def collate_fn(batch):
    features = torch.tensor(np.array([b['features'] for b in batch]))
    labels = torch.tensor([b['label'] for b in batch])
    return {'features': features, 'label': labels}


class DeepFM(torch.nn.Module):
    """DeepFM 模型"""
    
    def __init__(self, vocab_sizes, embedding_size=16, hidden_sizes=[256, 128, 64]):
        super().__init__()
        
        self.n_fields = len(vocab_sizes)
        self.embedding_size = embedding_size
        
        # Embeddings
        self.embeddings = torch.nn.ModuleList([
            torch.nn.Embedding(vocab_size, embedding_size)
            for vocab_size in vocab_sizes.values()
        ])
        
        # FM part
        self.fm_linear = torch.nn.Linear(self.n_fields * embedding_size, 1)
        
        # Deep part
        layers = []
        prev_dim = self.n_fields * embedding_size
        for h in hidden_sizes:
            layers.append(torch.nn.Linear(prev_dim, h))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(0.2))
            prev_dim = h
        self.deep = torch.nn.Sequential(*layers)
        
        # Output
        self.output = torch.nn.Linear(hidden_sizes[-1] + 1, 1)
        
        # Init
        for emb in self.embeddings:
            torch.nn.init.xavier_uniform_(emb.weight)
    
    def forward(self, features):
        # Embeddings
        embs = []
        for i, emb in enumerate(self.embeddings):
            embs.append(emb(features[:, i]))
        embs = torch.cat(embs, dim=1)  # [batch, n_fields * emb_size]
        
        # FM
        fm_out = self.fm_linear(embs)
        
        # Deep
        deep_out = self.deep(embs)
        
        # Combine
        x = torch.cat([fm_out, deep_out], dim=1)
        logits = self.output(x).squeeze(-1)
        return logits
    
    def calculate_loss(self, batch):
        logits = self.forward(batch['features'])
        return torch.nn.functional.binary_cross_entropy_with_logits(logits, batch['label'])
    
    def predict(self, batch):
        logits = self.forward(batch['features'])
        return torch.sigmoid(logits)


def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            preds = model.predict(batch)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch['label'].cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    return {
        'AUC': roc_auc_score(all_labels, all_preds),
        'LogLoss': log_loss(all_labels, all_preds),
    }


def train(model, train_loader, valid_loader, config, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    
    best_auc = 0
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
        print(f"Epoch {epoch+1}/{config['epochs']} - Loss: {total_loss/len(train_loader):.4f} - "
              f"AUC: {metrics['AUC']:.4f} - LogLoss: {metrics['LogLoss']:.4f}")
        
        if metrics['AUC'] > best_auc:
            best_auc = metrics['AUC']
    
    return best_auc


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8192)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--max_samples', type=int, default=None)
    args = parser.parse_args()
    
    print("=" * 60)
    print("DeepFM 训练 (预处理数据集)")
    print("=" * 60)
    
    data_root = Path('/mnt/workspace/dataset/ivr_sample_v16_ctcvr_sample')
    
    # 加载元信息
    with open(data_root / 'vocab_sizes.json') as f:
        vocab_sizes = json.load(f)
    
    # 加载数据
    total_start = time.time()
    
    print("\n[加载数据]")
    train_data = FastIVRDataset(data_root / 'train', max_samples=args.max_samples)
    valid_data = FastIVRDataset(data_root / 'test', max_samples=args.max_samples // 5 if args.max_samples else None)
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size * 2, shuffle=False, collate_fn=collate_fn, num_workers=4)
    
    load_time = time.time() - total_start
    print(f"\n总加载时间: {load_time:.2f}s")
    
    # 构建模型
    print("\n[构建模型]")
    model = DeepFM(vocab_sizes, embedding_size=16, hidden_sizes=[256, 128, 64])
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    model = model.to(args.device)
    
    # 训练
    print("\n[训练]")
    train_start = time.time()
    config = {'epochs': args.epochs, 'lr': args.lr}
    best_auc = train(model, train_loader, valid_loader, config, args.device)
    train_time = time.time() - train_start
    
    print(f"\n训练时间: {train_time:.2f}s")
    print(f"Best AUC: {best_auc:.4f}")
    
    total_time = time.time() - total_start
    print(f"总耗时: {total_time:.2f}s")


if __name__ == '__main__':
    main()
