#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IVR v16 数据集 CTCVR 预估实验

数据集路径: /mnt/data/oss_wanjun/pai_work/open_research/dataset/ivr_sample_v16_ctcvr/

用法:
    # 单模型
    python workshop/run_ivr_ctcvr.py --model deepfm --epochs 10

    # 快速测试
    python workshop/run_ivr_ctcvr.py --model deepfm --max_samples 100000 --epochs 3

    # GPU
    python workshop/run_ivr_ctcvr.py --model deepfm --device cuda
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ============================================================
# 数据集
# ============================================================

class IVRDataset(Dataset):
    """IVR v16 数据集"""
    
    def __init__(self, data_path, phase='train', max_samples=None, encoders=None, 
                 string_cols=None, numeric_cols=None):
        print(f"Loading {phase} data from {data_path}...")
        
        # 读取数据
        df = pd.read_parquet(data_path)
        
        if max_samples:
            df = df.head(max_samples)
        
        # 标签
        self.click_labels = df['click_label'].values.astype(np.float32)
        self.ctcvr_labels = df['ctcvr_label'].values.astype(np.float32)
        
        # 特征
        if string_cols is None:
            # 字符串特征 -> 类别特征
            self.string_cols = df.select_dtypes(include=['object']).columns.tolist()
        else:
            self.string_cols = string_cols
        
        if numeric_cols is None:
            # 数值特征
            self.numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            self.numeric_cols = [c for c in self.numeric_cols if c not in ['click_label', 'ctcvr_label']]
        else:
            self.numeric_cols = numeric_cols
        
        # 类别特征编码
        if encoders is None:
            self.encoders = {}
            self.sparse_data = {}
            for col in self.string_cols:
                le = LabelEncoder()
                # 处理缺失值，添加 __UNKNOWN__ 类别
                df[col] = df[col].fillna('__UNKNOWN__')
                # 确保 __UNKNOWN__ 在类别中
                unique_vals = df[col].astype(str).unique().tolist()
                if '__UNKNOWN__' not in unique_vals:
                    unique_vals.append('__UNKNOWN__')
                le.fit(unique_vals)
                self.sparse_data[col] = le.transform(df[col].astype(str)).astype(np.int64)
                self.encoders[col] = le
        else:
            self.encoders = encoders
            self.sparse_data = {}
            for col in self.string_cols:
                df[col] = df[col].fillna('__UNKNOWN__')
                # 将未知类别映射到 __UNKNOWN__ 的索引
                unknown_idx = 0  # __UNKNOWN__ 通常是第一个类别
                col_data = df[col].astype(str)
                # 使用 transform，对于未知类别用 unknown_idx 替代
                result = []
                for val in col_data:
                    if val in self.encoders[col].classes_:
                        result.append(self.encoders[col].transform([val])[0])
                    else:
                        result.append(unknown_idx)
                self.sparse_data[col] = np.array(result, dtype=np.int64)
        
        # 数值特征
        self.dense_data = df[self.numeric_cols].values.astype(np.float32)
        
        # 标准化
        if encoders is None:
            self.dense_mean = self.dense_data.mean(axis=0)
            self.dense_std = self.dense_data.std(axis=0) + 1e-8
        else:
            self.dense_mean = encoders.get('dense_mean', self.dense_data.mean(axis=0))
            self.dense_std = encoders.get('dense_std', self.dense_data.std(axis=0) + 1e-8)
        
        self.dense_data = (self.dense_data - self.dense_mean) / self.dense_std
        
        # 统计信息
        self.n_samples = len(df)
        self.n_sparse = len(self.string_cols)
        self.n_dense = len(self.numeric_cols)
        
        # 词汇表大小
        self.sparse_vocab_sizes = {col: len(enc.classes_) for col, enc in self.encoders.items()}
        
        print(f"  Samples: {self.n_samples:,}")
        print(f"  Sparse features: {self.n_sparse}")
        print(f"  Dense features: {self.n_dense}")
        print(f"  CTR: {self.click_labels.mean():.4f}")
        print(f"  CTCVR: {self.ctcvr_labels.mean():.4f}")
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        sparse = np.array([self.sparse_data[col][idx] for col in self.string_cols])
        
        return {
            'click_label': self.click_labels[idx],
            'ctcvr_label': self.ctcvr_labels[idx],
            'sparse_features': sparse,
            'dense_features': self.dense_data[idx],
        }
    
    def get_encoders(self):
        return {
            'encoders': self.encoders,
            'dense_mean': self.dense_mean,
            'dense_std': self.dense_std,
            'string_cols': self.string_cols,
            'numeric_cols': self.numeric_cols,
        }


def collate_fn(batch):
    """整理 batch"""
    click_labels = torch.tensor([b['click_label'] for b in batch])
    ctcvr_labels = torch.tensor([b['ctcvr_label'] for b in batch])
    sparse = torch.tensor(np.array([b['sparse_features'] for b in batch]))
    dense = torch.tensor(np.array([b['dense_features'] for b in batch]))
    
    return {
        'click_label': click_labels,
        'ctcvr_label': ctcvr_labels,
        'label': ctcvr_labels,  # 主标签
        'sparse_features': sparse,
        'dense_features': dense,
    }


# ============================================================
# DeepFM 模型（简化版）
# ============================================================

class SimpleDeepFM(torch.nn.Module):
    """简化版 DeepFM"""
    
    def __init__(self, sparse_vocab_sizes, n_dense, embedding_size=16, hidden_sizes=[256, 128, 64]):
        super().__init__()
        
        self.n_sparse = len(sparse_vocab_sizes)
        self.n_dense = n_dense
        self.embedding_size = embedding_size
        
        # Sparse embeddings
        self.embeddings = torch.nn.ModuleList([
            torch.nn.Embedding(vocab_size, embedding_size)
            for vocab_size in sparse_vocab_sizes.values()
        ])
        
        # Dense projection
        self.dense_proj = torch.nn.Linear(n_dense, embedding_size * n_dense)
        
        # FM 部分
        fm_input_dim = embedding_size * (self.n_sparse + n_dense)
        
        # Deep 部分
        layers = []
        prev_dim = fm_input_dim
        for h in hidden_sizes:
            layers.append(torch.nn.Linear(prev_dim, h))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(0.2))
            prev_dim = h
        self.deep = torch.nn.Sequential(*layers)
        
        # 输出层
        self.output_layer = torch.nn.Linear(hidden_sizes[-1], 1)
        
        # 初始化
        for emb in self.embeddings:
            torch.nn.init.xavier_uniform_(emb.weight)
    
    def forward(self, batch):
        # Sparse embeddings
        sparse_emb = []
        for i, emb in enumerate(self.embeddings):
            sparse_emb.append(emb(batch['sparse_features'][:, i]))
        sparse_emb = torch.cat(sparse_emb, dim=1)  # (B, n_sparse * emb)
        
        # Dense projection
        dense_emb = self.dense_proj(batch['dense_features'])  # (B, n_dense * emb)
        
        # Concat
        x = torch.cat([sparse_emb, dense_emb], dim=1)  # (B, (n_sparse + n_dense) * emb)
        
        # Deep
        deep_out = self.deep(x)
        
        # FM 一阶
        fm_first = x.sum(dim=1, keepdim=True)  # 简化
        
        # 输出
        logits = self.output_layer(deep_out) + fm_first * 0.1
        
        return logits.squeeze(-1)
    
    def calculate_loss(self, batch):
        logits = self.forward(batch)
        labels = batch['label']
        return torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    
    def predict(self, batch):
        logits = self.forward(batch)
        return torch.sigmoid(logits)
    
    def _print_param_count(self):
        n = sum(p.numel() for p in self.parameters())
        print(f"  Parameters: {n:,}")


# ============================================================
# 评估函数
# ============================================================

def evaluate(model, dataloader, device):
    """评估模型"""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_clicks = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}
            preds = model.predict(batch)
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch['label'].cpu().numpy())
            all_clicks.append(batch['click_label'].cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_clicks = np.concatenate(all_clicks)
    
    # 整体指标
    auc = roc_auc_score(all_labels, all_preds)
    logloss = log_loss(all_labels, all_preds)
    
    # 点击样本指标（CVR）
    click_mask = all_clicks > 0.5
    if click_mask.sum() > 0:
        cvr_auc = roc_auc_score(all_labels[click_mask], all_preds[click_mask])
    else:
        cvr_auc = 0.0
    
    return {
        'AUC': auc,
        'LogLoss': logloss,
        'CVR_AUC': cvr_auc,
    }


# ============================================================
# 训练函数
# ============================================================

def train(model, train_loader, valid_loader, config, device):
    """训练模型"""
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    epochs = config['epochs']
    best_auc = 0
    best_epoch = 0
    patience = 0
    max_patience = config.get('early_stop_patience', 3)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 0
        
        for batch in train_loader:
            batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}
            
            optimizer.zero_grad()
            loss = model.calculate_loss(batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        # 验证
        valid_metrics = evaluate(model, valid_loader, device)
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/n_batches:.4f} - "
              f"Valid AUC: {valid_metrics['AUC']:.4f} - LogLoss: {valid_metrics['LogLoss']:.4f}")
        
        # Early stop
        if valid_metrics['AUC'] > best_auc:
            best_auc = valid_metrics['AUC']
            best_epoch = epoch + 1
            patience = 0
        else:
            patience += 1
            if patience >= max_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    print(f"\nBest AUC: {best_auc:.4f} @ epoch {best_epoch}")
    return best_auc


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='IVR v16 CTCVR Experiment')
    parser.add_argument('--model', type=str, default='deepfm')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--max_samples', type=int, default=None, help='Max samples for quick test')
    args = parser.parse_args()
    
    print("=" * 80)
    print("IVR v16 CTCVR 预估实验")
    print("=" * 80)
    print(f"\n模型: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Device: {args.device}")
    if args.max_samples:
        print(f"Max samples: {args.max_samples:,} (快速测试模式)")
    
    # 数据路径
    data_root = Path('/mnt/data/oss_wanjun/pai_work/open_research/dataset/ivr_sample_v16_ctcvr')
    
    # 加载数据
    print("\n[加载数据]")
    train_data = IVRDataset(
        data_root / 'train',
        phase='train',
        max_samples=args.max_samples,
    )
    
    train_encoders = train_data.get_encoders()
    valid_data = IVRDataset(
        data_root / 'test',
        phase='valid',
        max_samples=args.max_samples // 5 if args.max_samples else None,
        encoders=train_encoders['encoders'],
        string_cols=train_encoders['string_cols'],
        numeric_cols=train_encoders['numeric_cols'],
    )
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size * 2, shuffle=False, collate_fn=collate_fn)
    
    # 模型
    print("\n[构建模型]")
    model = SimpleDeepFM(
        sparse_vocab_sizes=train_data.sparse_vocab_sizes,
        n_dense=train_data.n_dense,
        embedding_size=16,
        hidden_sizes=[256, 128, 64],
    )
    model._print_param_count()
    model = model.to(args.device)
    
    # 训练
    print("\n[训练]")
    config = {
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'early_stop_patience': 3,
    }
    train(model, train_loader, valid_loader, config, args.device)
    
    # 最终评估
    print("\n[评估]")
    test_metrics = evaluate(model, valid_loader, args.device)
    
    print("\n" + "=" * 80)
    print("实验结果")
    print("=" * 80)
    for k, v in test_metrics.items():
        print(f"  {k:<15} = {v:.6f}")
    print("=" * 80)


if __name__ == '__main__':
    main()
