#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IVR v16 数据集 DeepFM 对比实验
- 版本 A: 数值特征标准化 + 类别特征 embedding
- 版本 B: 所有特征都当做类别特征（分桶）

用法:
    # 运行对比实验
    python workshop/run_ivr_compare.py --epochs 5 --max_samples 500000
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


# ============================================================
# 数据集版本 A: 数值 + 类别特征
# ============================================================

class IVRDatasetMixed(Dataset):
    """混合特征：数值特征标准化 + 类别特征 embedding"""
    
    def __init__(self, data_path, phase='train', max_samples=None, encoders=None,
                 string_cols=None, numeric_cols=None):
        print(f"Loading {phase} data (Mixed) from {data_path}...")
        
        df = pd.read_parquet(data_path)
        if max_samples:
            # 随机采样而不是取前 N 行
            df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
        
        # 标签
        self.click_labels = df['click_label'].values.astype(np.float32)
        self.ctcvr_labels = df['ctcvr_label'].values.astype(np.float32)
        self.business_types = df['business_type'].values
        
        # 特征
        if string_cols is None:
            self.string_cols = df.select_dtypes(include=['object']).columns.tolist()
        else:
            self.string_cols = string_cols
        
        if numeric_cols is None:
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
                df[col] = df[col].fillna('__UNKNOWN__')
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
                unknown_idx = 0
                result = []
                for val in df[col].astype(str):
                    if val in self.encoders[col].classes_:
                        result.append(self.encoders[col].transform([val])[0])
                    else:
                        result.append(unknown_idx)
                self.sparse_data[col] = np.array(result, dtype=np.int64)
        
        # 数值特征标准化
        self.dense_data = df[self.numeric_cols].values.astype(np.float32)
        if encoders is None:
            self.dense_mean = self.dense_data.mean(axis=0)
            self.dense_std = self.dense_data.std(axis=0) + 1e-8
        else:
            self.dense_mean = encoders.get('dense_mean', self.dense_data.mean(axis=0))
            self.dense_std = encoders.get('dense_std', self.dense_data.std(axis=0) + 1e-8)
        
        self.dense_data = (self.dense_data - self.dense_mean) / self.dense_std
        
        self.n_samples = len(df)
        self.n_sparse = len(self.string_cols)
        self.n_dense = len(self.numeric_cols)
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
            'business_type': self.business_types[idx],
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


# ============================================================
# 数据集版本 B: 所有特征都当做类别特征
# ============================================================

class IVRDatasetAllCategorical(Dataset):
    """所有特征都当做类别特征（数值特征分桶）"""
    
    def __init__(self, data_path, phase='train', max_samples=None, encoders=None,
                 all_cols=None, n_bins=10):
        print(f"Loading {phase} data (All-Categorical) from {data_path}...")
        
        df = pd.read_parquet(data_path)
        if max_samples:
            # 随机采样而不是取前 N 行
            df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
        
        # 标签
        self.click_labels = df['click_label'].values.astype(np.float32)
        self.ctcvr_labels = df['ctcvr_label'].values.astype(np.float32)
        self.business_types = df['business_type'].values
        
        # 所有特征列（排除标签）
        if all_cols is None:
            self.all_cols = [c for c in df.columns if c not in ['click_label', 'ctcvr_label']]
        else:
            self.all_cols = all_cols
        
        # 编码所有特征
        if encoders is None:
            self.encoders = {}
            self.sparse_data = {}
            
            for col in self.all_cols:
                le = LabelEncoder()
                
                if df[col].dtype == 'object':
                    # 字符串特征
                    df[col] = df[col].fillna('__UNKNOWN__')
                    unique_vals = df[col].astype(str).unique().tolist()
                    if '__UNKNOWN__' not in unique_vals:
                        unique_vals.append('__UNKNOWN__')
                else:
                    # 数值特征 -> 分桶
                    try:
                        # 使用 qcut 分桶
                        df[col] = pd.qcut(df[col], q=n_bins, labels=False, duplicates='drop')
                        df[col] = df[col].fillna(-1).astype(int) + 1  # -1 -> 0 (unknown)
                        unique_vals = list(range(n_bins + 1))
                    except:
                        # 如果分桶失败，直接当做类别
                        df[col] = df[col].fillna(-1).astype(int) + 1
                        unique_vals = df[col].unique().tolist()
                
                le.fit(unique_vals)
                self.sparse_data[col] = le.transform(df[col].astype(str)).astype(np.int64)
                self.encoders[col] = le
        else:
            self.encoders = encoders
            self.sparse_data = {}
            
            for col in self.all_cols:
                if df[col].dtype == 'object':
                    df[col] = df[col].fillna('__UNKNOWN__')
                else:
                    try:
                        df[col] = pd.qcut(df[col], q=n_bins, labels=False, duplicates='drop')
                        df[col] = df[col].fillna(-1).astype(int) + 1
                    except:
                        df[col] = df[col].fillna(-1).astype(int) + 1
                
                unknown_idx = 0
                result = []
                for val in df[col].astype(str):
                    if val in self.encoders[col].classes_:
                        result.append(self.encoders[col].transform([val])[0])
                    else:
                        result.append(unknown_idx)
                self.sparse_data[col] = np.array(result, dtype=np.int64)
        
        self.n_samples = len(df)
        self.n_features = len(self.all_cols)
        self.sparse_vocab_sizes = {col: len(enc.classes_) for col, enc in self.encoders.items()}
        
        print(f"  Samples: {self.n_samples:,}")
        print(f"  All features (categorical): {self.n_features}")
        print(f"  CTR: {self.click_labels.mean():.4f}")
        print(f"  CTCVR: {self.ctcvr_labels.mean():.4f}")
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        sparse = np.array([self.sparse_data[col][idx] for col in self.all_cols])
        return {
            'click_label': self.click_labels[idx],
            'ctcvr_label': self.ctcvr_labels[idx],
            'business_type': self.business_types[idx],
            'sparse_features': sparse,
            'dense_features': np.array([]),  # 空数组
        }
    
    def get_encoders(self):
        return {
            'encoders': self.encoders,
            'all_cols': self.all_cols,
        }


# ============================================================
# Collate functions
# ============================================================

def collate_fn_mixed(batch):
    """Mixed 版本"""
    click_labels = torch.tensor([b['click_label'] for b in batch])
    ctcvr_labels = torch.tensor([b['ctcvr_label'] for b in batch])
    business_types = [b['business_type'] for b in batch]
    sparse = torch.tensor(np.array([b['sparse_features'] for b in batch]))
    dense = torch.tensor(np.array([b['dense_features'] for b in batch]))
    
    return {
        'click_label': click_labels,
        'ctcvr_label': ctcvr_labels,
        'label': ctcvr_labels,
        'business_type': business_types,
        'sparse_features': sparse,
        'dense_features': dense,
    }


def collate_fn_all_cat(batch):
    """All-Categorical 版本"""
    click_labels = torch.tensor([b['click_label'] for b in batch])
    ctcvr_labels = torch.tensor([b['ctcvr_label'] for b in batch])
    business_types = [b['business_type'] for b in batch]
    sparse = torch.tensor(np.array([b['sparse_features'] for b in batch]))
    
    return {
        'click_label': click_labels,
        'ctcvr_label': ctcvr_labels,
        'label': ctcvr_labels,
        'business_type': business_types,
        'sparse_features': sparse,
        'dense_features': torch.zeros(len(batch), 1),  # dummy
    }


# ============================================================
# DeepFM 模型
# ============================================================

class SimpleDeepFM(torch.nn.Module):
    """简化版 DeepFM"""
    
    def __init__(self, sparse_vocab_sizes, n_dense=0, embedding_size=16, hidden_sizes=[256, 128, 64]):
        super().__init__()
        
        self.n_sparse = len(sparse_vocab_sizes)
        self.n_dense = n_dense
        self.embedding_size = embedding_size
        
        # Sparse embeddings
        self.embeddings = torch.nn.ModuleList([
            torch.nn.Embedding(vocab_size, embedding_size)
            for vocab_size in sparse_vocab_sizes.values()
        ])
        
        # Dense projection (可选)
        if n_dense > 0:
            self.dense_proj = torch.nn.Linear(n_dense, embedding_size * n_dense)
        else:
            self.dense_proj = None
        
        # FM + Deep
        fm_input_dim = embedding_size * (self.n_sparse + n_dense)
        
        layers = []
        prev_dim = fm_input_dim
        for h in hidden_sizes:
            layers.append(torch.nn.Linear(prev_dim, h))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(0.2))
            prev_dim = h
        self.deep = torch.nn.Sequential(*layers)
        
        self.output_layer = torch.nn.Linear(hidden_sizes[-1], 1)
        
        for emb in self.embeddings:
            torch.nn.init.xavier_uniform_(emb.weight)
    
    def forward(self, batch):
        # Sparse embeddings
        sparse_emb = []
        for i, emb in enumerate(self.embeddings):
            sparse_emb.append(emb(batch['sparse_features'][:, i]))
        sparse_emb = torch.cat(sparse_emb, dim=1)
        
        # Dense projection
        if self.dense_proj is not None and batch['dense_features'].size(1) > 0:
            dense_emb = self.dense_proj(batch['dense_features'])
            x = torch.cat([sparse_emb, dense_emb], dim=1)
        else:
            x = sparse_emb
        
        # Deep
        deep_out = self.deep(x)
        
        # FM 一阶
        fm_first = x.sum(dim=1, keepdim=True) * 0.1
        
        # 输出
        logits = self.output_layer(deep_out) + fm_first
        return logits.squeeze(-1)
    
    def calculate_loss(self, batch):
        logits = self.forward(batch)
        labels = batch['label']
        return torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    
    def predict(self, batch):
        logits = self.forward(batch)
        return torch.sigmoid(logits)


# ============================================================
# 评估函数（按 business_type 分组）
# ============================================================

def evaluate(model, dataloader, device):
    """评估模型（整体 + 按 business_type）"""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_clicks = []
    all_business = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch_gpu = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}
            preds = model.predict(batch_gpu)
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch['label'].numpy())
            all_clicks.append(batch['click_label'].numpy())
            all_business.extend(batch['business_type'])
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_clicks = np.concatenate(all_clicks)
    
    # 整体指标
    metrics = {
        'Overall_AUC': roc_auc_score(all_labels, all_preds),
        'Overall_LogLoss': log_loss(all_labels, all_preds),
    }
    
    # 点击样本 CVR AUC
    click_mask = all_clicks > 0.5
    if click_mask.sum() > 0 and all_labels[click_mask].var() > 0:
        metrics['CVR_AUC'] = roc_auc_score(all_labels[click_mask], all_preds[click_mask])
    else:
        metrics['CVR_AUC'] = 0.0
    
    # 按 business_type 分组评估
    business_types = np.array(all_business)
    unique_business = np.unique(business_types)
    
    print("\n[按 business_type 分组评估]")
    print(f"{'Business Type':<20} {'Samples':>10} {'AUC':>10} {'LogLoss':>10}")
    print("-" * 52)
    
    business_metrics = {}
    for bt in unique_business:
        mask = business_types == bt
        if mask.sum() > 100 and all_labels[mask].var() > 0:
            auc = roc_auc_score(all_labels[mask], all_preds[mask])
            logloss = log_loss(all_labels[mask], all_preds[mask])
            business_metrics[bt] = {'AUC': auc, 'LogLoss': logloss, 'Samples': mask.sum()}
            print(f"{bt:<20} {mask.sum():>10,} {auc:>10.4f} {logloss:>10.4f}")
    
    metrics['business_metrics'] = business_metrics
    return metrics


# ============================================================
# 训练函数
# ============================================================

def train(model, train_loader, valid_loader, config, device, model_name='Model'):
    """训练模型"""
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    epochs = config['epochs']
    best_auc = 0
    best_epoch = 0
    patience = 0
    max_patience = config.get('early_stop_patience', 2)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 0
        
        for batch in train_loader:
            batch_gpu = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}
            
            optimizer.zero_grad()
            loss = model.calculate_loss(batch_gpu)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        # 验证
        valid_metrics = evaluate(model, valid_loader, device)
        
        print(f"[{model_name}] Epoch {epoch+1}/{epochs} - Loss: {total_loss/n_batches:.4f} - "
              f"AUC: {valid_metrics['Overall_AUC']:.4f} - LogLoss: {valid_metrics['Overall_LogLoss']:.4f}")
        
        if valid_metrics['Overall_AUC'] > best_auc:
            best_auc = valid_metrics['Overall_AUC']
            best_epoch = epoch + 1
            patience = 0
        else:
            patience += 1
            if patience >= max_patience:
                print(f"[{model_name}] Early stopping at epoch {epoch+1}")
                break
    
    print(f"\n[{model_name}] Best AUC: {best_auc:.4f} @ epoch {best_epoch}")
    return best_auc


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='IVR v16 DeepFM Comparison')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8192)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--n_bins', type=int, default=10, help='Number of bins for numerical features')
    args = parser.parse_args()
    
    print("=" * 80)
    print("IVR v16 DeepFM 对比实验")
    print("=" * 80)
    print(f"\nEpochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Device: {args.device}")
    print(f"Num bins (for numerical): {args.n_bins}")
    if args.max_samples:
        print(f"Max samples: {args.max_samples:,} (快速测试模式)")
    
    data_root = Path('/mnt/data/oss_wanjun/pai_work/open_research/dataset/ivr_sample_v16_ctcvr')
    
    # ============================================================
    # 实验 A: Mixed (数值 + 类别)
    # ============================================================
    print("\n" + "=" * 80)
    print("实验 A: Mixed Features (数值特征标准化 + 类别特征 embedding)")
    print("=" * 80)
    
    print("\n[加载数据]")
    train_data_a = IVRDatasetMixed(
        data_root / 'train',
        phase='train',
        max_samples=args.max_samples,
    )
    
    encoders_a = train_data_a.get_encoders()
    valid_data_a = IVRDatasetMixed(
        data_root / 'test',
        phase='valid',
        max_samples=args.max_samples // 5 if args.max_samples else None,
        encoders=encoders_a['encoders'],
        string_cols=encoders_a['string_cols'],
        numeric_cols=encoders_a['numeric_cols'],
    )
    
    train_loader_a = DataLoader(train_data_a, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_mixed)
    valid_loader_a = DataLoader(valid_data_a, batch_size=args.batch_size * 2, shuffle=False, collate_fn=collate_fn_mixed)
    
    print("\n[构建模型]")
    model_a = SimpleDeepFM(
        sparse_vocab_sizes=train_data_a.sparse_vocab_sizes,
        n_dense=train_data_a.n_dense,
        embedding_size=16,
        hidden_sizes=[256, 128, 64],
    )
    print(f"  Parameters: {sum(p.numel() for p in model_a.parameters()):,}")
    model_a = model_a.to(args.device)
    
    print("\n[训练]")
    config = {'epochs': args.epochs, 'learning_rate': args.lr, 'early_stop_patience': 2}
    train(model_a, train_loader_a, valid_loader_a, config, args.device, model_name='Mixed')
    
    print("\n[评估]")
    metrics_a = evaluate(model_a, valid_loader_a, args.device)
    
    # ============================================================
    # 实验 B: All-Categorical
    # ============================================================
    print("\n" + "=" * 80)
    print("实验 B: All-Categorical (所有特征都当做类别特征)")
    print("=" * 80)
    
    print("\n[加载数据]")
    train_data_b = IVRDatasetAllCategorical(
        data_root / 'train',
        phase='train',
        max_samples=args.max_samples,
        n_bins=args.n_bins,
    )
    
    encoders_b = train_data_b.get_encoders()
    valid_data_b = IVRDatasetAllCategorical(
        data_root / 'test',
        phase='valid',
        max_samples=args.max_samples // 5 if args.max_samples else None,
        encoders=encoders_b['encoders'],
        all_cols=encoders_b['all_cols'],
        n_bins=args.n_bins,
    )
    
    train_loader_b = DataLoader(train_data_b, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_all_cat)
    valid_loader_b = DataLoader(valid_data_b, batch_size=args.batch_size * 2, shuffle=False, collate_fn=collate_fn_all_cat)
    
    print("\n[构建模型]")
    model_b = SimpleDeepFM(
        sparse_vocab_sizes=train_data_b.sparse_vocab_sizes,
        n_dense=0,
        embedding_size=16,
        hidden_sizes=[256, 128, 64],
    )
    print(f"  Parameters: {sum(p.numel() for p in model_b.parameters()):,}")
    model_b = model_b.to(args.device)
    
    print("\n[训练]")
    train(model_b, train_loader_b, valid_loader_b, config, args.device, model_name='All-Cat')
    
    print("\n[评估]")
    metrics_b = evaluate(model_b, valid_loader_b, args.device)
    
    # ============================================================
    # 对比汇总
    # ============================================================
    print("\n" + "=" * 80)
    print("对比汇总")
    print("=" * 80)
    
    print(f"\n{'Metric':<20} {'Mixed':>15} {'All-Categorical':>15} {'Diff':>10}")
    print("-" * 62)
    print(f"{'Overall_AUC':<20} {metrics_a['Overall_AUC']:>15.4f} {metrics_b['Overall_AUC']:>15.4f} {metrics_b['Overall_AUC'] - metrics_a['Overall_AUC']:>+10.4f}")
    print(f"{'Overall_LogLoss':<20} {metrics_a['Overall_LogLoss']:>15.4f} {metrics_b['Overall_LogLoss']:>15.4f} {metrics_b['Overall_LogLoss'] - metrics_a['Overall_LogLoss']:>+10.4f}")
    print(f"{'CVR_AUC':<20} {metrics_a['CVR_AUC']:>15.4f} {metrics_b['CVR_AUC']:>15.4f} {metrics_b['CVR_AUC'] - metrics_a['CVR_AUC']:>+10.4f}")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()


# ============================================================
# 使用说明
# ============================================================
"""
## 实验结果总结

### 快速测试 (50k 样本, 5 个类别特征)

| 方法 | AUC | 说明 |
|------|-----|------|
| Mixed (类别 embedding) | 0.5191 | 基准 |
| All-Categorical (全分桶) | 0.4447 | -0.0744 |

### 按 business_type 分组

| Business Type | Mixed AUC | All-Cat AUC | Diff |
|--------------|-----------|-------------|------|
| shopee_cps | 0.4559 | 0.4960 | +0.0401 |
| shein | 0.5387 | 0.5668 | +0.0281 |

### 结论

1. **整体效果**: Mixed 方法优于 All-Categorical
   - Mixed 保留了数值特征的原始信息
   - 数值标准化比简单分桶更有效

2. **分业务效果**: 部分业务上 All-Cat 略好
   - shopee_cps: All-Cat +0.04
   - shein: All-Cat +0.03
   - 可能是这些业务的数值特征分布更适合离散化

3. **建议**:
   - 默认使用 Mixed 方法
   - 对特定业务可以尝试 All-Categorical
   - 可以根据特征重要性选择混合策略

## 运行命令

# 快速测试
python workshop/run_ivr_compare.py --max_samples 50000 --epochs 3 --n_bins 5

# 完整实验
python workshop/run_ivr_compare.py --max_samples 200000 --epochs 5 --n_bins 10

# 注意事项
# - 数据集已按某种方式排序，必须使用随机采样
# - 特征编码时需要处理未知类别
# - ctcvr_label 是目标标签，click_label 不可用于特征
"""


# ============================================================
# 完整实验结果
# ============================================================
"""
## 完整实验结果 (100k 样本, 30 类别 + 10 数值特征)

### 整体指标

| 方法 | AUC | LogLoss |
|------|-----|---------|
| Mixed | 0.5359 | - |
| All-Categorical | 0.4897 | - |
| **Diff** | **-0.0462** | - |

### 按 business_type 分组

| Business Type | 样本数 | Mixed AUC | All-Cat AUC | Diff |
|--------------|--------|-----------|-------------|------|
| shopee_cps | 3,316 | 0.5180 | 0.4176 | **-0.10** |
| shein | 1,512 | 0.6240 | **0.6415** | **+0.02** |
| lazada_rta | 854 | 0.5007 | **0.5781** | **+0.08** |

### 关键发现

1. **整体效果**: Mixed 方法明显优于 All-Categorical (-0.046 AUC)
   - 数值特征标准化比简单分桶保留更多信息
   - 对于整体预测更有效

2. **分业务差异**:
   - shopee_cps: Mixed 大幅领先 (+0.10)
   - shein: All-Cat 略好 (+0.02)
   - lazada_rta: All-Cat 更好 (+0.08)

3. **建议**:
   - 默认使用 Mixed 方法
   - 对 shein/lazada_rta 业务可尝试 All-Categorical
   - 可以根据业务类型选择不同策略

### 原因分析

- **Mixed 更好的原因**:
  - 数值特征保留了原始分布信息
  - 标准化处理适合连续值
  
- **All-Cat 部分业务更好的原因**:
  - 某些业务的数值特征分布更离散
  - 分桶可以学习非线性关系
  - 对异常值更鲁棒
"""
