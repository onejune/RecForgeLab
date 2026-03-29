#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Domain 模型对比实验：Ali-CCP 数据集

这是一个业务定制脚本示例，展示了如何复用 RecForgeLab 实验框架
同时实现自定义数据加载和评估逻辑。

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

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from recforgelab.experiment.modes.base import BaseExperiment
from recforgelab.experiment.modes.compare import CompareExperiment
from recforgelab.utils.config import Config
from recforgelab.utils.logger import get_logger, set_color


# ============================================================
# Ali-CCP 数据集
# ============================================================

class AliCCPMultiDomainDataset(Dataset):
    """Ali-CCP Multi-Domain 数据集"""
    
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
        
        # 稀疏特征
        sparse_cols = ['101', '121', '122', '124', '125', '126', '127', '128', '129',
                       '205', '206', '207', '210', '216', '508', '509', '702', '853']
        self.sparse_data = df[sparse_cols].values.astype(np.int64)
        
        # 稠密特征
        dense_cols = ['D109_14', 'D110_14', 'D127_14', 'D150_14', 'D508', 'D509', 'D702', 'D853']
        self.dense_data = df[dense_cols].values.astype(np.float32)
        
        # 统计
        self.num_samples = len(df)
        self.num_sparse = len(sparse_cols)
        self.num_dense = len(dense_cols)
        self.num_domains = 3
        
        # 词汇表大小
        self.sparse_vocab_sizes = {}
        for col in sparse_cols:
            self.sparse_vocab_sizes[col] = int(df[col].max()) + 1
        
        # 归一化
        self.dense_means = self.dense_data.mean(axis=0)
        self.dense_stds = self.dense_data.std(axis=0) + 1e-8
        self.dense_data = (self.dense_data - self.dense_means) / self.dense_stds
        
        print(f"  Samples: {self.num_samples}")
        print(f"  Domain distribution: {np.bincount(self.domain_ids)}")
    
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
    sparse = torch.tensor(np.array([b['sparse_features'] for b in batch]))
    dense = torch.tensor(np.array([b['dense_features'] for b in batch]))
    
    return {
        'label': labels,
        'click_label': labels,
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
        vocab_sizes = list(sparse_vocab_sizes.values())
        for i in range(num_sparse):
            self.embeddings.append(torch.nn.Embedding(vocab_sizes[i], embedding_size))
        
        # 稠密特征
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
    
    def forward(self, batch):
        sparse_emb = []
        for i, emb in enumerate(self.embeddings):
            sparse_emb.append(emb(batch['sparse_features'][:, i]))
        sparse_emb = torch.cat(sparse_emb, dim=1)
        
        dense_emb = self.dense_proj(batch['dense_features'])
        x = torch.cat([sparse_emb, dense_emb], dim=1)
        
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
    """简化版 STAR"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.domain_bn = torch.nn.ModuleList([
            torch.nn.BatchNorm1d(self.input_dim)
            for _ in range(self.num_domains)
        ])
        
        self.domain_mlps = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(self.input_dim, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 1),
            )
            for _ in range(self.num_domains)
        ])
    
    def forward(self, batch):
        sparse_emb = []
        for i, emb in enumerate(self.embeddings):
            sparse_emb.append(emb(batch['sparse_features'][:, i]))
        sparse_emb = torch.cat(sparse_emb, dim=1)
        
        dense_emb = self.dense_proj(batch['dense_features'])
        x = torch.cat([sparse_emb, dense_emb], dim=1)
        
        shared_logits = self.shared_mlp(x)
        
        domain_logits = torch.zeros(x.size(0), 1, device=x.device)
        domain_id = batch['domain_indicator']
        
        for d in range(self.num_domains):
            mask = (domain_id == d)
            if mask.sum() > 0:
                x_d = self.domain_bn[d](x[mask])
                x_d = self.domain_mlps[d](x_d)
                domain_logits[mask] = x_d
        
        logits = shared_logits * domain_logits
        return logits


class SimpleM3oE(SimpleMultiDomainModel):
    """简化版 M3oE"""
    
    def __init__(self, *args, num_experts=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_experts = num_experts
        
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
        
        self.domain_gates = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(self.input_dim, num_experts),
                torch.nn.Softmax(dim=-1)
            )
            for _ in range(self.num_domains)
        ])
        
        self.output_layer = torch.nn.Linear(expert_hidden[-1], 1)
    
    def forward(self, batch):
        sparse_emb = []
        for i, emb in enumerate(self.embeddings):
            sparse_emb.append(emb(batch['sparse_features'][:, i]))
        sparse_emb = torch.cat(sparse_emb, dim=1)
        
        dense_emb = self.dense_proj(batch['dense_features'])
        x = torch.cat([sparse_emb, dense_emb], dim=1)
        
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))
        expert_outputs = torch.stack(expert_outputs, dim=1)
        
        domain_id = batch['domain_indicator']
        final_output = torch.zeros(x.size(0), expert_outputs.size(-1), device=x.device)
        
        for d in range(self.num_domains):
            mask = (domain_id == d)
            if mask.sum() > 0:
                gate = self.domain_gates[d](x[mask])
                weighted = torch.matmul(gate.unsqueeze(1), expert_outputs[mask]).squeeze(1)
                final_output[mask] = weighted
        
        logits = self.output_layer(final_output)
        return logits


# ============================================================
# 自定义实验（复用 BaseExperiment）
# ============================================================

class AliCCPMultiDomainExperiment(BaseExperiment):
    """Ali-CCP Multi-Domain 实验（自定义数据加载）"""
    
    MODE_NAME = "ali_ccp_multi_domain"
    
    def __init__(self, config, model_name, data_path, max_samples=None):
        # 手动设置 config（简化版）
        self.config = config
        self.model_name = model_name
        self.data_path = Path(data_path)
        self.max_samples = max_samples
        
        self.logger = get_logger()
        self.device = torch.device('cpu')
        self.results = {}
    
    def load_data(self):
        """自定义数据加载"""
        self.logger.info(set_color("\n[Loading Ali-CCP Data]", "cyan"))
        
        self.train_ds = AliCCPMultiDomainDataset(
            self.data_path / 'ali_ccp_train.csv', self.max_samples
        )
        self.valid_ds = AliCCPMultiDomainDataset(
            self.data_path / 'ali_ccp_val.csv', self.max_samples
        )
        self.test_ds = AliCCPMultiDomainDataset(
            self.data_path / 'ali_ccp_test.csv', self.max_samples
        )
        
        self.train_loader = DataLoader(
            self.train_ds, batch_size=8192, shuffle=True, collate_fn=collate_fn
        )
        self.valid_loader = DataLoader(
            self.valid_ds, batch_size=16384, shuffle=False, collate_fn=collate_fn
        )
        self.test_loader = DataLoader(
            self.test_ds, batch_size=16384, shuffle=False, collate_fn=collate_fn
        )
        
        self.dataset_info = {
            'num_sparse': self.train_ds.num_sparse,
            'num_dense': self.train_ds.num_dense,
            'sparse_vocab_sizes': self.train_ds.sparse_vocab_sizes,
            'num_domains': self.train_ds.num_domains,
        }
    
    def build_model(self):
        """构建简化模型"""
        self.logger.info(set_color(f"\n[Building Model: {self.model_name}]", "cyan"))
        
        if self.model_name == 'deepfm':
            self.model = SimpleMultiDomainModel(**self.dataset_info)
        elif self.model_name == 'star':
            self.model = SimpleSTAR(**self.dataset_info)
        elif self.model_name == 'm3oe':
            self.model = SimpleM3oE(**self.dataset_info)
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
        
        n = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"  Parameters: {n:,}")
    
    def train(self):
        """训练"""
        self.logger.info(set_color("\n[Training]", "cyan"))
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        epochs = self.config.get("epochs", 3)
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for batch in self.train_loader:
                optimizer.zero_grad()
                loss = self.model.calculate_loss(batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            # 验证
            valid_metrics = self._evaluate(self.valid_loader)
            self.logger.info(f"  Epoch {epoch+1}/{epochs} - "
                           f"Loss: {total_loss/len(self.train_loader):.4f} - "
                           f"Valid AUC: {valid_metrics['AUC']:.4f}")
    
    def evaluate(self):
        """测试评估"""
        self.logger.info(set_color("\n[Evaluating]", "cyan"))
        return self._evaluate(self.test_loader)
    
    def _evaluate(self, dataloader):
        """评估辅助函数"""
        from sklearn.metrics import roc_auc_score, log_loss
        
        self.model.eval()
        all_preds = []
        all_labels = []
        all_domains = []
        
        with torch.no_grad():
            for batch in dataloader:
                preds = self.model.predict(batch)
                all_preds.append(preds.numpy())
                all_labels.append(batch['label'].numpy())
                all_domains.append(batch['domain_indicator'].numpy())
        
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        all_domains = np.concatenate(all_domains)
        
        results = {
            'AUC': roc_auc_score(all_labels, all_preds),
            'LogLoss': log_loss(all_labels, all_preds),
        }
        
        # 分域指标
        for d in range(3):
            mask = (all_domains == d)
            if mask.sum() > 0:
                results[f'domain_{d}_auc'] = roc_auc_score(all_labels[mask], all_preds[mask])
        
        return results
    
    def run(self):
        """运行实验"""
        self.load_data()
        self.build_model()
        self.train()
        results = self.evaluate()
        self.save_results(results)
        return results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Multi-Domain Experiments on Ali-CCP')
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--compare_all', action='store_true')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--max_samples', type=int, default=None)
    args = parser.parse_args()
    
    data_path = '/mnt/data/oss_wanjun/pai_work/open_research/dataset/ali_ccp'
    config = Config(config_dict={'epochs': args.epochs}, parse_cmd=False)
    
    if args.compare_all:
        models = ['deepfm', 'star', 'm3oe']
    elif args.model:
        models = [args.model]
    else:
        models = ['deepfm', 'star', 'm3oe']
    
    results = {}
    for model_name in models:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")
        
        exp = AliCCPMultiDomainExperiment(
            config, model_name, data_path, args.max_samples
        )
        results[model_name] = exp.run()
    
    # 汇总
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print(f"{'Model':<15} {'AUC':<10} {'LogLoss':<10} {'Domain_0_AUC':<15} {'Domain_1_AUC':<15} {'Domain_2_AUC':<15}")
    print("-"*80)
    for model_name, metrics in results.items():
        d0 = metrics.get('domain_0_auc', 0)
        d1 = metrics.get('domain_1_auc', 0)
        d2 = metrics.get('domain_2_auc', 0)
        print(f"{model_name:<15} {metrics['AUC']:<10.4f} {metrics['LogLoss']:<10.4f} "
              f"{d0:<15.4f} {d1:<15.4f} {d2:<15.4f}")
    print("="*80)


if __name__ == '__main__':
    main()
