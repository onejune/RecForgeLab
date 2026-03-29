# -*- coding: utf-8 -*-
"""
SSL 对比学习模型
包含多种对比学习方法用于 CVR 预估增强
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np


class InfoNCELoss(nn.Module):
    """InfoNCE 对比损失
    
    L = -log(exp(sim(z_i, z_j)/τ) / Σ_k exp(sim(z_i, z_k)/τ))
    """
    
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negatives: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            anchor: [batch, embed_dim] 锚点表征
            positive: [batch, embed_dim] 正样本表征
            negatives: [batch, num_neg, embed_dim] 负样本表征
        
        Returns:
            InfoNCE loss
        """
        batch_size = anchor.shape[0]
        
        # 正样本相似度
        pos_sim = F.cosine_similarity(anchor, positive) / self.temperature  # [batch]
        
        # 负样本相似度
        neg_sim = torch.einsum('bd,bnd->bn', anchor, negatives) / self.temperature  # [batch, num_neg]
        
        # logits: [batch, 1 + num_neg]
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        
        # 标签: 第 0 个是正样本
        labels = torch.zeros(batch_size, dtype=torch.long, device=anchor.device)
        
        return F.cross_entropy(logits, labels)


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss
    
    支持同一类别内有多个正样本
    """
    
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            features: [batch, embed_dim]
            labels: [batch] 类别标签
        
        Returns:
            SupCon loss
        """
        batch_size = features.shape[0]
        
        # 归一化
        features = F.normalize(features, dim=1)
        
        # 相似度矩阵
        sim_matrix = torch.matmul(features, features.T) / self.temperature  # [batch, batch]
        
        # 正样本掩码（同一标签为正样本）
        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.T).float()  # [batch, batch]
        
        # 移除对角线（自身）
        pos_mask.fill_diagonal_(0)
        
        # 负样本掩码
        neg_mask = 1 - pos_mask
        neg_mask.fill_diagonal_(0)
        
        # 计算损失
        exp_sim = torch.exp(sim_matrix)
        exp_sim.fill_diagonal_(0)  # 移除自身
        
        # 正样本相似度
        pos_sim = (exp_sim * pos_mask).sum(dim=1)
        
        # 所有样本相似度
        all_sim = exp_sim.sum(dim=1)
        
        # 避免除零
        pos_count = pos_mask.sum(dim=1).clamp(min=1)
        
        # 损失
        loss = -torch.log(pos_sim / all_sim + 1e-8) / pos_count
        loss = loss[pos_count > 0].mean()  # 只计算有正样本的
        
        return loss


class SSLContrastive(nn.Module):
    """对比学习增强的 CVR 模型
    
    两阶段训练:
    1. SSL 预训练：对比学习训练 embedding
    2. CVR 微调：只训练 CVR head
    """
    
    def __init__(
        self,
        embedding_layer: nn.Module,
        embed_dim: int = 128,
        projection_dim: int = 64,
        hidden_dims: list = [256, 128],
        dropout: float = 0.1,
        temperature: float = 0.1,
    ):
        super().__init__()
        
        self.embedding = embedding_layer
        self.embed_dim = embed_dim
        
        # Projection head (for contrastive learning)
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, projection_dim),
        )
        
        # CVR prediction head
        layers = []
        prev_dim = embed_dim
        for hidden in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden
        layers.append(nn.Linear(prev_dim, 1))
        self.cvr_head = nn.Sequential(*layers)
        
        # Loss
        self.contrastive_loss = InfoNCELoss(temperature)
    
    def encode(self, batch: Dict) -> torch.Tensor:
        """编码为 embedding"""
        return self.embedding(batch)
    
    def project(self, embed: torch.Tensor) -> torch.Tensor:
        """投影到对比学习空间"""
        return self.projection(embed)
    
    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """前向传播"""
        embed = self.encode(batch)
        cvr_pred = self.cvr_head(embed)
        return {
            "embed": embed,
            "cvr_pred": cvr_pred,
        }
    
    def predict(self, batch: Dict) -> torch.Tensor:
        """预测 CVR"""
        embed = self.encode(batch)
        return torch.sigmoid(self.cvr_head(embed)).squeeze(-1)
    
    def ssl_loss(
        self,
        batch1: Dict,
        batch2: Dict,
        hard_negatives: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """对比学习损失
        
        Args:
            batch1: 第一个视图
            batch2: 第二个视图（增强后的）
            hard_negatives: 硬负样本 [batch, num_neg, embed_dim]
        """
        # 编码
        embed1 = self.encode(batch1)
        embed2 = self.encode(batch2)
        
        # 投影
        proj1 = self.project(embed1)
        proj2 = self.project(embed2)
        
        # 负样本（batch 内其他样本）
        batch_size = proj1.shape[0]
        
        # 构建 negatives: [batch, num_neg, proj_dim]
        # 简单实现：使用 batch 内其他样本作为负样本
        neg_mask = ~torch.eye(batch_size, dtype=torch.bool, device=proj1.device)
        negatives = proj2.unsqueeze(1).expand(-1, batch_size, -1)[neg_mask]
        negatives = negatives.view(batch_size, batch_size - 1, -1)
        
        return self.contrastive_loss(proj1, proj2, negatives)
    
    def cvr_loss(self, batch: Dict) -> torch.Tensor:
        """CVR 预测损失"""
        pred = self.forward(batch)["cvr_pred"].squeeze(-1)
        label = batch.get("cvr_label", batch.get("label")).float()
        return F.binary_cross_entropy_with_logits(pred, label)


class SSLMomentumContrastive(nn.Module):
    """Momentum Contrast (MoCo) for CVR
    
    使用动量编码器和队列提高负样本多样性
    """
    
    def __init__(
        self,
        embedding_layer: nn.Module,
        embed_dim: int = 128,
        projection_dim: int = 64,
        hidden_dims: list = [256, 128],
        queue_size: int = 65536,
        momentum: float = 0.999,
        temperature: float = 0.07,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.encoder_q = embedding_layer  # Query encoder
        self.encoder_k = self._copy_encoder(embedding_layer)  # Key encoder (momentum)
        self.embed_dim = embed_dim
        self.momentum = momentum
        
        # Projection heads
        self.projection_q = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, projection_dim),
        )
        self.projection_k = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, projection_dim),
        )
        
        # CVR head
        layers = []
        prev_dim = embed_dim
        for hidden in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden
        layers.append(nn.Linear(prev_dim, 1))
        self.cvr_head = nn.Sequential(*layers)
        
        # Queue for negatives
        self.register_buffer("queue", torch.randn(projection_dim, queue_size))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
        self.temperature = temperature
    
    def _copy_encoder(self, encoder: nn.Module) -> nn.Module:
        """复制编码器"""
        import copy
        return copy.deepcopy(encoder)
    
    @torch.no_grad()
    def _momentum_update(self):
        """动量更新 key encoder"""
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1 - self.momentum)
        
        for param_q, param_k in zip(self.projection_q.parameters(), self.projection_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1 - self.momentum)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        """更新队列"""
        batch_size = keys.shape[0]
        queue_size = self.queue.shape[1]
        
        ptr = int(self.queue_ptr)
        
        # 替换队列中的元素
        if ptr + batch_size > queue_size:
            # 分成两次
            first_part = queue_size - ptr
            self.queue[:, ptr:] = keys[:first_part].T
            self.queue[:, :batch_size - first_part] = keys[first_part:].T
            ptr = batch_size - first_part
        else:
            self.queue[:, ptr:ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % queue_size
        
        self.queue_ptr[0] = ptr
    
    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """前向传播"""
        embed_q = self.encoder_q(batch)
        cvr_pred = self.cvr_head(embed_q)
        return {
            "embed": embed_q,
            "cvr_pred": cvr_pred,
        }
    
    def predict(self, batch: Dict) -> torch.Tensor:
        """预测 CVR"""
        embed = self.encoder_q(batch)
        return torch.sigmoid(self.cvr_head(embed)).squeeze(-1)
    
    def ssl_loss(self, batch: Dict) -> torch.Tensor:
        """MoCo contrastive loss"""
        # Query
        embed_q = self.encoder_q(batch)
        proj_q = self.projection_q(embed_q)  # [batch, proj_dim]
        proj_q = F.normalize(proj_q, dim=1)
        
        # Key (momentum encoder)
        with torch.no_grad():
            self._momentum_update()
            embed_k = self.encoder_k(batch)
            proj_k = self.projection_k(embed_k)  # [batch, proj_dim]
            proj_k = F.normalize(proj_k, dim=1)
        
        # 正样本相似度
        l_pos = torch.einsum('bd,bd->b', proj_q, proj_k).unsqueeze(1)  # [batch, 1]
        
        # 负样本相似度（队列中的样本）
        l_neg = torch.einsum('bd,dn->bn', proj_q, self.queue.clone().detach())  # [batch, queue_size]
        
        # Logits
        logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature  # [batch, 1 + queue_size]
        
        # Labels (正样本在第 0 位)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        
        # 更新队列
        self._dequeue_and_enqueue(proj_k)
        
        return F.cross_entropy(logits, labels)
    
    def cvr_loss(self, batch: Dict) -> torch.Tensor:
        """CVR 预测损失"""
        pred = self.forward(batch)["cvr_pred"].squeeze(-1)
        label = batch.get("cvr_label", batch.get("label")).float()
        return F.binary_cross_entropy_with_logits(pred, label)


class SSLUserBehaviorContrastive(nn.Module):
    """用户行为序列对比学习
    
    利用用户历史行为序列构建正样本对
    """
    
    def __init__(
        self,
        item_embedding: nn.Module,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        max_seq_len: int = 50,
        dropout: float = 0.1,
        temperature: float = 0.1,
    ):
        super().__init__()
        
        self.item_embedding = item_embedding
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        # 序列编码器 (Transformer)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 用户表征
        self.user_proj = nn.Linear(embed_dim, embed_dim)
        
        # CVR head
        self.cvr_head = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )
        
        self.contrastive_loss = InfoNCELoss(temperature)
    
    def encode_sequence(self, item_ids: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """编码用户行为序列
        
        Args:
            item_ids: [batch, seq_len]
            mask: [batch, seq_len] 1 表示有效位置
        
        Returns:
            [batch, embed_dim] 用户表征
        """
        # Item embedding
        item_emb = self.item_embedding(item_ids)  # [batch, seq_len, embed_dim]
        
        # Transformer
        if mask is not None:
            # 转换为 attention mask
            attn_mask = ~mask.bool()
        else:
            attn_mask = None
        
        seq_out = self.transformer(item_emb, src_key_padding_mask=attn_mask)  # [batch, seq_len, embed_dim]
        
        # 取最后一个位置或 mean pooling
        if mask is not None:
            # Mean pooling over valid positions
            seq_out = (seq_out * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
        else:
            seq_out = seq_out.mean(dim=1)
        
        return self.user_proj(seq_out)  # [batch, embed_dim]
    
    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """前向传播
        
        Args:
            batch: {
                item_ids: [batch, seq_len],
                target_item: [batch],
                mask: [batch, seq_len],
            }
        """
        user_emb = self.encode_sequence(batch["item_ids"], batch.get("mask"))
        target_emb = self.item_embedding(batch["target_item"])  # [batch, embed_dim]
        
        # CVR prediction
        cvr_input = torch.cat([user_emb, target_emb], dim=-1)
        cvr_pred = self.cvr_head(cvr_input)
        
        return {
            "user_emb": user_emb,
            "target_emb": target_emb,
            "cvr_pred": cvr_pred,
        }
    
    def predict(self, batch: Dict) -> torch.Tensor:
        """预测 CVR"""
        out = self.forward(batch)
        return torch.sigmoid(out["cvr_pred"]).squeeze(-1)
    
    def ssl_loss(self, batch: Dict) -> torch.Tensor:
        """用户行为对比损失
        
        正样本：同一用户的不同行为序列片段
        负样本：不同用户的行为序列
        """
        # 编码
        user_emb = self.encode_sequence(batch["item_ids"], batch.get("mask"))
        
        # 构建正负样本
        batch_size = user_emb.shape[0]
        
        # 随机打乱作为负样本
        neg_idx = torch.randperm(batch_size, device=user_emb.device)
        neg_emb = user_emb[neg_idx]
        
        # 构建负样本矩阵
        negatives = neg_emb.unsqueeze(1).expand(-1, 1, -1)  # [batch, 1, embed_dim]
        
        # 使用原始作为 anchor 和 positive（简化版）
        return self.contrastive_loss(user_emb, user_emb, negatives)
    
    def cvr_loss(self, batch: Dict) -> torch.Tensor:
        """CVR 预测损失"""
        pred = self.forward(batch)["cvr_pred"].squeeze(-1)
        label = batch.get("cvr_label", batch.get("label")).float()
        return F.binary_cross_entropy_with_logits(pred, label)
