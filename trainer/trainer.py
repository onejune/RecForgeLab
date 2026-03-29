# -*- coding: utf-8 -*-
"""
训练器
支持：
- 早停
- 检查点保存/恢复
- 混合精度训练 (AMP)
- TensorBoard 日志
- 学习率调度
"""

import os
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from typing import Dict, Optional, Tuple, Any
import numpy as np
from pathlib import Path
from datetime import datetime

from ..utils import get_logger, set_color
from ..evaluator import Evaluator


class Trainer:
    """训练器
    
    统一管理训练、验证、测试流程
    """
    
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.device = config["device"]
        self.logger = get_logger()
        
        # 优化器
        self.optimizer = self._build_optimizer()
        
        # 学习率调度器
        self.scheduler = self._build_scheduler()
        
        # 评估器
        self.evaluator = Evaluator(config)
        
        # 早停
        self.early_stop_patience = config.get("early_stop_patience", 5)
        self.best_score = -np.inf
        self.best_epoch = 0
        self.patience_counter = 0
        
        # 检查点
        self.checkpoint_dir = Path(config.get("checkpoint_dir", "./checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        
        # 混合精度训练
        self.use_amp = config.get("use_amp", False) and self.device.type == "cuda"
        if self.use_amp:
            self.scaler = GradScaler()
            self.logger.info("Mixed precision training enabled (AMP)")
        else:
            self.scaler = None
        
        # TensorBoard
        self.use_tensorboard = config.get("use_tensorboard", True)
        self.writer = None
        if self.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                log_dir = self.checkpoint_dir / "logs"
                self.writer = SummaryWriter(log_dir=str(log_dir))
                self.logger.info(f"TensorBoard logging to {log_dir}")
            except ImportError:
                self.logger.warning("TensorBoard not available, skipping logging")
                self.use_tensorboard = False
    
    def _build_optimizer(self) -> torch.optim.Optimizer:
        """构建优化器"""
        opt_name = self.config.get("optimizer", "adam").lower()
        lr = self.config.get("learning_rate", 1e-3)
        weight_decay = self.config.get("weight_decay", 1e-4)
        
        # 支持不同参数组有不同的 weight decay
        param_groups = self._get_param_groups(lr, weight_decay)
        
        if opt_name == "adam":
            return torch.optim.Adam(param_groups, lr=lr)
        elif opt_name == "adamw":
            return torch.optim.AdamW(param_groups, lr=lr)
        elif opt_name == "sgd":
            return torch.optim.SGD(param_groups, lr=lr, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")
    
    def _get_param_groups(self, lr: float, weight_decay: float) -> list:
        """获取参数组，支持 bias 不做 weight decay"""
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # bias 和 LayerNorm 不做 weight decay
            if 'bias' in name or 'LayerNorm' in name or 'layer_norm' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        return [
            {"params": decay_params, "lr": lr, "weight_decay": weight_decay},
            {"params": no_decay_params, "lr": lr, "weight_decay": 0.0},
        ]
    
    def _build_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """构建学习率调度器"""
        scheduler_name = self.config.get("scheduler")
        
        if scheduler_name is None or scheduler_name.lower() == "none":
            return None
        
        if scheduler_name.lower() == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size=self.config.get("scheduler_step", 10),
                gamma=self.config.get("scheduler_gamma", 0.1),
            )
        elif scheduler_name.lower() == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get("epochs", 10),
            )
        elif scheduler_name.lower() == "reduce_on_plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                factor=0.5,
                patience=2,
            )
        elif scheduler_name.lower() == "warmup_cosine":
            # 带 warmup 的 cosine scheduler
            from torch.optim.lr_scheduler import LambdaLR
            warmup_steps = self.config.get("warmup_steps", 1000)
            total_steps = self.config.get("total_steps", 10000)
            
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / warmup_steps
                return 0.5 * (1 + np.cos(np.pi * (step - warmup_steps) / (total_steps - warmup_steps)))
            
            return LambdaLR(self.optimizer, lr_lambda)
        else:
            return None
    
    def train(
        self,
        train_loader: DataLoader,
        valid_loader: Optional[DataLoader] = None,
        valid_metrics: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """训练循环
        
        Args:
            train_loader: 训练数据加载器
            valid_loader: 验证数据加载器（可选）
            valid_metrics: 验证集评估结果（可选，用于 ReduceLROnPlateau）
        
        Returns:
            训练结果字典
        """
        epochs = self.config.get("epochs", 10)
        log_steps = self.config.get("log_steps", 100)
        
        self.logger.info(set_color("Training started", "green"))
        self.logger.info(f"  Epochs: {epochs}")
        self.logger.info(f"  Device: {self.device}")
        self.logger.info(f"  AMP: {self.use_amp}")
        
        train_start_time = time.time()
        
        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # 训练一个 epoch
            train_loss = self._train_epoch(train_loader, epoch, log_steps)
            
            # 验证
            if valid_loader is not None:
                valid_results = self._evaluate_epoch(valid_loader, epoch)
                valid_score = self.evaluator.get_valid_score(valid_results)
                
                # 学习率调度（ReduceLROnPlateau）
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(valid_score)
                elif self.scheduler is not None:
                    self.scheduler.step()
                
                # TensorBoard 日志
                if self.writer:
                    self.writer.add_scalar("Loss/train", train_loss, epoch)
                    for metric, value in valid_results.items():
                        self.writer.add_scalar(f"Metrics/{metric}", value, epoch)
                    self.writer.add_scalar("LR", self.optimizer.param_groups[0]['lr'], epoch)
                
                # 检查是否最佳
                if self.evaluator.better(valid_score, self.best_score):
                    self.best_score = valid_score
                    self.best_epoch = epoch
                    self.patience_counter = 0
                    # 保存最佳模型
                    self._save_checkpoint(epoch, valid_results, is_best=True)
                else:
                    self.patience_counter += 1
                
                # 日志
                epoch_time = time.time() - epoch_start_time
                self.logger.info(
                    set_color(f"[Epoch {epoch+1}/{epochs}]", "yellow") +
                    f" loss={train_loss:.4f} | " +
                    self.evaluator.format_results(valid_results) +
                    f" | time={epoch_time:.1f}s"
                )
                
                # 早停
                if self.patience_counter >= self.early_stop_patience:
                    self.logger.info(set_color(f"Early stopping at epoch {epoch+1}", "red"))
                    break
            else:
                # 没有验证集，每个 epoch 都保存
                self._save_checkpoint(epoch, {"loss": train_loss})
                self.logger.info(f"[Epoch {epoch+1}/{epochs}] loss={train_loss:.4f}")
        
        total_time = time.time() - train_start_time
        self.logger.info(set_color(f"Training completed", "green"))
        self.logger.info(f"  Best epoch: {self.best_epoch + 1}")
        self.logger.info(f"  Best score: {self.best_score:.6f}")
        self.logger.info(f"  Total time: {total_time:.1f}s")
        
        if self.writer:
            self.writer.close()
        
        return {
            "best_epoch": self.best_epoch,
            "best_score": self.best_score,
            "total_time": total_time,
        }
    
    def _train_epoch(self, train_loader: DataLoader, epoch: int, log_steps: int) -> float:
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        for batch_idx, batch in enumerate(train_loader):
            # 转移到设备
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}
            
            self.optimizer.zero_grad()
            
            # 前向传播（支持混合精度）
            if self.use_amp:
                with autocast():
                    loss = self._compute_loss(batch)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss = self._compute_loss(batch)
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            self.global_step += 1
            
            # 日志
            if (batch_idx + 1) % log_steps == 0:
                self.logger.debug(
                    f"[Epoch {epoch+1}] Batch {batch_idx+1}/{num_batches} | "
                    f"loss={loss.item():.4f}"
                )
        
        return total_loss / num_batches
    
    def _compute_loss(self, batch: Dict) -> torch.Tensor:
        """计算损失"""
        # 由子类或模型实现
        return self.model.calculate_loss(batch)
    
    def _evaluate_epoch(self, data_loader: DataLoader, epoch: int) -> Dict:
        """评估一个 epoch"""
        self.model.eval()
        
        all_labels = []
        all_preds = []
        all_groups = []
        
        with torch.no_grad():
            for batch in data_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                         for k, v in batch.items()}
                
                # 预测
                preds = self.model.predict(batch)
                
                # 收集结果
                labels = batch.get("label", batch.get("ctcvr_label"))
                all_labels.append(labels.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
                
                # 收集分组信息（用于 GAUC）
                if "user_id" in batch:
                    all_groups.append(batch["user_id"].cpu().numpy())
        
        # 合并
        all_labels = np.concatenate(all_labels)
        all_preds = np.concatenate(all_preds)
        all_groups = np.concatenate(all_groups) if all_groups else None
        
        # 评估
        return self.evaluator.evaluate(all_labels, all_preds, all_groups)
    
    def _save_checkpoint(
        self, 
        epoch: int, 
        metrics: Dict, 
        is_best: bool = False
    ):
        """保存检查点"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "metrics": metrics,
            "config": self.config,
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        # 保存最新
        latest_path = self.checkpoint_dir / "latest.pt"
        torch.save(checkpoint, latest_path)
        
        # 保存最佳
        if is_best:
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model to {best_path}")
    
    def load_checkpoint(self, path: str, load_optimizer: bool = True):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.current_epoch = checkpoint["epoch"] + 1
        self.global_step = checkpoint["global_step"]
        
        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        self.logger.info(f"Loaded checkpoint from {path} (epoch {checkpoint['epoch']})")
        
        return checkpoint.get("metrics", {})


class MultiTaskTrainer(Trainer):
    """多任务训练器"""
    
    def _compute_loss(self, batch: Dict) -> torch.Tensor:
        """计算多任务损失"""
        return self.model.calculate_loss(batch)
    
    def _evaluate_epoch(self, data_loader: DataLoader, epoch: int) -> Dict:
        """多任务评估"""
        self.model.eval()
        
        all_labels = {"ctr": [], "cvr": [], "ctcvr": []}
        all_preds = {"ctr": [], "cvr": [], "ctcvr": []}
        all_groups = []
        
        with torch.no_grad():
            for batch in data_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                         for k, v in batch.items()}
                
                # 预测
                preds = self.model.predict(batch)
                
                # 收集结果
                for task in ["ctr", "cvr", "ctcvr"]:
                    label_key = f"{task}_label" if task != "ctcvr" else "ctcvr_label"
                    if label_key in batch:
                        all_labels[task].append(batch[label_key].cpu().numpy())
                    if task in preds:
                        all_preds[task].append(preds[task].cpu().numpy())
                
                if "user_id" in batch:
                    all_groups.append(batch["user_id"].cpu().numpy())
        
        # 合并
        labels = {}
        preds = {}
        for task in all_labels:
            if all_labels[task]:
                labels[task] = np.concatenate(all_labels[task])
                preds[task] = np.concatenate(all_preds[task]) if all_preds[task] else None
        
        all_groups = np.concatenate(all_groups) if all_groups else None
        
        # 多任务评估
        return self.evaluator.evaluate_multitask(labels, preds, all_groups)
