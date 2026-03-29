#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RecForgeLab 统一入口

用法::

    # 单模型实验
    python run.py --model deepfm --dataset criteo

    # Multi-Domain 模型
    python run.py --model star --dataset ali_ccp_multi_domain

    # 指定配置文件
    python run.py --config config/experiment/compare_models.yaml

    # 网格搜索
    python run.py --config config/experiment/grid_search.yaml --mode grid_search

    # 多模型对比
    python run.py --config config/experiment/compare_models.yaml --mode compare
    
    # Multi-Domain 模型对比
    python run.py --config config/experiment/compare_multi_domain.yaml --mode compare

    # SSL 两阶段训练
    python run.py --config config/experiment/ssl_cvr.yaml --mode ssl

    # 命令行覆盖配置
    python run.py --model deepfm --dataset criteo --learning_rate 0.0001 --epochs 5
    
    # 列出所有模型
    python run.py --list_models
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from recforgelab.utils.config import Config
from recforgelab.utils.logger import get_logger, set_color
from recforgelab.data import create_dataset, create_dataloader
from recforgelab.model import MODEL_REGISTRY, get_model, MultiDomainModel
from recforgelab.trainer.trainer import Trainer
from recforgelab.evaluator import Evaluator


# ============================================================
# 单次实验
# ============================================================

def run_single(config: Config) -> Dict[str, float]:
    """运行单次实验

    Args:
        config: 配置对象

    Returns:
        测试集指标字典
    """
    logger = get_logger()

    logger.info(set_color("=" * 60, "green"))
    logger.info(set_color("  RecForgeLab Experiment", "green"))
    logger.info(set_color("=" * 60, "green"))
    config.print_config()

    # 数据集
    logger.info(set_color("\n[1/4] Loading dataset...", "cyan"))
    train_dataset = create_dataset(config, phase="train")
    valid_dataset = create_dataset(config, phase="valid", encoders=train_dataset.feature_encoders)
    test_dataset = create_dataset(config, phase="test", encoders=train_dataset.feature_encoders)

    train_loader = create_dataloader(train_dataset, config, shuffle=True)
    valid_loader = create_dataloader(valid_dataset, config, shuffle=False)
    test_loader = create_dataloader(test_dataset, config, shuffle=False)

    # 模型
    logger.info(set_color("\n[2/4] Building model...", "cyan"))
    model_class = get_model(config["model"])
    model = model_class(config, train_dataset)
    model = model.to(config["device"])
    model._print_param_count()
    
    # 检测是否为 Multi-Domain 模型
    is_multi_domain = isinstance(model, MultiDomainModel)
    if is_multi_domain:
        logger.info(set_color("  [Multi-Domain Model Detected]", "yellow"))
        logger.info(f"  num_domains: {model.num_domains}")
        logger.info(f"  domain_field: {model.domain_field}")

    # 训练
    logger.info(set_color("\n[3/4] Training...", "cyan"))
    trainer = Trainer(config, model)
    train_result = trainer.train(train_loader, valid_loader)
    best_score = train_result.get("best_score", 0)
    best_epoch = train_result.get("best_epoch", 0)
    logger.info(f"Best valid score: {best_score:.6f} @ epoch {best_epoch + 1}")

    # 测试
    logger.info(set_color("\n[4/4] Testing...", "cyan"))
    
    # Multi-Domain 模型需要分域评估
    if is_multi_domain:
        test_results = evaluate_multi_domain(model, test_loader, config)
    else:
        test_results = trainer._evaluate_epoch(test_loader, epoch=-1)

    logger.info(set_color("\n[Results]", "green"))
    for k, v in test_results.items():
        if isinstance(v, dict):
            logger.info(f"  {k}:")
            for kk, vv in v.items():
                logger.info(f"    {kk:<18} = {vv:.6f}")
        else:
            logger.info(f"  {k:<20} = {v:.6f}")

    return test_results


def evaluate_multi_domain(model, dataloader, config) -> Dict[str, Any]:
    """Multi-Domain 模型分域评估
    
    Args:
        model: Multi-Domain 模型
        dataloader: 测试数据加载器
        config: 配置
    
    Returns:
        包含整体指标和分域指标的字典
    """
    import numpy as np
    from sklearn.metrics import roc_auc_score, log_loss
    
    logger = get_logger()
    model.eval()
    
    all_preds = []
    all_labels = []
    all_domains = []
    
    device = config["device"]
    
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}
            preds = model.predict(batch)
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch[config.get("label_field", "label")].cpu().numpy())
            all_domains.append(batch[config.get("domain_field", "domain_indicator")].cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_domains = np.concatenate(all_domains)
    
    # 整体指标
    overall_auc = roc_auc_score(all_labels, all_preds)
    overall_logloss = log_loss(all_labels, all_preds)
    
    # 分域指标
    num_domains = model.num_domains
    domain_metrics = {}
    
    for d in range(num_domains):
        mask = (all_domains == d)
        if mask.sum() > 0:
            d_preds = all_preds[mask]
            d_labels = all_labels[mask]
            d_auc = roc_auc_score(d_labels, d_preds)
            d_logloss = log_loss(d_labels, d_preds)
            d_size = mask.sum()
            domain_metrics[f"domain_{d}"] = {
                "auc": d_auc,
                "logloss": d_logloss,
                "samples": d_size,
                "ratio": d_size / len(all_labels),
            }
            logger.info(f"  Domain_{d}: AUC={d_auc:.4f}, LogLoss={d_logloss:.4f}, "
                       f"samples={d_size} ({d_size/len(all_labels)*100:.1f}%)")
    
    return {
        "AUC": overall_auc,
        "LogLoss": overall_logloss,
        "domains": domain_metrics,
    }


import torch


# ============================================================
# 网格搜索
# ============================================================

def run_grid_search(config: Config) -> List[Dict[str, Any]]:
    """运行网格搜索

    Args:
        config: 包含 grid_search 定义的配置

    Returns:
        所有实验结果列表
    """
    logger = get_logger()
    grid_configs = config.get_grid_search_configs()

    logger.info(set_color(f"Grid search: {len(grid_configs)} combinations", "yellow"))

    all_results = []
    for i, cfg in enumerate(grid_configs):
        logger.info(set_color(f"\n[Grid {i+1}/{len(grid_configs)}] {cfg['experiment_name']}", "cyan"))
        try:
            results = run_single(cfg)
            all_results.append({
                "experiment_name": cfg["experiment_name"],
                "config": cfg.to_dict(),
                "results": results,
            })
        except Exception as e:
            logger.error(f"Grid search failed for {cfg['experiment_name']}: {e}")
            all_results.append({
                "experiment_name": cfg["experiment_name"],
                "config": cfg.to_dict(),
                "error": str(e),
            })

    # 打印汇总
    logger.info(set_color("\n[Grid Search Summary]", "green"))
    valid_metric = config.get("valid_metric", "AUC").lower()
    sorted_results = sorted(
        [r for r in all_results if "results" in r],
        key=lambda x: x["results"].get(valid_metric, 0),
        reverse=True,
    )
    for r in sorted_results:
        score = r["results"].get(valid_metric, 0)
        logger.info(f"  {r['experiment_name']:<50} {valid_metric}={score:.6f}")

    return all_results


# ============================================================
# 多模型对比
# ============================================================

def run_compare(config: Config) -> Dict[str, Dict[str, float]]:
    """运行多模型对比实验

    Args:
        config: 包含 experiments 定义的配置

    Returns:
        {exp_name: results_dict}
    """
    logger = get_logger()
    exp_configs = config.get_experiment_configs()

    logger.info(set_color(f"Comparing {len(exp_configs)} experiments", "yellow"))

    all_results = {}
    for exp_name, cfg in exp_configs:
        logger.info(set_color(f"\n[Experiment] {exp_name}", "cyan"))
        try:
            results = run_single(cfg)
            all_results[exp_name] = results
        except Exception as e:
            logger.error(f"Experiment {exp_name} failed: {e}")
            all_results[exp_name] = {"error": str(e)}

    # 打印对比表
    logger.info(set_color("\n[Comparison Summary]", "green"))
    valid_metric = config.get("valid_metric", "AUC").lower()
    sorted_items = sorted(
        [(k, v) for k, v in all_results.items() if "error" not in v],
        key=lambda x: x[1].get(valid_metric, 0),
        reverse=True,
    )
    for name, results in sorted_items:
        score = results.get(valid_metric, 0)
        logger.info(f"  {name:<40} {valid_metric}={score:.6f}")

    return all_results


# ============================================================
# SSL 两阶段训练
# ============================================================

def run_ssl(config: Config) -> Dict[str, float]:
    """SSL 两阶段训练

    Phase 1: SSL 预训练（对比学习）
    Phase 2: 下游任务微调
    """
    logger = get_logger()

    logger.info(set_color("SSL Two-stage Training", "cyan"))

    # Phase 1: 预训练
    logger.info(set_color("\n[Phase 1] SSL Pretraining...", "yellow"))
    pretrain_config = Config(config_dict=config.to_dict(), parse_cmd=False)
    pretrain_config["training_phase"] = "pretrain"
    pretrain_config["epochs"] = config.get("pretrain_epochs", 5)
    pretrain_config["experiment_name"] = config.get("experiment_name", "ssl") + "_pretrain"

    train_dataset = create_dataset(pretrain_config, phase="train")
    valid_dataset = create_dataset(pretrain_config, phase="valid", encoders=train_dataset.feature_encoders)
    train_loader = create_dataloader(train_dataset, pretrain_config, shuffle=True)
    valid_loader = create_dataloader(valid_dataset, pretrain_config, shuffle=False)

    model_class = get_model(config["model"])
    model = model_class(pretrain_config, train_dataset)
    model = model.to(config["device"])

    trainer = Trainer(pretrain_config, model)
    trainer.fit(train_loader, valid_loader)

    # Phase 2: 微调
    logger.info(set_color("\n[Phase 2] Finetuning...", "yellow"))
    finetune_config = Config(config_dict=config.to_dict(), parse_cmd=False)
    finetune_config["training_phase"] = "finetune"
    finetune_config["epochs"] = config.get("finetune_epochs", config.get("epochs", 10))
    finetune_config["experiment_name"] = config.get("experiment_name", "ssl") + "_finetune"

    if hasattr(model, "set_phase"):
        model.set_phase("finetune")

    test_dataset = create_dataset(finetune_config, phase="test", encoders=train_dataset.feature_encoders)
    test_loader = create_dataloader(test_dataset, finetune_config, shuffle=False)

    finetune_trainer = Trainer(finetune_config, model)
    finetune_trainer.train(train_loader, valid_loader)
    results = finetune_trainer._evaluate_epoch(test_loader, epoch=-1)

    logger.info(set_color("\n[SSL Results]", "green"))
    for k, v in results.items():
        logger.info(f"  {k:<20} = {v:.6f}")

    return results


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="RecForgeLab: Recommendation System Experiment Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model", type=str, default=None, help="Model name (e.g., deepfm, star, m3oe)")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name")
    parser.add_argument("--config", type=str, default=None, help="Config file path")
    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=["single", "grid_search", "compare", "ssl"],
        help="Experiment mode",
    )
    parser.add_argument("--list_models", action="store_true", help="List all registered models")

    args, extra = parser.parse_known_args()

    # 列出所有模型
    if args.list_models:
        # 导入所有模型触发注册
        from recforgelab.model import MODEL_REGISTRY
        print("Registered models:")
        print("-" * 60)
        
        # 按类型分组
        ctr_models = []
        multitask_models = []
        multi_domain_models = []
        ssl_models = []
        
        for name in sorted(MODEL_REGISTRY.keys()):
            cls = MODEL_REGISTRY[name]
            model_type = getattr(cls, 'model_type', None)
            type_name = model_type.value if model_type else 'unknown'
            
            if type_name == 'ctr':
                ctr_models.append((name, cls))
            elif type_name == 'multitask':
                multitask_models.append((name, cls))
            elif type_name == 'multi_domain':
                multi_domain_models.append((name, cls))
            elif type_name == 'ssl':
                ssl_models.append((name, cls))
        
        if ctr_models:
            print("\n[CTR Models]")
            for name, cls in ctr_models:
                print(f"  {name:<20} -> {cls.__name__}")
        
        if multitask_models:
            print("\n[Multitask Models]")
            for name, cls in multitask_models:
                print(f"  {name:<20} -> {cls.__name__}")
        
        if multi_domain_models:
            print("\n[Multi-Domain Models]")
            for name, cls in multi_domain_models:
                print(f"  {name:<20} -> {cls.__name__}")
        
        if ssl_models:
            print("\n[SSL Models]")
            for name, cls in ssl_models:
                print(f"  {name:<20} -> {cls.__name__}")
        
        print("-" * 60)
        print(f"Total: {len(MODEL_REGISTRY)} models")
        return

    # 构建配置
    config_file_list = [args.config] if args.config else None
    config = Config(
        model=args.model,
        dataset=args.dataset,
        config_file_list=config_file_list,
        parse_cmd=True,
    )

    # 运行
    mode = args.mode or config.get("mode", "single")

    if mode == "grid_search":
        run_grid_search(config)
    elif mode == "compare":
        run_compare(config)
    elif mode == "ssl":
        run_ssl(config)
    else:
        run_single(config)


if __name__ == "__main__":
    main()
