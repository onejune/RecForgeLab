# -*- coding: utf-8 -*-
"""
综合实验脚本
支持：
- 多模型对比
- 多编码器对比
- SSL 对比学习实验
- 实验记录和报告生成
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch
import numpy as np
import pandas as pd

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from recforgelab.data import DSPDataset
from recforgelab.model import (
    DeepFM, DCN, DCNv2,
    ESMM, MMoE, PLE, SharedBottom, DirectCTCVR,
    AutoInt, xDeepFM,
)
from recforgelab.model.ssl import (
    SSLContrastive,
    SSLMomentumContrastive,
)
from recforgelab.trainer import Trainer, MultiTaskTrainer
from recforgelab.evaluator import Evaluator
from recforgelab.utils import Config, get_logger
from recforgelab.utils.experiment import ExperimentRecorder, ModelComparator


# 模型注册表
MODEL_REGISTRY = {
    # CTR models
    "deepfm": DeepFM,
    "dcn": DCN,
    "dcnv2": DCNv2,
    "autoint": AutoInt,
    "xdeepfm": xDeepFM,
    # Multi-task models
    "esmm": ESMM,
    "mmoe": MMoE,
    "ple": PLE,
    "shared_bottom": SharedBottom,
    "direct_ctcvr": DirectCTCVR,
    # SSL models (需要特殊处理)
    "ssl_contrastive": None,  # placeholder
    "ssl_momentum": None,
}


def run_single_experiment(
    model_name: str,
    dataset_name: str,
    config: Dict,
    recorder: Optional[ExperimentRecorder] = None,
) -> Dict:
    """运行单个实验
    
    Args:
        model_name: 模型名称
        dataset_name: 数据集名称
        config: 配置字典
        recorder: 实验记录器
    
    Returns:
        实验结果
    """
    logger = get_logger()
    logger.info(f"Running experiment: {model_name} on {dataset_name}")
    
    # 准备数据
    train_dataset = DSPDataset(config, phase="train")
    valid_dataset = DSPDataset(config, phase="valid", encoders=train_dataset.feature_encoders)
    test_dataset = DSPDataset(config, phase="test", encoders=train_dataset.feature_encoders)
    
    # 创建模型
    model_class = MODEL_REGISTRY[model_name]
    model = model_class(config, train_dataset).to(config["device"])
    
    # 训练
    trainer = MultiTaskTrainer(config, model) if model_name in ["esmm", "mmoe", "ple", "shared_bottom", "direct_ctcvr"] else Trainer(config, model)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4
    )
    
    train_result = trainer.train(train_loader, valid_loader)
    
    # 测试
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4
    )
    test_results = trainer._evaluate_epoch(test_loader, 0)
    
    logger.info(f"Test results: {test_results}")
    
    return {
        "model": model_name,
        "dataset": dataset_name,
        "train_result": train_result,
        "test_results": test_results,
    }


def run_model_comparison(
    models: List[str],
    dataset_name: str,
    config: Dict,
    output_dir: str = "./results",
) -> pd.DataFrame:
    """运行多模型对比实验
    
    Args:
        models: 模型名称列表
        dataset_name: 数据集名称
        config: 配置字典
        output_dir: 输出目录
    
    Returns:
        对比结果表格
    """
    recorder = ExperimentRecorder(log_dir=output_dir)
    comparator = ModelComparator(recorder)
    
    results = {}
    
    for model_name in models:
        if model_name not in MODEL_REGISTRY:
            print(f"Unknown model: {model_name}, skipping")
            continue
        
        # 开始记录
        recorder.start_experiment(
            exp_name=f"{model_name}_{dataset_name}",
            config=config,
            tags=[model_name, dataset_name],
            description=f"{model_name} on {dataset_name}",
        )
        
        try:
            # 运行实验
            exp_result = run_single_experiment(model_name, dataset_name, config, recorder)
            results[model_name] = exp_result["test_results"]
            
            # 记录结果
            recorder.finish_experiment(exp_result["test_results"], status="completed")
        except Exception as e:
            print(f"Error running {model_name}: {e}")
            recorder.finish_experiment({}, status="failed")
            results[model_name] = {"error": str(e)}
    
    # 生成对比报告
    df = comparator.compare_models(results)
    
    # 保存结果
    output_file = Path(output_dir) / f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    # 生成报告
    report_file = Path(output_dir) / f"experiment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    recorder.generate_report(str(report_file))
    
    return df


def run_encoder_comparison(
    model_name: str,
    encoders: List[str],
    dataset_name: str,
    config: Dict,
    output_dir: str = "./results",
) -> pd.DataFrame:
    """运行编码器对比实验
    
    Args:
        model_name: 基础模型名称
        encoders: 编码器名称列表
        dataset_name: 数据集名称
        config: 配置字典
        output_dir: 输出目录
    
    Returns:
        对比结果表格
    """
    recorder = ExperimentRecorder(log_dir=output_dir)
    comparator = ModelComparator(recorder)
    
    results = {}
    
    for encoder in encoders:
        # 更新配置
        encoder_config = config.copy()
        encoder_config["continuous_encoder"] = encoder
        
        # 开始记录
        recorder.start_experiment(
            exp_name=f"{model_name}_{encoder}_{dataset_name}",
            config=encoder_config,
            tags=[model_name, encoder, dataset_name],
            description=f"{model_name} with {encoder} encoder on {dataset_name}",
        )
        
        try:
            exp_result = run_single_experiment(model_name, dataset_name, encoder_config, recorder)
            results[encoder] = exp_result["test_results"]
            recorder.finish_experiment(exp_result["test_results"], status="completed")
        except Exception as e:
            print(f"Error with encoder {encoder}: {e}")
            recorder.finish_experiment({}, status="failed")
            results[encoder] = {"error": str(e)}
    
    # 生成对比报告
    df = comparator.compare_encoders(results)
    
    return df


def run_ssl_experiment(
    ssl_type: str,
    dataset_name: str,
    config: Dict,
    pretrain_epochs: int = 5,
    finetune_epochs: int = 3,
    output_dir: str = "./results",
) -> Dict:
    """运行 SSL 对比学习实验
    
    Args:
        ssl_type: SSL 类型 (contrastive / momentum / user_behavior)
        dataset_name: 数据集名称
        config: 配置字典
        pretrain_epochs: 预训练轮数
        finetune_epochs: 微调轮数
        output_dir: 输出目录
    
    Returns:
        实验结果
    """
    logger = get_logger()
    recorder = ExperimentRecorder(log_dir=output_dir)
    
    # 准备数据
    train_dataset = DSPDataset(config, phase="train")
    valid_dataset = DSPDataset(config, phase="valid", encoders=train_dataset.feature_encoders)
    
    # Stage 1: SSL Pre-training
    logger.info(f"Stage 1: SSL Pre-training ({ssl_type})")
    recorder.start_experiment(
        exp_name=f"ssl_{ssl_type}_pretrain_{dataset_name}",
        config=config,
        tags=["ssl", ssl_type, "pretrain"],
        description=f"SSL pre-training with {ssl_type}",
    )
    
    # ... (SSL pre-training logic)
    
    # Stage 2: CVR Fine-tuning
    logger.info(f"Stage 2: CVR Fine-tuning")
    # ... (fine-tuning logic)
    
    results = {"ssl_type": ssl_type, "auc": 0.0}  # placeholder
    recorder.finish_experiment(results)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="RecForgeLab Experiments")
    parser.add_argument("--mode", type=str, default="single", 
                        choices=["single", "compare_models", "compare_encoders", "ssl"],
                        help="Experiment mode")
    parser.add_argument("--model", type=str, default="esmm", help="Model name")
    parser.add_argument("--dataset", type=str, default="ivr_sample_v16", help="Dataset name")
    parser.add_argument("--models", type=str, nargs="+", help="Models to compare")
    parser.add_argument("--encoders", type=str, nargs="+", help="Encoders to compare")
    parser.add_argument("--config", type=str, help="Config file path")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory")
    parser.add_argument("--ssl_type", type=str, default="contrastive", 
                        choices=["contrastive", "momentum", "user_behavior"],
                        help="SSL type")
    
    args = parser.parse_args()
    
    # 加载配置
    if args.config:
        config = Config.load(args.config)
    else:
        config = {
            "data_path": "/mnt/data/oss_wanjun/pai_work/open_research/dataset/ivr_sample_v16_ctcvr/",
            "sparse_features": [f"cat_{i}" for i in range(99)],
            "dense_features": [f"num_{i}" for i in range(26)],
            "ctr_label_field": "click_label",
            "cvr_label_field": "ctcvr_label",
            "embedding_dim": 16,
            "hidden_dims": [256, 128],
            "batch_size": 4096,
            "learning_rate": 1e-3,
            "epochs": 10,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "metrics": ["auc", "logloss", "gauc", "pcoc"],
            "valid_metric": "auc",
        }
    
    # 运行实验
    if args.mode == "single":
        result = run_single_experiment(args.model, args.dataset, config)
        print(json.dumps(result["test_results"], indent=2))
    
    elif args.mode == "compare_models":
        models = args.models or ["deepfm", "dcn", "autoint", "xdeepfm", "esmm", "mmoe"]
        df = run_model_comparison(models, args.dataset, config, args.output_dir)
        print(df)
    
    elif args.mode == "compare_encoders":
        encoders = args.encoders or ["none", "scalar", "log", "bucket", "numeric"]
        df = run_encoder_comparison(args.model, encoders, args.dataset, config, args.output_dir)
        print(df)
    
    elif args.mode == "ssl":
        result = run_ssl_experiment(args.ssl_type, args.dataset, config, output_dir=args.output_dir)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
