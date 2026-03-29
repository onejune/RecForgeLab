# -*- coding: utf-8 -*-
"""
实验脚本：IVR Sample v16 多任务 CTCVR 预估对比实验

对比模型：
  - direct_ctcvr  (baseline: 直接预测 CTCVR)
  - shared_bottom
  - esmm
  - escm2
  - mmoe
  - ple

对比编码器（连续特征）：
  - log (默认，适合计数类特征)
  - scalar
  - numeric
  - autodis

用法：
  python experiments/run_ivr_multitask.py --model esmm
  python experiments/run_ivr_multitask.py --compare_models  # 全量对比
  python experiments/run_ivr_multitask.py --compare_encoders --model esmm  # 编码器对比
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from recforgelab.utils import Config, init_logger, get_logger
from recforgelab.data import create_dataset, create_dataloader
from recforgelab.model import get_model
from recforgelab.trainer import Trainer


BASE_CONFIG = {
    # 数据
    "dataset": "ivr_sample_v16",
    "data_path": "/mnt/data/oss_wanjun/pai_work/open_research/dataset/ivr_sample_v16_ctcvr",
    "data_format": "spark_dir",
    "label_field": "ctcvr_label",
    "ctr_label_field": "click_label",
    "cvr_label_field": "ctcvr_label",
    "model_type": "multitask",
    "train_date_range": ["2026-02-16", "2026-03-16"],
    "test_date_range": ["2026-03-17", "2026-03-19"],
    
    # 模型
    "embedding_size": 16,
    "mlp_hidden_size": [256, 128, 64],
    "dropout_prob": 0.2,
    "encoder_type": "log",
    
    # 训练
    "learning_rate": 0.001,
    "train_batch_size": 4096,
    "eval_batch_size": 8192,
    "epochs": 5,
    "early_stop_patience": 2,
    "optimizer": "adam",
    "weight_decay": 0.0,
    "num_workers": 4,
    
    # 评估
    "metrics": ["AUC", "LogLoss", "GAUC", "PCOC"],
    "valid_metric": "AUC",
    "eval_step": 1,
    
    # 系统
    "device": "cuda",
    "seed": 2024,
    "checkpoint_dir": "/mnt/workspace/open_research/recforgelab/saved",
    "log_dir": "/mnt/workspace/open_research/recforgelab/logs",
    "save_model": True,
}


def run_single(model_name: str, encoder_type: str = "log", extra_config: dict = None):
    """运行单个实验"""
    config_dict = {**BASE_CONFIG, "encoder_type": encoder_type}
    if extra_config:
        config_dict.update(extra_config)
    
    config = Config(
        model=model_name,
        config_file_list=["config/dataset/ivr_sample_v16.yaml"],
        config_dict=config_dict,
    )
    config["experiment_name"] = f"{model_name}_{encoder_type}_ivr"
    
    init_logger(config)
    logger = get_logger()
    logger.info(f"=== {model_name.upper()} | encoder={encoder_type} ===")
    
    # 数据集
    train_dataset = create_dataset(config, phase="train")
    test_dataset = create_dataset(config, phase="test", encoders=train_dataset.feature_encoders)
    
    train_loader = create_dataloader(train_dataset, config, shuffle=True)
    test_loader = create_dataloader(test_dataset, config, shuffle=False)
    
    # 模型
    model_class = get_model(model_name)
    model = model_class(config, train_dataset).to(config["device"])
    
    # 训练
    trainer = Trainer(config, model)
    trainer.train(train_loader)
    
    # 测试
    results = trainer.test(test_loader)
    
    return results


def compare_models():
    """对比所有多任务模型"""
    models = ["direct_ctcvr", "shared_bottom", "esmm", "escm2", "mmoe", "ple"]
    all_results = {}
    
    for model in models:
        print(f"\n{'='*60}")
        print(f"Model: {model}")
        print(f"{'='*60}")
        try:
            results = run_single(model)
            all_results[model] = results
        except Exception as e:
            print(f"Error running {model}: {e}")
            all_results[model] = {}
    
    # 打印汇总
    print("\n" + "=" * 80)
    print("Multi-Task Model Comparison on IVR Sample v16")
    print("=" * 80)
    
    metrics = ["ctr_auc", "cvr_auc", "ctcvr_auc", "ctr_logloss", "ctcvr_logloss", "ctcvr_pcoc"]
    header = f"{'Model':<16}" + "".join(f"{m:<14}" for m in metrics)
    print(header)
    print("-" * 80)
    
    for model, results in all_results.items():
        row = f"{model:<16}"
        for m in metrics:
            v = results.get(m, float("nan"))
            row += f"{v:<14.6f}" if isinstance(v, float) else f"{'N/A':<14}"
        print(row)


def compare_encoders(model_name: str = "esmm"):
    """对比连续特征编码器"""
    encoders = ["none", "scalar", "log", "bucket", "numeric", "numeric_deep", "autodis", "fttransformer", "periodic"]
    all_results = {}
    
    for enc in encoders:
        print(f"\n{'='*60}")
        print(f"Encoder: {enc}")
        print(f"{'='*60}")
        try:
            results = run_single(model_name, encoder_type=enc)
            all_results[enc] = results
        except Exception as e:
            print(f"Error with encoder {enc}: {e}")
            all_results[enc] = {}
    
    # 打印汇总
    print("\n" + "=" * 70)
    print(f"Encoder Comparison on IVR Sample v16 (model={model_name})")
    print("=" * 70)
    
    metrics = ["ctcvr_auc", "ctcvr_logloss", "ctcvr_pcoc"]
    header = f"{'Encoder':<20}" + "".join(f"{m:<16}" for m in metrics)
    print(header)
    print("-" * 70)
    
    for enc, results in all_results.items():
        row = f"{enc:<20}"
        for m in metrics:
            v = results.get(m, float("nan"))
            row += f"{v:<16.6f}" if isinstance(v, float) else f"{'N/A':<16}"
        print(row)


def main():
    parser = argparse.ArgumentParser(description="IVR Multi-Task Experiment")
    parser.add_argument("--model", type=str, default="esmm")
    parser.add_argument("--encoder", type=str, default="log")
    parser.add_argument("--compare_models", action="store_true")
    parser.add_argument("--compare_encoders", action="store_true")
    args = parser.parse_args()
    
    if args.compare_models:
        compare_models()
    elif args.compare_encoders:
        compare_encoders(args.model)
    else:
        run_single(args.model, args.encoder)


if __name__ == "__main__":
    main()
