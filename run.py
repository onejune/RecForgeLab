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

import sys
import argparse
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from recforgelab.experiment import ExperimentRunner
from recforgelab.experiment.runner import list_models


def main():
    parser = argparse.ArgumentParser(
        description="RecForgeLab: Recommendation System Experiment Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument("--model", type=str, default=None, 
                       help="Model name (e.g., deepfm, star, m3oe)")
    parser.add_argument("--dataset", type=str, default=None, 
                       help="Dataset name")
    parser.add_argument("--config", type=str, default=None, 
                       help="Config file path")
    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=["single", "grid_search", "compare", "ssl"],
        help="Experiment mode",
    )
    parser.add_argument("--list_models", action="store_true", 
                       help="List all registered models")

    args, extra = parser.parse_known_args()

    # 列出所有模型
    if args.list_models:
        list_models()
        return

    # 运行实验
    runner = ExperimentRunner(
        config_file=args.config,
        mode=args.mode,
        model=args.model,
        dataset=args.dataset,
    )
    
    runner.run()


if __name__ == "__main__":
    main()
