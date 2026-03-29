# -*- coding: utf-8 -*-
"""
实验脚本：对比连续特征编码方法
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from recforgelab import Config, run_experiment


def run_encoder_comparison():
    """对比不同连续特征编码器"""
    
    encoders = ["scalar", "bucket", "numeric", "autodis", "fttransformer"]
    
    results = {}
    
    for encoder in encoders:
        print(f"\n{'='*50}")
        print(f"Running with encoder: {encoder}")
        print(f"{'='*50}\n")
        
        config_dict = {
            "encoder_type": encoder,
            "experiment_name": f"encoder_comparison_{encoder}",
        }
        
        result = run_experiment(
            model="deepfm",
            dataset="criteo",
            config_file_list=["config/experiment/ctr_baseline.yaml"],
            config_dict=config_dict,
        )
        
        results[encoder] = result
    
    # 打印对比结果
    print("\n" + "=" * 50)
    print("Encoder Comparison Results")
    print("=" * 50)
    
    print(f"{'Encoder':<15} {'AUC':<10} {'LogLoss':<10}")
    print("-" * 35)
    
    for encoder, result in results.items():
        print(f"{encoder:<15} {result['auc']:.6f}   {result['logloss']:.6f}")


if __name__ == "__main__":
    run_encoder_comparison()
