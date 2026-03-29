# -*- coding: utf-8 -*-
"""
实验脚本：对比多任务模型
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from recforgelab import Config, run_experiment


def run_multitask_comparison():
    """对比不同多任务模型"""
    
    models = ["esmm", "mmoe", "ple"]
    
    results = {}
    
    for model in models:
        print(f"\n{'='*50}")
        print(f"Running model: {model}")
        print(f"{'='*50}\n")
        
        config_dict = {
            "experiment_name": f"multitask_comparison_{model}",
            "model_type": "multitask",
        }
        
        result = run_experiment(
            model=model,
            dataset="ali_ccp",
            config_file_list=["config/dataset/ali_ccp.yaml"],
            config_dict=config_dict,
        )
        
        results[model] = result
    
    # 打印对比结果
    print("\n" + "=" * 60)
    print("Multi-Task Model Comparison Results")
    print("=" * 60)
    
    print(f"{'Model':<10} {'CTR_AUC':<12} {'CVR_AUC':<12} {'CTCVR_AUC':<12}")
    print("-" * 50)
    
    for model, result in results.items():
        print(f"{model:<10} {result.get('ctr_auc', 0):.6f}     "
              f"{result.get('cvr_auc', 0):.6f}     "
              f"{result.get('ctcvr_auc', 0):.6f}")


if __name__ == "__main__":
    run_multitask_comparison()
