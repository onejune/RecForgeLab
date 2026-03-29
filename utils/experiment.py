# -*- coding: utf-8 -*-
"""
实验对比和记录工具
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd


class ExperimentRecorder:
    """实验记录器
    
    记录实验配置、结果，支持对比分析
    """
    
    def __init__(self, log_dir: str = "./experiment_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 当前实验
        self.current_exp = None
        self.start_time = None
    
    def start_experiment(
        self,
        exp_name: str,
        config: Dict,
        tags: Optional[List[str]] = None,
        description: str = "",
    ):
        """开始新实验
        
        Args:
            exp_name: 实验名称
            config: 实验配置
            tags: 标签列表
            description: 实验描述
        """
        self.current_exp = {
            "name": exp_name,
            "config": config,
            "tags": tags or [],
            "description": description,
            "start_time": datetime.now().isoformat(),
            "status": "running",
            "metrics": {},
            "log": [],
        }
        self.start_time = time.time()
    
    def log_metric(
        self,
        epoch: int,
        metrics: Dict[str, float],
        phase: str = "train",
    ):
        """记录指标
        
        Args:
            epoch: 轮次
            metrics: 指标字典
            phase: 阶段 (train/valid/test)
        """
        if self.current_exp is None:
            return
        
        log_entry = {
            "epoch": epoch,
            "phase": phase,
            "metrics": metrics,
            "timestamp": time.time() - self.start_time,
        }
        self.current_exp["log"].append(log_entry)
        
        # 更新最佳指标
        for key, value in metrics.items():
            if key not in self.current_exp["metrics"] or value > self.current_exp["metrics"].get(f"best_{key}", 0):
                self.current_exp["metrics"][f"best_{key}"] = value
    
    def finish_experiment(
        self,
        final_metrics: Dict[str, float],
        status: str = "completed",
    ):
        """结束实验
        
        Args:
            final_metrics: 最终指标
            status: 状态 (completed/failed/interrupted)
        """
        if self.current_exp is None:
            return
        
        self.current_exp["end_time"] = datetime.now().isoformat()
        self.current_exp["duration"] = time.time() - self.start_time
        self.current_exp["status"] = status
        self.current_exp["final_metrics"] = final_metrics
        
        # 保存到文件
        exp_file = self.log_dir / f"{self.current_exp['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(exp_file, "w") as f:
            json.dump(self.current_exp, f, indent=2, default=str)
        
        print(f"Experiment saved to {exp_file}")
        self.current_exp = None
    
    def load_experiments(self) -> List[Dict]:
        """加载所有实验"""
        experiments = []
        for file in self.log_dir.glob("*.json"):
            with open(file) as f:
                experiments.append(json.load(f))
        return experiments
    
    def compare_experiments(
        self,
        exp_names: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """对比实验结果
        
        Args:
            exp_names: 要对比的实验名称列表（None = 全部）
            metrics: 要对比的指标列表
        
        Returns:
            对比表格
        """
        experiments = self.load_experiments()
        
        if exp_names:
            experiments = [e for e in experiments if e["name"] in exp_names]
        
        if not experiments:
            return pd.DataFrame()
        
        # 提取关键信息
        rows = []
        for exp in experiments:
            row = {
                "name": exp["name"],
                "status": exp["status"],
                "duration": exp.get("duration", 0),
                "tags": ",".join(exp.get("tags", [])),
            }
            
            # 添加配置
            for key, value in exp.get("config", {}).items():
                if isinstance(value, (str, int, float, bool)):
                    row[f"config.{key}"] = value
            
            # 添加最终指标
            for key, value in exp.get("final_metrics", {}).items():
                row[key] = value
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # 筛选指标
        if metrics:
            cols = ["name", "status", "duration"] + [m for m in metrics if m in df.columns]
            df = df[cols]
        
        return df
    
    def generate_report(
        self,
        output_file: Optional[str] = None,
        exp_names: Optional[List[str]] = None,
    ) -> str:
        """生成实验报告
        
        Args:
            output_file: 输出文件路径
            exp_names: 要包含的实验
        
        Returns:
            Markdown 格式报告
        """
        experiments = self.load_experiments()
        
        if exp_names:
            experiments = [e for e in experiments if e["name"] in exp_names]
        
        if not experiments:
            return "No experiments found."
        
        # 生成 Markdown
        lines = [
            "# 实验报告",
            f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"\n实验数量: {len(experiments)}",
            "\n---\n",
        ]
        
        # 汇总表格
        lines.append("## 结果汇总\n")
        df = self.compare_experiments(exp_names)
        lines.append(df.to_markdown(index=False))
        lines.append("\n")
        
        # 每个实验详情
        lines.append("## 实验详情\n")
        for exp in experiments:
            lines.append(f"### {exp['name']}\n")
            lines.append(f"- **状态**: {exp['status']}")
            lines.append(f"- **时长**: {exp.get('duration', 0):.1f}s")
            lines.append(f"- **描述**: {exp.get('description', 'N/A')}")
            lines.append(f"- **标签**: {', '.join(exp.get('tags', []))}")
            
            lines.append("\n**配置**:")
            for key, value in exp.get("config", {}).items():
                if isinstance(value, (str, int, float, bool)):
                    lines.append(f"  - {key}: {value}")
            
            lines.append("\n**最终指标**:")
            for key, value in exp.get("final_metrics", {}).items():
                lines.append(f"  - {key}: {value:.6f}")
            
            lines.append("\n---\n")
        
        report = "\n".join(lines)
        
        if output_file:
            with open(output_file, "w") as f:
                f.write(report)
            print(f"Report saved to {output_file}")
        
        return report


class ModelComparator:
    """模型对比器"""
    
    def __init__(self, recorder: Optional[ExperimentRecorder] = None):
        self.recorder = recorder or ExperimentRecorder()
    
    def compare_models(
        self,
        model_results: Dict[str, Dict[str, float]],
        sort_by: str = "auc",
        ascending: bool = False,
    ) -> pd.DataFrame:
        """对比多个模型的结果
        
        Args:
            model_results: {model_name: {metric: value}}
            sort_by: 排序指标
            ascending: 是否升序
        
        Returns:
            对比表格
        """
        df = pd.DataFrame(model_results).T
        df.index.name = "model"
        df = df.reset_index()
        
        if sort_by in df.columns:
            df = df.sort_values(sort_by, ascending=ascending)
        
        return df
    
    def compare_encoders(
        self,
        encoder_results: Dict[str, Dict[str, float]],
    ) -> pd.DataFrame:
        """对比不同编码器的结果"""
        return self.compare_models(encoder_results, sort_by="auc")
    
    def statistical_significance_test(
        self,
        results_a: List[float],
        results_b: List[float],
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """统计显著性检验
        
        Args:
            results_a: 模型 A 的多次运行结果
            results_b: 模型 B 的多次运行结果
            alpha: 显著性水平
        
        Returns:
            检验结果
        """
        from scipy import stats
        
        # t 检验
        t_stat, p_value = stats.ttest_ind(results_a, results_b)
        
        # Wilcoxon 检验（非参数）
        try:
            w_stat, w_p_value = stats.ranksums(results_a, results_b)
        except Exception:
            w_stat, w_p_value = None, None
        
        return {
            "mean_a": np.mean(results_a),
            "mean_b": np.mean(results_b),
            "std_a": np.std(results_a),
            "std_b": np.std(results_b),
            "t_statistic": t_stat,
            "p_value": p_value,
            "wilcoxon_stat": w_stat,
            "wilcoxon_p_value": w_p_value,
            "significant": p_value < alpha,
            "improvement": (np.mean(results_a) - np.mean(results_b)) / np.mean(results_b) * 100,
        }
