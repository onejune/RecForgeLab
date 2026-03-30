#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IVR v16 数据集预处理脚本
- 所有特征当类别特征
- 一次性编码并保存，后续加载更快

用法:
    python workshop/preprocess_ivr.py
"""

import os
import json
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def encode_features(df, encoders=None, label_cols=['click_label', 'ctcvr_label']):
    """
    编码所有特征为整数
    
    Args:
        df: DataFrame
        encoders: 已有的编码器（用于测试集），None 表示训练集
        label_cols: 标签列名
    
    Returns:
        encoded_df: 编码后的 DataFrame
        encoders: 编码器字典
        vocab_sizes: 词表大小字典
    """
    # 特征列（排除标签）
    feature_cols = [c for c in df.columns if c not in label_cols]
    
    if encoders is None:
        # 训练集：构建编码器
        encoders = {}
        vocab_sizes = {}
        
        for col in tqdm(feature_cols, desc="Building encoders"):
            # 填充缺失值
            if df[col].dtype == 'object':
                df[col] = df[col].fillna('__UNKNOWN__')
                # 字符串特征：取唯一值
                unique_vals = df[col].astype(str).unique().tolist()
                if '__UNKNOWN__' not in unique_vals:
                    unique_vals.append('__UNKNOWN__')
            else:
                df[col] = df[col].fillna(-1)
                unique_vals = df[col].astype(str).unique().tolist()
            
            # 构建映射
            encoders[col] = {v: i for i, v in enumerate(unique_vals)}
            vocab_sizes[col] = len(unique_vals)
    else:
        # 测试集：使用已有编码器
        vocab_sizes = {col: len(enc) for col, enc in encoders.items()}
    
    # 编码
    encoded_data = {}
    for col in tqdm(feature_cols, desc="Encoding features"):
        if df[col].dtype == 'object':
            vals = df[col].fillna('__UNKNOWN__').astype(str).values
        else:
            vals = df[col].fillna(-1).astype(str).values
        
        # 映射，未知值映射到 0
        mapping = encoders[col]
        encoded_data[col] = np.array([mapping.get(v, 0) for v in vals], dtype=np.int32)
    
    # 标签
    for col in label_cols:
        if col in df.columns:
            encoded_data[col] = df[col].values.astype(np.float32)
    
    return pd.DataFrame(encoded_data), encoders, vocab_sizes


def main():
    input_root = Path('/mnt/workspace/dataset/ivr_sample_v16_ctcvr')
    output_root = Path('/mnt/workspace/dataset/ivr_sample_v16_ctcvr_sample')
    
    # 创建输出目录
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / 'train').mkdir(exist_ok=True)
    (output_root / 'test').mkdir(exist_ok=True)
    
    print("=" * 60)
    print("IVR v16 数据集预处理")
    print("=" * 60)
    print(f"输入: {input_root}")
    print(f"输出: {output_root}")
    
    # 1. 处理训练集
    print("\n[训练集]")
    train_files = sorted(input_root.glob('train/*.parquet'))
    print(f"  文件数: {len(train_files)}")
    
    all_encoders = None
    all_vocab_sizes = None
    
    for i, train_file in enumerate(train_files):
        print(f"\n  处理 {train_file.name}...")
        
        # 读取
        df = pd.read_parquet(train_file)
        print(f"    原始: {len(df):,} 行, {len(df.columns)} 列")
        
        # 编码
        encoded_df, encoders, vocab_sizes = encode_features(df, encoders=all_encoders)
        
        if all_encoders is None:
            all_encoders = encoders
            all_vocab_sizes = vocab_sizes
        
        print(f"    编码后: {len(encoded_df):,} 行, {len(encoded_df.columns)} 列")
        
        # 保存
        output_file = output_root / 'train' / train_file.name
        encoded_df.to_parquet(output_file, index=False)
        print(f"    保存: {output_file}")
    
    # 2. 处理测试集
    print("\n[测试集]")
    test_files = sorted(input_root.glob('test/*.parquet'))
    print(f"  文件数: {len(test_files)}")
    
    for test_file in test_files:
        print(f"\n  处理 {test_file.name}...")
        
        # 读取
        df = pd.read_parquet(test_file)
        print(f"    原始: {len(df):,} 行, {len(df.columns)} 列")
        
        # 编码（使用训练集的编码器）
        encoded_df, _, _ = encode_features(df, encoders=all_encoders)
        print(f"    编码后: {len(encoded_df):,} 行, {len(encoded_df.columns)} 列")
        
        # 保存
        output_file = output_root / 'test' / test_file.name
        encoded_df.to_parquet(output_file, index=False)
        print(f"    保存: {output_file}")
    
    # 3. 保存编码器和元信息
    print("\n[保存元信息]")
    
    # 编码器
    encoder_file = output_root / 'encoders.pkl'
    with open(encoder_file, 'wb') as f:
        pickle.dump(all_encoders, f)
    print(f"  编码器: {encoder_file}")
    
    # 词表大小
    vocab_file = output_root / 'vocab_sizes.json'
    with open(vocab_file, 'w') as f:
        json.dump(all_vocab_sizes, f, indent=2)
    print(f"  词表大小: {vocab_file}")
    
    # 特征列
    feature_cols = list(all_encoders.keys())
    meta_file = output_root / 'meta.json'
    meta = {
        'feature_cols': feature_cols,
        'n_features': len(feature_cols),
        'label_cols': ['click_label', 'ctcvr_label'],
    }
    with open(meta_file, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  元信息: {meta_file}")
    
    print("\n" + "=" * 60)
    print("预处理完成！")
    print("=" * 60)
    print(f"特征数: {len(feature_cols)}")
    print(f"总词表大小: {sum(all_vocab_sizes.values()):,}")


if __name__ == '__main__':
    main()
