# -*- coding: utf-8 -*-
"""
日志模块
"""

import os
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def init_logger(
    config,
    log_file: Optional[str] = None,
    log_level: int = logging.INFO,
):
    """初始化日志器
    
    Args:
        config: 配置对象
        log_file: 日志文件路径
        log_level: 日志级别
    """
    log_dir = Path(config["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{config['experiment_name']}_{timestamp}.log"
    
    # 创建 logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # 清除已有的 handlers
    logger.handlers.clear()
    
    # 控制台 handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_format = logging.Formatter(
        "[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # 文件 handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(log_level)
    file_format = logging.Formatter(
        "[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    return logger


def set_color(text: str, color: str) -> str:
    """设置终端输出颜色
    
    Args:
        text: 文本内容
        color: 颜色名称
    
    Returns:
        带颜色标记的文本
    """
    color_codes = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
    }
    reset_code = "\033[0m"
    
    if color in color_codes:
        return f"{color_codes[color]}{text}{reset_code}"
    return text


def get_logger(name: str = None) -> logging.Logger:
    """获取日志器"""
    return logging.getLogger(name)
