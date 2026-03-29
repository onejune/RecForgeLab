# workshop/ 目录说明

这个目录存放**针对具体数据集/场景的定制实验脚本**，是对 `run.py` 通用能力的补充。

---

## run.py vs workshop/ 的区别

| | `run.py` | `workshop/` |
|---|---|---|
| **定位** | 通用入口，适配任意模型/数据集 | 针对特定场景的定制脚本 |
| **配置方式** | YAML 配置文件 + 命令行参数 | 配置硬编码在脚本里，开箱即用 |
| **适用场景** | 标准实验流程 | 需要自定义逻辑（特殊预处理、特殊评估、多阶段训练）|
| **灵活性** | 高（通过配置控制一切）| 低（但更直接，改起来简单）|

**简单理解**：
- `run.py` = 通用工具，适合大多数情况
- `workshop/` = 针对具体任务的一次性脚本，适合需要定制逻辑时

---

## 当前脚本说明

### `run_ivr_multitask.py` ✅ 可用
针对 **IVR Sample v16** 数据集的多任务对比实验，配置已硬编码（数据路径、特征列表等），直接运行即可。

```bash
cd /mnt/workspace/open_research

# 运行单个模型
python recforgelab/workshop/run_ivr_multitask.py --model esmm

# 对比所有多任务模型
python recforgelab/workshop/run_ivr_multitask.py --compare_models

# 对比不同连续特征编码器（固定 esmm 模型）
python recforgelab/workshop/run_ivr_multitask.py --compare_encoders --model esmm
```

**适合场景**：快速在 IVR 数据集上跑实验，不想每次写配置文件。

---

### `run_encoder_comparison.py` ⚠️ 早期版本
早期写的编码器对比脚本，功能已被 `run.py --mode compare` + `config/experiment/compare_encoders.yaml` 覆盖。

**推荐替代**：
```bash
python recforgelab/run.py --config recforgelab/config/experiment/compare_encoders.yaml --mode compare
```

---

### `run_multitask_comparison.py` ⚠️ 早期版本
早期写的多任务模型对比脚本，功能已被 `run.py --mode compare` + `config/experiment/compare_models.yaml` 覆盖。

**推荐替代**：
```bash
python recforgelab/run.py --config recforgelab/config/experiment/compare_models.yaml --mode compare
```

---

### `run_comprehensive.py` ⚠️ 早期版本
综合实验脚本（多模型/编码器/SSL），功能已被 `run.py` 全面覆盖。

**推荐替代**：
```bash
python recforgelab/run.py --config recforgelab/config/experiment/ssl_cvr.yaml --mode ssl
```

---

## 什么时候应该在这里写新脚本？

以下情况适合在 `workshop/` 新建脚本，而不是用 `run.py`：

1. **特殊数据预处理**：需要在加载数据后做额外处理（如合并多个数据集、自定义负采样策略）
2. **多阶段训练**：SSL 预训练 → 微调，且需要在两阶段之间做额外操作
3. **自定义评估逻辑**：需要按业务规则分组评估（如按广告主、按投放渠道）
4. **一次性分析脚本**：数据分布分析、特征重要性分析、模型输出可视化
5. **特定数据集的固定配置**：像 `run_ivr_multitask.py` 这样，把 IVR 数据集的所有配置固化，方便团队成员直接运行

---

## 推荐使用方式

```
日常实验  →  run.py + config/experiment/*.yaml
IVR 专项  →  workshop/run_ivr_multitask.py
自定义逻辑 →  在这里新建脚本
```
