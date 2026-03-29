# RecForgeLab

> 面向 DSP 程序化广告的 CTR/CVR 预估实验框架

借鉴 [RecBole](https://github.com/RUCAIBox/RecBole) 的设计思想，专为广告推荐场景定制，核心目标是**极低门槛地扩展实验**——新增模型、数据集、指标，只需写一个文件 + 一行注册。

---

## 核心特性

| 特性 | 说明 |
|------|------|
| **插件式注册** | `@register_model` / `@register_dataset`，新增无需改框架代码 |
| **配置驱动** | YAML 配置 + 命令行覆盖，支持继承、网格搜索、多实验对比 |
| **16 种连续特征编码器** | 从 Scalar 到 AutoDis/FTTransformer，一行切换 |
| **12 个内置模型** | CTR 单任务 + 多任务，覆盖主流架构 |
| **完整评估指标** | AUC / GAUC / LogLoss / PCOC / ECE |
| **多种实验模式** | 单次 / 多模型对比 / 超参搜索 / SSL 两阶段训练 |
| **训练增强** | AMP 混合精度 / TensorBoard / Checkpoint / 早停 |

---

## 快速开始

### 安装依赖

```bash
pip install torch pandas pyarrow scikit-learn pyyaml tensorboard
```

### 运行第一个实验

```bash
cd /mnt/workspace/open_research

# 单模型实验
python recforgelab/run.py --model deepfm --dataset criteo

# 多模型对比
python recforgelab/run.py --config recforgelab/config/experiment/compare_models.yaml --mode compare

# 超参网格搜索
python recforgelab/run.py --config recforgelab/config/experiment/grid_search.yaml --mode grid_search

# 查看所有注册模型
python recforgelab/run.py --list_models
```

### Python API

```python
import sys
sys.path.insert(0, '/mnt/workspace/open_research')

from recforgelab.utils.config import Config
from recforgelab.model import get_model
from recforgelab.data import create_dataset, create_dataloader
from recforgelab.trainer import Trainer

# 配置
config = Config(config_dict={
    'model': 'esmm',
    'data_path': '/mnt/data/.../ivr_sample_v16_ctcvr/',
    'sparse_features': ['cat_0', 'cat_1', ...],
    'dense_features': ['num_0', 'num_1', ...],
    'ctr_label_field': 'click_label',
    'cvr_label_field': 'ctcvr_label',
    'encoder_type': 'log',
    'embedding_size': 16,
    'epochs': 5,
})

# 数据
train_ds = create_dataset(config, phase='train')
valid_ds = create_dataset(config, phase='valid', encoders=train_ds.feature_encoders)
train_loader = create_dataloader(train_ds, config, shuffle=True)
valid_loader = create_dataloader(valid_ds, config, shuffle=False)

# 模型 + 训练
model = get_model('esmm')(config, train_ds).to(config['device'])
trainer = Trainer(config, model)
trainer.train(train_loader, valid_loader)
```

---

## 项目结构

```
recforgelab/
├── run.py                      # 统一入口（单次/对比/网格搜索/SSL）
│
├── config/                     # 配置文件
│   ├── dataset/                # 数据集配置
│   │   ├── criteo.yaml
│   │   ├── ali_ccp.yaml
│   │   └── ivr_sample_v16.yaml
│   ├── model/                  # 模型默认超参
│   │   ├── deepfm.yaml
│   │   └── esmm.yaml
│   └── experiment/             # 实验配置（可直接运行）
│       ├── compare_models.yaml     # 多模型对比
│       ├── compare_encoders.yaml   # 编码器对比
│       ├── grid_search.yaml        # 超参搜索
│       ├── ssl_cvr.yaml            # SSL 两阶段训练
│       └── ctr_baseline.yaml       # CTR 基础实验
│
├── data/                       # 数据处理
│   ├── dataset.py              # DSPDataset + 注册机制
│   └── preprocess/             # 预处理工具
│
├── model/                      # 模型
│   ├── base.py                 # 基类 + 注册机制
│   ├── ctr/                    # CTR 单任务模型
│   │   ├── deepfm.py           # DeepFM
│   │   ├── dcn.py              # DCN / DCNv2
│   │   ├── autoint.py          # AutoInt / AutoInt+
│   │   └── xdeepfm.py          # xDeepFM (CIN + DNN)
│   ├── multitask/              # 多任务模型
│   │   ├── esmm.py             # ESMM / ESCM2
│   │   └── mmoe.py             # SharedBottom / MMoE / PLE / DirectCTCVR
│   ├── ssl/                    # 自监督学习
│   │   └── contrastive.py      # SSLContrastive / MoCo / UserBehavior
│   └── layers/                 # 通用层
│       ├── embedding.py        # FeatureEmbedding + 16种连续编码器
│       ├── mlp.py              # MLPLayers（含残差/BN/Dropout）
│       └── fm.py               # FM / CrossNetwork / CrossNetworkV2
│
├── trainer/
│   └── trainer.py              # Trainer（AMP/TensorBoard/Checkpoint/早停）
│
├── evaluator/
│   ├── metrics.py              # AUC/GAUC/PCOC/ECE/LogLoss/MSE/MAE
│   └── evaluator.py            # Evaluator
│
├── utils/
│   ├── config.py               # Config（继承/网格搜索/多实验/命令行覆盖）
│   ├── logger.py               # 日志
│   ├── enum.py                 # 枚举类型
│   └── experiment.py           # ExperimentRecorder / ModelComparator
│
└── experiments/                # 实验脚本（可直接运行）
    ├── run_ivr_multitask.py    # IVR v16 多任务对比
    ├── run_encoder_comparison.py
    ├── run_multitask_comparison.py
    └── run_comprehensive.py    # 综合实验（含 SSL）
```

---

## 内置模型

### CTR 单任务（6个）

| 模型 | 注册名 | 论文 | 特点 |
|------|--------|------|------|
| DeepFM | `deepfm` | IJCAI 2017 | FM + DNN |
| DCN | `dcn` | KDD 2017 | Cross Network + DNN |
| DCNv2 | `dcnv2` | WWW 2021 | 低秩 Cross + DNN |
| AutoInt | `autoint` | CIKM 2019 | 多头注意力特征交互 |
| AutoInt+ | `autoint+` | CIKM 2019 | AutoInt + DNN |
| xDeepFM | `xdeepfm` | KDD 2018 | CIN（向量级交叉）+ DNN |

### 多任务（6个）

| 模型 | 注册名 | 论文 | 特点 |
|------|--------|------|------|
| SharedBottom | `shared_bottom` | — | 共享底层 + 独立 Tower |
| ESMM | `esmm` | SIGIR 2018 | 全空间多任务，CTR × CVR = CTCVR |
| ESCM2 | `escm2` | SIGIR 2022 | ESMM + 反事实正则化 |
| MMoE | `mmoe` | KDD 2018 | 多门控混合专家 |
| PLE | `ple` | RecSys 2020 | 渐进式分层提取 |
| DirectCTCVR | `direct_ctcvr` | — | 直接预测 CTR + CTCVR |

---

## 连续特征编码器（16种）

通过 `encoder_type` 配置项一键切换，无需改模型代码：

```yaml
encoder_type: autodis   # 切换编码器
encoder_config:
  num_intervals: 16     # 编码器专属参数
```

| 编码器 | 注册名 | 说明 |
|--------|--------|------|
| 无编码 | `none` | 直接输入原始值 |
| 标准化 | `scalar` | Min-Max / Z-Score |
| 对数变换 | `log` | log(1+x)，适合计数类特征 |
| 分桶 | `bucket` | 等频/等宽分桶 → Embedding |
| 字段感知 | `field` | 字段级归一化 |
| NumericEmbedding | `numeric` | 数值 → Embedding（线性） |
| NumericEmbedding-Deep | `numeric_deep` | 数值 → Embedding（MLP） |
| NumericEmbedding-SiLU | `numeric_silu` | 数值 → Embedding（SiLU激活） |
| NumericEmbedding-LN | `numeric_ln` | 数值 → Embedding（LayerNorm） |
| NumericEmbedding-Ctx | `numeric_ctx` | 数值 → Embedding（上下文感知） |
| AutoDis | `autodis` | 自动离散化（SIGIR 2021） |
| FT-Transformer | `fttransformer` | 数值 → Token（NeurIPS 2021） |
| Periodic | `periodic` | 周期性编码（正弦/余弦） |
| PLR | `plr` | Piecewise Linear Representation |
| DLRM | `dlrm` | DLRM 风格数值处理 |
| MinMax | `minmax` | 全局 Min-Max 归一化 |

---

## 配置系统

### 配置优先级（从高到低）

```
命令行参数 > config_dict > 配置文件 > base_config（继承）> 默认值
```

### 配置继承

```yaml
# config/experiment/my_exp.yaml
base_config: config/dataset/ivr_sample_v16.yaml  # 继承数据集配置

model: esmm
encoder_type: autodis
epochs: 10
```

### 网格搜索

```yaml
# config/experiment/grid_search.yaml
model: deepfm
dataset: criteo

grid_search:
  learning_rate: [0.001, 0.0005, 0.0001]
  embedding_size: [16, 32]
  dropout_prob: [0.1, 0.2]
  # → 自动生成 3×2×2 = 12 组实验，按 valid_metric 排序输出
```

```bash
python run.py --config config/experiment/grid_search.yaml --mode grid_search
```

### 多实验对比

```yaml
# config/experiment/compare_models.yaml
dataset: criteo
embedding_size: 16
epochs: 10

experiments:
  - name: deepfm
    config: {model: deepfm}
  - name: autoint
    config: {model: autoint, num_heads: 4}
  - name: xdeepfm
    config: {model: xdeepfm, cin_layer_sizes: [128, 128, 64]}
```

```bash
python run.py --config config/experiment/compare_models.yaml --mode compare
```

### 命令行覆盖

```bash
# 覆盖任意配置项
python run.py --model deepfm --dataset criteo --learning_rate=0.0001 --epochs=5 --embedding_size=32
```

---

## 扩展指南

### 新增模型（3步）

**Step 1**：创建模型文件

```python
# recforgelab/model/ctr/my_model.py
from ..base import CTRModel, register_model
from ..layers import MLPLayers, FeatureEmbedding

@register_model("my_model")
class MyModel(CTRModel):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.embedding = FeatureEmbedding(config, dataset)
        input_dim = self.embedding.output_dim
        self.mlp = MLPLayers([input_dim, 256, 128, 1])

    def forward(self, batch):
        embed = self.embedding(batch)          # [B, num_fields, D]
        flat = embed.view(embed.size(0), -1)   # [B, num_fields * D]
        return self.mlp(flat)                  # [B, 1]

    def calculate_loss(self, batch):
        pred = self.forward(batch).squeeze(-1)
        label = batch[self.label_field].float()
        return F.binary_cross_entropy_with_logits(pred, label)

    def predict(self, batch):
        return torch.sigmoid(self.forward(batch)).squeeze(-1)
```

**Step 2**：在 `__init__.py` 加一行

```python
# recforgelab/model/ctr/__init__.py
from .my_model import MyModel   # 加这一行
```

**Step 3**：运行

```bash
python run.py --model my_model --dataset criteo
```

---

### 新增数据集（2步）

**Step 1**：注册数据集

```python
# recforgelab/data/datasets/my_dataset.py
from ..dataset import DSPDataset, register_dataset

@register_dataset("my_dataset")
class MyDataset(DSPDataset):
    # 可覆盖 _load_data / _build_features 等方法
    pass
```

**Step 2**：创建配置文件

```yaml
# recforgelab/config/dataset/my_dataset.yaml
dataset: my_dataset
data_path: /path/to/my/data
data_format: parquet          # parquet / csv / spark_dir
label_field: label
sparse_features: [user_id, item_id, category]
dense_features: [price, ctr_14d, cvr_7d]
encoder_type: log             # 推荐：计数类特征用 log
```

---

### 新增评估指标

```python
# recforgelab/evaluator/metrics.py
from . import register_metric

@register_metric("my_metric")
class MyMetric(BaseMetric):
    higher_is_better = True

    def calculate(self, labels, preds, groups=None):
        # 实现指标计算逻辑
        return float(my_score)
```

---

## 评估指标说明

| 指标 | 说明 | 越高越好 |
|------|------|---------|
| **AUC** | Area Under ROC Curve | ✅ |
| **GAUC** | Group AUC（按 user_id 分组加权） | ✅ |
| **LogLoss** | Binary Cross Entropy | ❌ |
| **PCOC** | Prediction / Click-Over-Click，校准度（=1.0 最佳） | 越接近1越好 |
| **ECE** | Expected Calibration Error，期望校准误差 | ❌ |
| **MSE** | Mean Squared Error | ❌ |
| **MAE** | Mean Absolute Error | ❌ |

---

## 数据集配置

### IVR Sample v16（内部数据集）

```yaml
# config/dataset/ivr_sample_v16.yaml
dataset: ivr_sample_v16
data_path: /mnt/data/oss_wanjun/pai_work/open_research/dataset/ivr_sample_v16_ctcvr/
data_format: spark_dir

label_field: ctcvr_label
ctr_label_field: click_label
cvr_label_field: ctcvr_label

sparse_features: [cat_0, cat_1, ..., cat_98]   # 99个类别特征
dense_features: [num_0, num_1, ..., num_25]     # 26个数值特征

encoder_type: log    # 计数类特征，推荐 log 变换

train_date_range: ["2026-02-16", "2026-03-16"]
test_date_range: ["2026-03-17", "2026-03-19"]
```

**数据集信息**：
- 训练集：~286万样本 / 测试集：~145万样本
- CTR 正样本率：~38.8% / CTCVR 正样本率：~7.1%
- 特征：99个稀疏特征 + 26个稠密特征（计数类，长尾分布）

### Criteo（公开数据集）

```yaml
dataset: criteo
data_path: /path/to/criteo
label_field: label
sparse_features: [C1, C2, ..., C26]
dense_features: [I1, I2, ..., I13]
encoder_type: minmax
```

### Ali-CCP（公开数据集）

```yaml
dataset: ali_ccp
data_path: /path/to/ali_ccp
ctr_label_field: click
cvr_label_field: purchase
sparse_features: [user_id, item_id, ...]
dense_features: [...]
```

---

## 训练器配置

```yaml
# 优化器
optimizer: adam          # adam / sgd / adagrad / rmsprop
learning_rate: 0.001
weight_decay: 1e-5

# 学习率调度
scheduler: cosine        # step / cosine / warmup_cosine / null
scheduler_config:
  T_max: 10              # cosine 周期

# 正则化
dropout_prob: 0.2
use_bn: true

# 训练增强
use_amp: true            # AMP 混合精度（GPU 加速）
grad_clip: 1.0           # 梯度裁剪

# 早停
early_stop_patience: 3
valid_metric: AUC

# 可视化
use_tensorboard: true
log_dir: ./logs

# 检查点
checkpoint_dir: ./saved
save_model: true
load_model: null         # 恢复训练：填写 checkpoint 路径
```

---

## 推荐系统常见优化方向

框架设计时充分考虑了以下实验方向，均可通过配置文件快速切换：

### 1. 特征工程
```bash
# 对比 16 种连续特征编码器
python run.py --config config/experiment/compare_encoders.yaml --mode compare
```

### 2. 模型结构
```bash
# 对比 CTR 单任务 vs 多任务
python run.py --config config/experiment/compare_models.yaml --mode compare
```

### 3. 超参调优
```bash
# 网格搜索 lr / batch_size / embedding_size / dropout
python run.py --config config/experiment/grid_search.yaml --mode grid_search
```

### 4. 多任务学习
```yaml
# 调整任务权重
tasks: [ctr, cvr, ctcvr]
task_weights: [1.0, 0.5, 1.0]
```

### 5. 自监督学习（SSL）
```bash
# 两阶段训练：对比学习预训练 → CVR 微调
python run.py --config config/experiment/ssl_cvr.yaml --mode ssl
```

### 6. 校准实验
```yaml
# 监控 PCOC 和 ECE
metrics: [AUC, GAUC, PCOC, ECE]
```

---

## 与 RecBole 的对比

| 方面 | RecBole | RecForgeLab |
|------|---------|-------------|
| 目标场景 | 通用推荐（协同过滤为主） | DSP 广告 CTR/CVR 预估 |
| 多任务学习 | ❌ | ✅ 6种多任务架构 |
| 连续特征编码 | Scalar/Bucket | ✅ 16种编码器 |
| 评估指标 | AUC/NDCG/HR | ✅ + GAUC/PCOC/ECE |
| 数据格式 | AtomicFile | ✅ Parquet/CSV/Spark目录 |
| 超参搜索 | ❌ | ✅ 内置网格搜索 |
| SSL 支持 | ❌ | ✅ InfoNCE/MoCo |
| 实验记录 | ❌ | ✅ 自动记录 + Markdown报告 |

---

## 参考论文

- **DeepFM**: Guo et al., IJCAI 2017
- **DCN**: Wang et al., KDD 2017
- **DCNv2**: Wang et al., WWW 2021
- **xDeepFM**: Lian et al., KDD 2018
- **AutoInt**: Song et al., CIKM 2019
- **ESMM**: Ma et al., SIGIR 2018
- **ESCM2**: Wang et al., SIGIR 2022
- **MMoE**: Ma et al., KDD 2018
- **PLE**: Tang et al., RecSys 2020
- **AutoDis**: Guo et al., SIGIR 2021
- **FT-Transformer**: Gorishniy et al., NeurIPS 2021
