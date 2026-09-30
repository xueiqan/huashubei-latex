# 华数杯数学建模代码说明

## 1. 代码简介

本项目用于组织华数杯数学建模竞赛的题目数据处理、模型计算、模型检验和结果绘图工作。项目采用“总入口 + 分题入口”的结构：总入口负责统一调度，各题目录负责本题的完整计算流程。

项目按数据使用范围划分预处理职责：多道题共同使用的基础数据在项目外层统一处理；只服务于某一道题的数据和变量在对应题目目录内处理。共享数据经过外层预处理后，由需要使用它的题目继续完成本题处理，不同题目之间不直接读取彼此的中间结果。

## 2. 运行环境

- Python 3.12及以上
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`

依赖配置位于 `pyproject.toml`。在项目根目录执行以下命令即可同步运行环境：

```powershell
uv sync
```

## 3. 项目结构

```text
huashubei/
├── data/
│   ├── raw/                    # 多道题共同使用的原始数据
│   └── processed/              # 共享数据预处理结果
├── outputs/
│   └── logs/                   # 总流程和共享预处理日志
├── main.py                     # 全部题目统一入口
├── preprocess.py               # 共享数据预处理入口
├── pyproject.toml              # 项目和依赖配置
└── question/
    ├── question_01/
    ├── question_02/
    ├── question_03/
    └── question_04/
        ├── data/
        │   ├── raw/            # 本题专属原始数据
        │   └── processed/      # 本题建模输入数据
        ├── outputs/
        │   ├── tables/         # 本题结果表
        │   ├── figures/        # 本题结果图
        │   └── logs/           # 本题运行日志
        ├── main.py             # 本题流程入口
        ├── preprocess.py       # 本题专属预处理
        ├── model.py            # 本题模型计算
        ├── validation.py       # 模型检验和灵敏度分析
        └── plot.py             # 结果绘图
```

四个题目目录具有相同的代码结构，题目之间的原始数据、处理结果、结果表、图片和日志分别存放，便于单独运行和复核。

## 4. 数据处理流程

项目采用以下数据流：

```text
共享原始数据 data/raw/
        ↓
外层 preprocess.py
        ↓
共享处理结果 data/processed/
        ↓
各题 preprocess.py ← 各题专属原始数据 question/question_xx/data/raw/
        ↓
各题处理结果 question/question_xx/data/processed/
        ↓
各题 model.py
        ↓
结果表、模型检验和结果图
```

外层预处理负责多道题都需要的基础读取、字段整理、公共质量检查和共享变量构造。各题预处理负责本题的样本筛选、特征构造、指标转换和建模数据整理。模型只读取本题 `data/processed/` 中的数据，计算结果只写入本题 `outputs/`。

这种组织方式能够同时支持两类数据：

1. 一份附件被多道题共同使用时，先在外层完成公共处理，再分流到相关题目。
2. 某道题有独立附件或独特处理规则时，直接在该题的 `preprocess.py` 中完成处理。

## 5. 运行方式

在项目根目录执行：

```powershell
Set-Location 'D:\PyCharm\PyCharm2026.2\Project\huashubei'
```

运行全部题目：

```powershell
uv run python main.py
```

只运行指定题目：

```powershell
uv run python main.py --question 1
uv run python main.py --question 2
uv run python main.py --question 3
uv run python main.py --question 4
```

也可以直接运行某一题：

```powershell
uv run python question\question_01\main.py
```

如果共享数据已经完成预处理，可以跳过外层共享预处理：

```powershell
uv run python main.py --skip-shared-preprocess
```

每道题的执行顺序为：

```text
preprocess.py → model.py → validation.py → plot.py
```

## 6. 输出结果

总入口和共享预处理的日志写入项目根目录的 `outputs/logs/`。每道题的结果分别写入对应目录：

- `question/question_xx/outputs/tables/`：模型参数、统计结果、预测结果和检验结果表。
- `question/question_xx/outputs/figures/`：模型关系图、拟合图、误差图和敏感性分析图等。
- `question/question_xx/outputs/logs/`：本题预处理、模型和流程运行日志。

不同题目的结果不会混写到同一个结果目录中，论文中的表格和图片可以按题号直接追溯到对应题目的输出目录。

## 7. 复现要求

复现实验时，应保持原始附件内容不变，并在同一项目根目录执行对应运行命令。程序会按照固定的数据路径读取输入，按固定的阶段顺序生成处理数据、结果表、图形和日志。论文中的数值应以程序输出结果为准，并结合对应日志和验证结果进行核对。
