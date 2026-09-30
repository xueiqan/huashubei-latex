# 华数杯C题Python程序

这里保存需求预测、算力调度、储能优化和算储电联合优化的计算程序。项目总览、论文和冻结提交材料的入口见[仓库README](../README.md)。

## 环境与运行

用PyCharm打开本目录，项目解释器选择本目录的`.venv/Scripts/python.exe`。已有本机环境已随目录移动；首次下载时，在本目录执行`uv sync --frozen`，使用`pyproject.toml`与`uv.lock`同步环境。Python要求3.12及以上。

推荐按问题一到问题四依次运行各题的`main.py`：

```powershell
uv run python question/question_01/main.py
uv run python question/question_02/main.py
uv run python question/question_03/main.py
uv run python question/question_04/main.py
```

已有环境且需要明确禁止依赖同步时，可用`uv run --no-sync python`代替上述`uv run python`。在PyCharm中也可直接右键对应`main.py`，选择“运行”。这些入口会启动实际计算，请保留历史成果并为求解预留时间。

## 源码、输入与结果

每问的处理顺序为`preprocess.py → model.py → validation.py → plot.py`。共24个活动Python文件；Q2另有精确/启发式对照、窗口启发式和隔离验证工具。

```text
question/
├── question_01/
├── question_02/
├── question_03/
└── question_04/
    ├── main.py
    ├── preprocess.py
    ├── model.py
    ├── validation.py
    ├── plot.py
    ├── data/
    │   ├── raw/
    │   └── processed/
    └── outputs/
```

Q1原始6份Excel位于[question/question_01/data/raw/](question/question_01/data/raw/)。Q1生成`question/question_01/data/processed/shared/`的共享输入，Q2—Q4各自读取这些共享数据并生成本题输入。保留文件名与内部层级，模型参数仍在相应源码中配置。

各题的结果表、图片、日志、正式版本目录及检查点位于自己的`outputs/`。历史Q1结果移至[archive/legacy-results/question_01](../archive/legacy-results/question_01/)，Q4旧源码移至[archive/source-backups/question_04](../archive/source-backups/question_04/)，这些旧目录不参与默认流程。

Q2的`validation_runs.py`可准备或归档K24/K72隔离验证。它按代码目录名称生成仓库根的`code_q2_validation_runs/`；该目录由实验命令生成，不包含在当前整理的正式源码目录中。原有Q2运行配置已改为项目相对路径，运行隔离实验前需先按该工具的`prepare`入口准备相应目录。

## Q4专项复核

```powershell
uv run python question/question_04/model.py --help
uv run python question/question_04/validation.py --help
uv run python question/question_04/model.py --self-test
uv run python question/question_04/model.py --baseline-audit
uv run python question/question_04/validation.py --skip-reference-windows
```

自检用于结构与算法局部检查；基准审计及独立检验依赖对应版本的完整结果。`--skip-reference-windows`跳过代表窗口MILP求解，仍会审计已有文件。默认独立检验可能启动代表窗口求解，不属于只读查看参数。

## 证据与版本

Q4当前源码采用`matheuristic_v4_doccompliant`，同时保留V3历史结果和固定任务方案的V4能源复算材料。完整联合搜索、固定任务能源复算、历史归档检验是不同的证据，不混用其完成状态或指标。

Q2—Q4主入口遇到检验/绘图失败可能只记录警告并返回0，请同时核对实际日志、报告和预期输出。模型检验通过不等于全局最优。

本次目录整理已检查移动后的Python环境、关键入口导入及源码结构，未重新执行四问完整求解。历次模型修改与验证范围见[CHANGELOG.md](CHANGELOG.md)。
