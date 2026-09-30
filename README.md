# 华数杯C题：多区域数据中心算储电协同优化

本仓库整理了2026年第七届华数杯数学建模竞赛C题“面向算电协同的多目标调度优化研究”的Python程序、LaTeX论文、计算结果与复核材料。研究围绕多区域数据中心的算力任务调度、碳排放、网络时延、新能源消纳和储能运行展开。

项目可用于学习数学建模的完整计算流程，以及研究算力、储能与电力之间的多目标权衡。仓库包含竞赛提交时的冻结材料和赛后修订内容，使用时应核对代码与结果版本。

## 研究内容

| 问题 | 主要内容 |
| --- | --- |
| 问题一 | 分层需求预测、基础算力调度与压力边界检验 |
| 问题二 | 成本、碳排放、网络时延和新能源利用的多目标调度；精确求解与启发式对照 |
| 问题三 | 给定IT负荷下的储能、购售电协同优化与能源约束复核 |
| 问题四 | 算力任务、储能与电力的联合优化；顺序基准对照、情景分析和独立审计 |

每问均包含预处理、模型、检验与绘图模块。当前可浏览的题目程序共24个Python文件；历史源码备份另存于结果目录，不作为默认运行入口。

## 从哪里开始

- **查看程序**：[huashubei code](huashubei%20code/)。
- **阅读论文**：[main.pdf](main.pdf)；LaTeX入口为[main.tex](main.tex)，正文位于[sections](sections/)，图片位于[figures](figures/)。
- **查看提交材料与归档**：[haushubei submit](haushubei%20submit/)。
- **区分竞赛原版与赛后修订版**：[归档说明](haushubei%20submit/归档/2026华数杯_C题_CCM2601033_20260818/00_归档说明/README_归档说明.md)。

目录名中的空格以及`haushubei submit`的现有拼写均按仓库实际结构保留。

```text
huashubei-latex/
├── README.md
├── huashubei code/
│   ├── pyproject.toml
│   ├── uv.lock
│   ├── main.py
│   └── question/
│       ├── question_01/
│       ├── question_02/
│       ├── question_03/
│       └── question_04/
│           ├── main.py
│           ├── preprocess.py
│           ├── model.py
│           ├── validation.py
│           ├── plot.py
│           ├── data/
│           └── outputs/
├── haushubei submit/
├── main.tex
├── main.pdf
├── references.tex
├── sections/
└── figures/
```

## 运行环境与数据

Python版本要求为3.12及以上。依赖由[pyproject.toml](huashubei%20code/pyproject.toml)声明，[uv.lock](huashubei%20code/uv.lock)记录锁定版本，包括NumPy、pandas、SciPy、HiGHS/highspy、Matplotlib、Seaborn和OpenPyXL。

在已经安装Git、Python与uv的环境中，下载并进入代码目录：

```powershell
git clone https://github.com/xueiqan/huashubei-latex.git
Set-Location './huashubei-latex/huashubei code'
uv sync --frozen
```

通过网页下载ZIP时，先解压仓库，再进入其中的`huashubei code`目录。也可以用PyCharm打开该目录，选用uv同步出的项目环境，按问题一至问题四依次运行各题的`main.py`。

问题一读取的6份原始Excel已位于[question/question_01/data/raw](huashubei%20code/question/question_01/data/raw/)：

```text
GPU_information.xlsx
network_latency.xlsx
power_mapping.xlsx
region_time_data.xlsx
storage_information.xlsx
workload_trace.xlsx
```

保留原始文件名、工作表和数据内容。问题一生成`question/question_01/data/processed/shared/`中的共享输入，后续问题在此基础上处理各自的建模输入。

## 运行与复核

建议在单独的工作副本中运行，以保留随仓库提供的历史结果。进入`huashubei code`后，按顺序执行：

```powershell
uv run python question/question_01/main.py
uv run python question/question_02/main.py
uv run python question/question_03/main.py
uv run python question/question_04/main.py
```

这些命令会启动实际计算，各题流程为：

```text
preprocess.py → model.py → validation.py → plot.py
```

问题二至问题四的入口在检验或绘图失败后可能只记录警告，并仍返回0。完成复核时需检查日志、检验报告和预期结果文件，不能仅凭进程退出码判断全部阶段通过。

### 问题四的专项入口

查看参数：

```powershell
uv run python question/question_04/model.py --help
uv run python question/question_04/validation.py --help
```

源码提供了结构自检、既有基准审计和独立检验入口：

```powershell
uv run python question/question_04/model.py --self-test
uv run python question/question_04/model.py --baseline-audit
uv run python question/question_04/validation.py --skip-reference-windows
```

结构自检不等于完整求解。基准审计和独立检验需要对应版本的完整结果；缺少文件时应根据报错补齐。最后一条命令跳过代表窗口MILP对照，仍会审计已有结果。默认运行`validation.py`还会求解代表窗口，耗时取决于求解设置和机器性能。

## 结果与版本边界

各题结果位于各自的`outputs/`，包括结果表、图形、日志、检验报告及部分历史版本目录。问题四的模型和检验程序当前以`matheuristic_v4_doccompliant`为主要命名空间，仓库同时保留V3结果与竞赛支撑材料。

- 竞赛提交ZIP是冻结版本，不代表当前展开源码的最新版本。
- 当前快照中的V4目录包含顺序方案与既有联合任务方案的能源复算材料，未见V4全量联合运行的完成标记；这些材料不能替代完整V4联合运行结果。
- 归档检验报告记录的是对应历史运行，不能直接作为修改后源码在新环境中通过检验的证明。
- 论文数值应追溯到同版本结果表与独立核验材料，不从图片估读，不混用不同版本的基准与联合方案。
- 数学启发式结果在通过检验后可说明给定数据、参数和边界下的可行性；不据此宣称全局最优。
- 当前README依据仓库源码与文件结构整理，尚未完成新环境下的全流程复现验证。

LaTeX主文件指定XeLaTeX，并使用Windows中文字体配置以及Times New Roman/Consolas。编译需要相应TeX环境与字体；仓库已有PDF可直接阅读。主文件中的参赛编号、题名和正文是本项目内容，复用排版时需自行替换。本项目并非赛方发布的官方模板。

## 许可与反馈

当前仓库尚未提供统一LICENSE。程序、团队论文、赛题数据、赛方文件及引用资料应分别核对权利归属和适用授权；本说明不替这些材料设定新的许可。

欢迎通过[Issues](https://github.com/xueiqan/huashubei-latex/issues)反馈复现问题、结果口径差异和文档错误，或提交改进程序与说明的Pull Request。反馈时请附代码提交版本、问题编号、运行命令、Python及依赖版本、相关日志和预期行为，便于复核。
