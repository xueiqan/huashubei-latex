# C题问题1论文手交接清单

更新时间：2026-08-08




### 3. 当前结果表

- [forecast_window_selection.csv](D:/PyCharm/PyCharm2026.2/Project/huashubei/question/question_01/outputs/tables/forecast_window_selection.csv)：窗口选择；
- [forecast_metrics.csv](D:/PyCharm/PyCharm2026.2/Project/huashubei/question/question_01/outputs/tables/forecast_metrics.csv)：验证/测试WAPE、RMSE、MAE；
- `forecast_predictions.csv`、`forecast_hierarchy_predictions.csv`：逐小时预测和分层边际；
- `forecast_hierarchy_metrics.csv`、`forecast_aggregation_sensitivity.csv`：聚合尺度和层级指标；
- [dispatch_summary.csv](D:/PyCharm/PyCharm2026.2/Project/huashubei/question/question_01/outputs/tables/dispatch_summary.csv)：调度目标值和资源摘要；
- `dispatch_assignments.csv`、`dispatch_resource_profile.csv`：任务分配和资源剖面；
- [validation_summary.csv](D:/PyCharm/PyCharm2026.2/Project/huashubei/question/question_01/outputs/tables/validation_summary.csv)：全部检验状态和证据；
- [validation_critical_pressure.csv](D:/PyCharm/PyCharm2026.2/Project/huashubei/question/question_01/outputs/tables/validation_critical_pressure.csv)：压力下界与容量上界；
- `validation_dispatch_certificate.csv`及其他`validation_*.csv`：调度约束和检验明细。

### 4. 当前论文图

- `outputs/figures/q1_group1_demand_structure.pdf`：需求结构、累计矩阵、验证期—测试期预测；
- `outputs/figures/q1_group2_core_results.pdf`：分层代表任务调度、执行矩阵、GPU利用率和资源瓶颈；
- `outputs/figures/q1_group3_validation_robustness.pdf`：84窗口检验、分层恢复和压力区间。

## 四、给论文手的核心数字

### 1. 预测模型

`H*=72`由验证期总体WAPE在候选窗口`24/72/168/336`中选出；测试期没有参与窗口选择。

| 数据区间 | 分层局部均值WAPE/RMSE/MAE | 24小时同刻基线WAPE/RMSE/MAE |
|---|---:|---:|
| 验证期2352—2375 | 0.7241 / 45.7417 / 23.6915 | 1.0957 / 69.9249 / 35.8495 |
| 测试期2376—2399 | 0.6519 / 53.4704 / 26.3323 | 0.8740 / 72.8628 / 35.3032 |

可写结论：在当前验证期和测试期样本上，分层局部均值相对24小时同刻基线的WAPE、RMSE和MAE均更低。

### 2. 分层恢复检验

- 84个滚动窗口中，分层模型相对24小时基线的WAPE差值全部小于0：`84/84<0`；
- 单侧Wilcoxon检验`p=8.55328×10^-16`；
- 分层恢复与直接预测的中位WAPE差为`-3.04989×10^-4`，`p=0.21625`；
- 区域和任务类型最大闭合误差均为`6.82121×10^-13`，18类汇总误差为`4.11035×10^-10`。

可写结论：分层恢复保持区域、任务类型和系统边际一致，在18类序列上没有显著精度损失。

### 3. 基础算力调度

- 调度任务数：538；
- 第一目标：`F1*=0 GPU·h`；
- 迁移任务数：0；本地执行率：100%；
- 第二目标：`F2*=3 h`；
- 两阶段求解状态均为最优，MIP gap为0；
- 任务唯一执行、网络时延、最早开工、LatestFinish、实时立即启动、终端边界、GPU、IT和设施功率约束的违反数均为0；
- `2406`小时未被占用。

可写结论：在问题1给定的基础算力约束下，2376—2399小时到达任务存在全本地可行解，因此最优迁移工作量为0；弹性任务总等待时间为3小时。

### 4. 压力边界

两种时间粒度均得到相同的证据边界：

- 已验证可行下界：`α=1.0103`；
- 保持全本地执行的容量上界：`α=1.6320`；
- 系统整体容量上界：`α=2.6031`；
- `ExactCriticalValueSolved=False`。

可写结论：当前结果给出已验证下界和容量上界构成的压力区间证据，不能把`1.6320`或`2.6031`写成精确临界压力。

## 五、论文表述边界

### 可以写

- “在当前验证/测试划分和数据范围内，72小时历史窗口的分层局部均值模型优于24小时同刻基线。”
- “分层恢复满足区域、任务类型和系统边际闭合，并且相对直接预测没有显著精度损失。”
- “在基础算力调度模型下，最优解实现100%本地执行，`F1*=0`。”
- “不同区域的GPU、AI类IT容量和设施功率紧张程度存在异质性。”

### 不能写

- 不能把`H*=72`写成对所有场景都普遍最优；
- 不能把分层恢复写成相对直接预测“显著提升”，因为`p=0.21625`；
- 不能把压力图的上界写成精确临界压力；
- 不能把浅色压力区间线段写成整段已验证可行；
- 不能写“占用了第2406小时”；正确说法是允许在2406时刻结束，但本次结果没有占用2406小时；
- 不能只凭`F1*=0`推断网络时延没有作用；正确含义是在当前容量和候选区域约束下无需迁移；
- 不能把当前代码的`HierarchicalLocalMean`直接改名为复合泊松，除非同步修改模型代码、检验、结果表和论文公式。

## 六、建模手交接时的验收顺序


3. 以`forecast_metrics.csv`和`dispatch_summary.csv`为正文数字唯一来源，不手抄图片上的近似值；
4. 打开`validation_summary.csv`，区分`PASS`和`QUALIFIED`，不能把`QUALIFIED`改写成无条件证明；
5. 检查3份PDF的标题、字体、图例和结论是否与正文一致；
6. 论文完成后重新运行一次Q1入口，确认正文数字没有脱离最新输出表；
7. 最终交稿前保留代码、输入、结果表、PDF和日志的对应关系，避免只提交图片而无法追溯。
