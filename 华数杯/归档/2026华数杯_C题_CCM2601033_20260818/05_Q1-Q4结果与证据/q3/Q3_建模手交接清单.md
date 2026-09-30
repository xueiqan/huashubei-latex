# C题问题3论文手交接清单

更新时间：2026-08-08

## 一、当前交付结构

本目录按照问题1的交接模板组织：

- `问题3建模.docx`：沿用问题1模板的页面、页眉、页脚和正文样式，正文替换为问题3模型内容；
- `figures/`：问题3三组论文图；
- `tables/`：问题3模型结果、基准对照和独立检验结果表。

## 二、当前结果表

- [q3_model_configuration.csv](tables/q3_model_configuration.csv)：运行时域、模型规模和容差；
- [q3_objective_summary.csv](tables/q3_objective_summary.csv)：无储能、附件基准和优化方案的四项目标；
- [q3_regional_metrics.csv](tables/q3_regional_metrics.csv)：六区域成本、碳排、峰值净购电、波动和吞吐量；
- `q3_balanced_region_hour.csv`、`q3_baseline_region_hour.csv`、`q3_no_storage_region_hour.csv`：三类方案逐区域逐小时运行状态；
- [q3_validation_summary.csv](tables/q3_validation_summary.csv)：独立检验状态汇总；
- [q3_validation_profile_checks.csv](tables/q3_validation_profile_checks.csv)：三类方案的物理约束复核；
- [q3_validation_baseline_reconciliation.csv](tables/q3_validation_baseline_reconciliation.csv)：附件基准口径复现；
- [q3_validation_multiobjective.csv](tables/q3_validation_multiobjective.csv)：三级择优保持性；
- `q3_validation_solver_checks.csv`及其他`q3_validation_*.csv`：求解阶段和检验明细。

## 三、当前论文图

- `figures/q3_group1_storage_mechanism.pdf`：六区域储能功率时序概览、RegionD–F的SOC与充放电时移特征；
- `figures/q3_group2_optimization_effect.pdf`：四指标改善率、六区域削峰改善和净购电波动改善；
- `figures/q3_group3_validation_credibility.pdf`：附件基准口径诊断、三级择优保持性检验。

## 四、给论文手的核心数字

### 1. 模型规模与时域

- 正式运行时段：0—2405小时，共2406个运行小时；
- 终端结算小时：2406小时，不参与运行调度；
- 区域数：6；变量数：144367，其中二元变量14436；
- 基础约束数：187668；审计容差：`1e-5`，能量平衡容差：`1e-6`。

### 2. 三级均衡优化结果

- 优化方案：Cost=`-459055485.8262`，Carbon=`0`，Peak=`0`，Ramp=`0.0112528`，Throughput=`346120.9884`；
- 附件基准：Cost=`1801786872.6919`，Carbon=`2045367.4753`，Peak=`2021.2628`，Ramp=`28.1476`；
- 无储能反事实：Cost=`-419693592.9393`，Carbon=`8.8506`，Peak=`31.5979`，Ramp=`5.3981`；
- 成本指标按`购电成本−售电收益`计算，因此出现负值时应解释为净收益，不要直接写成“成本为负的物理耗费”。

### 3. 独立检验

- 检验汇总：`116 PASS、4 WARN、0 FAIL`；
- 7个求解阶段均为`status=0`且检查状态为`PASS`；
- 4项WARN全部来自附件基准：新能源平衡残差约`2.0×10^-4 MW`、负荷平衡残差约`1.5×10^-4 MW`、SOC递推偏差约`0.999944 MWh`、终端SOC缺口`193.3246 MWh`；
- 优化方案、无储能方案的物理闭合和指标重算均通过；
- 附件基准字段复现最大差异`1.0×10^-4 MW`，低于复现容差`2.0×10^-4`。

### 4. 三级择优保持性

- `|Δz|=1.0000×10^-6`，容差`1.0001×10^-6`；
- `|ΔΦ|=1.0000×10^-6`，容差`1.0001×10^-6`；
- `ΔG=-900.1601 MWh`，吞吐量没有增加。

可写结论：第三级在数值容差内保持前两级目标，并进一步减少储能吞吐量。

## 五、论文表述边界

### 可以写

- “在固定算力负荷条件下，储能通过跨时段充放电实现新能源消纳、购电时移、削峰和平滑。”
- “优化方案在成本、碳排、区域峰值净购电和净购电波动之间形成三级均衡结果。”
- “独立复核表明优化方案和无储能方案的物理约束、指标重算和终端状态闭合。”
- “附件基准的4项WARN主要来自输入小数精度和SOC终端口径差异。”

### 不能写

- 不能把附件基准的4项WARN写成本文优化方案失败；
- 不能把附件基准直接当作完全满足本文终端SOC约束的可行最优解；
- 不能把`Cost<0`直接解释为实际负电费，必须说明其中包含售电收益；
- 不能只凭单一综合指标声称四项目标同时达到各自单目标最优；
- 不能把三级择优的微小容差差异写成前两级目标发生实质恶化。

## 六、最终交稿检查

1. 正文数字只从`tables/`中的CSV读取，不手抄PDF近似值；
2. 检查图1、图2、图3的标题、图例和正文结论是否一致；
3. 保留模型代码、输入数据、结果表、PDF和检验日志之间的对应关系；
4. 最终提交前重新运行Q3检验，并确认`PASS/WARN/FAIL`数量没有变化。
