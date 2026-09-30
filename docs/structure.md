# 目录与版本说明

## 当前目录

| 目录 | 用途 |
| --- | --- |
| `code/` | 活动Python源码及其运行数据、结果与本机环境 |
| `paper/` | 当前论文、LaTeX章节、图库与编译产物 |
| `materials/problems/` | 赛题与赛方原始附件 |
| `materials/rules/` | 比赛规则、论文格式和赛方模板 |
| `archive/2026华数杯_C题_CCM2601033_20260818/` | 冻结归档，保留提交版本与证据链 |
| `archive/submission-copies/` | 原比赛过程目录中的论文与压缩提交件 |
| `archive/source-backups/question_04/` | 不参与运行的Q4历史源码 |
| `archive/legacy-results/question_01/` | 不参与运行的Q1历史结果 |
| `docs/` | 赛前执行清单与本说明 |

## 2026-09-30迁移对应关系

| 原位置 | 新位置 |
| --- | --- |
| `huashubei code/` | `code/` |
| 根目录`main.tex`、`references.tex`、`main.pdf`、`sections/`、`figures/` | `paper/`下同名文件和目录 |
| 根目录`main.aux`等编译缓存 | `paper/build/` |
| `haushubei submit/2026年第七届华数杯数学建模竞赛赛题/` | `materials/problems/` |
| `haushubei submit/赛方文件/` | `materials/rules/` |
| `haushubei submit/归档/` | `archive/` |
| `haushubei submit/比赛过程文件/` | `archive/submission-copies/` |
| `haushubei submit/2026华数杯赛前与赛中执行清单.md` | `docs/`下同名文件 |
| Q4`outputs/source_backups/` | `archive/source-backups/question_04/` |
| Q1`old_outputs_archive_20260808_fixed/` | `archive/legacy-results/question_01/` |

迁移前后核对了8554个原有文件的SHA-256，全部在新位置保持原内容；随后更新活动文档、忽略规则与Q2的IDE运行路径。既有算法源码、输入、正式计算结果和冻结归档内容未改写。原有本机虚拟环境、IDE设置及缓存随代码目录保留。

## 冻结归档与运行目录

归档内部的相对布局与校验清单保持不变。报告中的旧绝对路径记录历史计算来源，不作为当前目录入口，也不因目录迁移批量改写。

各题的`data/`与`outputs/`保留在题目目录内；活动程序通过源码位置定位它们。再次整理时，先检查共享输入、结果命名空间与检查点，再更新活动文档和IDE配置，核对文件完整性并验证入口。避免只移动某问数据而遗漏跨题共享引用。

旧Q4 V2情景签名含绝对路径，目录移动可能影响旧缓存命中；当前V3/V4签名主要基于配置与数据。历史缓存和证据原文保留，迁移本身不改签名，也不自动触发重新求解。

## 本机验证与使用

本次整理目标为`D:\下载\qq\huashubei-latex`。在PyCharm中打开`code/`，确认项目解释器选中`code/.venv/Scripts/python.exe`；Q2保存的运行配置已经改用`$PROJECT_DIR$`。IDE界面操作未在本轮实际验证。

移动后的Python3.12.13及既有依赖导入通过；Q1主入口、Q4模型和独立检验的`--help`通过。论文在`paper/`下用现有TeX Live/XeLaTeX编译成功，得到28页`paper/build/main.pdf`。

LaTeX插件脚本捕获Windows编译输出时出现UTF-8解码错误，随后直接使用已安装的latexmk完成并确认编译。未修改插件或系统编码设置，未新增依赖，未重新执行四问完整求解。
