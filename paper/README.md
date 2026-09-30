# 论文与LaTeX源文件

- [main.pdf](main.pdf)：保留的赛后修订版论文。
- [main.tex](main.tex)：XeLaTeX主入口。
- `sections/`：分章节正文。
- `references.tex`：参考文献。
- `figures/`：论文引用的图形。
- `build/`：本地编译输出与日志，不提交Git。

在TeXstudio中打开本目录的`main.tex`，使用XeLaTeX编译。主文件、章节与图库应保持当前相对位置。

已安装TeX Live与latexmk时，在本目录运行：

```powershell
latexmk -xelatex -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build main.tex
```

新生成的PDF为`build/main.pdf`。根于本目录的`main.pdf`保留历史版本，编译不会自动覆盖它。字体配置使用Windows中文字体和Times New Roman/Consolas，其他系统应按其字体环境调整后再编译。

本次迁移后已用现有TeX Live/XeLaTeX成功编译28页PDF；原有字体字号替代警告仍存在。论文正文、图库及既有PDF未因目录整理改写。

竞赛提交原版、正式支撑材料和归档说明见[archive](../archive/)。本目录的论文、当前Python源码和冻结ZIP可能属于不同版本，请按结果口径对应使用。
