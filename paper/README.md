# Paper and LaTeX Source

- [main.tex](main.tex): current English XeLaTeX entry point.
- `sections/`: editable paper sections.
- `references.tex`: references.
- `figures/`: figure assets used by the paper.
- [main.pdf](main.pdf): preserved PDF from before the English translation.
- `build/`: local compilation outputs and logs, ignored by Git.

Open `main.tex` in TeXstudio and compile with XeLaTeX. Keep the main file, sections, and figures in their current relative locations.

With TeX Live and latexmk installed, run from this directory:

```powershell
latexmk -xelatex -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build main.tex
```

Read the newly compiled English paper at `build/main.pdf`. Compilation does not overwrite the preserved `main.pdf`. The font configuration uses Windows Chinese fonts, Times New Roman, and Consolas; adjust the font configuration for other systems.

The English translation preserves mathematical definitions, numerical findings, and version boundaries. Existing figure files are historical assets and may retain Chinese labels. English plotting labels are available in the active Python source for subsequent figure generation; translation alone does not rerun the models or regenerate historical figures.

The original competition paper, supporting materials, and archive guide are in [archive/](../archive/). The editable paper, active Python source, and frozen ZIP files may belong to different versions; align their evidence before comparing results.
