# Huashu Cup Problem C: Coordinated Computing, Storage, and Power Optimization

Python programs, a LaTeX paper, computational results, and verification materials for Problem C of the 2026 Seventh Huashu Cup Mathematical Modeling Competition. The project studies multiobjective scheduling across regional data centers, including computing workloads, carbon emissions, network latency, renewable energy utilization, and battery storage.

Use this repository to study the modeling workflow and the tradeoffs between computing, storage, and electricity. It contains both frozen competition submissions and later revisions; match each result to its source version.

## Research tasks

| Task | Focus |
| --- | --- |
| Q1 | Hierarchical demand forecasting, basic computing dispatch, and stress-boundary checks |
| Q2 | Multiobjective dispatch for cost, carbon, latency, and renewable utilization; exact versus heuristic comparisons |
| Q3 | Storage and electricity trading under a fixed IT load, with independent energy-constraint checks |
| Q4 | Joint computing, storage, and power optimization; sequential-baseline comparisons, scenarios, and independent audits |

The active code contains 24 Python files. Each question provides preprocessing, modeling, validation, and plotting stages. Historical source backups are stored in `archive/source-backups/`.

## Start here

| Directory | Contents and entry points |
| --- | --- |
| [code/](code/) | Python source, dependencies, inputs, and results; [run guide](code/README.md) |
| [paper/](paper/) | English LaTeX source, references, figures, and the preserved historical PDF; [build guide](paper/README.md) |
| [materials/](materials/) | Original [problem statements and attachments](materials/problems/) and [competition rules](materials/rules/) |
| [archive/](archive/) | Frozen submissions, historical results, and source backups |
| [docs/](docs/) | [Repository layout](docs/structure.md) and the [historical competition checklist](docs/competition-checklist.md) |

The current paper source is [paper/main.tex](paper/main.tex). The English revision compiles to 36 pages at `paper/build/main.pdf`. Rebuild it using the paper guide. The preserved [paper/main.pdf](paper/main.pdf) predates the English translation. See the [archive guide](docs/archive-guide.md) for the distinction between the competition submission and later revisions.

```text
huashubei-latex/
├── README.md
├── code/
│   ├── README.md
│   ├── CHANGELOG.md
│   ├── pyproject.toml
│   ├── uv.lock
│   ├── main.py
│   └── question/
│       └── question_01 ... question_04/
│           ├── main.py / preprocess.py / model.py
│           ├── validation.py / plot.py
│           ├── data/          # Raw and processed inputs
│           └── outputs/       # Tables, figures, logs, and checkpoints
├── paper/
│   ├── README.md
│   ├── main.tex / references.tex / main.pdf
│   ├── sections/
│   ├── figures/
│   └── build/                 # Local builds, ignored by Git
├── materials/
│   ├── problems/
│   └── rules/
├── archive/
│   ├── submission-copies/
│   ├── source-backups/question_04/
│   └── legacy-results/question_01/
└── docs/
```

Question-specific inputs and outputs remain beside their source files to preserve shared-data and checkpoint paths. Frozen evidence is kept separately in `archive/`.

## Environment and data

Python 3.12 or later is required. [pyproject.toml](code/pyproject.toml) declares the dependencies and [uv.lock](code/uv.lock) locks their versions: NumPy, pandas, SciPy, HiGHS/highspy, Matplotlib, Seaborn, and OpenPyXL.

With Git, Python, and uv available:

```powershell
git clone https://github.com/xueiqan/huashubei-latex.git
Set-Location './huashubei-latex/code'
uv sync --frozen
```

For a ZIP download, extract it and enter `code/`. In PyCharm, open that directory and select its project interpreter, then run each question's `main.py` in order.

Q1 reads six original Excel attachments from [question/question_01/data/raw/](code/question/question_01/data/raw/):

```text
GPU_information.xlsx
network_latency.xlsx
power_mapping.xlsx
region_time_data.xlsx
storage_information.xlsx
workload_trace.xlsx
```

Keep their filenames, worksheets, and contents intact. Q1 generates shared inputs under `question/question_01/data/processed/shared/`; subsequent questions derive their own inputs from that shared layer.

## Run and validate

Run in a separate working copy if you need to preserve the supplied historical results. From `code/`:

```powershell
uv run python question/question_01/main.py
uv run python question/question_02/main.py
uv run python question/question_03/main.py
uv run python question/question_04/main.py
```

These commands perform actual computations. The per-question workflow is:

```text
preprocess.py -> model.py -> validation.py -> plot.py
```

Q2-Q4 entry points may log a warning after a validation or plotting failure and still exit with status 0. Check the logs, validation reports, and expected outputs before treating a run as complete.

### Q4 commands

Inspect the available options:

```powershell
uv run python question/question_04/model.py --help
uv run python question/question_04/validation.py --help
```

Additional checks:

```powershell
uv run python question/question_04/model.py --self-test
uv run python question/question_04/model.py --baseline-audit
uv run python question/question_04/validation.py --skip-reference-windows
```

The self-test checks local structure and algorithm behavior; it is not a full solve. Baseline auditing and independent validation require complete results from the matching version. `--skip-reference-windows` skips representative-window MILP comparisons while auditing existing outputs. Default validation also solves representative windows; runtime depends on solver settings and hardware. See the [Q4 runbook](code/question/question_04/q4-runbook.md) for the detailed sequence.

## Results and evidence boundaries

Each question's `outputs/` contains tables, figures, logs, reports, and historical version directories. Current Q4 source primarily uses `matheuristic_v4_doccompliant`; V3 results and competition evidence are also retained.

- Submitted ZIP files are frozen versions and may differ from the current expanded source.
- The inspected V4 snapshot contains sequential-plan and fixed joint-task energy recomputations, without a completion marker for a full V4 joint search. These are different kinds of evidence.
- Archived validation reports describe historical runs; they do not prove that modified source passes in a new environment.
- Trace paper values to tables and independent checks from the same version. Do not estimate values from plots or mix baseline and joint results across versions.
- Validated heuristic results establish feasibility under the stated data, parameters, and boundaries; they do not establish global optimality.
- A complete fresh-environment reproduction has not been performed.

Active documentation, editable paper text, source comments, command help, and display messages are maintained in English. Original organizer files, input data, historical outputs, frozen archives, and archive-internal names remain unchanged; repository filenames are English. Existing figure assets can still contain Chinese labels; translated plotting source applies to future figure generation.

The paper uses XeLaTeX with Windows font settings, Times New Roman, and Consolas. See its build guide for prerequisites. This is a project paper, not an official competition template; replace the team number, title, and content when reusing its layout.

## License and feedback

There is currently no repository-wide LICENSE. Check the ownership and applicable permissions of source code, team papers, problem data, organizer documents, and cited material separately. This README does not grant new permissions for those materials.

Report reproduction issues, inconsistent result definitions, or documentation errors through [Issues](https://github.com/xueiqan/huashubei-latex/issues), or propose improvements in a pull request. Include the commit, question number, command, Python/dependency versions, relevant logs, and expected behavior.
