# Repository Layout and Version Boundaries

## Current layout

| Directory | Purpose |
| --- | --- |
| `code/` | Active Python source, runtime inputs, outputs, and the local environment |
| `paper/` | Editable paper, LaTeX sections, figures, and local builds |
| `materials/problems/` | Original problem statements and organizer attachments |
| `materials/rules/` | Competition rules, paper-format guidance, and organizer templates |
| `archive/` | Frozen submissions and their evidence records |
| `archive/submission-copies/` | Paper and compressed submissions from the former competition-work directory |
| `archive/source-backups/question_04/` | Historical Q4 source outside the active workflow |
| `archive/legacy-results/question_01/` | Historical Q1 results outside the active workflow |
| `docs/` | Project notes and the historical competition checklist |

## Directory migration on 2026-09-30

| Previous location | Current location |
| --- | --- |
| `huashubei code/` | `code/` |
| Root `main.tex`, `references.tex`, `main.pdf`, `sections/`, and `figures/` | Same names under `paper/` |
| Root LaTeX auxiliary files such as `main.aux` | `paper/build/` |
| Problem statements and attachments under the former submission directory | `materials/problems/` |
| Organizer documents under the former submission directory | `materials/rules/` |
| Frozen archive under the former submission directory | `archive/` |
| Working paper and submission copies | `archive/submission-copies/` |
| Team competition checklist | `docs/competition-checklist.md` |
| Q4 `outputs/source_backups/` | `archive/source-backups/question_04/` |
| Q1 `old_outputs_archive_20260808_fixed/` | `archive/legacy-results/question_01/` |

SHA-256 checks confirmed that all 8,554 original files retained their contents immediately after the move. Documentation, ignore rules, and Q2 IDE paths were then updated. Algorithms, inputs, formal results, and frozen archives were unchanged by that migration. The existing local virtual environment, IDE settings, and caches moved with `code/`.

## English text revision

The subsequent English revision covers active documentation, historical authored change notes, editable LaTeX prose, Python comments/docstrings, help text, display messages, and run-configuration names. The workflow figure filename is now `paper/figures/workflow.pdf`; its contents are unchanged. The authored competition checklist and Q4 runbook use English filenames.

Original organizer documents, datasets, historical computational outputs, archived source, and frozen submissions retain their original content; their repository filenames are now English, with original paths recorded in [filename-map.json](filename-map.json). Required legacy input/schema/path values are retained for compatibility. Existing figure assets may contain Chinese labels; translated plotting source applies when figures are next generated. `paper/main.pdf` remains the historical paper; build the English source to obtain `paper/build/main.pdf`.

## Frozen evidence and active runs

ZIP/RAR contents remain unchanged. Filesystem names were translated; [filename-map.json](filename-map.json) maps original manifest paths to their current locations. Frozen text can still mention original names; use the mapping when tracing an archived reference or restoring its original layout in a separate working copy. Old absolute paths in historical reports record their original computational context; they are not current entry points.

Each question keeps its `data/` and `outputs/` beside its source. Active programs locate those paths relative to their own files. Shared inputs, output namespaces, and checkpoints must remain compatible when moving a question.

Legacy Q4 V2 scenario signatures include absolute paths, so relocation can affect old cache hits. Legacy Q4 source-byte signatures also change after translation. Current V4 `_matheuristic_signature` and `_rollout_signature` use inputs/configuration rather than source text, so translation alone does not invalidate them. Historical caches retain their original signatures; moving directories does not trigger a new solve.

## Local use and verification

Open `code/` in PyCharm and select `code/.venv/Scripts/python.exe`. All 22 shared Q2/Q4 run configurations are in `code/.run/`; Q2 paths use `$PROJECT_DIR$`. The IDE interface itself was not exercised during the migration.

After the move, Python 3.12.13 and existing dependencies imported successfully. Help commands for the Q1 entry point, Q4 model, and Q4 independent validation passed. All 24 active Python files parsed, 24 local documentation links resolved, and 11 Q2 XML configurations passed structural/path checks.

Before translation, TeX Live/XeLaTeX compiled the paper into a 28-page PDF under `paper/build/`. A LaTeX wrapper encountered a Windows output-decoding error, so compilation was verified directly with the installed latexmk. For the English revision, all 24 model-source ASTs retained control flow, numeric constants, identifiers, and formatting; 42 local documentation links resolved and all 22 shared run configurations retained their options. All 902 checked input/result/figure/archive files and 242 original archive-manifest entries retained their hashes. English and historical duration-log cases passed the runnable check in `code/tests/test_runtime_log_compatibility.py`. The translated paper compiled to 36 pages, with its cover and key tables visually reviewed. See the English revision entry in `code/CHANGELOG.md` for these checks. No dependency was added, and the complete four-question solve was not rerun.
