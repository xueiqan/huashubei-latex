# Change Log

Historical entries describe checks performed at the time of each change. They are not new verification claims for the current translated source. Relative paths below are rooted at `code/` unless stated otherwise. No dependency was added by the English revision.

## 2026-09-30 (English text and filenames)

- Scope: active documentation, this change log, editable LaTeX text, Python comments/docstrings and display messages, run-configuration names, and repository filenames.
- Translated authored text into English and updated active references after filename changes. Numerical algorithms, command flags, and required data contracts remain intact. Original organizer files, inputs, historical results, frozen submissions, and ZIP/RAR contents retain their content.
- Renamed 95 original-material/archive file and directory entries; verified 288 relocated files byte-for-byte. `docs/filename-map.json` records original paths, English paths, and content hashes for traceability.
- Verification: all 24 model sources retained their nontext AST, numeric constants, identifiers, and f-string formats; eight guides, 11 TeX files, and 22 run configurations contain English authored text. All 42 local documentation links resolve, configuration options/CLI flags are preserved, and repository filenames contain no Chinese characters. The original hashes of 902 input/result/figure/archive files and all 242 archived manifest entries match. Legacy/English duration-log checks and safe CLI help passed. The English paper compiled to 36 pages; its cover and key tables were visually checked.
- Shared Q2 configurations now join Q4 in `code/.run/`, explicitly included by the ignore rules. The existing historical `paper/main.pdf` and figure contents remain unchanged. Legacy source-hash caches may reject earlier signatures after translation; current V4 signature functions do not hash source text. No full model solve or historical figure regeneration occurred. Changes remain local and have not been committed or pushed.

## 2026-09-30 (Repository directory split)

- Scope: repository layout, root/code/paper READMEs, layout notes, `.gitignore`, and Q2 PyCharm run configurations.
- Grouped active programs under `code/`, paper files under `paper/`, statements/rules under `materials/`, frozen submissions/old results/source backups under `archive/`, and working notes under `docs/`. Per-question `data/outputs` relationships were preserved.
- All 8,554 original files had identical SHA-256 hashes immediately after moving. Active documentation and configuration were updated afterward; algorithms, raw data, formal results, and frozen archives were not rewritten by the move.
- Python 3.12.13/dependency imports, Q1 entry-point help, and Q4 model/validation help passed. Existing latexmk/XeLaTeX compiled a 28-page paper. A LaTeX wrapper encountered Windows output-decoding problems; direct latexmk verification succeeded.
- No full solve/scenario rerun, IDE-interface test, new dependency, commit, or GitHub push was performed.

## 2026-08-10 (Q4 V4 consistent energy recomputation and paper results)

- Scope: Q4 `model.py`, `validation.py`, `plot.py`, V4 sequential/joint fixed-task energy references, figure groups 1/3, and the then-current paper source.
- Added V4 energy recomputation for an existing joint task plan using the same energy-response rule as the sequential plan. Group 1 no longer mixed V4 sequential and V3 joint energy trajectories. Task plans were fixed; this was not a new full V4 joint search.
- Joint versus sequential plans reduced net operating settlement by CNY656,651.67, average latency by 2.98506 ms, QoS loss by 0.00437750, and renewable unused rate by 0.025766 percentage points; carbon increased by 41.0772 tCO2 and peak net import by 30.6119 MW. Removed superseded paper claims of `90555.27 -> 428.01 tCO2`, CNY133.82 million, and 2551.32 MW.
- Checks: 28/28 automated self-tests; `validation.py --comparison-only` passed 67 checks per plan, with maximum hard-constraint discrepancy `2.3842e-7`; groups 1/3 were redrawn and visually reviewed; XeLaTeX produced 28 pages. No full V4 joint search, scenario rerun, or dependency addition.

## 2026-08-10 (Q3 Figure 9 cost definition and layout)

- Scope: Q3 `plot.py`, group-2 PDF, the paper figure copy, and `sections/7_problem3.tex`.
- Replaced an improvement percentage spanning zero with the absolute difference between the attachment baseline and plan, in CNY10,000. Carbon, peak import, and import variability retained relative improvement measures.
- Moved panels A/B downward and the shared legend to the top to avoid labels; regional peaks/variability used absolute changes.
- Ran only `plot.py --groups group2` against existing CSVs and generated 1/1 group. `py_compile`, source/copy SHA-256 comparison, and PNG visual review passed. No model/validation rerun or new dependency.

## 2026-08-10 (Q4 V4 physical scaling and output namespace)

- Scope: Q4 `model.py`, `plot.py`, and `validation.py`.
- Added `matheuristic_v4_doccompliant` while retaining V3. Replaced degenerate cost/carbon/peak scales with full-horizon renewable-deficit scales; added deterministic final tie-breaking by cost, carbon, and storage throughput. Plotting/validation read V4, and `q4_scaling.csv` recorded scale provenance and optimization roles.
- Passed the missing `scaling` argument into the energy objective, fixing an undefined variable during formal energy response.
- Three entry points parsed; 28/28 self-tests and a targeted energy-objective smoke check passed. No long model/scenario/validation run or dependency addition.

## 2026-08-10 (Q4 sequential-reference repair and group-1 replacement)

- Scope: Q4 model/validation/plot source, V4 sequential tables, independent checks, and group 1.
- Added `--repair-sequential-reference`, `validation.py --sequential-only`, and `plot.py --groups group1` to reuse Q2 task seeds with V4 scaling and avoid unrelated full solves.
- Sequential Carbon changed from 90555.2663 to 30.1233 tCO2 and Peak from 2800 to 66.5709 MW. Independent validation passed; PDF/PNG output and paper copies were updated.
- Joint results had not been solved under V4. Group 1 temporarily used audited V3 joint trajectories and only their originally optimized `L/J/Q` metrics; this did not establish V4 six-objective joint completion.
- AST checks and 28/28 self-tests passed; no active Q4 model/validation/plot process remained. No new dependency.

## 2026-08-10 (Q4 overall titles and legends)

- Scope: Q4 `plot.py` and three PDF/PNG groups.
- Removed overall in-figure titles from groups 1/2, moved the group-1 objective legend below its explanation box, and aligned five panel captions with the actual metrics/scenario/validation content.
- Compile/plot execution and three PNG visual reviews passed. No model or validation run.

## 2026-08-10 (Q4 group-3 duplicate captions)

- Removed repeated titles above the panels, retaining the lower (a)/(b) captions; moved the SOC legend above the right panel.
- Compile/plot execution and group-3 PNG review passed. No model, scenario, or validation run; no new dependency.

## 2026-08-10 (Q4 group-3 legend and paper wording)

- Moved the SOC region legend below the right panel. Replaced implementation references such as validation/plot filenames and CSV names with model-validation and data-meaning descriptions.
- Compile/plot execution and group-3 review passed. No model/validation run or new dependency.

## 2026-08-10 (Q4 negative energy-change labels)

- Adjusted positive/negative bar-label margins in group 1 so the `-242.17k` label no longer collided with the bottom axis/ticks.
- Compile, plot exit status 0, and group-1 review passed. Result data, model, and validation were unchanged.

## 2026-08-10 (Q4 three-group visual layout)

- Moved group-1/3 legends above their panels, avoiding tall bars, reference lines, and SOC curves. Shortened group-1 legend text; retained the group-2 heatmap layout.
- Regenerated and visually checked all three PNGs. No result-data changes, model/validation run, or new dependency.

## 2026-08-10 (Q4 group-3 validation-derived plots)

- Group 3 preferred `outputs/validation/q4_reference_window_comparison.csv`, falling back to `q4_validation_window_summary.csv`, rather than relying on legacy `q4_mh_ref_comparison.csv`.
- Full MILP remained the validator's responsibility. Plotting read completed reports/CSV files, avoided incomplete historical intermediates, and did not launch solves.
- Left panel showed actual validation MIP gaps, objective gaps, solver status, and independent audit status; right panel showed full-horizon terminal SOC restoration.
- Compile/plot execution passed. Two validation windows were read; three PDFs were single-page and PNG review passed. Plotting did not start model/validation work. No dependency addition.

## 2026-08-10 (Q4 V3 plot inputs and labels)

- Read formal V3 baseline/sequential/scenario inputs; loaded nine completed, audited scenarios; added a `QoSLoss` alias to prevent false missing-data diagnoses.
- Replaced generic change labels with maximum-deviation reductions. Missing full MILP window comparisons retained an evidence-pending panel instead of substituted rolling logs.
- Compile and plotting passed; three PDFs/PNGs were regenerated and visually checked. Plotting started no model/validation work; no new dependency.

## 2026-08-10 (Q4 default full validation)

- Direct `validation.py` execution now solved the two representative joint MILP windows by default rather than only inspecting existing window files.
- Added selection/start/completion/failure/summary logs and `--skip-reference-windows` for explicitly skipping the solves.
- Compile passed; full validation was not automatically run for this change.

## 2026-08-10 (Q4 scenario-specific independent audit inputs)

- Reconstructed flat-price, carbon-constrained, and renewable-smoothing inputs from V3 completion markers before checking task/energy trajectories and six objectives, instead of always using baseline inputs.
- Added audit-input context and explicit failures for missing metadata, inconsistent `ChangedColumns`, or failed single-factor reconstruction.
- Model/validation compile checks passed. The manual audit was not automatically run.

## 2026-08-10 (Q4 VALIDATION_ONLY progress)

- Added logs for validation entry, baseline inspection, window identification/solves, independent audit status, and scenario-interface smoke progress.
- Compile passed; no validation-window solve was run.

## 2026-08-10 (Q4 scenario-comparison progress)

- Added start, baseline-read, candidate-check, per-scenario, and output-write progress. Comparison continued to read existing results without MILP calls.
- Compile passed; scenario comparison was not rerun.

## 2026-08-10 (Q4 V3 scenario return-value order)

- Aligned `_read_matheuristic_result()` and `_scenario_baseline_reference()` return ordering so scenario code received the table directory, metric dictionary, and schedule correctly, instead of treating a DataFrame as a `q4_scaling.csv` path.
- Compile and regressions for returned types, scaling paths, and schedule interfaces passed against completed V3 baseline files. No full scenario optimization rerun.

## 2026-08-10 (Q4 paper plotting entry)

- Added three figure groups covering joint effects, scenario responses, MH/REF checks, and full-horizon SOC, with five panels and PDF/450-dpi PNG output.
- Missing formal Q2-to-Q3 sequential baselines, scenarios, and MH/REF pairs were shown as pending evidence; neither the fixed Q3 attachment baseline nor `q4_window_solver.csv` substituted for them.
- SOC used Q4 hourly dispatch and Q1 shared storage parameters, extending `SOCEnd_MWh` from operating hours 0-2405 to terminal t=2406. Plotting did not call models or MILP.
- Compile/plot execution passed; three single-page PDFs and PNG visual checks passed. No new dependency.

## 2026-08-10 (Q2 plot conclusion definitions)

- Group 1 stated a net operating-cost reduction of CNY5,073,400. Group 2 distinguished local task counts on the gray diagonal from cross-region GPU-hour color values.
- Moved group-3 exact-window statistics to the lower right to avoid the 293.40% WindowID=9 outlier; changed stability wording to sensitivity.
- No raw/model data changed. Claims did not imply global near-optimality or unconditional K=24/48/72 stability. Static checks only; plotting was not run.

## 2026-08-10 (Q2 isolated window-9 heuristic rerun)

- Added `heuristic_window9.py` and its PyCharm configuration. Fixed historical exact-window 0-8 assignments and reran only marginal greedy/LNS for WindowID=9, tau=216, without exact MILP.
- Intended outputs under `outputs/validation_runs/heuristic_window9` included window results, fixed-history copies, metadata, and logs, without overwriting exact comparisons or formal Q2 results.
- Static interface checks only; the actual rerun was not started. No new dependency.

## 2026-08-10 (Q2 paper-figure layout)

- Group 1 moved the right explanation box to the lower left and added label margins. Group 2 used separate heatmap/colorbar/migration columns with larger GPU-workload/task-count annotations.
- Group 3 retained all exact-window points but labeled only the largest three deviations; moved the look-ahead legend below its title and retained the full K=72 carbon-outlier range.
- Unified panel-caption spacing and changed PDF/PNG safe margin from 0.03 to 0.06. Layout-only changes; no plotting run, data/model change, or dependency addition.

## 2026-08-09 (Q2 paper plotting entry)

- Added three groups for objectives/renewables, cross-region dispatch, and exact-window/look-ahead validation. Read existing CSVs without calling Q2 models or MILP.
- Used the supplied two/three/four-series and blue-gradient palettes, with dark-blue text, light-blue grids, and red references consistent with Q1/Q3.
- Missing exact-window or K=24/K=72 evidence retained pending panels. Intended outputs were PDF/450-dpi PNG under Q2 figures; code was written without running plots. No new dependency.

## 2026-08-09 (Q2 validation-experiment entries)

- Added `validation_runs.py`, `exact_heuristic_benchmark.py`, and ten optional PyCharm configurations for K24/K72 preparation, first runs, resume, archive, exact comparison, and final validation, using the project `.venv`.
- Isolated K24/K72 projects archived completed tables/logs without deleting or overwriting formal K48 results/checkpoints.
- Exact/heuristic comparisons used matched H=24, K=48, task pools, historical exact assignments, and global scales for windows 0-9. Exact evidence required `status=0`; checkpoints supported resume.
- Both entries and existing validation passed compile/help checks. No long solve or dependency addition.

## 2026-08-09 (Q2 validation for the new heuristic)

- Rebuilt validation around global continuous scaling, multiobjective marginal greedy, capacity repair, and LNS-MILP: independent full-horizon conservation, matched baselines/mechanisms, exact-window/LNS ablation, and K=24/48/72 sensitivity.
- Independently reconstructed 50,000 TaskID assignments across hours 0-2405 and six regions. `LatestFinishHour=2406` remained legal while hour 2406 was excluded from occupancy; resource profiles/objectives were checked.
- Read `ZBeforeLNS`, `ZStar`, and `LNSAccepted` for 91 windows. Exact comparison required separate exact/heuristic files for matched windows 0-9, otherwise PENDING.
- Audited existing K48; K24/K72 required isolated experiment outputs. Validation did not launch models or overwrite checkpoints; console/file progress and `--progress-every`/`--max-seconds` remained.
- Compile/help and actual independent validation passed core constraints/conservation. LNS improved 45 of 91 windows; exact/K24/K72 gaps were PENDING. No new dependency or model/MILP import.

## 2026-08-09 (Q2 stage-four low-quality timeout guard)

- Windows 1/2 had committed `z=1e6`, `MIPGap` approximately 1, which later rolling decisions inherited. Stopped that run and moved only four `q2_refactored_balanced_v3_*` checkpoint files to `outputs/checkpoints/archive_balanced_v3_bad_gap_20260809_174427/`; retained baseline/scaling caches.
- Restored the stage-four guard limit to 900 s. Rejected timeout incumbents with `status=1` and `MIPGap>0.02`, retaining the prior checkpoint. Optimal `status=0` solutions remained accepted; objectives/constraints were unchanged.
- Compile, constants, and guard static checks passed. No full Q2 computation or new dependency.

## 2026-08-09 (Q2 automatic extension without an incumbent)

- Normal guard stayed at 180 s; HiGHS `status=1` with `x=None` triggered a logged retry at 900 s instead of immediate failure.
- Variables, objectives, constraints, and acceptance rules were unchanged. Missing feasible solutions after 900 s still failed explicitly.
- Compile/branch checks passed; no full computation or dependency addition.

## 2026-08-09 (Q2 default per-window MILP limit)

- Set defaults to `mip-rel-gap=0.005`, `solver-time-limit=180s`, and `max-solver-time-limit=180s`, retaining automatic checkpoint resume.
- Objectives/constraints were unchanged. An incumbent at timeout could continue the rolling process; no incumbent remained an error.
- Compile/default checks only; no full run or dependency addition.

## 2026-08-09 (Q2 stage-four closing-log object lifetime)

- Balanced-window closing logs accessed `result.mip_gap`/`model.option_count` after releasing the objects, causing `UnboundLocalError` after window 0.
- Saved candidate count and MIP-gap scalars before release; fixed the analogous baseline log. Model, parameters, and checkpoint data were unchanged.
- Compile/static lifetime checks passed; no full computation or new dependency.

## 2026-08-09 (Q2 renewable-unused scaling tolerance)

- Changed the `RenewableUnusedRate` global continuous-scaling certificate tolerance from `1e-5` to `0.0025`, allowing at most 0.25 percentage points of ideal-value error to end an already stable, slow column-generation process.
- Rolling variables, constraints, and objective structure were unchanged. Running processes required restart/resume to use the new threshold.
- Compile/constants checks passed; no full computation or new dependency.

## 2026-08-09 (Q2 default checkpoint resume)

- No-argument PyCharm launches defaulted to `resume=True`; added `--no-resume` for an explicit fresh start. Startup logs recorded the policy.
- Compile/help checks passed; no full computation or new dependency.

## 2026-08-09 (Q2 window-completion log lifetime)

- Baseline/balanced logs accessed `model.option_count` and `result.mip_gap` after `del result, model`. Saved scalars before deletion without changing objectives, constraints, or solver settings.
- Compile passed; `--resume --max-windows 2` advanced the existing baseline checkpoint through window 1, logged completion, and exited normally. No full run or dependency addition.

## 2026-08-09 (Q4 calibration-timeout fallback)

- When a continuous calibration or integer reference subproblem timed out without a feasible point, reused an already validated continuous feasible solution from the same window when available.
- Recorded `ContinuousFeasibleFallback`/`FeasibleReferenceFallback`, source metrics, and original solver messages in `q4_calibration_records.csv`; never presented fallback as an exact bound/reference.
- Still raised errors if no feasible continuous point existed or the failure was not a timeout.

## 2026-08-09 (Q4 LP solver for continuous calibration)

- Continuous `milp` calls could return no feasible renewable-unused calibration point within 60 s. Switched `relax=True` to SciPy `linprog(method="highs")`, splitting two-sided `LinearConstraint` into equality/inequality matrices. Integer rolling MILP stayed unchanged.
- Reproduced the window-768/renewable-unused 60-s failure on real Q4 data; controlled fallback, syntax, and small constraint-conversion checks passed. No full model or new dependency.

## 2026-08-09 (Q4 optional continuous-result fields)

- SciPy/HiGHS could return `mip_gap=None` for continuous relaxation, so `float(None)` interrupted calibration. Converted `mip_gap`/`mip_node_count` through optional-numeric handling and retained missing values as `NaN`.
- A small MILP smoke check covered `None` fields; progress output remained. No new dependency.

## 2026-08-09 (Q4 solver progress)

- Added input/calibration/window/output/constraint-check stages and window start/completion logs. Blocking MILP calls printed elapsed time, variable/constraint counts every 30 s; failures logged retries.
- Display-only changes; no objective, constraint, solver-parameter, or dependency change.

## 2026-08-08 (Q2 relaxation-guided rolling model)

- Decision horizon H retained integer start variables; look-ahead K used continuous forecast variables. Four LP ideal points guided candidate filtering before balanced solving.
- Defaulted to all legal start times followed by relaxation filtering; `--use-six-start-options=1` retained the legacy six-point mode. Added `--mip-rel-gap`, `--relaxation-zero-threshold`, and `--relaxation-top-k`, initially 0.01, `1e-6`, and 12; representative-window comparisons would determine formal settings.
- Removed terminal hour 2406 from constraints/objectives/profiles; checked main hours 0-2399, tail 2400-2405, and the complete Hour-by-Region grid.
- Secondary refinement ran only after stage B reached optimality or the configured gap; otherwise retained its feasible solution with a skip reason.
- Returned the filtered candidate list with its solution vector, preventing reduced-vector/original-list mismatches; added length protection. Wrote RUNNING checkpoints before each window and converted mapping/missing-fixing/duplicate TaskID failures into window retries.
- Used version-2 checkpoint names while retaining old files. Reused baseline solutions, matrices, anchors, and filtered matrices during extended-time retries; cleared caches only when expanding candidate sets. Continuous ideals used `linprog`.
- Compile, two-sided conversion/LP, window LP, stage-cache, and baseline-cache smoke checks passed. Ruff was unavailable; no new dependency. Did not interrupt or parallel-run the old active process, or validate all 50,000 tasks numerically.

## 2026-08-08 (Q2 validation rewrite)

- Independently checked task uniqueness, immediate real-time starts, candidate regions, latency, earliest/latest limits, terminal hour, overlap-based GPU/AI-IT/facility use, import/export/unused energy, and energy balance.
- Read precomputed production/reference windows for normalized four-objective gaps, maximum-deviation gaps, and regional hourly AI-load gaps, without default MILP reruns.
- Read K=12/24/36 results or stability tables; missing common anchors/profiles allowed only raw-metric reporting. Compared Q2 balanced/Q1 computing baselines using full-horizon anchors when available, otherwise marked unassessed.
- Added `--max-seconds` and progress every 1,000 tasks. Validation did not import/call `scipy.optimize.milp`; console/file logs covered all stages.
- Compile/help passed; Q2 validation execution was intentionally not started. No new dependency.

## 2026-08-08 (Q3 independent validation and attachment baseline)

- Scope: Q3 model/validation and the original external validation specification.
- Recomputed energy allocation/balance, SOC, hard bounds, five metrics, attachment reproduction, and three-stage multiobjective ordering; wrote summary/detail/report outputs.
- Attachment-baseline physical/terminal discrepancies were WARN, distinct from Balanced/NoStorage hard failures. A 24-hour run produced 116 PASS, 4 WARN, 0 FAIL.
- BaselineReference terminal metrics used the attachment terminal SOC rather than repeating the last operating-hour SOC. Added `1e-10` floating comparison protection around the model's `1e-6` bound.
- Compile, Ruff, and 24-hour seven-stage MILP plus audit passed. Formal 2406-hour validation was `PASS_WITH_WARNINGS` with 116 PASS/4 WARN/0 FAIL. No new dependency.

## 2026-08-08 (Q3 MILP progress)

- Added problem size, stage changes, status, and MIP-gap logs for four anchors and three balanced stages, with 30-s heartbeats.
- Added `--progress-interval` and optional native HiGHS `--solver-disp`, retaining unlimited time and `mip_rel_gap=1e-5` defaults.
- Compile, Ruff, help, and a 24-hour CLI smoke run passed, including logs, writes, and constraint audit.

## 2026-08-08 (Q2 model-document revision)

- Continuous task/energy auxiliaries formed four LP ideal lower bounds, not rolling dispatch results. Used the matched computing-only baseline for one integer minimax solve and a sum-deviation refinement only after first-stage optimality; removed migration/wait tertiary optimization.
- Removed V/X/energy-state binaries; retained continuous B/W epigraph constraints and derived export X. Final energy settlement was independently recomputed piecewise from facility load.
- Default six-point candidates preserved earliest/latest starts and representative low-price/low-carbon/high-renewable/high-slack points; `--use-six-start-options 0` retained all legal integer starts.
- Set gap `1e-3` and default 60-s subproblem limits. Without an incumbent, expanded candidates and retried at 60/120/240/480/600 s. Final metrics covered 0-2405; 2406 remained a completion boundary.
- Window energy logs covered actual overlaps beyond the look-ahead end; final summaries stayed full-horizon. Removed old full-horizon single-objective anchor commands/branches while retaining window ideals.
- Added start/30-s heartbeat logs, `--progress-interval` (0 disables), expansion counts, and actual limits. Saved CSV/JSON checkpoints each successful window and FAILED state before final failure; `--resume` continued only unfinished work.
- Syntax/help/startup checks only; no full Q2 solve/validation.

## 2026-08-08 (Q3 storage multiobjective model)

- Implemented hourly renewable-storage-grid MILP at fixed computing load, including SOC, charge/discharge exclusivity, independent export bounds, peak import, and adjacent-hour import ramp.
- Solved Cost/Carbon/Peak/Ramp anchors, then minimax deviation, total deviation, and storage-throughput tie-breaking.
- Wrote Balanced, attachment-baseline, and NoStorage hourly profiles, regional metrics, anchors, comparisons, solver logs, and audits; `--max-hour` supported small smoke runs.
- Existing `scipy.optimize.milp` only; compile, Ruff, and 24-hour seven-stage MILP passed all constraint checks. No 2406-hour solve or new dependency.

## 2026-08-08 (Q2 unlimited MILP time)

- Removed the main 30-s and feasibility-retry 5-s limits; omitted time limits from `scipy.optimize.milp`. Removed `--solver-time-limit` and marked startup/results as unlimited.
- Retained `mip_rel_gap=1e-5` as accuracy, not a time limit. Syntax/help/options smoke checks passed; no full unlimited solve rerun.

## 2026-08-08 (Q2 model consistency)

- Unified candidate mode as `--use-six-start-options {0,1}`: all legal integer starts or six representatives, default 1 for the 50,000-task model.
- Used two hourly binary states to linearize `B=max(F-P_RE,0)`, `V=min(F,P_RE)`, and `X=min(P_RE-V,ExportLimit)`, allowing grid supply when renewables were insufficient.
- Added `--global-anchors` for four full rolling single-objective policies and `q2_global_anchor_summary.csv`; window anchors remained normalization inputs.
- Distinguished feasible/nonoptimal windows in sensitivity reports, retaining requested H/K. A feasibility retry without an optimal anchor kept `status=1` and missing gap; optimality still depended on status/gap review.
- AST/compile, two extreme renewable scenarios, minimal full-horizon anchors, and minimal balanced rolling checks passed. Full candidates for 50,000 tasks yielded no first incumbent within 30 s, so were not the default.

## 2026-08-08 (Q2 MILP timeout fallback)

- A feasible stage-C timeout skipped D; a stage-D timeout without a new incumbent retained C. Baseline stage 2 similarly retained stage 1 when needed.
- Kept `status=1` with explicit fallback/skipped messages; did not claim proven tie-break optimality.
- Syntax/help checks passed; no full rolling rerun.

## 2026-08-08 (Q2 model logs)

- Added console/file logs for inputs, candidates, anchors, balanced/baseline stages, sensitivity, independent checks, and writes.
- MILP records included status/success, tasks/variables, elapsed time, gap, dual bound, and solver messages; timeout incumbents retained explicit `status=1`.
- Added `--log-level`, default INFO. AST passed; no full solve or new dependency.

## 2026-08-07

- Q1 `preprocess.py` produced cleaned tasks, hourly arrivals, final-window regional capacities, and task-level candidate regions according to the Q1 Gate A PASS plan.
- Aggregated `GPU_Demand` by `ArrivalHour x SourceRegion x TaskType`; scheduling used actual tasks arriving 2376-2399 and checked GPU/IT/facility capacities without Q2-Q4 energy fields.
- Source/output names/time boundaries/fields/exclusions were statically checked; no project execution or generated results.
- Added user-authorized `openpyxl>=3.1` so pandas could read the supplied Excel files.

## 2026-08-07 (Shared preprocessing migration)

- Q1 generated `question_01/data/processed/shared/`, with its own outputs under `q1/`; Q2-Q4 derived `q2/`, `q3/`, and `q4/` interfaces from that layer.
- Candidate regions covered all tasks; Q1 selected 2376-2399 arrivals only during modeling. Q2-Q4 retained energy inputs through 2406 but no execution occupancy at that terminal hour; `LatestFinishHour=2406` remained legal.
- Removed four obsolete nonshared CSVs from the Q1 processed root without deleting raw attachments/logs/other directories.
- Static source/path/field/boundary checks only. The user generated runtime results in the existing contest environment.

## 2026-08-08 (Q1 model)

- Implemented compound-Poisson forecasting, a same-hour 24-h naive baseline, and a 168-h local-linear-trend challenger. Wrote validation/test WAPE, RMSE, MAE, and 68 historical 24-h rolling comparisons.
- Scheduling used actual arrivals 2376-2399, candidate regions/integer starts, minute-overlap occupancy, GPU/IT/facility bounds, and lexicographic minimization of migrated GPU workload then flexible-task waiting.
- Test data served only final evaluation; dispatch did not read forecasts; 2406 was a finish boundary without execution capacity.
- Exact scheduling called `scipy.optimize.milp`; SciPy had not yet been added to dependencies at that point. No model execution pending the user's environment confirmation.

## 2026-08-08 (Q1 validation)

- Independently recomputed compound-Poisson parameters, dispersion indices for 18 series, empirical 95% intervals from 24-h blocks, validation/test metrics, the naive baseline, and 68 rolling windows, without random K-fold or test-based model selection.
- Audited unique task execution, candidate regions/latency, immediate real-time starts, earliest/latest limits, terminal hour, and overlap-based GPU/IT/facility use.
- Added all-local feasibility and 1-h/0.5-h start-resolution reruns across 2376-2405; summaries recorded two-stage solver messages/gaps/dual bounds.
- AST/Ruff passed; initial model/MILP/validation execution was deferred and no dependency added. Later fixes corrected nonexistent `*_input` and `StartHour_out` references because the relevant fields came only from one merge input; AST/Ruff/suffix checks passed again.

## 2026-08-08 (Q1 progress and PDF plots)

- Added input/forecast/backtest/two-stage MILP/write progress with status, objectives, gaps, and dual bounds.
- Implemented ten evidence figure groups for workflow, demand, compound-Poisson assumptions, forecasts/errors, candidate regions, utilization, Gantt/migration, waiting/resources, and validation. Output was PDF only.
- Read existing tables without model/validation reruns. Fixed unsupported `Line2D` hatch and a dual-axis `tight_layout` error, then generated ten single-page PDFs; pages/sizes/PDF-only output were checked.

## 2026-08-08 (Q1 plot evidence density)

- Figure 4 focused on validation/test hours 2352-2399 with black observed/blue compound-Poisson curves and separated boundaries/ticks. Figure 7 showed six-region GPU utilization from 0-105%, with dashed 100% limits.
- Figure 8 replaced an empty no-migration panel with a 6-by-6 GPU-hour execution matrix showing `F1*=0`/local execution. Figure 9 combined resource peaks, discrete waiting bars, total wait, and task-type means. Figure 10 combined F2 resolution sensitivity and constraint certificates.
- Corrected boundary text, four-decimal aggregate metrics, one-way latency, and full-horizon task context in Figures 2/5/6.
- Validation WAPE/RMSE/MAE: CP `0.7278/45.8519/23.8110`, LT168 `0.7213/45.7409/23.5995`, naive `1.0957/69.9249/35.8495`. Test: CP `0.6565/53.7868/26.5186`, LT168 `0.6469/53.6794/26.1302`, naive `0.8740/72.8628/35.3032`. LT168 was slightly more accurate overall; the paper needed to justify compound-Poisson as a structural model.
- Plot execution, AST/Ruff, single-page/size checks, and selected visual reviews passed; ten PDFs only. No new dependency.

## 2026-08-08 (Q1 color palette)

- Used the third seven-color palette in the user's supplied color document, with light blue/yellow/green/grayscale fills/grids/backgrounds.
- Regions used red/orange/green/cyan/blue/purple; task/model/resource colors were consistent. Demand/latency/execution matrices used the blue gradient; observed/baseline/reference lines used black/gray, red for violations.
- Regenerated ten single-page PDFs only, with no new dependency. Compile, 10/10 generation, size/page checks, and all-PDF raster reviews passed. Ruff was unavailable in the current environment and was not rerun.

## 2026-08-08 (Q1 final three-group figures)

- Group 1 combined 18-series hourly demand, a 6-by-3 cumulative matrix, and whole-system validation/test forecasts with observed/hierarchical/naive curves and WAPE.
- Group 2 combined representative-task Gantt, source-to-execution GPU-hours, six-region utilization, and GPU/AI-IT/facility peak dots, showing `F1*=0` and 100% local execution.
- Group 3 used four verified stress intervals with solid feasible lower/empty capacity upper endpoints, without claiming exact critical stress; added 84-window WAPE differences and hierarchical-restoration scatter with p-values/closure errors.
- Retained supplied colors/blue gradients without black/gray plotting elements; 2406 remained terminal. Generated 3/3 single-page PDFs, rasterized/reviewed them with Poppler, and checked PDF-only output. No new dependency.

## 2026-08-08 (Q1 A4 three-group review)

- Group 1 removed unhelpful heatmap boundaries, restored validation hours 2352-2375 for observed/hierarchical/naive forecasts, and retained the 2376 boundary/WAPE/legend.
- Group 2 selected 36 tasks across real-time inference, batch inference, and AI training; included `F2*=3h`, all-local feasibility, `F1*=0`, and 100% local execution while keeping resource panels focused on bottlenecks.
- Group 3 stated `84/84<0`, shortened axes, used equal-aspect restoration scatter with 18-series closure, and marked unresolved stress intervals without implying exact thresholds.
- A4 layouts avoided overlaps using the supplied colors with no black/gray elements. Compile, 3/3 single-page generation, Poppler page checks, and 160-dpi reviews passed. No new dependency.

## 2026-08-08 (Q1 independent handoff entry)

- Rewrote Q1 `main.py` as a strict preprocessing/model/validation/plot sequence. Added `--skip-preprocess`, `--skip-model`, `--skip-validation`, and `--skip-plot` with downstream-file checks.
- Stage failures or missing key results returned nonzero immediately; child output streamed to console/Q1 log instead of reporting successful completion after validation/plot warnings.
- Compile/help and plot-only skip paths passed. A complete Q1 entry run completed preprocessing, modeling, 84-window validation, and three PDF groups with `F1*=0`, `F2*=3`, and the required validation status.
- Added the original external modeler handoff checklist recording commands, outputs, values, and paper-claim boundaries.
