# Q4 Handoff, Serial Execution, and Result Integration

This is the historical V3 handoff checklist. Its V3 directories and completion markers describe that version's artifacts. The current `model.py` uses V4 for formal baseline and scenario execution and rejects V2/V3 baselines on that path. Use [the code guide](../../README.md) for current V4 commands; do not present this historical checklist as evidence that V4 was run.

## 1. Baseline Audit

The baseline audit corresponds to `--baseline-audit`.

- It does not optimize again or invoke MILP, and it does not modify, overwrite, or delete baseline results.
- It reads completed baseline task assignments, energy dispatch, objective summaries, and the completion marker. It independently checks unique task execution, immediate starts for realtime tasks, network latency, earliest starts and deadlines, GPU/IT/facility capacity, renewable balance, grid import/export bounds, SOC recurrence, charging/discharging exclusivity, terminal SOC, and agreement among the six metrics.
- Success requires a complete baseline from the accepted version and `audit_passed=true`. A missing or incompatible baseline is explicitly rejected; old V2 results are not substituted.
- Baseline execution solves and writes the baseline. The audit is the read-only check before handoff.

## 2. Historical V3 Baseline Acceptance

For the original V3 workflow, complete the baseline run and the read-only audit before packaging:

1. Run the V3 baseline configuration with automatic checkpoint recovery.
2. Run the V3 read-only baseline audit. Require `passed=True` and process exit code 0.
3. Check that all of these files exist:

```text
code/question/question_04/outputs/matheuristic_v3_doccompliant/tables/
  .q4_matheuristic_v3_complete.json
  q4_task_assignments.csv
  q4_region_hour_dispatch.csv
  q4_objective_summary.csv
  q4_window_solver.csv
  q4_scaling.csv
  q4_model_configuration.csv
  q4_simple_validation.csv
  q4_calibration_records.csv
  q4_forecast_profile.csv
```

4. Confirm these fields in `.q4_matheuristic_v3_complete.json`:

```json
{
  "complete": true,
  "audit_passed": true,
  "model_version": "Q4_MATHEURISTIC_V3_DOCUMENT_COMPLIANT"
}
```

5. Package only after every requirement above is satisfied. A running baseline or a checkpoint alone is not a completed shared baseline.

These are historical acceptance criteria. The current V4 command requires the corresponding V4 artifacts and marker rather than the V3 files above.

## 3. Packaging

Keep relative paths intact. A package containing only the computational project should unpack to a `code/` directory. Exclude `.venv/` to avoid transferring a large, machine-specific Python environment.

Required computational inputs and artifacts:

```text
code/
  pyproject.toml
  uv.lock
  .run/
    Q4_*.run.xml
  question/question_01/data/processed/shared/
    tasks_clean.csv
    task_candidate_regions.csv
    storage_params.csv
  question/question_02/outputs/tables/
    q2_assignments.csv
  question/question_04/
    model.py
    preprocess.py
    data/processed/q4/q4_region_hour_input.csv
    outputs/matheuristic_v3_doccompliant/
    q4-runbook.md
```

Optional artifacts:

- `code/question/question_04/outputs/matheuristic_v2/`: historical diagnostics, not required for V3 scenario execution.
- `archive/source-backups/question_04/`: preserved source backups, not required for execution.
- Completed `code/question/question_04/outputs/scenarios/<scenario-name>/` directories: for review or integration. Do not distribute incomplete scenarios as finished results.

Exclude `.venv/`, `.idea/`, and `__pycache__/`. Transfer a running scenario checkpoint only for an explicit continuation on another machine, with execution stopped on the source machine first.

## 4. Environment Check

Use Python 3.12 and the dependencies locked in `code/uv.lock`. The historical V3 environment recorded Python 3.12.13, NumPy 2.5.1, pandas 3.0.5, and SciPy 1.18.0; this is a historical record rather than a new compatibility guarantee.

From `code/`, prepare an environment when required:

```powershell
uv sync --frozen
```

In PyCharm, select `$PROJECT_DIR$/.venv/Scripts/python.exe` with `code/` opened as the project. Audit the accepted baseline before taking over a scenario. Existing V3 artifacts require their matching historical implementation; the current implementation audits V4.

## 5. Serial Execution Order

For the original single-machine workflow, run these steps serially:

1. Baseline.
2. Read-only baseline audit.
3. Low-carbon reference.
4. Carbon constraint, `lambda=0.25`.
5. Carbon constraint, `lambda=0.50`.
6. Carbon constraint, `lambda=1.00`.
7. Parity electricity prices.
8. Renewable smoothing, `gamma=0.25`.
9. Renewable smoothing, `gamma=0.50`.
10. Renewable smoothing, `gamma=1.00`.
11. Read-only scenario comparison.

The low-carbon reference must finish before any carbon-constrained scenario. Parity pricing and renewable smoothing depend on the baseline only. They do not logically require the low-carbon reference, but one large solve at a time is recommended on each machine.

The corresponding current command options are `--run-scenario low_carbon_reference`, `--run-scenario carbon_constraint`, `--run-scenario flat_price`, `--run-scenario low_variability_renewable`, and `--compare-scenarios`. These commands produce V4 results with the current source, not historical V3 results.

## 6. Two-Machine Division of Work

After both machines share the same audited baseline:

| Operator | Serial tasks | Prerequisites |
| --- | --- | --- |
| A | Low-carbon reference -> lambda=0.25 -> lambda=0.50 -> lambda=1.00 | Shared baseline; each lambda waits for the completed low-carbon reference |
| B | Parity pricing -> gamma=0.25 -> gamma=0.50 -> gamma=1.00 | Shared baseline |

Keep tasks serial within each machine. Different scenarios can run concurrently across machines. Do not run the same `scenario-name` on both machines, and do not synchronize the same output directory bidirectionally.

## 7. Interruption, Failure, and Recovery

- After interruption, abnormal exit, or restart, rerun the same configuration on the original machine to resume.
- Keep the scenario's version-specific `tables/` directory, checkpoint, and progress CSV files.
- Do not reuse the same scenario name with a different signature. The program rejects incompatible signatures and unsafe overwrites to protect existing results.
- To transfer unfinished work, copy the entire `outputs/scenarios/<scenario-name>/` directory. Stop the source configuration before starting the matching configuration on the destination machine.
- Record the complete error message for a failure. Do not manually edit CSV, JSON, or PKL checkpoints.
- Legacy cache paths that hash source bytes will reject signatures created before the source translation. Current V4 matheuristic and rollout signatures hash inputs and configuration rather than source text. Preserve existing artifacts; never edit historical markers or signatures to bypass compatibility checks.

## 8. Integrating Completed Results

Each completed historical V3 scenario must contain:

```text
outputs/scenarios/<scenario-name>/matheuristic_v3_doccompliant/tables/
  .q4_matheuristic_v3_complete.json
  q4_task_assignments.csv
  q4_region_hour_dispatch.csv
  q4_objective_summary.csv
  q4_scenario_metadata.json
```

For current V4 results, require the corresponding V4 directory and marker. Never mix versions.

Integration rules:

1. Copy only the entire scenario directory after completion and `audit_passed=true`.
2. Place it under the main project's `code/question/question_04/outputs/scenarios/`. Preserve its directory name and internal filenames.
3. Do not overwrite the main project's baseline directory.
4. Exclude V2 results, old `low_carbon_reference_guarded` directories, and incomplete checkpoints from formal scenario results.
5. Once all compatible scenarios are integrated, run the read-only scenario comparison. The historical V3 comparison was written to:

```text
code/question/question_04/outputs/scenarios/q4_matheuristic_v3_scenario_comparison.csv
```

The current comparison command writes the corresponding V4 comparison table.

## 9. Handoff Record

Copy this template into the handoff message or README:

```text
Scenario name:
Run configuration:
Machine / operator:
Start and finish times:
Completion marker: complete=true; audit_passed=true / incomplete
Model version:
Result directory:
Resumed from checkpoint: yes / no
Exceptions: none / complete error message
Notes:
```

## 10. Final Integration Checklist

- [ ] The shared baseline is complete and its read-only audit reports `passed=True`.
- [ ] Every required scenario has a completion marker from the same version and `audit_passed=true`.
- [ ] No old-version energy results or incomplete checkpoints are mixed in.
- [ ] All machines used identical source, input CSV files, and baseline artifacts.
- [ ] All completed scenarios are integrated into one main project's `outputs/scenarios/` directory.
- [ ] The read-only comparison command generated the matching-version comparison table.
- [ ] The paper cites only audited metrics and scenario comparisons supported by the matching artifacts.
