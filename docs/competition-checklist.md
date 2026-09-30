# 2026 Huashu Cup Competition Checklist

> Historical team working notes prepared on 2026-08-07. This is not a competition paper or a supporting-submission document. The rule summaries below reflect the local organizer documents inspected at that time; verify the applicable official notices before using them for another competition.

## 0. Confirm the schedule

The original workspace timestamp was **2026-08-07 00:21 (UTC+8)**. The locally inspected organizer documents stated:

- Start: 2026-08-07 18:00.
- Deadline: 2026-08-10 20:00.
- The team captain submits the paper and supporting files through `https://new.saikr.com/vse/chinamcm26`.

At that timestamp, the event started later the same day. Check pinned registration notices and the official competition group for the final schedule.

## 1. Organizer requirements

- Select one of Problems A, B, or C. The first page contains the category, team number, title, abstract, and keywords.
- Keep the abstract on the first page. Number pages consecutively from 1 with Arabic numerals centered in the footer.
- Begin the main text on page 2. Omit the table of contents; the main text should normally stay within 20 pages. Do not include headers or personal/institutional identification in the main text or appendices.
- Upload the paper separately as PDF or Word, preferably PDF, no larger than 20 MB. Use the problem letter followed by the team number, for example `ACM2600001.pdf`.
- Supporting files are optional. If submitted, use one RAR/ZIP archive no larger than 20 MB containing all source programs actually used in the paper. Original problem data need not be duplicated.
- The supporting-archive filename uses the problem letter, team number, and the organizer-required Chinese attachment suffix. Preserve that exact suffix from the original submission guidance. Both paper and supporting files must be anonymous.
- Upload the signed commitment form separately. Verify the problem letter, `CM` plus seven-digit team number, full institution name, signatures, and date. The inspected guidance stated that submitted commitment forms could not be changed.
- After the contest starts, do not discuss the problems with anyone outside the team, including advisers, by phone, email, online consultation, or solution-discussion groups.
- Use the fallback email only if platform upload fails. Do not duplicate a successful platform submission. The fallback deadline was August 10 at 20:00; only one submission per person was allowed, with the first treated as final.

## 2. Preparation before the start

### Accounts and files

- [ ] Log in and confirm the team number, category, captain's account, and upload entry.
- [ ] Preserve original organizer files: handbook, format guidance, AI policy, paper template, and commitment form.
- [ ] Create working copies of the template and commitment form.
- [ ] Prepare anonymous paper and archive directories; use only the problem letter and team number in submission filenames.
- [ ] Check that the existing Python/MATLAB environment can read tables, optimize, analyze statistics, and plot. Avoid unconfirmed environment changes or dependencies during the contest.
- [ ] Establish PDF export, page-number checks, compression, file-opening checks, and upload verification procedures.

### Three-person roles

For a two-person team, combine adjacent responsibilities.

- Captain/paper coordinator: problem selection, version consolidation, abstract, and submission; control the single final version.
- Modeler: problem interpretation, assumptions, notation, equations, and explanations; the team must develop the core method independently.
- Programmer: data interfaces, models, tables, validation, and necessary figures; save every result and connect it to source.
- Everyone: include only material they understand and have checked. Keep brief records of key decisions.

## 3. First two hours after release

1. Download all statements and attachments, preserve originals, and record filenames and download times.
2. Read A/B/C independently. Record inputs, outputs, constraints, data granularity, and dependencies between questions before selecting algorithms.
3. Score each problem from 0 to 2 for data clarity, end-to-end solvability, team familiarity, verifiability, and clarity of the eventual paper.
4. Select a problem that can be completed and validated. Break ties using fewer uncertain parameters, clearer constraints, and useful existing code experience.
5. Establish a minimal structure for the actual questions:

```text
project/
├── main.py
├── README.md
└── questions/
    ├── question_01/
    │   ├── main.py
    │   ├── preprocess.py
    │   ├── model.py
    │   ├── validation.py
    │   ├── plot.py
    │   ├── data/raw/
    │   ├── data/processed/
    │   ├── outputs/tables/
    │   └── outputs/figures/
    └── question_xx/
```

This is the historical planning example, not the current repository layout. A dispatcher should call each question's existing workflow without duplicating modeling logic.

## 4. Historical problem-selection patterns

These categories summarize the 2023-2025 statements and excellent-paper abstracts in the handbook. They are not predictions of future questions.

| Type | Common subjects | Useful team strengths | First evidence to establish |
| --- | --- | --- | --- |
| A | Heat transfer, robotic arms, materials, optics, mechanisms, and processes | Physical interpretation, boundary conditions, and parameter meaning | Units, boundaries, identifiability, and physical constraints |
| B | Resource allocation, dynamic scheduling, layout, power/energy, and multiple objectives | Discrete decisions, constraints, simulation, and optimization | Feasibility, objective definitions, discrete constraints, and stability |
| C | Indicators, ranking, forecasting/classification, routing, and statistical experiments | Data cleaning, statistical judgment, and result interpretation | Granularity, indicator direction, train/validation boundaries, and route feasibility |

Strong historical papers turn the statement into computable objects, report a main model and results, then test conclusions using comparisons, sensitivity, stability, error analysis, or physical constraints. Algorithm complexity alone is not evidence.

## 5. Minimum evidence chain for each question

Keep all six stages before including a result in the abstract:

1. **Input:** row meaning, fields, units, and temporal/spatial granularity.
2. **Processing:** missing values, outliers, units, alignment, aggregation, and features that materially affect the model.
3. **Model:** objectives, decision variables, constraints, parameter sources, and solution procedure.
4. **Results:** save core values, plans, forecasts, or rankings to `outputs/tables/`.
5. **Validation:** choose feasibility, error, perturbation, scenarios, repeatability, stability, or appropriate controls. Actual execution evidence is needed before claiming a pass.
6. **Paper:** explain what each table/figure shows, why it occurs, and how it answers the question. Trace it to the current source and outputs.

### Common scoring and eligibility risks

- Mixing raw data, model outputs, and figures, then citing obsolete results after changing units or definitions.
- Describing a method different from the executed code, or providing incomplete source that cannot be checked.
- Treating file existence, program startup, or successful plotting as model validation.
- Claiming good performance without numbers, thresholds, controls, or constraint checks.
- Reporting only methods in the abstract. Include at least one actual quantitative result per question.

## 6. Paper-writing and submission sequence

Once results stabilize:

1. Write the restatement, analysis, assumptions, and notation. Keep unresolved checks distinct from established facts.
2. Complete each question's sequence: objective, method choice, equations/variables, solution, results, explanation, and validation.
3. Write the abstract: background, overall approach, per-question methods and real quantitative results, validation conclusions, and keywords.
4. Number references in first-citation order. Include book pages and website access dates.
5. Place the AI-use statement before the references. Use the organizer's exact original wording when no AI was used; otherwise report actual uses.
6. Include software, commands, all programs actually used, and AI-use details where applicable in the appendices.
7. Remove template instructions, placeholders, personal/institutional information, identifying image details, and author metadata. Metadata removal was a precaution inferred from anonymity requirements; consult the official notice before submission.

## 7. AI-use compliance

### Assistance requiring human understanding and records

- Brainstorming, editing, translation, abstract/keyword preparation, background summaries, and published-literature summaries.
- Organizing/analyzing collected data, code generation/debugging/optimization, and figure preparation. Label AI-generated figures as required by the organizer.

### Material the policy did not allow AI to replace

- Core method design, core theoretical innovations, key materials/methods, and central results/discussion.
- Fabricated collection records, altered raw data/figures, generated author information, direct generation of the entire paper or appendix.
- Formulas or models without reliable sources that the team cannot explain.

### Records to keep when AI was used

- Tool name, version/model, developer, and use date; verify the version rather than guessing.
- Paper sections/code locations, purposes, key prompts, and response summaries.
- Accepted suggestions, human changes, and rejected suggestions.
- Organizer-required comments preceding programs, the AI statement before references, and detailed appendix disclosure.
- The inspected policy also required an AI service registered with the national cyberspace authority. The local review did not establish whether the team's tool met that condition; the team must verify it.

## 8. Final hour before submission

### Paper

- [ ] Consistent problem letter, team number, and category.
- [ ] Abstract fits the first page and includes real quantitative results for each question.
- [ ] Page numbering begins at 1 in the centered footer; main text starts on page 2; no contents page, headers, or identifying details.
- [ ] Main text normally stays within 20 pages; equation, figure, table, reference, and appendix numbers agree with citations.
- [ ] Every reported value traces to current source, tables, or validation outputs.
- [ ] PDF opens without clipping, overlap, or encoding problems; size is below 20 MB and the filename is correct.

### Supporting files and commitment form

- [ ] Include necessary source, non-organizer data actually used, and essential intermediate results. Avoid duplicating original attachments.
- [ ] Archive is below 20 MB and contains no names, institutions, identifying personal paths, or chat histories.
- [ ] Commitment-form problem/team/institution/signature/date fields are correct; upload it separately.
- [ ] Upload and verify the paper, archive, and commitment form separately. Reopen or download each file after submission.
- [ ] Record final filenames, sizes, and successful upload evidence; preserve the confirmed version.

## 9. Original organizer files

The files in `materials/rules/` retain their original official filenames and content. Use:

- The 2026 competition handbook for registration, rules, historical statements, and sample papers.
- The paper-format/submission guidance for deadlines, sizes, naming, layout, and fallback email rules.
- The AI-use policy for permitted/prohibited uses, statements, source comments, appendix disclosures, and review rules.
- The paper template for first-page, AI-statement, reference, and appendix structure; remove instructional text before use.
- The commitment-form document for required fields and signatures.
