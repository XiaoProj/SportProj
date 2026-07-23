# Reproducibility

## Analysis Scope

The project uses Route A. Pose movement-quality analyses and 5742821 tabular exploratory analyses are independent.

No current script should claim that pose features predict fatigue, because Human-M3/EPFL and 5742821 do not share verified subject/session alignment.

## Environment

The current Codex run used:

```text
D:\anaconda3\python.exe
```

Run this command to regenerate the environment report:

```powershell
python -B scripts/check_and_report_python_environment.py
```

See:

- `docs/python_environment_report.md`
- `outputs/results/python_environment_packages.csv`

## Fresh Setup

```powershell
cd E:\sportpro
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-optional.txt
```

If optional dependencies fail, rerun the project anyway. Missing optional models are recorded in skipped-model reports.

## Determinism

Modeling scripts use fixed random seeds where applicable. Cross-validation uses small-sample leave-one-subject-out when `subject_id` is available.

## Input Data

Raw `dataset/` files are read-only. Current manuscript figures and models are based on processed sample tables and prior schema-probe outputs.

## Outputs

Primary reproducible outputs:

- `outputs/results/`
- `outputs/models/tabular_fatigue_sleep_performance/`
- `outputs/figures/manuscript_ready/`
- `docs/manuscript_ready_captions.md`

## Limitations

- 5742821 labels are candidate/exploratory labels or scores, not ground-truth fatigue labels.
- Shot 1/Shot 2 interpretation is protocol-derived and should remain described as candidate.
- Pose knee valgus remains a proxy feature, not a clinical valgus measurement.
- Current pose features are small-sample examples, not full-dataset estimates.
