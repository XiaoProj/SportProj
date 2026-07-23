# Run Project

This project uses Route A:

- Human-M3 and EPFL are used for pose movement-quality analysis only.
- 5742821 is used for tabular fatigue/sleep/free-throw performance exploratory analysis only.
- Pose features are not merged with fatigue/sleep labels.

## 1. Enter The Project

```powershell
cd E:\sportpro
```

## 2. Create And Activate A Virtual Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\.venv\Scripts\Activate.ps1
```

## 3. Install Dependencies

Install required dependencies:

```powershell
python -m pip install -r requirements.txt
```

Install optional modeling/interpretability dependencies:

```powershell
python -m pip install -r requirements-optional.txt
```

If LightGBM fails to install on Windows, the project still runs; LightGBM rows are recorded as skipped optional models.

## 4. Select Python In VS Code

If using the current Codex/Anaconda environment, select:

```text
D:\anaconda3\python.exe
```

If using a project virtual environment, select:

```text
E:\sportpro\.venv\Scripts\python.exe
```

Use `Ctrl+Shift+P` in VS Code, then choose `Python: Select Interpreter`.

## 5. Check The Environment

```powershell
python -B scripts/check_and_report_python_environment.py
```

Outputs:

- `docs/python_environment_report.md`
- `logs/python_environment_report.txt`
- `outputs/results/python_environment_packages.csv`

## 6. Pose Movement Quality Pipeline

Run the existing small-sample pose movement-quality analysis:

```powershell
python -B scripts/run_pose_movement_quality_analysis.py
```

This uses processed sample pose feature tables only. It does not read MP4 files and does not train a fatigue classifier.

## 7. Tabular Fatigue/Sleep/Performance Pipeline

Run Route A tabular exploratory modeling:

```powershell
python -B scripts/run_phase7_5_integrated_fix.py
```

This uses:

- `outputs/processed/fatigue_sleep_label_rules.csv`
- 5742821-derived tabular variables only

It does not use Human-M3 or EPFL pose features.

## 8. Generate Manuscript-Ready Figures

```powershell
python -B scripts/make_manuscript_ready_figures.py
python -B scripts/make_manuscript_contact_sheet.py
```

Outputs are written to:

```text
outputs/figures/manuscript_ready/
```

## 9. View Model Files

Exploratory tabular model bundles are saved under:

```text
outputs/models/tabular_fatigue_sleep_performance/
```

Each bundle includes a `.joblib` model/preprocessing pipeline, feature list JSON, metadata JSON, and model card Markdown.

## 10. Confirm Dataset Was Not Modified

The project scripts use `ensure_not_inside_dataset()` guards for report/model/figure outputs.

Manual checks:

```powershell
Get-ChildItem -Recurse dataset | Measure-Object
Get-Content logs\phase7_5_integrated_fix.txt -TotalCount 5
Get-Content docs\data_alignment_feasibility.md
```

All derived files should be under `outputs/`, `logs/`, `docs/`, `src/`, `scripts/`, or `configs/`.
