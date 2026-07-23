"""Report the active Python environment for reproducible project runs.

This script records interpreter paths, pip paths, package versions, and import
locations. It does not read or write `dataset/`.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.inventory import ensure_not_inside_dataset


DATASET_ROOT = PROJECT_ROOT / "dataset"
DOC_PATH = PROJECT_ROOT / "docs" / "python_environment_report.md"
LOG_PATH = PROJECT_ROOT / "logs" / "python_environment_report.txt"
CSV_PATH = PROJECT_ROOT / "outputs" / "results" / "python_environment_packages.csv"


COMMANDS = [
    ["where.exe", "python"],
    ["where.exe", "pip"],
    ["python", "--version"],
    ["pip", "--version"],
    ["python", "-c", "import sys; print(sys.executable)"],
    ["python", "-c", "import sys; print(sys.path)"],
    ["python", "-c", "import site; print(site.getsitepackages())"],
    ["python", "-c", "import pandas, numpy, sklearn, matplotlib, scipy, joblib; print('core packages ok')"],
    ["python", "-c", "import shap; print('shap', shap.__version__)"],
    ["python", "-c", "import xgboost; print('xgboost', xgboost.__version__)"],
    ["python", "-c", "import lightgbm; print('lightgbm', lightgbm.__version__)"],
]

PACKAGES = [
    "pandas",
    "numpy",
    "sklearn",
    "matplotlib",
    "scipy",
    "joblib",
    "openpyxl",
    "yaml",
    "shap",
    "xgboost",
    "lightgbm",
    "statsmodels",
    "seaborn",
]


def _run_command(command: list[str]) -> dict[str, Any]:
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "command": " ".join(command),
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def _package_row(package_name: str) -> dict[str, Any]:
    import importlib

    display_name = "pyyaml" if package_name == "yaml" else package_name
    try:
        module = importlib.import_module(package_name)
    except Exception as exc:
        return {
            "package": display_name,
            "import_name": package_name,
            "installed": False,
            "version": "",
            "module_file": "",
            "site_packages_location": "",
            "error": repr(exc),
        }
    module_file = Path(getattr(module, "__file__", "") or "")
    version = getattr(module, "__version__", "unknown")
    site_location = ""
    parts = list(module_file.parts)
    if "site-packages" in parts:
        idx = parts.index("site-packages")
        site_location = str(Path(*parts[: idx + 1]))
    return {
        "package": display_name,
        "import_name": package_name,
        "installed": True,
        "version": str(version),
        "module_file": str(module_file),
        "site_packages_location": site_location,
        "error": "",
    }


def _environment_details() -> dict[str, Any]:
    import site

    return {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "sys_path": sys.path,
        "site_packages": site.getsitepackages(),
        "user_site_packages": site.getusersitepackages(),
    }


def _write_csv(rows: list[dict[str, Any]]) -> None:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_logs(command_results: list[dict[str, Any]], package_rows: list[dict[str, Any]], env: dict[str, Any]) -> None:
    lines = ["Python environment report", "=" * 60, ""]
    for result in command_results:
        lines.extend(
            [
                f"$ {result['command']}",
                f"returncode: {result['returncode']}",
                "stdout:",
                result["stdout"] or "",
                "stderr:",
                result["stderr"] or "",
                "",
            ]
        )
    lines.extend(["Environment details", "-" * 60, json.dumps(env, indent=2), ""])
    lines.extend(["Package rows", "-" * 60])
    for row in package_rows:
        lines.append(json.dumps(row, ensure_ascii=False))
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_markdown(command_results: list[dict[str, Any]], package_rows: list[dict[str, Any]], env: dict[str, Any]) -> None:
    python_where = next((row["stdout"] for row in command_results if row["command"] == "where.exe python"), "")
    pip_where = next((row["stdout"] for row in command_results if row["command"] == "where.exe pip"), "")
    package_by_name = {row["package"]: row for row in package_rows}
    lines = [
        "# Python Environment Report",
        "",
        "This report records the Python environment used by Codex for the current project run.",
        "No raw `dataset/` files are read or written by this check.",
        "",
        "## Active Interpreter",
        "",
        f"- Codex `python.exe`: `{env['python_executable']}`",
        f"- Python version: `{env['python_version'].splitlines()[0]}`",
        f"- `where python`:",
        "",
        "```text",
        python_where,
        "```",
        "",
        f"- `where pip`:",
        "",
        "```text",
        pip_where,
        "```",
        "",
        "## Package Status",
        "",
        "| Package | Installed | Version | Module file | Site-packages | Error |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in package_rows:
        lines.append(
            f"| {row['package']} | {row['installed']} | {row['version'] or '-'} | "
            f"`{row['module_file']}` | `{row['site_packages_location']}` | {row['error'] or '-'} |"
        )

    lines.extend(
        [
            "",
            "## Key Optional Packages",
            "",
            f"- SHAP installed: `{package_by_name.get('shap', {}).get('installed')}`, version `{package_by_name.get('shap', {}).get('version') or '-'}`",
            f"- XGBoost installed: `{package_by_name.get('xgboost', {}).get('installed')}`, version `{package_by_name.get('xgboost', {}).get('version') or '-'}`",
            f"- LightGBM installed: `{package_by_name.get('lightgbm', {}).get('installed')}`, version `{package_by_name.get('lightgbm', {}).get('version') or '-'}`",
            "",
            "## VS Code Interpreter Recommendation",
            "",
            f"Select this interpreter in VS Code if you want to reproduce the current Codex environment exactly: `{env['python_executable']}`.",
            "",
            "If you prefer an isolated project environment, create `.venv` in the project root and select `.venv\\Scripts\\python.exe` in VS Code.",
            "",
            "## Rebuild Commands",
            "",
            "```powershell",
            "cd E:\\sportpro",
            "python -m venv .venv",
            ".\\.venv\\Scripts\\Activate.ps1",
            "python -m pip install --upgrade pip",
            "python -m pip install -r requirements.txt",
            "python -m pip install -r requirements-optional.txt",
            "```",
            "",
            "## Python Search Paths",
            "",
            "```text",
            json.dumps(env["sys_path"], indent=2),
            "```",
            "",
            "## Site Packages",
            "",
            "```text",
            json.dumps(
                {
                    "site.getsitepackages": env["site_packages"],
                    "site.getusersitepackages": env["user_site_packages"],
                },
                indent=2,
            ),
            "```",
            "",
        ]
    )
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    for path in (DOC_PATH, LOG_PATH, CSV_PATH):
        ensure_not_inside_dataset(path, DATASET_ROOT)
    command_results = [_run_command(command) for command in COMMANDS]
    package_rows = [_package_row(package) for package in PACKAGES]
    env = _environment_details()
    _write_csv(package_rows)
    _write_logs(command_results, package_rows, env)
    _write_markdown(command_results, package_rows, env)
    print("Python environment report complete.")
    print(f"Codex python: {env['python_executable']}")
    print(f"Wrote: {DOC_PATH}")
    print(f"Wrote: {LOG_PATH}")
    print(f"Wrote: {CSV_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
