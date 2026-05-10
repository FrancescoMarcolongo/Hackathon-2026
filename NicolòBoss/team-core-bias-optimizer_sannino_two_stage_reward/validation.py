"""Validation report helpers for the core bias optimizer."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_json_default)


def _json_default(obj):
    try:
        import numpy as np

        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
    except Exception:
        pass
    if isinstance(obj, complex):
        return {"real": obj.real, "imag": obj.imag}
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def result_row(label: str, result: Dict[str, object], reward: Dict[str, object]) -> dict:
    return {
        "label": label,
        "g2": f"{float(result['g2_real']):.8g}{float(result['g2_imag']):+.8g}j",
        "epsilon_d": f"{float(result['eps_d_real']):.8g}{float(result['eps_d_imag']):+.8g}j",
        "T_X_us": float(result["T_X"]),
        "T_Z_us": float(result["T_Z"]),
        "bias": float(result["bias"]),
        "geo_lifetime_us": float(result["geo_lifetime"]),
        "reward": float(reward["reward"]),
        "loss_to_minimize": float(reward["loss_to_minimize"]),
        "target_achieved": bool(reward["is_feasible"]),
        "fit_penalty": float(result["fit_penalty"]),
        "fit_x_r2": float(result["fit_x_r2"]),
        "fit_z_r2": float(result["fit_z_r2"]),
    }


def write_markdown_report(
    path: Path,
    *,
    command: str,
    package_versions: dict,
    reward_formula: str,
    reward_config: dict,
    target_bias: float,
    table_rows: list[dict],
    selected_config: dict,
    figures: dict,
    notes: list[str],
) -> None:
    optimized_row = table_rows[-1]
    achieved = "yes" if optimized_row["target_achieved"] else "no"
    lines = [
        "# Core bias optimizer validation report",
        "",
        "## Reproduction command",
        "",
        f"```bash\n{command}\n```",
        "",
        "## Package versions",
        "",
    ]
    for name, version in package_versions.items():
        lines.append(f"- {name}: {version}")
    lines += [
        "",
        "## Reward formula",
        "",
        reward_formula,
        "",
        "Selected reward config:",
        "",
        "```json",
        json.dumps(reward_config, indent=2),
        "```",
        "",
        f"Target bias: `{target_bias:g}`",
        f"Target achieved by optimized candidate: `{achieved}`",
        "",
        "## Baseline vs optimized",
        "",
        "| label | g2 | epsilon_d | T_X (us) | T_Z (us) | bias | reward | target achieved |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        *[_table_line(row) for row in table_rows],
        "",
        "## Selected sweep run",
        "",
        "```json",
        json.dumps(selected_config, indent=2),
        "```",
        "",
        "## Figures",
        "",
    ]
    for label, fig_path in figures.items():
        lines.append(f"- {label}: `{fig_path}`")
    if notes:
        lines += ["", "## Notes", ""]
        lines.extend(f"- {note}" for note in notes)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _table_line(row: dict) -> str:
    return (
        f"| {row['label']} | `{row['g2']}` | `{row['epsilon_d']}` | "
        f"{row['T_X_us']:.4g} | {row['T_Z_us']:.4g} | {row['bias']:.4g} | "
        f"{row['reward']:.4g} | {row['target_achieved']} |"
    )
