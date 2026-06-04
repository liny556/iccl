"""Plot public ICCL experiment result files.

The script intentionally covers the result formats produced by the public
entry points in this repository without depending on local legacy plotting
helpers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def numeric_key(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return float("nan")


def sorted_numeric_items(payload: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    items = [(str(key), value) for key, value in payload.items() if isinstance(value, dict)]
    return sorted(items, key=lambda item: numeric_key(item[0]))


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_plot.pdf")


def plot_metric_sweep(payload: Dict[str, Any], output_path: Path) -> bool:
    items = sorted_numeric_items(payload)
    rows = [
        (float(key), value)
        for key, value in items
        if "overall_mse" in value or "avg_task_mse" in value
    ]
    if not rows:
        return False

    xs = [row[0] for row in rows]
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    if any("overall_mse" in row[1] for row in rows):
        ax.plot(
            xs,
            [row[1].get("overall_mse", float("nan")) for row in rows],
            marker="o",
            label="Overall MSE",
        )
    if any("avg_task_mse" in row[1] for row in rows):
        ax.plot(
            xs,
            [row[1].get("avg_task_mse", float("nan")) for row in rows],
            marker="s",
            label="Average task MSE",
        )

    task_keys = sorted(
        {
            key
            for _, row in rows
            for key in row.keys()
            if key.startswith("task_") and key.endswith("_mse")
        }
    )
    for task_key in task_keys:
        ax.plot(
            xs,
            [row.get(task_key, float("nan")) for _, row in rows],
            linewidth=1.0,
            alpha=0.45,
            label=task_key.replace("_mse", "").replace("_", " ").title(),
        )

    ax.set_xlabel("Sweep value")
    ax.set_ylabel("MSE")
    ax.set_title("ICCL metric sweep")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_task_order(payload: Dict[str, Any], output_path: Path) -> bool:
    per_task = payload.get("per_task_mse")
    if not isinstance(per_task, dict):
        return False

    task_items = sorted(per_task.items(), key=lambda item: numeric_key(item[0].split("_")[-1]))
    xs = [item[0].replace("_", " ").title() for item in task_items]
    ys = [item[1] for item in task_items]

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.bar(xs, ys)
    ax.set_xlabel("Task position")
    ax.set_ylabel("MSE")
    ax.set_title("Task order analysis")
    ax.grid(True, axis="y", alpha=0.25)
    fig.autofmt_xdate(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_qwen_rows(payload: Dict[str, Any], output_path: Path) -> bool:
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        return False

    xs = [row["M"] for row in rows]
    fig, (ax_b, ax_a) = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True)

    ax_b.plot(xs, [row["task_b_baseline_accuracy"] for row in rows], marker="o", label="Baseline")
    ax_b.plot(xs, [row["task_b_iccl_accuracy"] for row in rows], marker="s", label="ICCL")
    ax_b.set_title("Task B: AG News")
    ax_b.set_xlabel("M")
    ax_b.set_ylabel("Accuracy")
    ax_b.grid(True, alpha=0.25)
    ax_b.legend(fontsize=8)

    ax_a.plot(xs, [row["task_a_baseline_accuracy"] for row in rows], marker="o", label="Baseline")
    ax_a.plot(xs, [row["task_a_final_accuracy"] for row in rows], marker="s", label="Final")
    ax_a.set_title("Task A: SST-2")
    ax_a.set_xlabel("M")
    ax_a.grid(True, alpha=0.25)
    ax_a.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_file(input_path: Path, output_path: Path) -> None:
    with input_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    plotters = (plot_qwen_rows, plot_task_order, plot_metric_sweep)
    for plotter in plotters:
        if plotter(payload, output_path):
            print(f"Saved plot to {output_path}")
            return

    raise ValueError(f"Unsupported result format: {input_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot public ICCL result JSON files.")
    parser.add_argument("input", type=Path, help="Path to a result JSON file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output plot path. Defaults to '<input_stem>_plot.pdf'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output or default_output_path(args.input)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_file(args.input, output_path)


if __name__ == "__main__":
    main()
