"""
Plot per-task MSE vs context length M for each task family in task_family_context_sweep.json.

For each task family writes two PDFs:
- task_family_<name>_per_task_mse_vs_M.pdf — normalized MSE (matches six_2.plot_exp_6_2_normal)
- task_family_<name>_per_task_mse_vs_M_raw.pdf — raw MSE (no normalization)
"""
import argparse
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.rcParams["axes.unicode_minus"] = False
sns.set_style("whitegrid")
sns.set_palette("husl")


def _infer_n_tasks(metrics: dict) -> int:
    max_idx = -1
    for k in metrics:
        m = re.match(r"^task_(\d+)_mse$", k)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1 if max_idx >= 0 else 0


def plot_task_family_block(
    normal_data: dict,
    output_path: str,
    title: str,
    normalize: bool = True,
):
    """
    normal_data: M_str -> metrics dict (same shape as exp_6_2 JSON).
    If normalize=True, MSE is min-max scaled within each task family (matches exp_6_2_normal).
    If normalize=False, plots raw per-task MSE.
    """
    M_values = sorted(int(k) for k in normal_data.keys() if str(k).isdigit())
    if not M_values:
        print(f"Skip (no M keys): {title}")
        return

    first_m = str(M_values[0])
    n_tasks = _infer_n_tasks(normal_data[first_m])
    if n_tasks <= 0:
        print(f"Skip (no task_*_mse): {title}")
        return

    task_mses = {t: [] for t in range(n_tasks)}
    for M in M_values:
        M_str = str(M)
        block = normal_data.get(M_str, {})
        for t in range(n_tasks):
            key = f"task_{t}_mse"
            task_mses[t].append(block.get(key, np.nan))

    all_mses = []
    for t in range(n_tasks):
        all_mses.extend(m for m in task_mses[t] if not np.isnan(m))
    if not all_mses:
        print(f"Skip (empty MSE): {title}")
        return

    if normalize:
        m_min, m_max = np.min(all_mses), np.max(all_mses)
        for t in range(n_tasks):
            normed = []
            for mse in task_mses[t]:
                if np.isnan(mse):
                    normed.append(np.nan)
                elif m_max - m_min > 0:
                    normed.append((mse - m_min) / (m_max - m_min))
                else:
                    normed.append(0.0)
            task_mses[t] = normed
        ylabel = "Normalized MSE"
    else:
        ylabel = "MSE"

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, n_tasks))

    for t in range(n_tasks):
        ax.plot(
            M_values,
            task_mses[t],
            marker="o",
            markersize=16,
            linewidth=5,
            label=f"Task {t + 1}",
            color=colors[t],
            alpha=0.8,
            markeredgewidth=3.5,
            markeredgecolor="white",
            linestyle="-",
        )

    ax.set_xlabel("Context Length M", fontsize=30, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=30, fontweight="bold")
    ax.set_title(title, fontsize=32, fontweight="bold")
    ax.grid(True, alpha=0.4, linestyle="--", linewidth=1.8)
    ax.legend(fontsize=28, loc="best", ncol=2, framealpha=0.9)
    ax.tick_params(axis="both", which="major", labelsize=26, width=2, length=8)
    ax.tick_params(axis="both", which="minor", labelsize=24, width=1.5, length=6)

    if len(M_values) > 20:
        step = max(1, len(M_values) // 20)
        ax.set_xticks(M_values[::step])
        ax.tick_params(axis="x", rotation=45)
    else:
        ax.set_xticks(M_values)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", format="pdf")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json_path",
        type=str,
        default="task_family_context_sweep.json",
        help="Path to task_family_context_sweep.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory for output PDFs",
    )
    args = parser.parse_args()

    with open(args.json_path, "r", encoding="utf-8") as f:
        bundle = json.load(f)

    for task_family, block in bundle.items():
        if not isinstance(block, dict):
            continue
        safe_name = task_family.replace("/", "_")
        base = f"task_family_{safe_name}_per_task_mse_vs_M"
        title_base = f"Per-Task MSE vs Context Length M ({task_family})"
        plot_task_family_block(
            block,
            os.path.join(args.output_dir, f"{base}.pdf"),
            title=title_base,
            normalize=True,
        )
        plot_task_family_block(
            block,
            os.path.join(args.output_dir, f"{base}_raw.pdf"),
            title=f"{title_base} [raw]",
            normalize=False,
        )


if __name__ == "__main__":
    main()
