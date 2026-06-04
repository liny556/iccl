"""
Task-family experiments for in-context continual learning.

This script extends Section 5-style experiments from linear regression
to the task families used in the public experiments.
"""
import argparse
import json
import os
from typing import Dict, List, Optional

from eval import get_model_from_run
from multi_task_eval import eval_multi_task_model


def parse_task_model_pairs(raw_pairs: List[str]) -> Dict[str, str]:
    pairs: Dict[str, str] = {}
    for item in raw_pairs:
        if ":" not in item:
            raise ValueError(f"Invalid --task_model_pairs item: {item}. Expected task_name:model_path")
        task_name, model_path = item.split(":", 1)
        pairs[task_name.strip()] = model_path.strip()
    return pairs


def evaluate_task_family(
    task_model_pairs: Dict[str, str],
    data_name: str,
    n_tasks: int,
    n_context_values: List[int],
    num_eval_batches: int,
    batch_size: int,
    device: str,
    output_dir: str,
    n_dims_override: Optional[int] = None,
):
    os.makedirs(output_dir, exist_ok=True)
    aggregate_results: Dict[str, Dict] = {}

    for task_name, model_path in task_model_pairs.items():
        model, conf = get_model_from_run(model_path)
        model = model.to(device).eval()

        n_dims = n_dims_override if n_dims_override is not None else conf.model.n_dims
        task_results: Dict[str, Dict] = {}

        for n_context in n_context_values:
            metrics = eval_multi_task_model(
                model=model,
                task_name=task_name,
                data_name=data_name,
                n_dims=n_dims,
                n_tasks=n_tasks,
                n_context=n_context,
                num_eval_batches=num_eval_batches,
                batch_size=batch_size,
                device=device,
            )
            task_results[str(n_context)] = metrics
            print(
                f"[{task_name}] M={n_context} overall={metrics['overall_mse']:.4f} "
                f"avg-task={metrics['avg_task_mse']:.4f}"
            )

        aggregate_results[task_name] = task_results

    output_path = os.path.join(output_dir, "task_family_context_sweep.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(aggregate_results, f, indent=2)
    print(f"Saved results to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run ICCL experiments across task families.")
    parser.add_argument(
        "--task_model_pairs",
        nargs="+",
        required=True,
        help=(
            "Task-model pairs in task_name:model_path format. "
            "Example: linear_regression:models/linear_regression/pretrained "
            "sparse_linear_regression:models/sparse_linear_regression/pretrained "
            "relu_2nn_regression:models/relu_2nn_regression/pretrained"
        ),
    )
    parser.add_argument("--data_name", type=str, default="gaussian")
    parser.add_argument("--n_tasks", type=int, default=5)
    parser.add_argument("--n_context_values", nargs="+", type=int, default=[1, 3, 5, 7, 9])
    parser.add_argument("--num_eval_batches", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--n_dims", type=int, default=None)
    args = parser.parse_args()

    task_model_pairs = parse_task_model_pairs(args.task_model_pairs)
    evaluate_task_family(
        task_model_pairs=task_model_pairs,
        data_name=args.data_name,
        n_tasks=args.n_tasks,
        n_context_values=args.n_context_values,
        num_eval_batches=args.num_eval_batches,
        batch_size=args.batch_size,
        device=args.device,
        output_dir=args.output_dir,
        n_dims_override=args.n_dims,
    )


if __name__ == "__main__":
    main()
