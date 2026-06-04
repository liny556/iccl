"""
Multi-task continual learning experiments.
Implements the experiments described in Section 6 of the paper.
"""
import os
import json
import argparse
from typing import Dict, List, Optional
import numpy as np
import torch
from tqdm import tqdm

from eval import get_model_from_run
from multi_task_eval import eval_multi_task_model
from tasks import get_task_sampler
from samplers import get_data_sampler
from theorem_4_3 import tensor_weights_to_numpy_list, theorem_4_3_decomposition


def experiment_6_1_training_sample_size(
    model_path: str,
    n_dims: Optional[int] = None,
    n_tasks: int = 5,
    n_context: int = 5,
    n_values: List[int] = [100, 500, 1000, 5000, 10000],
    num_eval_batches: int = 100,
    batch_size: int = 32,
    output_dir: str = "./results",
    device: str = "cuda",
):
    """
    Experiment 6.1: Effect of training sample size N
    
    Varies N (number of pretraining examples) and evaluates per-task MSE.
    """
    print("=" * 60)
    print("Experiment 6.1: Effect of training sample size N")
    print("=" * 60)
    
    # Load model
    model, conf = get_model_from_run(model_path)
    model = model.to(device).eval()
    
    n_dims = _resolve_n_dims(n_dims, conf)
    
    results = {}
    
    for N in tqdm(n_values, desc="Varying N"):
        print(f"\nEvaluating with N={N}")
        
        # Note: In practice, N affects the pretraining, so we would need
        # models trained with different N values. For now, we evaluate
        # the same model and note that N affects the bias term in Gamma.
        
        metrics = eval_multi_task_model(
            model=model,
            task_name="linear_regression",
            data_name="gaussian",
            n_dims=n_dims,
            n_tasks=n_tasks,
            n_context=n_context,
            num_eval_batches=num_eval_batches,
            batch_size=batch_size,
            device=device,
        )
        
        results[N] = metrics
        print(f"  Overall MSE: {metrics['overall_mse']:.4f} +/- {metrics['overall_mse_std']:.4f}")
        print(f"  Avg Task MSE: {metrics['avg_task_mse']:.4f} +/- {metrics['avg_task_mse_std']:.4f}")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "exp_6_1_training_sample_size.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    
    return results


def experiment_6_2_context_length(
    model_path: str,
    n_dims: Optional[int] = None,
    n_tasks: int = 5,
    m_values: Optional[List[int]] = None,
    num_eval_batches: int = 100,
    batch_size: int = 32,
    output_dir: str = "./results",
    device: str = "cuda",
):
    """
    Experiment 6.2: Effect of Single Context Length M
    
    Tests the M/(t^2*(M+1)^2) variance scaling.
    
    This experiment FIXES the number of tasks T and varies M to show
    how context length affects performance.
    """
    print("=" * 60)
    print("Experiment 6.2: Effect of Single Context Length M")
    print(f"Fixed: T = {n_tasks} tasks")
    print("=" * 60)
    
    # Load model
    model, conf = get_model_from_run(model_path)
    model = model.to(device).eval()
    
    n_dims = _resolve_n_dims(n_dims, conf)
    
    # Check model's maximum sequence length
    max_positions = conf.model.n_positions
    max_sequence_tokens = max_positions * 2  # Model interleaves x and y
    print(f"Model's maximum sequence length: {max_sequence_tokens} tokens ({max_positions} points)")
    
    # Calculate maximum M value based on model constraints
    # Sequence length = 2 * n_tasks * (M + 1) <= max_sequence_tokens
    max_M = (max_sequence_tokens // (2 * n_tasks)) - 1
    print(f"With T={n_tasks} tasks, maximum M = {max_M}")
    
    # Generate M values if not provided
    # Use a dense range to better show the effect of M
    if m_values is None:
        # Generate values from 1 to max_M with reasonable spacing
        if max_M <= 5:
            m_values = list(range(1, max_M + 1))  # All values if range is small
        elif max_M <= 10:
            # Sample more densely: every 1 or 2
            m_values = list(range(1, max_M + 1, 1))
        else:
            # For larger ranges, sample more points but not all
            # Include: 1, 2, 3, then every 2-3 values up to max
            m_values = [1, 2, 3]
            m_values.extend(range(5, max_M + 1, 2))
            if max_M not in m_values:
                m_values.append(max_M)
    
    # Filter m_values to ensure sequence length doesn't exceed model limit
    valid_m_values = []
    for M in sorted(set(m_values)):  # Remove duplicates and sort
        seq_length = 2 * n_tasks * (M + 1)
        if seq_length <= max_sequence_tokens:
            valid_m_values.append(M)
        else:
            print(f"Warning: Skipping M={M} (sequence length {seq_length} > {max_sequence_tokens})")
    
    if not valid_m_values:
        raise ValueError(
            f"No valid M values. Model can only handle sequences up to {max_sequence_tokens} tokens. "
            f"With {n_tasks} tasks, maximum M is {max_M}"
        )
    
    print(f"\nTesting M values: {valid_m_values}")
    print(f"Sequence lengths: {[2 * n_tasks * (M + 1) for M in valid_m_values]} tokens")
    print()
    
    results = {}
    
    for M in tqdm(valid_m_values, desc="Varying M"):
        seq_length = 2 * n_tasks * (M + 1)
        print(f"\nEvaluating M={M} (sequence length: {seq_length} tokens, {seq_length/max_sequence_tokens*100:.1f}% of max)")
        
        metrics = eval_multi_task_model(
            model=model,
            task_name="linear_regression",
            data_name="gaussian",
            n_dims=n_dims,
            n_tasks=n_tasks,
            n_context=M,
            num_eval_batches=num_eval_batches,
            batch_size=batch_size,
            device=device,
        )
        
        results[M] = metrics
        print(f"  Overall MSE: {metrics['overall_mse']:.4f} +/- {metrics['overall_mse_std']:.4f}")
        print(f"  Avg Task MSE: {metrics['avg_task_mse']:.4f} +/- {metrics['avg_task_mse_std']:.4f}")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "exp_6_2_context_length.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    
    # Print summary table
    print("\n" + "=" * 60)
    print("Summary: MSE vs Context Length M (T fixed at {})".format(n_tasks))
    print("=" * 60)
    print(f"{'M':>4} | {'Overall MSE':>12} | {'Avg Task MSE':>14} | {'Seq Length':>11}")
    print("-" * 60)
    for M in sorted(valid_m_values):
        seq_len = 2 * n_tasks * (M + 1)
        print(f"{M:>4} | {results[M]['overall_mse']:>12.4f} | {results[M]['avg_task_mse']:>14.4f} | {seq_len:>11}")
    print("=" * 60)
    
    return results


def _resolve_n_dims(n_dims: Optional[int], conf) -> int:
    """
    Evaluation x and task weights must use the same d as the checkpoint's embedding
    (conf.model.n_dims); otherwise Linear read_in hits matmul shape errors.
    """
    d_model = int(conf.model.n_dims)
    if n_dims is None:
        print(f"Using model's n_dims: {d_model}")
        return d_model
    if n_dims != d_model:
        print(
            f"Warning: n_dims={n_dims} != checkpoint model.n_dims={d_model}; "
            f"using {d_model} from config."
        )
        return d_model
    return n_dims


def _pretraining_n_from_conf(conf) -> float:
    """Paper's N: pretraining sample count for Γ; fallback if missing in config."""
    tr = getattr(conf, "training", None)
    if tr is None:
        return 100_000.0
    for key in ("num_examples", "n_examples", "total_examples"):
        if hasattr(tr, key):
            v = getattr(tr, key, None)
            if v is not None:
                return float(v)
    return 100_000.0


def _fixed_task_weights_independent(
    n_tasks: int,
    n_dims: int,
    batch_size: int,
    seed: int,
    w_scale: float = 1.0,
    w_task_means: Optional[List[float]] = None,
) -> List[torch.Tensor]:
    """
    Same w_t for every batch item. Per task t (0-based), each coordinate is
    Normal(mu_t, w_scale^2) with independent noise: w = mu_t * 1 + w_scale * eps, eps ~ N(0,I).

    If w_task_means is None, mu_t = 0 for all tasks (original behavior).
    """
    g = torch.Generator()
    g.manual_seed(seed)
    out: List[torch.Tensor] = []
    for t in range(n_tasks):
        noise = torch.randn(n_dims, 1, generator=g) * float(w_scale)
        if w_task_means is not None:
            mu = float(w_task_means[t])
            w = noise + mu
        else:
            w = noise
        out.append(w.unsqueeze(0).expand(batch_size, -1, -1).clone())
    return out


def experiment_6_2_context_length_fixed_w(
    model_path: str,
    n_dims: Optional[int] = None,
    n_tasks: int = 5,
    m_values: Optional[List[int]] = None,
    num_eval_batches: int = 100,
    batch_size: int = 32,
    output_dir: str = "./results",
    device: str = "cuda",
    fixed_w_seed: int = 42,
    training_n: Optional[float] = None,
    n_mc_sigma: int = 80_000,
    w_scale: float = 1.0,
    w_task_means: Optional[List[float]] = None,
):
    """
    Experiment 6.2 with fixed task weights across all batches and M values.

    Aligns with Definition 3.2 / Theorem 4.3: μ_t = Λ w_t uses the same {w_t} as evaluation.
    Default data is Gaussian with Λ = I (see `GaussianSampler` without scale).
    Theory block uses that Λ and N from config (or `training_n`).

    w_scale: noise std on each coordinate; w_j = mu_{t,j} + w_scale * eps (here mu is scalar per task, all dims).
    w_task_means: optional length-T list, task t uses mean mu_t on every coordinate (e.g. [1,2,3,4,5]).
    """
    if w_task_means is not None and len(w_task_means) != n_tasks:
        raise ValueError(
            f"w_task_means must have length n_tasks={n_tasks}, got {len(w_task_means)}"
        )

    print("=" * 60)
    print("Experiment 6.2 (fixed w): Effect of Context Length M")
    print(
        f"Fixed: T = {n_tasks}, fixed_w_seed = {fixed_w_seed}, w_scale = {w_scale}, "
        f"w_task_means = {w_task_means}"
    )
    print("=" * 60)

    model, conf = get_model_from_run(model_path)
    model = model.to(device).eval()

    n_dims = _resolve_n_dims(n_dims, conf)

    if training_n is None:
        training_n = _pretraining_n_from_conf(conf)
    print(f"Using pretraining N = {training_n} for Γ (Theorem 4.3)")

    max_positions = conf.model.n_positions
    max_sequence_tokens = max_positions * 2
    max_M = (max_sequence_tokens // (2 * n_tasks)) - 1

    if m_values is None:
        if max_M <= 5:
            m_values = list(range(1, max_M + 1))
        elif max_M <= 10:
            m_values = list(range(1, max_M + 1, 1))
        else:
            m_values = [1, 2, 3]
            m_values.extend(range(5, max_M + 1, 2))
            if max_M not in m_values:
                m_values.append(max_M)

    valid_m_values = []
    for M in sorted(set(m_values)):
        seq_length = 2 * n_tasks * (M + 1)
        if seq_length <= max_sequence_tokens:
            valid_m_values.append(M)
        else:
            print(f"Warning: Skipping M={M} (sequence length {seq_length} > {max_sequence_tokens})")

    if not valid_m_values:
        raise ValueError("No valid M values for this model.")

    fixed_task_weights = _fixed_task_weights_independent(
        n_tasks,
        n_dims,
        batch_size,
        fixed_w_seed,
        w_scale=w_scale,
        w_task_means=w_task_means,
    )
    w_numpy = tensor_weights_to_numpy_list(fixed_task_weights)
    Lambda = np.eye(n_dims)

    print(f"\nTesting M values: {valid_m_values}")
    print()

    meta_dict: Dict = {
            "experiment": "6.2_context_length_fixed_w",
            "fixed_w_seed": fixed_w_seed,
            "w_scale": w_scale,
            "n_tasks": n_tasks,
            "n_dims": n_dims,
            "batch_size": batch_size,
            "num_eval_batches": num_eval_batches,
            "pretraining_N": training_n,
            "Lambda": "I_d (GaussianSampler default; matches μ_t = w_t when Λ=I)",
            "task_weights_row0": [w.tolist() for w in w_numpy],
    }
    if w_task_means is not None:
        meta_dict["w_task_means"] = list(w_task_means)
    else:
        meta_dict["w_task_means"] = None

    results: Dict = {"meta": meta_dict}

    for M in tqdm(valid_m_values, desc="Varying M (fixed w)"):
        seq_length = 2 * n_tasks * (M + 1)
        print(f"\nEvaluating M={M} (seq tokens: {seq_length})")

        metrics = eval_multi_task_model(
            model=model,
            task_name="linear_regression",
            data_name="gaussian",
            n_dims=n_dims,
            n_tasks=n_tasks,
            n_context=M,
            num_eval_batches=num_eval_batches,
            batch_size=batch_size,
            device=device,
            fixed_task_weights=fixed_task_weights,
        )

        theory_43 = {}
        for t in range(1, n_tasks + 1):
            theory_43[f"task_{t}"] = theorem_4_3_decomposition(
                t=t,
                M=M,
                w_list=w_numpy,
                Lambda=Lambda,
                N=training_n,
                n_mc_sigma=n_mc_sigma,
            )

        results[str(M)] = {
            **metrics,
            "theorem_4_3": theory_43,
        }
        print(f"  Overall MSE: {metrics['overall_mse']:.4f} +/- {metrics['overall_mse_std']:.4f}")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "exp_6_2_context_length_fixed_w.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    return results


def experiment_6_2_context_length_same_w_task0_task2(
    model_path: str,
    n_dims: Optional[int] = None,
    n_tasks: int = 5,
    m_values: Optional[List[int]] = None,
    num_eval_batches: int = 100,
    batch_size: int = 32,
    output_dir: str = "./results",
    device: str = "cuda",
    fixed_seed: int = 42,
):
    """
    Experiment 6.2 variant: Effect of Context Length M with task0 and task2 having the same w
    
    This is a modified version where task0 and task2 use the same weight vector w,
    while other tasks have different weights.
    """
    print("=" * 60)
    print("Experiment 6.2 (Variant): Effect of Context Length M")
    print(f"Fixed: T = {n_tasks} tasks")
    print(f"Special: Task 0 and Task 2 use the same weight w")
    print("=" * 60)
    
    # Load model
    model, conf = get_model_from_run(model_path)
    model = model.to(device).eval()
    
    n_dims = _resolve_n_dims(n_dims, conf)
    
    # Check model's maximum sequence length
    max_positions = conf.model.n_positions
    max_sequence_tokens = max_positions * 2  # Model interleaves x and y
    print(f"Model's maximum sequence length: {max_sequence_tokens} tokens ({max_positions} points)")
    
    # Calculate maximum M value based on model constraints
    max_M = (max_sequence_tokens // (2 * n_tasks)) - 1
    print(f"With T={n_tasks} tasks, maximum M = {max_M}")
    
    # Generate M values if not provided
    if m_values is None:
        if max_M <= 5:
            m_values = list(range(1, max_M + 1))
        elif max_M <= 10:
            m_values = list(range(1, max_M + 1, 1))
        else:
            m_values = [1, 2, 3]
            m_values.extend(range(5, max_M + 1, 2))
            if max_M not in m_values:
                m_values.append(max_M)
    
    # Filter m_values to ensure sequence length doesn't exceed model limit
    valid_m_values = []
    for M in sorted(set(m_values)):
        seq_length = 2 * n_tasks * (M + 1)
        if seq_length <= max_sequence_tokens:
            valid_m_values.append(M)
        else:
            print(f"Warning: Skipping M={M} (sequence length {seq_length} > {max_sequence_tokens})")
    
    if not valid_m_values:
        raise ValueError(
            f"No valid M values. Model can only handle sequences up to {max_sequence_tokens} tokens. "
            f"With {n_tasks} tasks, maximum M is {max_M}"
        )
    
    print(f"\nTesting M values: {valid_m_values}")
    print(f"Sequence lengths: {[2 * n_tasks * (M + 1) for M in valid_m_values]} tokens")
    
    # Generate fixed task weights where task0, task2, task4 have the same w, and task1, task3 have the same w
    print(f"\nGenerating fixed task weights (seed={fixed_seed})...")
    print(f"Task pattern: 0,1,0,1,0 (tasks 0,2,4 share w0, tasks 1,3 share w1)")
    torch.manual_seed(fixed_seed)
    np.random.seed(fixed_seed)
    
    fixed_task_weights = []
    w0 = None  # Weight for tasks 0, 2, 4
    w1 = None  # Weight for tasks 1, 3
    
    for t in range(n_tasks):
        if t == 0:
            # Task 0: generate w0
            w0 = torch.randn(batch_size, n_dims, 1)
            fixed_task_weights.append(w0.clone())
            print(f"  Task {t}: weight norm = {w0.norm().item():.4f} (w0)")
        elif t == 1:
            # Task 1: generate w1
            w1 = torch.randn(batch_size, n_dims, 1)
            fixed_task_weights.append(w1.clone())
            print(f"  Task {t}: weight norm = {w1.norm().item():.4f} (w1)")
        elif t == 2:
            # Task 2: use w0 (same as task 0)
            w_t = w0.clone()
            fixed_task_weights.append(w_t)
            print(f"  Task {t}: using w0 (norm = {w_t.norm().item():.4f})")
        elif t == 3:
            # Task 3: use w1 (same as task 1)
            w_t = w1.clone()
            fixed_task_weights.append(w_t)
            print(f"  Task {t}: using w1 (norm = {w_t.norm().item():.4f})")
        elif t == 4:
            # Task 4: use w0 (same as task 0)
            w_t = w0.clone()
            fixed_task_weights.append(w_t)
            print(f"  Task {t}: using w0 (norm = {w_t.norm().item():.4f})")
        else:
            # For n_tasks > 5, alternate pattern
            if t % 2 == 0:
                w_t = w0.clone()
                fixed_task_weights.append(w_t)
                print(f"  Task {t}: using w0 (norm = {w_t.norm().item():.4f})")
            else:
                w_t = w1.clone()
                fixed_task_weights.append(w_t)
                print(f"  Task {t}: using w1 (norm = {w_t.norm().item():.4f})")
    
    print()
    
    results = {}
    
    for M in tqdm(valid_m_values, desc="Varying M"):
        seq_length = 2 * n_tasks * (M + 1)
        print(f"\nEvaluating M={M} (sequence length: {seq_length} tokens, {seq_length/max_sequence_tokens*100:.1f}% of max)")
        
        metrics = eval_multi_task_model(
            model=model,
            task_name="linear_regression",
            data_name="gaussian",
            n_dims=n_dims,
            n_tasks=n_tasks,
            n_context=M,
            num_eval_batches=num_eval_batches,
            batch_size=batch_size,
            device=device,
            fixed_task_weights=fixed_task_weights,  # Use fixed weights
        )
        
        results[M] = metrics
        print(f"  Overall MSE: {metrics['overall_mse']:.4f} +/- {metrics['overall_mse_std']:.4f}")
        print(f"  Avg Task MSE: {metrics['avg_task_mse']:.4f} +/- {metrics['avg_task_mse_std']:.4f}")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "exp_6_2_context_length_same_w_task0_task2.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    
    # Print summary table
    print("\n" + "=" * 60)
    print("Summary: MSE vs Context Length M (T fixed at {}, Task 0 and Task 2 have same w)".format(n_tasks))
    print("=" * 60)
    print(f"{'M':>4} | {'Overall MSE':>12} | {'Avg Task MSE':>14} | {'Seq Length':>11}")
    print("-" * 60)
    for M in sorted(valid_m_values):
        seq_len = 2 * n_tasks * (M + 1)
        print(f"{M:>4} | {results[M]['overall_mse']:>12.4f} | {results[M]['avg_task_mse']:>14.4f} | {seq_len:>11}")
    print("=" * 60)
    
    return results


def experiment_4_2_variation_1_context_length_forgetting_same_w(
    model_path: str,
    n_dims: Optional[int] = None,
    n_tasks: int = 5,
    m_values: Optional[List[int]] = None,
    num_eval_batches: int = 100,
    batch_size: int = 32,
    output_dir: str = "./results",
    device: str = "cuda",
    fixed_seed: int = 42,
):
    """
    Experiment 4.2 Variation 1: Forgetting vs Context Length M
    with same task weight pattern (tasks 1,3,5 use w0, tasks 2,4 use w1)
    
    This is a modified version where:
    - Tasks 1, 3, 5 use the same weight vector w0
    - Tasks 2, 4 use the same weight vector w1
    - Evaluates forgetting metrics for different context lengths M
    """
    print("=" * 60)
    print("Experiment 4.2 Variation 1: Forgetting vs Context Length M")
    print(f"Fixed: T = {n_tasks} tasks")
    if n_tasks == 4:
        print(f"Special: Task 1 and 4 (indices 1 and 3) share w1, Task 2 and 3 are different")
    else:
        print(f"Special: Tasks 1,3,5 use w0, Tasks 2,4 use w1")
    print("=" * 60)
    
    # Load model
    model, conf = get_model_from_run(model_path)
    model = model.to(device).eval()
    
    n_dims = _resolve_n_dims(n_dims, conf)
    
    # Check model's maximum sequence length
    max_positions = conf.model.n_positions
    max_sequence_tokens = max_positions * 2  # Model interleaves x and y
    print(f"Model's maximum sequence length: {max_sequence_tokens} tokens ({max_positions} points)")
    
    # Calculate maximum M value based on model constraints
    # Sequence length = 2 * n_tasks * (M + 1) + 2 (for forgetting query) <= max_sequence_tokens
    # Solving: 2 * n_tasks * (M + 1) + 2 <= max_sequence_tokens
    # => 2 * n_tasks * (M + 1) <= max_sequence_tokens - 2
    # => M + 1 <= (max_sequence_tokens - 2) / (2 * n_tasks)
    # => M <= (max_sequence_tokens - 2) / (2 * n_tasks) - 1
    max_M = ((max_sequence_tokens - 2) // (2 * n_tasks)) - 1
    print(f"With T={n_tasks} tasks (including forgetting query), maximum M = {max_M}")
    
    # Generate M values if not provided
    if m_values is None:
        if max_M <= 5:
            m_values = list(range(1, max_M + 1))
        elif max_M <= 10:
            m_values = list(range(1, max_M + 1, 1))
        else:
            m_values = [1, 2, 3]
            m_values.extend(range(5, max_M + 1, 2))
            if max_M not in m_values:
                m_values.append(max_M)
    
    # Filter m_values to ensure sequence length doesn't exceed model limit
    valid_m_values = []
    for M in sorted(set(m_values)):
        seq_length = 2 * n_tasks * (M + 1) + 2  # +2 for forgetting query
        if seq_length <= max_sequence_tokens:
            valid_m_values.append(M)
        else:
            print(f"Warning: Skipping M={M} (sequence length {seq_length} > {max_sequence_tokens})")
    
    if not valid_m_values:
        raise ValueError(
            f"No valid M values. Model can only handle sequences up to {max_sequence_tokens} tokens. "
            f"With {n_tasks} tasks, maximum M is {max_M}"
        )
    
    print(f"\nTesting M values: {valid_m_values}")
    print(f"Sequence lengths: {[2 * n_tasks * (M + 1) + 2 for M in valid_m_values]} tokens")
    
    # Generate fixed task weights where task 1 and 4 (indices 1 and 3) share the same w,
    # and task 2 and 3 (indices 2 and 2) are different, i.e., task 1,2,3 are all different
    print(f"\nGenerating fixed task weights (seed={fixed_seed})...")
    if n_tasks == 4:
        print(f"Task pattern: Task 1 and 4 (indices 1 and 3) share w1, Task 2 and 3 (indices 2 and 2) are different")
        print(f"  -> Task 0: w0 (unique)")
        print(f"  -> Task 1: w1 (unique)")
        print(f"  -> Task 2: w2 (unique)")
        print(f"  -> Task 3: w1 (same as task 1)")
    else:
        print(f"Task pattern: 1,3,5 use w0, 2,4 use w1")
    torch.manual_seed(fixed_seed)
    np.random.seed(fixed_seed)
    
    fixed_task_weights = []
    w0 = None  # Weight for task 0
    w1 = None  # Weight for tasks 1 and 3 (when n_tasks=4)
    w2 = None  # Weight for task 2 (when n_tasks=4)
    
    if n_tasks == 4:
        # Special pattern for 4 tasks: task 1 and 4 (indices 1 and 3) share w1, task 2 and 3 are different
        for t in range(n_tasks):
            if t == 0:
                # Task 0: generate w0
                w0 = torch.randn(batch_size, n_dims, 1)
                fixed_task_weights.append(w0.clone())
                print(f"  Task {t}: weight norm = {w0.norm().item():.4f} (w0)")
            elif t == 1:
                # Task 1: generate w1
                w1 = torch.randn(batch_size, n_dims, 1)
                fixed_task_weights.append(w1.clone())
                print(f"  Task {t}: weight norm = {w1.norm().item():.4f} (w1)")
            elif t == 2:
                # Task 2: generate w2 (different from w1)
                w2 = torch.randn(batch_size, n_dims, 1)
                fixed_task_weights.append(w2.clone())
                print(f"  Task {t}: weight norm = {w2.norm().item():.4f} (w2)")
            elif t == 3:
                # Task 3: use w1 (same as task 1)
                w_t = w1.clone()
                fixed_task_weights.append(w_t)
                print(f"  Task {t}: using w1 (norm = {w_t.norm().item():.4f}, same as task 1)")
    else:
        # Original pattern for other n_tasks
        w0_pattern = None  # Weight for tasks 1, 3, 5
        w1_pattern = None  # Weight for tasks 2, 4
        
        for t in range(n_tasks):
            if t == 0:
                # Task 0: generate a unique weight (not part of the pattern)
                w_t = torch.randn(batch_size, n_dims, 1)
                fixed_task_weights.append(w_t.clone())
                print(f"  Task {t}: weight norm = {w_t.norm().item():.4f} (unique)")
            elif t == 1:
                # Task 1: generate w0
                w0_pattern = torch.randn(batch_size, n_dims, 1)
                fixed_task_weights.append(w0_pattern.clone())
                print(f"  Task {t}: weight norm = {w0_pattern.norm().item():.4f} (w0)")
            elif t == 2:
                # Task 2: generate w1
                w1_pattern = torch.randn(batch_size, n_dims, 1)
                fixed_task_weights.append(w1_pattern.clone())
                print(f"  Task {t}: weight norm = {w1_pattern.norm().item():.4f} (w1)")
            elif t == 3:
                # Task 3: use w0 (same as task 1)
                w_t = w0_pattern.clone()
                fixed_task_weights.append(w_t)
                print(f"  Task {t}: using w0 (norm = {w_t.norm().item():.4f})")
            elif t == 4:
                # Task 4: use w1 (same as task 2)
                w_t = w1_pattern.clone()
                fixed_task_weights.append(w_t)
                print(f"  Task {t}: using w1 (norm = {w_t.norm().item():.4f})")
            elif t == 5:
                # Task 5: use w0 (same as task 1)
                w_t = w0_pattern.clone()
                fixed_task_weights.append(w_t)
                print(f"  Task {t}: using w0 (norm = {w_t.norm().item():.4f})")
            else:
                # For n_tasks > 6, alternate pattern starting from task 1
                if (t - 1) % 2 == 0:  # Tasks 1, 3, 5, 7, ... use w0
                    w_t = w0_pattern.clone()
                    fixed_task_weights.append(w_t)
                    print(f"  Task {t}: using w0 (norm = {w_t.norm().item():.4f})")
                else:  # Tasks 2, 4, 6, 8, ... use w1
                    w_t = w1_pattern.clone()
                    fixed_task_weights.append(w_t)
                    print(f"  Task {t}: using w1 (norm = {w_t.norm().item():.4f})")
    
    print()
    
    results = {}
    
    for M in tqdm(valid_m_values, desc="Varying M"):
        seq_length = 2 * n_tasks * (M + 1) + 2
        print(f"\nEvaluating M={M} (sequence length: {seq_length} tokens, {seq_length/max_sequence_tokens*100:.1f}% of max)")
        
        metrics = eval_multi_task_model(
            model=model,
            task_name="linear_regression",
            data_name="gaussian",
            n_dims=n_dims,
            n_tasks=n_tasks,
            n_context=M,
            num_eval_batches=num_eval_batches,
            batch_size=batch_size,
            device=device,
            fixed_task_weights=fixed_task_weights,  # Use fixed weights
            eval_forgetting=True,  # Evaluate forgetting
        )
        
        results[M] = metrics
        if "forgetting_overall_mse" in metrics:
            print(f"  Forgetting Overall MSE: {metrics['forgetting_overall_mse']:.4f} +/- {metrics['forgetting_overall_mse_std']:.4f}")
        else:
            print(f"  Warning: No forgetting metrics found")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "exp_4_2_variation_1_context_length_forgetting_same_w.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    
    # Print summary table
    print("\n" + "=" * 60)
    if n_tasks == 4:
        print("Summary: Forgetting vs Context Length M (T fixed at {}, Task 1 and 4 share w1, Task 2 and 3 are different)".format(n_tasks))
    else:
        print("Summary: Forgetting vs Context Length M (T fixed at {}, Tasks 1,3,5 share w0, Tasks 2,4 share w1)".format(n_tasks))
    print("=" * 60)
    print(f"{'M':>4} | {'Forgetting MSE':>14} | {'Forgetting Std':>14} | {'Seq Length':>11}")
    print("-" * 60)
    for M in sorted(valid_m_values):
        seq_len = 2 * n_tasks * (M + 1) + 2
        if "forgetting_overall_mse" in results[M]:
            print(f"{M:>4} | {results[M]['forgetting_overall_mse']:>14.4f} | {results[M]['forgetting_overall_mse_std']:>14.4f} | {seq_len:>11}")
        else:
            print(f"{M:>4} | {'N/A':>14} | {'N/A':>14} | {seq_len:>11}")
    print("=" * 60)
    
    return results


def experiment_6_3_task_similarity(
    model_path: str,
    n_dims: Optional[int] = None,
    n_tasks: int = 5,
    n_context: Optional[int] = None,
    theta_values: List[float] = [0, np.pi/12, np.pi/6, np.pi/4, np.pi/3, np.pi/2],
    num_eval_batches: int = 100,
    batch_size: int = 32,
    output_dir: str = "./results",
    device: str = "cuda",
    fixed_seed: int = 42,
):
    """
    Experiment 6.3: Task similarity
    
    Quantifies how task similarity (theta) affects bias and variance.
    Each task's weight w is kept fixed across different theta values.
    
    Args:
        n_tasks: Number of tasks T (default: 5)
        n_context: Number of in-context examples M per task. If None, M = n_tasks (adaptive)
        fixed_seed: Random seed for generating fixed task weights
    """
    print("=" * 60)
    print("Experiment 6.3: Task similarity")
    print("=" * 60)
    
    # Load model
    model, conf = get_model_from_run(model_path)
    model = model.to(device).eval()
    
    n_dims = _resolve_n_dims(n_dims, conf)
    
    # Adaptive context length: if M is not specified, use M = number of tasks.
    if n_context is None:
        n_context = n_tasks
        print(f"Adaptive M: n_context = n_tasks = {n_context}")
    else:
        print(f"Using fixed n_context = {n_context}")
    
    print(f"T = {n_tasks}, M = {n_context}")
    
    # Check model's maximum sequence length
    max_positions = conf.model.n_positions
    max_sequence_tokens = max_positions * 2  # Model interleaves x and y
    print(f"Model's maximum sequence length: {max_sequence_tokens} tokens ({max_positions} points)")
    
    # Check if sequence length is within model limits
    seq_length = 2 * n_tasks * (n_context + 1)
    print(f"Sequence length: {seq_length} tokens ({seq_length/max_sequence_tokens*100:.1f}% of max)")
    
    if seq_length > max_sequence_tokens:
        max_M = (max_sequence_tokens // (2 * n_tasks)) - 1
        raise ValueError(
            f"Sequence length {seq_length} exceeds model's maximum {max_sequence_tokens}. "
            f"With T={n_tasks} tasks, maximum M is {max_M}. "
            f"Requested M={n_context} is too large."
        )
    
    # Generate one fixed base weight so theta sweeps are comparable.
    # Different theta values still induce different rotated task weights.
    # Each batch uses fresh random directions and data to preserve variance.
    print(f"\nGenerating fixed base weights (seed={fixed_seed})...")
    print("Each theta value is generated from the same base_w.")
    print("Each batch uses fresh random directions and data.\n")
    
    # Fixed base_w shared by all theta values.
    torch.manual_seed(fixed_seed)
    np.random.seed(fixed_seed)
    fixed_base_w = torch.randn(batch_size, n_dims, 1)
    # Reset RNG state so later random directions and data keep stochasticity.
    # torch.manual_seed does not accept None, so draw a fresh integer seed.
    import random
    random_seed = random.randint(0, 2**31 - 1)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    results = {}
    
    for theta in tqdm(theta_values, desc="Varying theta"):
        print(f"\nEvaluating with theta={theta:.4f} (cos={np.cos(theta):.4f})")
        
        # Use task-similarity generation while sharing the fixed base_w.
        # Each batch still draws fresh random directions.
        metrics = eval_multi_task_model(
            model=model,
            task_name="linear_regression",
            data_name="gaussian",
            n_dims=n_dims,
            n_tasks=n_tasks,
            n_context=n_context,
            num_eval_batches=num_eval_batches,
            batch_size=batch_size,
            task_similarity=theta,
            device=device,
            fixed_base_w=fixed_base_w,
        )
        
        results[theta] = metrics
        print(f"  Overall MSE: {metrics['overall_mse']:.4f} +/- {metrics['overall_mse_std']:.4f}")
        print(f"  Avg Task MSE: {metrics['avg_task_mse']:.4f} +/- {metrics['avg_task_mse_std']:.4f}")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "exp_6_3_task_similarity.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    
    return results


def experiment_6_4_task_order(
    model_path: str,
    n_dims: Optional[int] = None,
    n_tasks: int = 5,
    n_context: Optional[int] = None,
    num_eval_batches: int = 100,
    batch_size: int = 32,
    output_dir: str = "./results",
    device: str = "cuda",
):
    """
    Experiment 6.4: Task order t
    
    Tests the role of causal masking vs unmasked attention in respecting task order.
    Analyzes whether order affects results (e.g., later tasks easier to learn).
    
    Args:
        n_tasks: Number of tasks T (default: 10)
        n_context: Number of in-context examples M per task. If None, M = n_tasks (adaptive)
    """
    print("=" * 60)
    print("Experiment 6.4: Task order t")
    print("=" * 60)
    
    # Load model
    model, conf = get_model_from_run(model_path)
    model = model.to(device).eval()
    
    n_dims = _resolve_n_dims(n_dims, conf)
    
    # Adaptive context length: if M is not specified, use M = number of tasks.
    if n_context is None:
        n_context = n_tasks
        print(f"Adaptive M: n_context = n_tasks = {n_context}")
    else:
        print(f"Using fixed n_context = {n_context}")
    
    print(f"T = {n_tasks}, M = {n_context}")
    
    # Check model's maximum sequence length
    max_positions = conf.model.n_positions
    max_sequence_tokens = max_positions * 2  # Model interleaves x and y
    print(f"Model's maximum sequence length: {max_sequence_tokens} tokens ({max_positions} points)")
    
    # Check if sequence length is within model limits
    seq_length = 2 * n_tasks * (n_context + 1)
    print(f"Sequence length: {seq_length} tokens ({seq_length/max_sequence_tokens*100:.1f}% of max)")
    
    if seq_length > max_sequence_tokens:
        raise ValueError(
            f"Sequence length {seq_length} exceeds model's maximum {max_sequence_tokens}. "
            f"Reduce n_tasks ({n_tasks}) or n_context ({n_context}). "
            f"Maximum T with M={n_context}: {max_sequence_tokens // (2 * (n_context + 1))}"
        )
    
    print("\nEvaluating task order effects...")
    
    metrics = eval_multi_task_model(
        model=model,
        task_name="linear_regression",
        data_name="gaussian",
        n_dims=n_dims,
        n_tasks=n_tasks,
        n_context=n_context,
        num_eval_batches=num_eval_batches,
        batch_size=batch_size,
        device=device,
    )
    
    results = {
        "per_task_mse": {f"task_{t}": metrics[f"task_{t}_mse"] for t in range(n_tasks)},
        "per_task_mse_std": {f"task_{t}": metrics[f"task_{t}_mse_std"] for t in range(n_tasks)},
        "overall_mse": metrics["overall_mse"],
        "overall_mse_std": metrics["overall_mse_std"],
        "avg_task_mse": metrics["avg_task_mse"],
        "avg_task_mse_std": metrics["avg_task_mse_std"],
    }
    
    # Print summary table
    print("\n" + "=" * 60)
    print("Summary: Per-Task MSE (Task Order Analysis)")
    print("=" * 60)
    print(f"{'Task':>6} | {'MSE':>12} | {'Std':>12}")
    print("-" * 60)
    for t in range(n_tasks):
        print(f"{t:>6} | {results['per_task_mse'][f'task_{t}']:>12.4f} | {results['per_task_mse_std'][f'task_{t}']:>12.4f}")
    print("-" * 60)
    print(f"{'Overall':>6} | {results['overall_mse']:>12.4f} | {results['overall_mse_std']:>12.4f}")
    print(f"{'Avg Task':>6} | {results['avg_task_mse']:>12.4f} | {results['avg_task_mse_std']:>12.4f}")
    print("=" * 60)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "exp_6_4_task_order.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    
    return results


def experiment_6_5_number_of_tasks(
    model_path: str,
    n_dims: Optional[int] = None,
    n_context: Optional[int] = None,
    t_values: Optional[List[int]] = None,
    num_eval_batches: int = 100,
    batch_size: int = 32,
    output_dir: str = "./results",
    device: str = "cuda",
):
    """
    Experiment 6.5: Number of tasks T
    
    Validates the dependence of interference terms on T.
    
    Args:
        n_context: Number of in-context examples M per task. If None, M = T (adaptive for each T)
        t_values: List of T values to test. If None, auto-generate based on model constraints
    """
    print("=" * 60)
    print("Experiment 6.5: Number of tasks T")
    print("=" * 60)
    
    # Load model
    model, conf = get_model_from_run(model_path)
    model = model.to(device).eval()
    
    n_dims = _resolve_n_dims(n_dims, conf)
    
    # Check model's maximum sequence length
    max_positions = conf.model.n_positions
    max_sequence_tokens = max_positions * 2  # Model interleaves x and y
    print(f"Model's maximum sequence length: {max_sequence_tokens} tokens ({max_positions} points)")
    
    # Generate t_values if not provided
    if t_values is None:
        # Generate a reasonable range of T values
        # Start with small values and increase
        if n_context is None:
            # Adaptive M: M = T, so sequence length = 2 * T * (T + 1) = 2T^2 + 2T
            # Solve: 2T^2 + 2T <= max_sequence_tokens
            # T^2 + T <= max_sequence_tokens / 2
            max_T = int(np.sqrt(max_sequence_tokens / 2))
            # Generate values: 1, 2, 3, then every few values up to max
            t_values = [1, 2, 3]
            if max_T > 3:
                t_values.extend(range(5, max_T + 1, 2))
                if max_T not in t_values:
                    t_values.append(max_T)
        else:
            # Fixed M: sequence length = 2 * T * (M + 1)
            max_T = max_sequence_tokens // (2 * (n_context + 1))
            # Generate values: 1, 2, 3, then every few values up to max
            t_values = [1, 2, 3]
            if max_T > 3:
                t_values.extend(range(5, max_T + 1, 2))
                if max_T not in t_values:
                    t_values.append(max_T)
    
    # Filter t_values to ensure sequence length doesn't exceed model limit
    valid_t_values = []
    for T in sorted(set(t_values)):  # Remove duplicates and sort
        if n_context is None:
            # Adaptive M: M = T
            M = T
        else:
            M = n_context
        
        seq_length = 2 * T * (M + 1)
        if seq_length <= max_sequence_tokens:
            valid_t_values.append(T)
        else:
            print(f"Warning: Skipping T={T} (sequence length {seq_length} > {max_sequence_tokens})")
    
    if not valid_t_values:
        raise ValueError(
            f"No valid T values. Model can only handle sequences up to {max_sequence_tokens} tokens."
        )
    
    print(f"\nTesting T values: {valid_t_values}")
    if n_context is None:
        print("Adaptive M: M = T for each T")
    else:
        print(f"Fixed M = {n_context}")
    print()
    
    results = {}
    
    for T in tqdm(valid_t_values, desc="Varying T"):
        # Determine M for this T
        if n_context is None:
            M = T  # Adaptive: M = T
        else:
            M = n_context
        
        seq_length = 2 * T * (M + 1)
        print(f"\nEvaluating T={T}, M={M} (sequence length: {seq_length} tokens, {seq_length/max_sequence_tokens*100:.1f}% of max)")
        
        metrics = eval_multi_task_model(
            model=model,
            task_name="linear_regression",
            data_name="gaussian",
            n_dims=n_dims,
            n_tasks=T,
            n_context=M,
            num_eval_batches=num_eval_batches,
            batch_size=batch_size,
            device=device,
        )
        
        results[T] = metrics
        print(f"  Overall MSE: {metrics['overall_mse']:.4f} +/- {metrics['overall_mse_std']:.4f}")
        print(f"  Avg Task MSE: {metrics['avg_task_mse']:.4f} +/- {metrics['avg_task_mse_std']:.4f}")
        
        # Show first and last task performance
        if T > 1:
            print(f"  First task MSE: {metrics['task_0_mse']:.4f}")
            print(f"  Last task MSE: {metrics[f'task_{T-1}_mse']:.4f}")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "exp_6_5_number_of_tasks.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    
    # Print summary table
    print("\n" + "=" * 60)
    print("Summary: MSE vs Number of Tasks T")
    print("=" * 60)
    if n_context is None:
        print(f"{'T':>4} | {'M':>4} | {'Overall MSE':>12} | {'Avg Task MSE':>14} | {'Seq Length':>11}")
        print("-" * 60)
        for T in sorted(valid_t_values):
            M = T
            seq_len = 2 * T * (M + 1)
            print(f"{T:>4} | {M:>4} | {results[T]['overall_mse']:>12.4f} | {results[T]['avg_task_mse']:>14.4f} | {seq_len:>11}")
    else:
        print(f"{'T':>4} | {'Overall MSE':>12} | {'Avg Task MSE':>14} | {'Seq Length':>11}")
        print("-" * 60)
        for T in sorted(valid_t_values):
            seq_len = 2 * T * (n_context + 1)
            print(f"{T:>4} | {results[T]['overall_mse']:>12.4f} | {results[T]['avg_task_mse']:>14.4f} | {seq_len:>11}")
    print("=" * 60)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run multi-task continual learning experiments")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model directory")
    parser.add_argument("--experiment", type=str, required=True,
                       choices=["6.1", "6.2", "6.2fw", "6.3", "6.4", "6.5", "all"],
                       help="Which experiment to run (6.2fw = 6.2 with fixed w, aligned to Thm 4.3)")
    parser.add_argument("--fixed_w_seed", type=int, default=42,
                       help="Only for 6.2fw: RNG seed for task weights w_1..w_T")
    parser.add_argument(
        "--w_scale",
        type=float,
        default=1.0,
        help="Only for 6.2fw: noise std per dim: w = mu_t + w_scale * N(0,1) (default 1)",
    )
    parser.add_argument(
        "--w_task_means",
        type=str,
        default=None,
        help='Only for 6.2fw: comma-separated mean per task on each coordinate, e.g. "1,2,3,4,5"',
    )
    parser.add_argument("--output_dir", type=str, default="./results",
                       help="Output directory for results")
    parser.add_argument(
        "--n_dims",
        type=int,
        default=None,
        help="Input dimension; default: model's n_dims from config (must match checkpoint)",
    )
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda or cpu)")
    parser.add_argument("--num_eval_batches", type=int, default=100,
                       help="Number of evaluation batches")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for evaluation")
    
    args = parser.parse_args()
    
    if args.experiment == "6.1" or args.experiment == "all":
        experiment_6_1_training_sample_size(
            model_path=args.model_path,
            n_dims=args.n_dims,
            output_dir=args.output_dir,
            device=args.device,
            num_eval_batches=args.num_eval_batches,
            batch_size=args.batch_size,
        )
    
    if args.experiment == "6.2" or args.experiment == "all":
        experiment_6_2_context_length(
            model_path=args.model_path,
            n_dims=args.n_dims,
            output_dir=args.output_dir,
            device=args.device,
            num_eval_batches=args.num_eval_batches,
            batch_size=args.batch_size,
        )

    if args.experiment == "6.2fw":
        w_task_means = None
        if args.w_task_means is not None:
            w_task_means = [float(x.strip()) for x in args.w_task_means.split(",")]
        experiment_6_2_context_length_fixed_w(
            model_path=args.model_path,
            n_dims=args.n_dims,
            output_dir=args.output_dir,
            device=args.device,
            num_eval_batches=args.num_eval_batches,
            batch_size=args.batch_size,
            fixed_w_seed=args.fixed_w_seed,
            w_scale=args.w_scale,
            w_task_means=w_task_means,
        )
    
    if args.experiment == "6.3" or args.experiment == "all":
        experiment_6_3_task_similarity(
            model_path=args.model_path,
            n_dims=args.n_dims,
            output_dir=args.output_dir,
            device=args.device,
            num_eval_batches=args.num_eval_batches,
            batch_size=args.batch_size,
        )
    
    if args.experiment == "6.4" or args.experiment == "all":
        experiment_6_4_task_order(
            model_path=args.model_path,
            n_dims=args.n_dims,
            output_dir=args.output_dir,
            device=args.device,
            num_eval_batches=args.num_eval_batches,
            batch_size=args.batch_size,
        )
    
    if args.experiment == "6.5" or args.experiment == "all":
        experiment_6_5_number_of_tasks(
            model_path=args.model_path,
            n_dims=args.n_dims,
            output_dir=args.output_dir,
            device=args.device,
            num_eval_batches=args.num_eval_batches,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
