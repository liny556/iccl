"""
Multi-task continual learning evaluation module.
Implements evaluation for in-context continual learning with sequential tasks.
"""
import json
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from tqdm import tqdm

from samplers import get_data_sampler
from tasks import get_task_sampler, LinearRegression


def generate_multi_task_sequence(
    task_sampler,
    data_sampler,
    n_tasks: int,
    n_context: int,
    n_dims: int,
    batch_size: int = 1,
    task_similarity: Optional[float] = None,
    seed: Optional[int] = None,
    fixed_task_weights: Optional[List[torch.Tensor]] = None,
    fixed_base_w: Optional[torch.Tensor] = None,
    add_forgetting_query: bool = False,
    forgetting_query_task_ids: Optional[torch.Tensor] = None,
    task_scale: float = 1.0,  # Scale parameter for task evaluation (default=1)
) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], torch.Tensor]:
    """
    Generate a sequence of T tasks, each with M in-context examples and a query.
    Optionally adds an additional query at the end for forgetting evaluation.
    
    Args:
        task_sampler: Task sampler function
        data_sampler: Data sampler
        n_tasks: Number of tasks T
        n_context: Number of in-context examples M per task
        n_dims: Input dimension
        batch_size: Batch size
        task_similarity: Task similarity parameter (theta in paper). 
                        If None, tasks are independent.
        seed: Random seed for reproducibility
        fixed_task_weights: Fixed task weights to use [T, batch_size, n_dims, 1]
                           If provided, these weights will be used instead of generating new ones.
        fixed_base_w: Fixed base weight vector [batch_size, n_dims, 1]
                     If provided and task_similarity is not None, use this base_w instead of generating a new one.
                     This allows fair comparison across different theta values while still generating random_vec per batch.
        add_forgetting_query: If True, adds an additional query at the end for forgetting evaluation.
                             This query can belong to any previous task (identified by task_indices).
        forgetting_query_task_ids: Task IDs for the forgetting query [batch_size].
                                  If None and add_forgetting_query=True, randomly samples from [0, n_tasks-1].
        task_scale: Scale parameter for task evaluation (default=1.0). 
                   When task_similarity is used, this ensures consistency with task.evaluate() which applies scale.
        
    Returns:
        xs_all: All input sequences [batch_size, T*(M+1) + (1 if add_forgetting_query else 0), n_dims]
        ys_all: All label sequences [batch_size, T*(M+1) + (1 if add_forgetting_query else 0)]
        task_weights: List of task weight vectors [T, batch_size, n_dims, 1]
        task_indices: Task indices for each point [batch_size, T*(M+1) + (1 if add_forgetting_query else 0)]
                     where each value indicates which task the point belongs to
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    # Note: We do NOT fix seed when task_similarity is set, to allow variance across batches
    # Each batch will generate different base_w and data, providing real variance
    # If you need consistent base_w across different theta values for fair comparison,
    # generate it at the experiment level and pass as fixed_task_weights
    
    xs_list = []
    ys_list = []
    task_weights = []
    task_indices_list = []
    
    # Use fixed task weights if provided
    if fixed_task_weights is not None:
        for t in range(n_tasks):
            w_t = fixed_task_weights[t].clone()
            task_weights.append(w_t)
            
            # Generate M in-context examples + 1 query for this task
            xs_t = data_sampler.sample_xs(n_context + 1, batch_size, n_dims_truncated=n_dims)
            # Apply scale to match task.evaluate() behavior
            ys_t = task_scale * (xs_t @ w_t.to(xs_t.device))[:, :, 0]
            
            xs_list.append(xs_t)
            ys_list.append(ys_t)
            # Create task index tensor: all points in this task have index t
            task_idx_t = torch.full((batch_size, n_context + 1), t, dtype=torch.long)
            task_indices_list.append(task_idx_t)
    else:
        # Original logic: generate task weights dynamically
        # Generate base task weight for similarity
        if task_similarity is not None:
            # Create tasks with controlled similarity
            # For similarity, we rotate a base weight vector
            # IMPORTANT: Do NOT normalize base_w to match LinearRegression behavior
            # LinearRegression uses torch.randn without normalization, so w has expected norm sqrt(n_dims)
            if fixed_base_w is not None:
                # Use provided fixed base_w (for fair comparison across theta values)
                base_w = fixed_base_w.clone()
            else:
                # Generate new base_w for this batch
                base_w = torch.randn(batch_size, n_dims, 1)
            # Do NOT normalize: base_w = base_w / (base_w.norm(dim=1, keepdim=True) + 1e-8)
        else:
            base_w = None
        
        # Special handling for theta=90 degrees: generate orthogonal basis
        # This ensures all task pairs are orthogonal when theta=90 degrees
        if task_similarity is not None and abs(task_similarity - np.pi/2) < 1e-6:
            # Generate orthogonal basis using Gram-Schmidt process
            # This ensures all task pairs are orthogonal
            ortho_basis = []
            for t in range(n_tasks):
                if t == 0:
                    # First task uses base_w (normalized for orthogonal basis)
                    w_t = base_w.clone()
                    w_t = w_t / (w_t.norm(dim=1, keepdim=True) + 1e-8)
                else:
                    # Generate random vector and orthogonalize against all previous vectors
                    random_vec = torch.randn(batch_size, n_dims, 1)
                    random_vec = random_vec / (random_vec.norm(dim=1, keepdim=True) + 1e-8)
                    
                    # Orthogonalize against all previous vectors in the basis
                    for prev_vec in ortho_basis:
                        # Project random_vec onto prev_vec and subtract
                        proj = (random_vec * prev_vec).sum(dim=1, keepdim=True) * prev_vec
                        random_vec = random_vec - proj
                    
                    # Normalize
                    w_t = random_vec / (random_vec.norm(dim=1, keepdim=True) + 1e-8)
                
                ortho_basis.append(w_t)
                # Scale to preserve expected norm sqrt(n_dims) (matching LinearRegression behavior)
                w_t_scaled = w_t * np.sqrt(n_dims)
                task_weights.append(w_t_scaled)
                
                # Generate M in-context examples + 1 query for this task
                xs_t = data_sampler.sample_xs(n_context + 1, batch_size, n_dims_truncated=n_dims)
                # Directly compute y = scale * w^T x (apply scale to match task.evaluate())
                ys_t = task_scale * (xs_t @ w_t_scaled.to(xs_t.device))[:, :, 0]
                
                xs_list.append(xs_t)
                ys_list.append(ys_t)
                # Create task index tensor: all points in this task have index t
                task_idx_t = torch.full((batch_size, n_context + 1), t, dtype=torch.long)
                task_indices_list.append(task_idx_t)
        else:
            # Original logic for other theta values
            for t in range(n_tasks):
                # Generate task weight
                if task_similarity is not None:
                    # Use similarity-based weight generation for ALL tasks (including t=0)
                    if t == 0:
                        # First task uses base_w directly
                        w_t = base_w.clone()
                    elif task_similarity == 0.0:
                        # When theta=0 (completely similar), all tasks use the same base_w
                        # This avoids numerical errors from computing ortho
                        w_t = base_w.clone()
                    else:
                        # Rotate base weight by similarity angle
                        # theta controls the angle between consecutive tasks
                        cos_theta = np.cos(task_similarity)
                        sin_theta = np.sin(task_similarity)
                        # Simple 2D rotation in the plane spanned by base_w and a random vector
                        random_vec = torch.randn(batch_size, n_dims, 1)
                        random_vec = random_vec / (random_vec.norm(dim=1, keepdim=True) + 1e-8)
                        # Orthogonalize
                        proj = (base_w * random_vec).sum(dim=1, keepdim=True) * base_w
                        ortho = random_vec - proj
                        ortho = ortho / (ortho.norm(dim=1, keepdim=True) + 1e-8)
                        w_t = cos_theta * base_w + sin_theta * ortho
                        # Do NOT normalize to preserve expected norm sqrt(n_dims)
                        # w_t = w_t / (w_t.norm(dim=1, keepdim=True) + 1e-8)
                
                    task_weights.append(w_t)
                else:
                    # Independent task - sample a new task
                    task = task_sampler()
                    if hasattr(task, 'w_b'):
                        w_t = task.w_b.clone()
                    else:
                        # Fallback: generate random weight
                        w_t = torch.randn(batch_size, n_dims, 1)
                    task_weights.append(w_t)
                
                # Generate M in-context examples + 1 query for this task
                xs_t = data_sampler.sample_xs(n_context + 1, batch_size, n_dims_truncated=n_dims)
                # When using task_similarity, compute y directly with w_t
                # (task object is only created in the else branch for independent tasks)
                if task_similarity is not None:
                    # Directly compute y = scale * w^T x (apply scale to match task.evaluate())
                    # This ensures consistency with training when scale != 1
                    ys_t = task_scale * (xs_t @ w_t.to(xs_t.device))[:, :, 0]
                else:
                    # Use task.evaluate() to ensure consistency with training (handles scale parameter)
                    # This is important because task.evaluate() applies the scale factor
                    if hasattr(task, 'evaluate'):
                        ys_t = task.evaluate(xs_t)
                    else:
                        # Fallback: directly compute y = w^T x
                        ys_t = (xs_t @ w_t.to(xs_t.device))[:, :, 0]
                
                xs_list.append(xs_t)
                ys_list.append(ys_t)
                # Create task index tensor: all points in this task have index t
                task_idx_t = torch.full((batch_size, n_context + 1), t, dtype=torch.long)
                task_indices_list.append(task_idx_t)
    
    # Concatenate all tasks
    xs_all = torch.cat(xs_list, dim=1)  # [batch_size, T*(M+1), n_dims]
    ys_all = torch.cat(ys_list, dim=1)  # [batch_size, T*(M+1)]
    task_indices = torch.cat(task_indices_list, dim=1)  # [batch_size, T*(M+1)]
    
    # Add forgetting query if requested
    if add_forgetting_query:
        # Determine which task the forgetting query belongs to
        if forgetting_query_task_ids is None:
            # Randomly sample task IDs from [0, n_tasks-1] for each sample in batch
            forgetting_query_task_ids = torch.randint(0, n_tasks, (batch_size,), dtype=torch.long)
        else:
            # Use provided task IDs
            forgetting_query_task_ids = forgetting_query_task_ids.clone()
        
        # Generate the forgetting query for each sample
        # For each sample, use the task weight corresponding to its task ID
        forgetting_xs_list = []
        forgetting_ys_list = []
        forgetting_task_indices_list = []
        
        # Generate forgetting queries for all samples at once
        # Sample task IDs for each sample in batch
        forgetting_xs = data_sampler.sample_xs(1, batch_size, n_dims_truncated=n_dims)  # [batch_size, 1, n_dims]
        
        # Compute ys for each sample using the corresponding task weight
        forgetting_ys_list = []
        for b in range(batch_size):
            task_id = forgetting_query_task_ids[b].item()
            w_task = task_weights[task_id][b:b+1]  # [1, n_dims, 1]
            xs_forget_b = forgetting_xs[b:b+1]  # [1, 1, n_dims]
            
            # Compute y using the task weight (consistent with how we generated regular queries)
            # Apply scale to match task.evaluate() behavior
            ys_forget_b = task_scale * (xs_forget_b @ w_task.to(xs_forget_b.device))[:, :, 0]  # [1, 1]
            forgetting_ys_list.append(ys_forget_b)
        
        # Stack forgetting ys: [batch_size, 1]
        forgetting_ys = torch.cat(forgetting_ys_list, dim=0)  # [batch_size, 1]
        
        # Create task indices for forgetting query: [batch_size, 1]
        forgetting_task_indices = forgetting_query_task_ids.unsqueeze(1)  # [batch_size, 1]
        
        # Append to sequences
        xs_all = torch.cat([xs_all, forgetting_xs], dim=1)  # [batch_size, T*(M+1)+1, n_dims]
        ys_all = torch.cat([ys_all, forgetting_ys], dim=1)  # [batch_size, T*(M+1)+1]
        task_indices = torch.cat([task_indices, forgetting_task_indices], dim=1)  # [batch_size, T*(M+1)+1]
    
    return xs_all, ys_all, task_weights, task_indices


def eval_multi_task_batch(
    model,
    xs_all: torch.Tensor,
    ys_all: torch.Tensor,
    n_tasks: int,
    n_context: int,
    task_metric,
    device: str = "cuda",
    debug: bool = False,
    task_indices: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    Evaluate model on a multi-task sequence.
    
    Args:
        model: Model to evaluate
        xs_all: All inputs [batch_size, T*(M+1), n_dims]
        ys_all: All labels [batch_size, T*(M+1)]
        n_tasks: Number of tasks T
        n_context: Number of in-context examples M per task
        task_metric: Metric function
        device: Device to run on
        debug: Whether to print debug information
        task_indices: Task indices for each point [batch_size, T*(M+1)], optional
        
    Returns:
        Dictionary with per-task metrics
    """
    if hasattr(model, 'eval'):
        model.eval()
    
    # Check sequence length: model interleaves x and y, so actual length is 2 * num_points
    num_points = xs_all.shape[1]
    
    # Check if model has n_positions attribute (for transformer models)
    if hasattr(model, 'n_positions'):
        max_sequence_length = model.n_positions * 2  # Model config uses 2 * n_positions
        if num_points * 2 > max_sequence_length:
            raise ValueError(
                f"Sequence length {num_points * 2} exceeds model's maximum {max_sequence_length}. "
                f"Reduce n_tasks ({n_tasks}) or n_context ({n_context}). "
                f"Current: {n_tasks} tasks × {n_context + 1} points = {num_points} points → {num_points * 2} tokens"
            )
    
    xs_all = xs_all.to(device)
    ys_all = ys_all.to(device)
    
    # Build prompt with zeros for query labels (to be predicted)
    # For each task: [x1, y1, x2, y2, ..., xM, yM, x_query, 0]
    ys_prompt = ys_all.clone()
    # Set query labels to 0 (they will be predicted)
    query_indices = [t * (n_context + 1) + n_context for t in range(n_tasks)]
    for idx in query_indices:
        ys_prompt[:, idx] = 0.0
    
    with torch.no_grad():
        # Forward pass - model expects xs and ys, and returns predictions for all positions
        # We need to specify which indices to predict (query indices)
        pred = model(xs_all, ys_prompt, inds=query_indices)
    
    # Extract predictions for query points only
    query_preds = []
    query_targets = []
    query_task_ids = []  # Track which task each query belongs to
    task_metrics = {}
    
    for t in range(n_tasks):
        query_idx = query_indices[t]
        # pred shape is [batch_size, len(inds)]
        query_pred = pred[:, t]
        query_target = ys_all[:, query_idx]
        
        query_preds.append(query_pred)
        query_targets.append(query_target)
        
        # Get task ID for this query (if task_indices provided)
        if task_indices is not None:
            query_task_id = task_indices[:, query_idx]  # [batch_size]
            query_task_ids.append(query_task_id)
        else:
            # Default: assume query at index t belongs to task t
            query_task_id = torch.full((xs_all.shape[0],), t, dtype=torch.long, device=device)
            query_task_ids.append(query_task_id)
        
        # Compute metric for this task
        task_pred = query_pred.unsqueeze(1).to(device)
        task_target = query_target.unsqueeze(1).to(device)
        metric_val = task_metric(task_pred, task_target)
        task_metrics[f"task_{t}"] = metric_val
        
        # Debug: print statistics for each task
        if debug:
            query_pred_cpu = query_pred.cpu()
            query_target_cpu = query_target.cpu()
            print(f"\n=== Task {t} ===")
            print(f"  Query index: {query_idx}")
            print(f"  Prediction stats: mean={query_pred_cpu.mean().item():.4f}, std={query_pred_cpu.std().item():.4f}, min={query_pred_cpu.min().item():.4f}, max={query_pred_cpu.max().item():.4f}")
            print(f"  Target stats: mean={query_target_cpu.mean().item():.4f}, std={query_target_cpu.std().item():.4f}, min={query_target_cpu.min().item():.4f}, max={query_target_cpu.max().item():.4f}")
            print(f"  Error (target - pred): mean={(query_target_cpu - query_pred_cpu).mean().item():.4f}, std={(query_target_cpu - query_pred_cpu).std().item():.4f}")
            print(f"  MSE (this task): {metric_val.mean().item():.4f}")
            print(f"  First 3 samples:")
            for i in range(min(3, len(query_pred_cpu))):
                print(f"    Sample {i}: pred={query_pred_cpu[i].item():.4f}, target={query_target_cpu[i].item():.4f}, error={query_target_cpu[i].item() - query_pred_cpu[i].item():.4f}, sq_error={metric_val[i, 0].item():.4f}")
    
    # Overall metrics
    all_preds = torch.stack(query_preds, dim=1).to(device)  # [batch_size, T]
    all_targets = torch.stack(query_targets, dim=1).to(device)  # [batch_size, T]
    
    task_metrics["overall"] = task_metric(all_preds, all_targets)
    task_metrics["per_task_mse"] = task_metric(all_preds, all_targets).mean(dim=0)
    
    # Store query task IDs for forgetting analysis
    if task_indices is not None:
        all_query_task_ids = torch.stack(query_task_ids, dim=1)  # [batch_size, T]
        task_metrics["query_task_ids"] = all_query_task_ids
    
    # Debug: print overall statistics
    if debug:
        all_preds_cpu = all_preds.cpu()
        all_targets_cpu = all_targets.cpu()
        print(f"\n=== Overall (all tasks) ===")
        print(f"  All predictions shape: {all_preds_cpu.shape}")
        print(f"  All targets shape: {all_targets_cpu.shape}")
        print(f"  Overall MSE: {task_metrics['overall'].mean().item():.4f}")
        print(f"  Overall prediction stats: mean={all_preds_cpu.mean().item():.4f}, std={all_preds_cpu.std().item():.4f}")
        print(f"  Overall target stats: mean={all_targets_cpu.mean().item():.4f}, std={all_targets_cpu.std().item():.4f}")
        if task_indices is not None:
            print(f"  Query task IDs: {all_query_task_ids[0].cpu().tolist()}")
    
    return task_metrics


def eval_forgetting_batch(
    model,
    xs_all: torch.Tensor,
    ys_all: torch.Tensor,
    task_indices: torch.Tensor,
    n_tasks: int,
    n_context: int,
    task_metric,
    device: str = "cuda",
    debug: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Evaluate forgetting on a multi-task sequence using the original definition:
    Forgetting = E[(f_after(x) - f_before(x))^2]
    
    This measures the change in predictions on the same query point before and after
    processing subsequent tasks.
    
    Args:
        model: Model to evaluate
        xs_all: All inputs [batch_size, T*(M+1)+1, n_dims] (includes forgetting query)
        ys_all: All labels [batch_size, T*(M+1)+1] (includes forgetting query)
        task_indices: Task indices for each point [batch_size, T*(M+1)+1]
                      Each value indicates which task the point belongs to
        n_tasks: Number of tasks T
        n_context: Number of in-context examples M per task
        task_metric: Metric function
        device: Device to run on
        debug: Whether to print debug information
        
    Returns:
        Dictionary with forgetting metrics:
        - forgetting_task_{t}: Forgetting for samples where last query belongs to task t
        - forgetting_task_{t}_count: Number of samples for task t
        - forgetting_overall: Overall forgetting (mean across all samples)
        - last_query_task_ids: Task IDs of the last query for each sample
    """
    if hasattr(model, 'eval'):
        model.eval()
    
    xs_all = xs_all.to(device)
    ys_all = ys_all.to(device)
    task_indices = task_indices.to(device)
    
    # The last point in the sequence is the forgetting query
    last_query_idx = xs_all.shape[1] - 1  # Last index in sequence
    
    # Get task ID of the last query using task_indices
    last_query_task_id = task_indices[:, last_query_idx]  # [batch_size]
    
    # Get the forgetting query point (same for all samples in this batch)
    forgetting_query_x = xs_all[:, last_query_idx:last_query_idx+1, :]  # [batch_size, 1, n_dims]
    forgetting_query_y = ys_all[:, last_query_idx]  # [batch_size]
    
    # Regular query indices (excluding the forgetting query)
    query_indices = [t * (n_context + 1) + n_context for t in range(n_tasks)]
    
    # Compute AFTER prediction: using all tasks
    ys_prompt_after = ys_all.clone()
    for idx in query_indices:
        ys_prompt_after[:, idx] = 0.0
    ys_prompt_after[:, last_query_idx] = 0.0  # Set forgetting query label to 0
    
    with torch.no_grad():
        # Predict the forgetting query using all tasks (AFTER)
        pred_after = model(xs_all, ys_prompt_after, inds=[last_query_idx])
        after_pred = pred_after[:, 0].to(device)  # [batch_size]
    
    # Compute BEFORE predictions for each sample based on its forgetting query task
    before_preds = []
    
    for b in range(xs_all.shape[0]):  # For each sample in batch
        target_task_id = last_query_task_id[b].item()
        
        # BEFORE sequence: only tasks 0 to target_task_id (inclusive)
        n_tasks_before = target_task_id + 1
        seq_length_before = n_tasks_before * (n_context + 1)
        
        # Extract before sequence
        xs_before = xs_all[b:b+1, :seq_length_before, :]  # [1, seq_length_before, n_dims]
        ys_before = ys_all[b:b+1, :seq_length_before]  # [1, seq_length_before]
        
        # Append the forgetting query to the before sequence
        xs_before_with_query = torch.cat([xs_before, forgetting_query_x[b:b+1, :, :]], dim=1)  # [1, seq_length_before+1, n_dims]
        ys_before_with_query = torch.cat([ys_before, forgetting_query_y[b:b+1].unsqueeze(0)], dim=1)  # [1, seq_length_before+1]
        
        # Build prompt for before sequence
        ys_prompt_before = ys_before_with_query.clone()
        # Set all query labels to 0
        for t in range(n_tasks_before):
            query_idx = t * (n_context + 1) + n_context
            if query_idx < seq_length_before:
                ys_prompt_before[:, query_idx] = 0.0
        ys_prompt_before[:, -1] = 0.0  # Set forgetting query label to 0
        
        # Predict the forgetting query using only tasks 0 to target_task_id (BEFORE)
        query_idx_before = xs_before_with_query.shape[1] - 1
        pred_before = model(xs_before_with_query, ys_prompt_before, inds=[query_idx_before])
        before_pred = pred_before[:, 0].to(device)  # [1]
        before_preds.append(before_pred)
    
    before_pred_all = torch.cat(before_preds, dim=0)  # [batch_size]
    
    # Compute forgetting: (after_pred - before_pred)^2
    pred_diff = after_pred - before_pred_all  # [batch_size]
    forgetting_values = pred_diff ** 2  # [batch_size]
    
    # Compute forgetting metrics grouped by task
    forgetting_metrics = {}
    
    # For each possible task, compute forgetting when the last query belongs to that task
    for task_id in range(n_tasks):
        # Find samples where last query belongs to this task
        mask = (last_query_task_id == task_id).to(device)  # [batch_size]
        
        if mask.sum() > 0:
            task_forgetting = forgetting_values[mask]  # [n_samples]
            forgetting_metrics[f"forgetting_task_{task_id}"] = task_forgetting.mean().item()
            forgetting_metrics[f"forgetting_task_{task_id}_count"] = mask.sum().item()
        else:
            # No samples for this task
            forgetting_metrics[f"forgetting_task_{task_id}"] = None
            forgetting_metrics[f"forgetting_task_{task_id}_count"] = 0
    
    # Overall forgetting metric: mean of (after_pred - before_pred)^2
    forgetting_metrics["forgetting_overall"] = forgetting_values.mean().item()
    forgetting_metrics["forgetting_overall_std"] = forgetting_values.std().item()
    forgetting_metrics["last_query_task_ids"] = last_query_task_id
    
    if debug:
        print(f"\n=== Forgetting Analysis (Original Definition) ===")
        print(f"  Last query index: {last_query_idx}")
        print(f"  Last query task IDs: {last_query_task_id.cpu().tolist()[:10]}...")
        print(f"  Overall forgetting: {forgetting_metrics['forgetting_overall']:.4f} +/- {forgetting_metrics['forgetting_overall_std']:.4f}")
        for task_id in range(n_tasks):
            count = forgetting_metrics[f"forgetting_task_{task_id}_count"]
            if count > 0:
                forgetting_val = forgetting_metrics[f"forgetting_task_{task_id}"]
                print(f"  Task {task_id} forgetting: {forgetting_val:.4f} (n={count})")
    
    return forgetting_metrics


def eval_multi_task_model(
    model,
    task_name: str,
    data_name: str,
    n_dims: int,
    n_tasks: int,
    n_context: int,
    num_eval_batches: int = 100,
    batch_size: int = 32,
    task_similarity: Optional[float] = None,
    device: str = "cuda",
    fixed_task_weights: Optional[List[torch.Tensor]] = None,
    fixed_base_w: Optional[torch.Tensor] = None,
    eval_forgetting: bool = False,
    **kwargs
) -> Dict[str, float]:
    """
    Evaluate a model on multi-task continual learning.
    
    Args:
        model: Model to evaluate
        task_name: Name of the task (e.g., "linear_regression")
        data_name: Name of data sampler (e.g., "gaussian")
        n_dims: Input dimension
        n_tasks: Number of tasks T
        n_context: Number of in-context examples M per task
        num_eval_batches: Number of evaluation batches
        batch_size: Batch size
        task_similarity: Task similarity parameter (theta)
        device: Device to run on
        fixed_task_weights: Fixed task weights to use
        fixed_base_w: Fixed base weight vector to use (for fair comparison across theta values)
        eval_forgetting: Whether to evaluate forgetting metrics
        **kwargs: Additional arguments for samplers
        
    Returns:
        Dictionary with aggregated metrics
    """
    data_sampler = get_data_sampler(data_name, n_dims, **kwargs.get("data_sampler_kwargs", {}))
    task_sampler = get_task_sampler(
        task_name, n_dims, batch_size, **kwargs.get("task_sampler_kwargs", {})
    )
    
    # Get metric function and extract scale parameter
    temp_task = task_sampler()
    metric_func = temp_task.get_metric()
    # Get scale parameter from task (default to 1.0 if not available)
    task_scale = getattr(temp_task, 'scale', 1.0)
    
    all_metrics = []
    per_task_metrics = {f"task_{t}": [] for t in range(n_tasks)}
    
    for i in tqdm(range(num_eval_batches), desc="Evaluating multi-task"):
        xs_all, ys_all, task_weights, task_indices = generate_multi_task_sequence(
            task_sampler,
            data_sampler,
            n_tasks=n_tasks,
            n_context=n_context,
            n_dims=n_dims,
            batch_size=batch_size,
            task_similarity=task_similarity,
            seed=None,  # Random for each batch
            fixed_task_weights=fixed_task_weights,
            fixed_base_w=fixed_base_w,  # Pass fixed base_w if provided
            task_scale=task_scale,  # Pass scale parameter
        )
        
        metrics = eval_multi_task_batch(
            model, xs_all, ys_all, n_tasks, n_context, metric_func, device,
            debug=(i == 0),  # Debug only for first batch
            task_indices=task_indices,
        )
        
        all_metrics.append(metrics["overall"].cpu())
        for t in range(n_tasks):
            per_task_metrics[f"task_{t}"].append(metrics[f"task_{t}"].cpu())
    
    # Aggregate results
    # Fix: overall_mse_std should be the std of batch-level MSEs, not the std of all squared errors
    all_metrics_stacked = torch.stack(all_metrics)  # [num_batches, batch_size, T]
    batch_mses = all_metrics_stacked.mean(dim=(1, 2))  # [num_batches] - MSE for each batch
    results = {
        "overall_mse": batch_mses.mean().item(),
        "overall_mse_std": batch_mses.std().item(),  # std across batches
    }
    
    for t in range(n_tasks):
        task_vals = torch.stack(per_task_metrics[f"task_{t}"])  # [num_batches, batch_size, 1]
        batch_task_mses = task_vals.mean(dim=1).squeeze(-1)  # [num_batches] - MSE for each batch
        results[f"task_{t}_mse"] = batch_task_mses.mean().item()
        results[f"task_{t}_mse_std"] = batch_task_mses.std().item()  # std across batches
    
    # Average over tasks
    task_means = [results[f"task_{t}_mse"] for t in range(n_tasks)]
    results["avg_task_mse"] = np.mean(task_means)
    results["avg_task_mse_std"] = np.std(task_means)
    
    # Evaluate forgetting if requested
    if eval_forgetting:
        forgetting_metrics_list = []
        per_task_forgetting = {f"forgetting_task_{t}": [] for t in range(n_tasks)}
        
        for i in tqdm(range(num_eval_batches), desc="Evaluating forgetting"):
            xs_all, ys_all, task_weights, task_indices = generate_multi_task_sequence(
                task_sampler,
                data_sampler,
                n_tasks=n_tasks,
                n_context=n_context,
                n_dims=n_dims,
                batch_size=batch_size,
                task_similarity=task_similarity,
                seed=None,  # Random for each batch
                fixed_task_weights=fixed_task_weights,
                fixed_base_w=fixed_base_w,  # Pass fixed base_w if provided
                add_forgetting_query=True,  # Add additional query for forgetting evaluation
                forgetting_query_task_ids=None,  # Randomly sample from previous tasks
                task_scale=task_scale,  # Pass scale parameter
            )
            
            forgetting_metrics = eval_forgetting_batch(
                model, xs_all, ys_all, task_indices, n_tasks, n_context,
                metric_func, device, debug=(i == 0)
            )
            
            # forgetting_overall is now a scalar (float), not a tensor
            forgetting_metrics_list.append(forgetting_metrics["forgetting_overall"])
            for t in range(n_tasks):
                if forgetting_metrics[f"forgetting_task_{t}"] is not None:
                    per_task_forgetting[f"forgetting_task_{t}"].append(
                        forgetting_metrics[f"forgetting_task_{t}"]
                    )
        
        # Aggregate forgetting results
        if forgetting_metrics_list:
            # forgetting_metrics_list contains scalars, convert to numpy array for aggregation
            forgetting_array = np.array(forgetting_metrics_list)  # [num_batches]
            results["forgetting_overall_mse"] = forgetting_array.mean().item()
            results["forgetting_overall_mse_std"] = forgetting_array.std().item()  # std across batches
        
        for t in range(n_tasks):
            if per_task_forgetting[f"forgetting_task_{t}"]:
                # per_task_forgetting contains scalars
                task_forgetting_array = np.array(per_task_forgetting[f"forgetting_task_{t}"])
                results[f"forgetting_task_{t}_mse"] = task_forgetting_array.mean().item()
                results[f"forgetting_task_{t}_mse_std"] = task_forgetting_array.std().item()
            else:
                results[f"forgetting_task_{t}_mse"] = None
                results[f"forgetting_task_{t}_mse_std"] = None
    
    return results
