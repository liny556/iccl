# Understanding Generalization and Forgetting in In-Context Continual Learning

This repository contains the code used to reproduce the experiments for
[**Understanding Generalization and Forgetting in In-Context Continual Learning**](https://arxiv.org/abs/2605.28705).

The paper studies how frozen attention-based models behave when a prompt
contains a sequence of tasks, focusing on generalization, interference,
negative transfer, and forgetting.

The repository is intended to contain source code only. Model checkpoints,
generated metrics, generated plots, and server-side run artifacts are ignored by
git and should be regenerated locally.

## Repository Layout

```text
.
├── environment.yml          # Conda environment
├── requirements.txt         # pip-only dependency list
├── src/
│   ├── train.py             # GPT-style synthetic ICL training
│   ├── multi_task_eval.py   # Multi-task prompt construction and evaluation
│   ├── multi_task_experiments.py
│   ├── theorem_4_3.py       # Fixed-weight theory comparison helpers
│   ├── task_family_experiments.py
│   ├── plot_paper_results.py
│   ├── plot_task_family_context_sweep.py
│   └── real_world_llm_eval.py
└── src/conf/                # Training configs
```

## Setup

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate in-context-continual-learning
```

Or install with pip in an existing environment:

```bash
pip install -r requirements.txt
```

## Checkpoints

Synthetic experiments require trained checkpoints. Checkpoints are not tracked
in git. Place local checkpoints under paths such as:

```text
models/linear_regression/pretrained/
models/sparse_linear_regression/pretrained/
models/relu_2nn_regression/pretrained/
```

Each checkpoint directory should contain the files produced by `src/train.py`,
including `config.yaml`, `state.pt`, and `metrics.json`.

## Train a Synthetic ICL Model

Run from the repository root:

```bash
python src/train.py --config src/conf/linear_regression.yaml
```

The default linear regression config writes to `models/linear_regression/`.
For a quick smoke run:

```bash
python src/train.py --config src/conf/linear_regression.yaml --test_run true
```

## Run Synthetic ICCL Experiments

Context length, task similarity, task order, and number-of-task experiments are
available through one entry point:

```bash
python src/multi_task_experiments.py \
  --model_path models/linear_regression/pretrained \
  --experiment 6.2 \
  --output_dir results \
  --num_eval_batches 100 \
  --batch_size 32
```

Valid experiment names are `6.1`, `6.2`, `6.2fw`, `6.3`, `6.4`, `6.5`, and
`all`. The `6.2fw` option runs the fixed-weight context-length experiment used
for the theory comparison:

```bash
python src/multi_task_experiments.py \
  --model_path models/linear_regression/pretrained \
  --experiment 6.2fw \
  --output_dir results \
  --fixed_w_seed 42
```

For nonlinear task-family experiments:

```bash
python src/task_family_experiments.py \
  --task_model_pairs linear_regression=models/linear_regression/pretrained \
                     sparse_linear_regression=models/sparse_linear_regression/pretrained \
                     relu_2nn_regression=models/relu_2nn_regression/pretrained \
  --n_context_values 1 2 3 5 7 9 11 13 15 19 \
  --output_dir results
```

## Generate Figures

Plot a synthetic experiment JSON file:

```bash
python src/plot_paper_results.py results/exp_6_2_context_length.json
```

Choose an explicit output path if needed:

```bash
python src/plot_paper_results.py \
  results/exp_6_3_task_similarity.json \
  --output results/exp_6_3_task_similarity_plot.pdf
```

Generated figures are ignored by git.

## Real-World Qwen2.5 Evaluation

The real-world validation in the paper uses Qwen2.5-1.5B-Instruct on a two-task
sequence: SST-2 followed by AG News. The public script matches that protocol
and reports Table 2 style metrics.

Preview the prompt without downloading a model or datasets:

```bash
python src/real_world_llm_eval.py --dry_run_prompt --m_values 1
```

Run the evaluation:

```bash
python src/real_world_llm_eval.py \
  --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
  --m_values 1 3 5 19 \
  --num_test_samples 50 \
  --output_dir results/qwen2_5
```

If the model is already downloaded locally, either pass its path with
`--model_name_or_path` or set:

```bash
export ICCL_MODEL_PATH=/path/to/Qwen2.5-1.5B-Instruct
```

The script evaluates:

- Task B baseline: AG News examples followed by an AG News query.
- Task B ICCL: SST-2 examples, then AG News examples, then an AG News query.
- Task A baseline: SST-2 examples followed by an SST-2 query.
- Task A final: SST-2 examples, then AG News examples, then an SST-2 query.

## Citation

Please consider citing our paper if you find this repo useful in your work 😀.

``
@misc{li2026understandinggeneralizationforgettingincontext,
      title={Understanding Generalization and Forgetting in In-Context Continual Learning}, 
      author={Guangyu Li and Meng Ding and Lijie Hu},
      year={2026},
      eprint={2605.28705},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.28705}, 
}
``

## Contact

If you have any questions, please feel free to open an issue or contact me via email at flipped@mail.ustc.edu.cn.
