"""Real-world ICCL evaluation with Qwen2.5 on SST-2 and AG News.

This script implements the two-task protocol described in the paper:
Task A is SST-2 sentiment classification and Task B is AG News topic
classification. It reports Table 2 style baseline, ICCL/final, and delta
metrics for each context length M.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple


DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SST2_LABELS = {0: "Negative", 1: "Positive"}
AG_NEWS_LABELS = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Science and Technology",
}


@dataclass(frozen=True)
class LabeledText:
    text: str
    label: int


def sst2_label(label_idx: int) -> str:
    return SST2_LABELS[int(label_idx)]


def ag_news_label(label_idx: int) -> str:
    return AG_NEWS_LABELS[int(label_idx)]


def build_iccl_prompt(
    sst2_shots: Sequence[LabeledText],
    ag_news_shots: Sequence[LabeledText],
    query_text: str,
    query_task: str,
) -> str:
    """Build the paper's two-task ICCL prompt."""
    lines = [
        "You are an expert text classifier. Read the examples and classify",
        "the final text accordingly.",
        "",
    ]

    if sst2_shots:
        lines.append("--- Task 1: Sentiment Analysis ---")
        for sample in sst2_shots:
            lines.append(f"Text: {sample.text}")
            lines.append(f"Label: {sst2_label(sample.label)}")
            lines.append("")

    if ag_news_shots:
        lines.append("--- Task 2: News Topic Classification ---")
        for sample in ag_news_shots:
            lines.append(f"Text: {sample.text}")
            lines.append(f"Label: {ag_news_label(sample.label)}")
            lines.append("")

    if query_task == "sst2":
        lines.append("--- Task 1: Sentiment Analysis ---")
        lines.append(f"Text: {query_text}")
        lines.append("Label:")
    elif query_task == "ag_news":
        lines.append("--- Task 2: News Topic Classification ---")
        lines.append(f"Text: {query_text}")
        lines.append("Label:")
    else:
        raise ValueError(f"Unsupported query task: {query_task}")

    return "\n".join(lines)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def parse_sst2_prediction(completion: str) -> int:
    text = normalize_text(completion)
    if re.search(r"\bpositive\b", text):
        return 1
    if re.search(r"\bnegative\b", text):
        return 0
    return -1


def parse_ag_news_prediction(completion: str) -> int:
    text = normalize_text(completion)
    if re.search(r"\b(science|technology|tech)\b", text):
        return 3
    if re.search(r"\bbusiness\b", text):
        return 2
    if re.search(r"\b(sport|sports)\b", text):
        return 1
    if re.search(r"\bworld\b", text):
        return 0
    return -1


def apply_chat_template(tokenizer: Any, prompt: str, enabled: bool) -> str:
    if not enabled or getattr(tokenizer, "chat_template", None) is None:
        return prompt
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def model_input_device(model: Any) -> Any:
    for param in model.parameters():
        if getattr(param, "device", None) is not None and param.device.type != "meta":
            return param.device
    return "cpu"


def generate_completion(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    use_chat_template: bool,
) -> str:
    import torch

    model_prompt = apply_chat_template(tokenizer, prompt, use_chat_template)
    inputs = tokenizer(model_prompt, return_tensors="pt")
    inputs = inputs.to(model_input_device(model))

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def evaluate_accuracy(
    model: Any,
    tokenizer: Any,
    samples: Sequence[LabeledText],
    prompt_fn: Callable[[LabeledText], str],
    parse_fn: Callable[[str], int],
    max_new_tokens: int,
    use_chat_template: bool,
    description: str,
) -> Dict[str, Any]:
    from tqdm import tqdm

    correct = 0
    parse_failures = 0
    examples = []

    for sample in tqdm(samples, desc=description):
        prompt = prompt_fn(sample)
        completion = generate_completion(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            use_chat_template=use_chat_template,
        )
        prediction = parse_fn(completion)
        if prediction < 0:
            parse_failures += 1
        if prediction == sample.label:
            correct += 1
        if len(examples) < 3:
            examples.append(
                {
                    "gold": sample.label,
                    "prediction": prediction,
                    "completion": completion.strip(),
                }
            )

    total = len(samples)
    return {
        "accuracy": float(correct / total) if total else 0.0,
        "correct": int(correct),
        "total": int(total),
        "parse_failures": int(parse_failures),
        "examples": examples,
    }


def load_datasets_and_samples(
    m_values: Sequence[int],
    num_test_samples: int,
    seed: int,
    local_files_only: bool,
) -> Tuple[List[LabeledText], List[LabeledText], List[LabeledText], List[LabeledText]]:
    from datasets import DownloadConfig, load_dataset

    download_config = DownloadConfig(local_files_only=local_files_only)
    sst2 = load_dataset("glue", "sst2", download_config=download_config)
    ag_news = load_dataset("ag_news", download_config=download_config)

    rng = random.Random(seed)
    train_sst2 = list(sst2["train"])
    train_ag_news = list(ag_news["train"])
    rng.shuffle(train_sst2)
    rng.shuffle(train_ag_news)

    max_m = max(m_values)
    sst2_shot_pool = [
        LabeledText(text=item["sentence"], label=int(item["label"]))
        for item in train_sst2[:max_m]
    ]
    ag_news_shot_pool = [
        LabeledText(text=item["text"], label=int(item["label"]))
        for item in train_ag_news[:max_m]
    ]

    eval_sst2_raw = list(sst2["validation"])
    eval_ag_news_raw = list(ag_news["test"])
    if num_test_samples > 0:
        eval_sst2_raw = eval_sst2_raw[:num_test_samples]
        eval_ag_news_raw = eval_ag_news_raw[:num_test_samples]

    eval_sst2 = [
        LabeledText(text=item["sentence"], label=int(item["label"]))
        for item in eval_sst2_raw
    ]
    eval_ag_news = [
        LabeledText(text=item["text"], label=int(item["label"]))
        for item in eval_ag_news_raw
    ]
    return sst2_shot_pool, ag_news_shot_pool, eval_sst2, eval_ag_news


def resolve_model_name(model_name_or_path: str) -> str:
    local_override = os.environ.get("ICCL_MODEL_PATH", "").strip()
    if local_override:
        return local_override
    return model_name_or_path


def dtype_from_name(dtype_name: str) -> Any:
    import torch

    if dtype_name == "auto":
        return "auto"
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def load_tokenizer_and_model(args: argparse.Namespace) -> Tuple[Any, Any, str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_ref = resolve_model_name(args.model_name_or_path)
    local_files_only = args.local_files_only or os.path.isdir(model_ref)
    common_kwargs = {
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": local_files_only,
    }

    tokenizer = AutoTokenizer.from_pretrained(model_ref, **common_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = dict(common_kwargs)
    model_kwargs["torch_dtype"] = dtype_from_name(args.dtype)
    if args.device_map:
        model_kwargs["device_map"] = args.device_map

    model = AutoModelForCausalLM.from_pretrained(model_ref, **model_kwargs)
    if not args.device_map:
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
    model.eval()
    return tokenizer, model, model_ref


def run_evaluation(args: argparse.Namespace) -> Dict[str, Any]:
    tokenizer, model, model_ref = load_tokenizer_and_model(args)
    sst2_pool, ag_pool, eval_sst2, eval_ag = load_datasets_and_samples(
        m_values=args.m_values,
        num_test_samples=args.num_test_samples,
        seed=args.seed,
        local_files_only=args.local_files_only,
    )

    rows = []
    for m_value in args.m_values:
        print(f"\n=== Evaluating M={m_value} ===")
        sst2_shots = sst2_pool[:m_value]
        ag_shots = ag_pool[:m_value]

        task_b_baseline = evaluate_accuracy(
            model=model,
            tokenizer=tokenizer,
            samples=eval_ag,
            prompt_fn=lambda sample: build_iccl_prompt(
                sst2_shots=[],
                ag_news_shots=ag_shots,
                query_text=sample.text,
                query_task="ag_news",
            ),
            parse_fn=parse_ag_news_prediction,
            max_new_tokens=args.max_new_tokens,
            use_chat_template=not args.no_chat_template,
            description=f"Task B baseline M={m_value}",
        )
        task_b_iccl = evaluate_accuracy(
            model=model,
            tokenizer=tokenizer,
            samples=eval_ag,
            prompt_fn=lambda sample: build_iccl_prompt(
                sst2_shots=sst2_shots,
                ag_news_shots=ag_shots,
                query_text=sample.text,
                query_task="ag_news",
            ),
            parse_fn=parse_ag_news_prediction,
            max_new_tokens=args.max_new_tokens,
            use_chat_template=not args.no_chat_template,
            description=f"Task B ICCL M={m_value}",
        )
        task_a_baseline = evaluate_accuracy(
            model=model,
            tokenizer=tokenizer,
            samples=eval_sst2,
            prompt_fn=lambda sample: build_iccl_prompt(
                sst2_shots=sst2_shots,
                ag_news_shots=[],
                query_text=sample.text,
                query_task="sst2",
            ),
            parse_fn=parse_sst2_prediction,
            max_new_tokens=args.max_new_tokens,
            use_chat_template=not args.no_chat_template,
            description=f"Task A baseline M={m_value}",
        )
        task_a_final = evaluate_accuracy(
            model=model,
            tokenizer=tokenizer,
            samples=eval_sst2,
            prompt_fn=lambda sample: build_iccl_prompt(
                sst2_shots=sst2_shots,
                ag_news_shots=ag_shots,
                query_text=sample.text,
                query_task="sst2",
            ),
            parse_fn=parse_sst2_prediction,
            max_new_tokens=args.max_new_tokens,
            use_chat_template=not args.no_chat_template,
            description=f"Task A final M={m_value}",
        )

        row = {
            "M": int(m_value),
            "task_b_baseline_accuracy": task_b_baseline["accuracy"],
            "task_b_iccl_accuracy": task_b_iccl["accuracy"],
            "task_b_delta": task_b_iccl["accuracy"] - task_b_baseline["accuracy"],
            "task_a_baseline_accuracy": task_a_baseline["accuracy"],
            "task_a_final_accuracy": task_a_final["accuracy"],
            "task_a_delta": task_a_final["accuracy"] - task_a_baseline["accuracy"],
            "parse_failures": {
                "task_b_baseline": task_b_baseline["parse_failures"],
                "task_b_iccl": task_b_iccl["parse_failures"],
                "task_a_baseline": task_a_baseline["parse_failures"],
                "task_a_final": task_a_final["parse_failures"],
            },
        }
        rows.append(row)
        print(
            "M={M}: Task B baseline={b0:.3f}, ICCL={b1:.3f}, delta={bd:+.3f}; "
            "Task A baseline={a0:.3f}, final={a1:.3f}, delta={ad:+.3f}".format(
                M=m_value,
                b0=row["task_b_baseline_accuracy"],
                b1=row["task_b_iccl_accuracy"],
                bd=row["task_b_delta"],
                a0=row["task_a_baseline_accuracy"],
                a1=row["task_a_final_accuracy"],
                ad=row["task_a_delta"],
            )
        )

    return {
        "model": model_ref,
        "m_values": [int(m) for m in args.m_values],
        "config": {
            "seed": args.seed,
            "num_test_samples": args.num_test_samples,
            "max_new_tokens": args.max_new_tokens,
            "use_chat_template": not args.no_chat_template,
            "sst2_eval_split": "validation",
            "ag_news_eval_split": "test",
            "task_order": ["sst2", "ag_news"],
        },
        "rows": rows,
    }


def dry_run_prompt(args: argparse.Namespace) -> None:
    m_value = args.m_values[0]
    sst2_shots = [
        LabeledText("a warm , funny , and sharply observed comedy", 1),
        LabeledText("the movie is dull and painfully predictable", 0),
    ][:m_value]
    ag_shots = [
        LabeledText("Stocks rose after the company reported strong earnings.", 2),
        LabeledText("The home team won the final after overtime.", 1),
    ][:m_value]

    print("Task B ICCL prompt:\n")
    print(
        build_iccl_prompt(
            sst2_shots=sst2_shots,
            ag_news_shots=ag_shots,
            query_text="Global leaders met to discuss trade policy.",
            query_task="ag_news",
        )
    )
    print("\n" + "=" * 80 + "\n")
    print("Task A final prompt:\n")
    print(
        build_iccl_prompt(
            sst2_shots=sst2_shots,
            ag_news_shots=ag_shots,
            query_text="a beautifully acted film with a generous heart",
            query_task="sst2",
        )
    )


def write_results(payload: Dict[str, Any], output_dir: str) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    output_path = path / "qwen2_5_sst2_agnews_iccl_results.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the paper-aligned Qwen2.5 SST-2/AG News ICCL evaluation."
    )
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL)
    parser.add_argument("--m_values", nargs="+", type=int, default=[1, 3, 5, 19])
    parser.add_argument("--num_test_samples", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="results/qwen2_5")
    parser.add_argument("--device", default=None)
    parser.add_argument("--device_map", default="auto")
    parser.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--no_chat_template", action="store_true")
    parser.add_argument(
        "--dry_run_prompt",
        action="store_true",
        help="Print representative prompts without loading the model or datasets.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run_prompt:
        dry_run_prompt(args)
        return

    try:
        payload = run_evaluation(args)
    except Exception as exc:
        print(
            "Evaluation failed. If this is a Hugging Face access issue, set "
            "ICCL_MODEL_PATH to a local Qwen2.5 directory or use "
            "--local_files_only after caching the model and datasets.",
            file=sys.stderr,
        )
        raise exc

    output_path = write_results(payload, args.output_dir)
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
