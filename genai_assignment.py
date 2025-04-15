# Install libraries
!pip install transformers
!pip install torch
!git clone https://github.com/EleutherAI/lm-evaluation-harness
!cd lm-evaluation-harness && pip install -e .
!pip install deepspeed

# Import and deploy Qwen models
import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import deepspeed

sys.path.append("/content/lm-evaluation-harness")
from lm_eval import evaluator, tasks
from lm_eval.models.huggingface import HFLM

ds_config = {
    "fp16": {
        "enabled": True
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu"
        }
    },
    "train_batch_size": 8,
    "train_micro_batch_size_per_gpu": 8
}

MODELS = [
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-7B"
]

TASKS = {
    "NLI": ["hellaswag"],
    "understanding": ["mmlu"],
    "code_generation": ["mbpp"]
}

results = {}

# Benchmark for Qwen
os.environ["HF_ALLOW_CODE_EVAL"] = "1"

for model_name in MODELS:
    print(f"Evaluating {model_name}...")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16
    )

    ds_engine = deepspeed.init_inference(
        model=model,
        mp_size=1,
        dtype=torch.float16,
        replace_with_kernel_inject=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    hf_model = HFLM(
        pretrained=ds_engine.module,
        tokenizer=tokenizer,
        batch_size=8,
        device="cuda"
    )

    model_results = {}

    for category, task_list in TASKS.items():
        print(f"Evaluating {category} tasks...")

        num_fewshot = 2 if category == "code_generation" else 0

        results_dict = evaluator.simple_evaluate(
            model=hf_model,
            tasks=task_list,
            num_fewshot=num_fewshot,
            batch_size=8,
            device="cuda",
            confirm_run_unsafe_code=True,
            gen_kwargs="temperature=0.1,top_p=0.95,max_length=512",
            random_seed=42,
            torch_random_seed=42,
            fewshot_random_seed=42
        )

        model_results[category] = results_dict

    results[model_name] = model_results

    del hf_model
    del ds_engine
    del model
    torch.cuda.empty_cache()

# Value Function
def get_metric_value(task_results, metric_name):
    formats = [
        f"{metric_name},none",
        metric_name
    ]

    for fmt in formats:
        if fmt in task_results:
            return task_results[fmt]
    return None

# Print Model Evaluation Results
print("\n============= Model Evaluation Results with 7B vs 1.5B Improvement =============")

# Determine all task categories
all_categories = set()
for model_results in results.values():
    all_categories.update(model_results.keys())

# Get models in correct order for comparison
models = list(results.keys())
model_headers = [model.split('/')[-1] for model in models]  # Only take the last part of model names

# Ensure we have exactly 2 models for comparison
if len(models) != 2:
    print("Warning: Expected exactly 2 models for comparison")

# Find the indices for the 1.5B and 7B models
model_1_5B_idx = -1
model_7B_idx = -1
for i, header in enumerate(model_headers):
    if "1.5B" in header:
        model_1_5B_idx = i
    elif "7B" in header:
        model_7B_idx = i

# Print headers
header = "Task/Metric".ljust(25)
for model_header in model_headers:
    header += model_header.ljust(20)
header += "Improvement(pp)".ljust(20)  # Add improvement column
print(header)
print("-" * (25 + 20 * len(models) + 20))  # Extend line for new column

# Process and print results for all categories
for category in sorted(all_categories):
    print(f"\n【{category}】")

    # Collect all tasks in this category
    category_tasks = set()
    for model in models:
        if category in results[model] and 'results' in results[model][category]:
            category_tasks.update(results[model][category]['results'].keys())

    # Print results for each task
    for task in sorted(category_tasks):
        # Special handling for hellaswag - distinguish between standard and normalized
        if task == "hellaswag":
            # Standard acc
            task_line = f"  {task} (standard)".ljust(25)
            model_values = []

            for model in models:
                if (category in results[model] and
                    'results' in results[model][category] and
                    task in results[model][category]['results']):
                    task_results = results[model][category]['results'][task]

                    # Get value and stderr
                    value = get_metric_value(task_results, "acc")
                    stderr = get_metric_value(task_results, "acc_stderr")

                    if value is not None:
                        # Store raw value for improvement calculation
                        model_values.append(value)
                        # Convert to percentage for display
                        score = f"{value*100:.2f}% ± {stderr*100:.2f}%" if stderr else f"{value*100:.2f}%"
                    else:
                        model_values.append(None)
                        score = "N/A"
                    task_line += score.ljust(20)
                else:
                    model_values.append(None)
                    task_line += "N/A".ljust(20)

            # Calculate improvement (in percentage points)
            if len(model_values) >= 2 and model_values[model_1_5B_idx] is not None and model_values[model_7B_idx] is not None:
                improvement = (model_values[model_7B_idx] - model_values[model_1_5B_idx]) * 100
                task_line += f"+{improvement:.2f}pp".ljust(20) if improvement >= 0 else f"{improvement:.2f}pp".ljust(20)
            else:
                task_line += "N/A".ljust(20)

            print(task_line)

            # Normalized acc
            task_line = f"  {task} (normalized)".ljust(25)
            model_values = []

            for model in models:
                if (category in results[model] and
                    'results' in results[model][category] and
                    task in results[model][category]['results']):
                    task_results = results[model][category]['results'][task]

                    # Get value and stderr
                    value = get_metric_value(task_results, "acc_norm")
                    stderr = get_metric_value(task_results, "acc_norm_stderr")

                    if value is not None:
                        # Store raw value for improvement calculation
                        model_values.append(value)
                        # Convert to percentage for display
                        score = f"{value*100:.2f}% ± {stderr*100:.2f}%" if stderr else f"{value*100:.2f}%"
                    else:
                        model_values.append(None)
                        score = "N/A"
                    task_line += score.ljust(20)
                else:
                    model_values.append(None)
                    task_line += "N/A".ljust(20)

            # Calculate improvement (in percentage points)
            if len(model_values) >= 2 and model_values[model_1_5B_idx] is not None and model_values[model_7B_idx] is not None:
                improvement = (model_values[model_7B_idx] - model_values[model_1_5B_idx]) * 100
                task_line += f"+{improvement:.2f}pp".ljust(20) if improvement >= 0 else f"{improvement:.2f}pp".ljust(20)
            else:
                task_line += "N/A".ljust(20)

            print(task_line)

        # Special handling for mbpp
        elif task == "mbpp":
            task_line = f"  {task} (pass@1)".ljust(25)
            model_values = []

            for model in models:
                if (category in results[model] and
                    'results' in results[model][category] and
                    task in results[model][category]['results']):
                    task_results = results[model][category]['results'][task]

                    # Get value and stderr
                    value = get_metric_value(task_results, "pass_at_1")
                    stderr = get_metric_value(task_results, "pass_at_1_stderr")

                    if value is not None:
                        # Store raw value for improvement calculation
                        model_values.append(value)
                        # Convert to percentage for display
                        score = f"{value*100:.2f}% ± {stderr*100:.2f}%" if stderr else f"{value*100:.2f}%"
                    else:
                        model_values.append(None)
                        score = "N/A"
                    task_line += score.ljust(20)
                else:
                    model_values.append(None)
                    task_line += "N/A".ljust(20)

            # Calculate improvement (in percentage points)
            if len(model_values) >= 2 and model_values[model_1_5B_idx] is not None and model_values[model_7B_idx] is not None:
                improvement = (model_values[model_7B_idx] - model_values[model_1_5B_idx]) * 100
                task_line += f"+{improvement:.2f}pp".ljust(20) if improvement >= 0 else f"{improvement:.2f}pp".ljust(20)
            else:
                task_line += "N/A".ljust(20)

            print(task_line)

        # Handle MMLU and other tasks that use acc
        else:
            task_line = f"  {task}".ljust(25)
            model_values = []

            for model in models:
                if (category in results[model] and
                    'results' in results[model][category] and
                    task in results[model][category]['results']):
                    task_results = results[model][category]['results'][task]

                    # Get value and stderr
                    value = get_metric_value(task_results, "acc")
                    stderr = get_metric_value(task_results, "acc_stderr")

                    if value is not None:
                        # Store raw value for improvement calculation
                        model_values.append(value)
                        # Convert to percentage for display
                        score = f"{value*100:.2f}% ± {stderr*100:.2f}%" if stderr else f"{value*100:.2f}%"
                    else:
                        model_values.append(None)
                        score = "N/A"
                    task_line += score.ljust(20)
                else:
                    model_values.append(None)
                    task_line += "N/A".ljust(20)

            # Calculate improvement (in percentage points)
            if len(model_values) >= 2 and model_values[model_1_5B_idx] is not None and model_values[model_7B_idx] is not None:
                improvement = (model_values[model_7B_idx] - model_values[model_1_5B_idx]) * 100
                task_line += f"+{improvement:.2f}pp".ljust(20) if improvement >= 0 else f"{improvement:.2f}pp".ljust(20)
            else:
                task_line += "N/A".ljust(20)

            print(task_line)
