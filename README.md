<div align="center">
<h1>DSI-Bench: A Benchmark for Dynamic Spatial Intelligence</h1>

[**Ziang Zhang**](https://scholar.google.com/citations?hl=zh-CN&user=DptGMnYAAAAJ) <sup>1*</sup> · [**Zehan Wang**](https://scholar.google.com/citations?user=euXK0lkAAAAJ&hl=zh-CN) <sup>1*</sup> · Guanghao Zhang<sup>2*</sup> · Weilong Dai<sup>2</sup> · Yan Xia<sup>2</sup> · Ziang Yan<sup>1,3</sup> · Minjie Hong<sup>1</sup> · Zhou Zhao <sup>1,3†</sup>

<sup>1</sup>Zhejiang University  <sup>2</sup>Alibaba Group  <sup>3</sup>Shanghai AI Lab

*Equal Contribution †Corresponding author.


<a href='https://arxiv.org/abs/2412.18605'><img src='https://img.shields.io/badge/arXiv-Paper PDF-red' alt='Paper PDF'></a>
<a href='https://dsibench.github.io'><img src='https://img.shields.io/badge/Project_Page-DSI_Bench-green' alt='Project Page'></a>
<a href='https://huggingface.co/datasets/Viglong/DSI-Bench'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue'></a>
</div>


## Quick Start

### 1. Install dependency
```shell
pip install -r requirements.txt
```

### 2. Download full dataset
```shell
huggingface-cli download --repo-type dataset Viglong/DSI-Bench --local-dir DSI-Bench
```


### 3. Inference with Qwen API
Here we provide a sample for testing on DSI-bench using the Qwen API.

```python
import os
import re
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import dashscope

# Configuration
DASHSCOPE_API_KEY = "YOUR_DASHSCOPE_API_KEY_HERE"  # Replace or load from env

VLM_MODEL = "qwen2.5-vl-32b-instruct"   # e.g., "qwen2.5-vl-72b-instruct"
VIDEO_AUG = "std"                       # Video variant: 'std', 'hflip', etc.
NUM_WORKERS = 2
MAX_RETRIES = 10
FPS = 5                                 # Frames per second for video input

# Relative paths (relative to this script)
METADATA_BASE_DIR = "/path/to/metadatas"# Contains {VIDEO_AUG}.csv
VIDEO_BASE_DIR = "/path/to/videos"      # Structure: videos/{VIDEO_AUG}/xxx.mp4
OUTPUT_BASE_DIR = "/path/to/outputs"    # Output: outputs/{VIDEO_AUG}/{model}.csv

# Prompt
rawqa_prompt = """You are a vision-language expert.
You are given a clip of video and your task is to answer a question about the video.
You only need to provide *ONE* correct answer selecting from the options listed below. 
For example, if you think the correct answer is 'A' from 'A. Above B. Under C. Front D. Behind', 
your response should **only** be '<answer>A</answer>'.
Please answer the question in this format strictly:
<answer>[A, B, C, or D]</answer>
"""

def extract_single_choice_with_word_boundary(text):
    """
    Extract the answer letter (A/B/C/D) from <answer>X</answer> in the response.
    Returns None if not found.
    """
    match = re.search(r"<answer>\s*([A-D])\s*</answer>", text, re.IGNORECASE)
    return match.group(1).upper() if match else None

# Call VLM with video and question
def query_vlm(mp4_path, question, options):
    """
    Send video and question to Qwen-VL via DashScope API.
    Returns raw model response text.
    """
    question_and_options = f"\nQuestion:\n{question} {options}"
    messages = [
        {
            "role": "user",
            "content": [
                {"video": f"file://{mp4_path}", "fps": FPS},
                {"text": rawqa_prompt + question_and_options}
            ]
        }
    ]
    
    response = dashscope.MultiModalConversation.call(
        api_key=DASHSCOPE_API_KEY,
        model=VLM_MODEL,
        messages=messages,
        max_length=2048,
        stream=False,
        top_k=1
    )
    return response.output.choices[0].message.content[0]["text"]

# Process a single sample
def process_sample(meta, idx):
    """
    Process one video-question sample.
    Returns (raw_response, extracted_answer) or (None, None) on failure.
    """
    mp4_path = os.path.join(VIDEO_BASE_DIR, VIDEO_AUG, meta["relative_path"][idx])
    question = meta["question"][idx]
    options = meta["options"][idx]

    try:
        raw_response = query_vlm(mp4_path, question, options)
        final_answer = extract_single_choice_with_word_boundary(raw_response)
        return raw_response, final_answer
    except Exception as e:
        return None, None

# Batch processing with retries
def run_batch_inference(meta, num_workers=NUM_WORKERS, max_retries=MAX_RETRIES):
    """
    Run inference on all samples with parallel execution and retry logic.
    Ensures output order matches input order.
    """
    print(f"Running inference with model: {VLM_MODEL}, video variant: {VIDEO_AUG}")
    n = len(meta)
    results = [None] * n

    def worker(idx):
        raw, ans = process_sample(meta, idx)
        return idx, raw, ans

    def run_parallel(indices, desc="Processing"):
        temp = [None] * n
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(worker, i): i for i in indices}
            for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
                idx, raw, ans = future.result()
                temp[idx] = {"result_text": raw, "final_answer": ans}
        return temp

    # Initial run
    current_results = run_parallel(list(range(n)), desc="Initial Run")
    for i, res in enumerate(current_results):
        results[i] = res

    # Retry failed samples
    for retry in range(1, max_retries + 1):
        failed = [i for i in range(n) if results[i]["final_answer"] is None]
        if not failed:
            print(f"All samples succeeded after {retry - 1} retries.")
            break

        print(f"Retry {retry}/{max_retries} for {len(failed)} failed samples...")
        retry_results = run_parallel(failed, desc=f"Retry {retry}")
        for i in failed:
            results[i] = retry_results[i]

    # Handle permanently failed samples
    for i in range(n):
        if results[i]["final_answer"] is None:
            results[i]["final_answer"] = "E"  # Default error answer
            if results[i]["result_text"] is None:
                results[i]["result_text"] = ""

    success_count = sum(1 for r in results if r["final_answer"] != "E")
    print(f"Completed. Success: {success_count}/{n}")
    return results

if __name__ == "__main__":
    for aug in ["std", "hflip", "reverse", "reverse_hflip"]:
        VIDEO_AUG = aug
        meta_path = os.path.join(METADATA_BASE_DIR, f"{VIDEO_AUG}.csv")
        df = pd.read_csv(meta_path)
        print(f"Loaded metadata: {meta_path} with {len(df)} samples")

        # Set directories
        output_dir = os.path.join(OUTPUT_BASE_DIR, VIDEO_AUG)
        os.makedirs(output_dir, exist_ok=True)

        # Run inference
        results = run_batch_inference(df)

        # Save results
        output_file = os.path.join(output_dir, f"{VLM_MODEL}.csv")
        pd.DataFrame(results).to_csv(output_file, index=False)
        print(f"Results saved to: {output_file}")
```

### 4. Evaluate model performance
Use the following code to get the Sample-wise Accuracy and Group-wise Accuracy of the model.

```python
import pandas as pd
import os
from typing import Dict

# ==============================
# Configuration
# ==============================
VLM_MODEL = "qwen2.5-vl-32b-instruct"  # Model name
VIDEO_AUGS = ["std", "reverse", "hflip", "reverse_hflip"]

# Base paths (relative to project root)
META_BASE_PATH = "/path/to/metadatas"        # Contains {aug}.csv
OUTPUT_BASE_PATH = "/path/to/outputs"        # Contains {aug}/{VLM_MODEL}.csv

# Category name mapping
CATE_NAMES = [
    "Obj:static cam",
    "Obj:moving cam",
    "Cam:static scene",
    "Cam:dynamic scene",
    "Obj-Cam distance",
    "Obj-Cam orientation"
]

def load_all_data() -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Load metadata and prediction results for all augmentations.
    Returns: {aug: {'meta': DataFrame, 'result': DataFrame}}
    """
    data_dict = {}

    for aug in VIDEO_AUGS:
        meta_path = os.path.join(META_BASE_PATH, f"{aug}.csv")
        res_path = os.path.join(OUTPUT_BASE_PATH, aug, f"{VLM_MODEL}.csv")

        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata not found: {meta_path}")
        if not os.path.exists(res_path):
            raise FileNotFoundError(f"Result file not found: {res_path}")

        meta = pd.read_csv(meta_path)
        result = pd.read_csv(res_path)

        if len(meta) != len(result):
            raise ValueError(f"Length mismatch in {aug}: meta={len(meta)}, result={len(result)}")
        if "GT" not in meta.columns:
            raise ValueError(f"'GT' column missing in metadata: {meta_path}")
        if "final_answer" not in result.columns:
            raise ValueError(f"'final_answer' column missing in results: {res_path}")

        data_dict[aug] = {"meta": meta, "result": result}

    # Ensure all augmentations have the same number of samples
    lengths = [len(data_dict[aug]["meta"]) for aug in VIDEO_AUGS]
    if len(set(lengths)) > 1:
        raise ValueError(f"Inconsistent sample counts: {dict(zip(VIDEO_AUGS, lengths))}")

    return data_dict

def sample_wise_evaluation():
    """
    Treat all samples across all augmentations as independent.
    Compute per-category and overall accuracy.
    """
    print("=== Method 1: Independent Samples ===")
    data_dict = load_all_data()

    records = []
    for aug in VIDEO_AUGS:
        meta = data_dict[aug]["meta"]
        res = data_dict[aug]["result"]
        for i in range(len(meta)):
            gt = str(meta.iloc[i]["GT"]).strip()
            pred = str(res.iloc[i]["final_answer"]).strip()
            pred_letter = pred[0] if pred else ""
            correct = int(gt == pred_letter)
            records.append({"cate": meta.iloc[i]["cate"], "correct": correct})

    df = pd.DataFrame(records)
    acc_by_cat = df.groupby("cate")["correct"].mean()
    overall_acc = df["correct"].mean()
    print_metrics(acc_by_cat, overall_acc)

def group_wise_evaluation(n: int):
    """
    For each original question (4 views), count how many augmented views are correct
    under their own ground truth. If >= n are correct, count as robustly correct.
    """
    print(f"=== Method 2: Ensemble Voting (n>={n}) ===")
    data_dict = load_all_data()
    num_samples = len(data_dict[VIDEO_AUGS[0]]["meta"])

    records = []
    total_robust_correct = 0

    for i in range(num_samples):
        correct_count = 0
        cate = None
        for aug in VIDEO_AUGS:
            meta = data_dict[aug]["meta"]
            res = data_dict[aug]["result"]
            gt = str(meta.iloc[i]["GT"]).strip()
            pred = str(res.iloc[i]["final_answer"]).strip()
            pred_letter = pred[0] if pred else ""
            if gt == pred_letter:
                correct_count += 1
            if cate is None:
                cate = meta.iloc[i]["cate"]

        is_robust_correct = int(correct_count >= n)
        records.append({"cate": cate, "correct": is_robust_correct})
        total_robust_correct += is_robust_correct

    df = pd.DataFrame(records)
    acc_by_cat = df.groupby("cate")["correct"].mean()
    overall_acc = total_robust_correct / num_samples
    print_metrics(acc_by_cat, overall_acc)

def single_evaluation(aug: str):
    """
    Evaluate performance on a single augmentation variant.
    """
    if aug not in VIDEO_AUGS:
        raise ValueError(f"Invalid augmentation: {aug}. Choose from {VIDEO_AUGS}")

    print(f"=== Method 3: Single View Evaluation ({aug}) ===")
    data_dict = load_all_data()
    meta = data_dict[aug]["meta"]
    res = data_dict[aug]["result"]

    correct_list = []
    for i in range(len(meta)):
        gt = str(meta.iloc[i]["GT"]).strip()
        pred = str(res.iloc[i]["final_answer"]).strip()
        pred_letter = pred[0] if pred else ""
        correct_list.append(int(gt == pred_letter))

    df = pd.DataFrame({"cate": meta["cate"], "correct": correct_list})
    acc_by_cat = df.groupby("cate")["correct"].mean()
    overall_acc = df["correct"].mean()
    print_metrics(acc_by_cat, overall_acc)

def print_metrics(acc_by_cat: pd.Series, overall_acc: float):
    """
    Print per-category and overall accuracy in a readable format.
    """
    for cat, ratio in acc_by_cat.items():
        name = CATE_NAMES[cat] if cat < len(CATE_NAMES) else f"Category {cat}"
        print('category = {0} {1:<20}  Acc = {2:.2%}'.format(cat, name, ratio))
    print(f'\nOverall Acc = {overall_acc:.2%}\n')

if __name__ == "__main__":
    print(f"Model: {VLM_MODEL}")

    sample_wise_evaluation()
    group_wise_evaluation(n=3)


```


## Citation

If you find this repository useful for your research, please use the following.
```

```