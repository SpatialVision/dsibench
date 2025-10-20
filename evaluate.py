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