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
FPS = 6                                 # Frames per second for video input

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