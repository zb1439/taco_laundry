"""
Filter TACO training dataset based on OpenAI API evaluation.
Supports resumable execution with checkpointing.
"""
import json
import time
import re
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm
from datasets import Dataset, concatenate_datasets

# Configuration
root = Path(".")
CHECKPOINT_FILE = root / "filter_checkpoint.json"
RUBRICS = """
I will provide you a programming question and possibly a Python code solution. You need to determine whether this
question is challenging to evaluate using traditional unit tests. Apply the following criteria to
identify questions that are hard to evaluate using unit tests:
1. Functions involving randomness or probability:
1) Random number generators; 2) Shuffling algorithms; 3) Probability-based functions
2. Time-dependent functions:
1) Functions that get the current time; 2) Timer functions
3. Functions relying on external resources:
1) Network request functions; 2) File system operations; 3) Database queries
4. Concurrency and multithreading functions:
1) Thread synchronization functions; 2) Concurrent operation functions
5. Hardware-related functions:
1) Device driver functions; 2) Hardware sensor reading functions
6. User interface related functions:
1) Graphics rendering functions; 2) User input processing functions
7. Functions with side effects:
1) Functions modifying global state; 2) Logging functions
8. Cryptography-related functions:
1) Functions generating encryption keys; 2) Certain encryption algorithm implementations
9. Machine learning and adaptive algorithm functions:
1) Model training functions; 2) Neural network backpropagation algorithms; 3) Self-tuning
algorithms
10. Complex mathematical or simulation functions:
1) High-precision floating-point calculations; 2) Physical simulations (e.g., fluid dynamics, particle
collisions); 3) Complex optimization algorithms

For the response, let's think step by step. If the question and answer meet the above criteria, please answer YES;
otherwise, answer NO. Please first give the reason for your judgments, followed by your decision.
Your decision should be in the last line of the reply, which ONLY contains one word: YES or NO.

Output Format:
- REASON: one-sentence justification
- YES or NO
"""


def fulfill_prompt(sample):
    """Create prompt from sample."""
    assert "question" in sample
    prompt = "### Programming Question\n" + sample["question"] + "\n\n"
    if "solutions" in sample and len(sample["solutions"]) > 0:
        prompt += "### Solution\n" + sample["solutions"][0] + "\n"
    return prompt


def parse_response(response_text):
    """
    Parse OpenAI response to extract YES/NO decision.
    Returns (decision, reason) where decision is True for YES, False for NO.
    """
    response_text = response_text.strip().upper()
    
    # Try to find YES or NO in the last line
    lines = response_text.split('\n')
    for line in reversed(lines):
        line = line.strip().upper()
        if line == "YES":
            # Extract reason from earlier lines
            reason = "\n".join(lines[:-1]) if len(lines) > 1 else ""
            return True, reason
        elif line == "NO":
            reason = "\n".join(lines[:-1]) if len(lines) > 1 else ""
            return False, reason
    
    # Fallback: search for YES/NO anywhere
    if "YES" in response_text and "NO" not in response_text:
        return True, response_text
    elif "NO" in response_text:
        return False, response_text
    
    # Default to NO (keep the sample) if unclear
    print(f"Warning: Could not parse response, defaulting to NO (keep): {response_text[:100]}")
    return False, response_text


def load_checkpoint():
    """Load checkpoint if it exists."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {
        "processed_indices": {},
        "kept_indices": [],
        "filtered_indices": [],
        "last_processed_idx": -1,
        "total_samples": 0
    }


def save_checkpoint(checkpoint):
    """Save checkpoint to file."""
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f, indent=2)


def call_openai_api(client, prompt, model_name, max_retries=3, retry_delay=5):
    """
    Call OpenAI API with retry logic.
    Returns (decision, reason) where decision=True means filter out (YES), False means keep (NO).
    """
    for attempt in range(max_retries):
        try:
            # Try chat completions API (standard format)
            response = client.chat.completions.create(
                model=model_name,  # Using a more standard model name
                messages=[
                    {"role": "system", "content": RUBRICS},
                    {"role": "user", "content": prompt}
                ],
            )
            response_text = response.choices[0].message.content
            is_hard_to_evaluate, reason = parse_response(response_text)
            return is_hard_to_evaluate, reason
        
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"API call failed after {max_retries} attempts: {e}")
                raise


def filter_dataset(
    start_idx=0,
    max_samples=None,
    batch_size=100,
    api_delay=0.1,
    model_name='gpt-5-nano'
):
    """
    Filter the training dataset using OpenAI API.
    
    Args:
        start_idx: Index to start from (for resuming)
        max_samples: Maximum number of samples to process (None for all)
        batch_size: Number of samples to process before saving checkpoint
        api_delay: Delay between API calls in seconds
        model_name: OpenAI model to use
    """
    # Load dataset
    print("Loading training dataset...")
    train_files = sorted((root / "train").glob("data-*.arrow"))
    train_datasets = [Dataset.from_file(str(f)) for f in train_files]
    train_ds = concatenate_datasets(train_datasets)
    
    total_samples = len(train_ds)
    print(f"Total samples in training set: {total_samples}")
    
    # Load checkpoint
    checkpoint = load_checkpoint()
    if checkpoint["total_samples"] == 0:
        checkpoint["total_samples"] = total_samples
    
    # Determine end index
    end_idx = total_samples
    if max_samples:
        end_idx = min(start_idx + max_samples, total_samples)
    
    # Initialize OpenAI client
    client = OpenAI()
    
    # Process samples
    processed_count = 0
    kept_count = 0
    filtered_count = 0
    
    print(f"Starting from index {start_idx}, processing until {end_idx}")
    print(f"Resuming from checkpoint: {len(checkpoint['processed_indices'])} samples already processed")
    
    for idx in tqdm(range(start_idx, end_idx)):
        # Skip if already processed
        if str(idx) in checkpoint["processed_indices"]:
            decision = checkpoint["processed_indices"][str(idx)]["decision"]
            if decision == "keep":
                kept_count += 1
            else:
                filtered_count += 1
            continue
        
        try:
            # Load and parse sample
            sample = train_ds[idx]
            sample = dict(sample)  # Convert to dict
            sample["solutions"] = json.loads(sample["solutions"])
            sample["input_output"] = json.loads(sample["input_output"])
            sample["raw_tags"] = eval(sample["raw_tags"])
            sample["tags"] = eval(sample["tags"])
            sample["skill_types"] = eval(sample["skill_types"])
            
            # Create prompt
            prompt = fulfill_prompt(sample)
            
            # Call OpenAI API
            is_hard_to_evaluate, reason = call_openai_api(client, prompt, model_name)
            
            # Decision: keep if NOT hard to evaluate (NO response)
            should_keep = not is_hard_to_evaluate
            
            # Update checkpoint
            checkpoint["processed_indices"][str(idx)] = {
                "decision": "keep" if should_keep else "filter",
                "reason": reason,
                "timestamp": time.time()
            }
            
            if should_keep:
                if idx not in checkpoint["kept_indices"]:
                    checkpoint["kept_indices"].append(idx)
                kept_count += 1
            else:
                if idx not in checkpoint["filtered_indices"]:
                    checkpoint["filtered_indices"].append(idx)
                filtered_count += 1
            
            checkpoint["last_processed_idx"] = idx
            processed_count += 1
            
            # Print progress
            if processed_count % 10 == 0:
                print(f"Processed {processed_count} samples | "
                      f"Kept: {kept_count} | Filtered: {filtered_count} | "
                      f"Progress: {idx+1}/{end_idx} ({100*(idx+1)/end_idx:.1f}%)")
            
            # Save checkpoint periodically
            if processed_count % batch_size == 0:
                save_checkpoint(checkpoint)
                print(f"Checkpoint saved at index {idx}")
            
            # Rate limiting
            time.sleep(api_delay)
            
        except KeyboardInterrupt:
            print("\nInterrupted by user. Saving checkpoint...")
            save_checkpoint(checkpoint)
            print(f"Checkpoint saved. Last processed index: {idx}")
            print(f"To resume, run with start_idx={idx+1}")
            return
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            print(f"Saving checkpoint before exiting...")
            save_checkpoint(checkpoint)
            raise
    
    # Final save
    save_checkpoint(checkpoint)
    
    # Print summary
    print("\n" + "="*60)
    print("Filtering complete!")
    print(f"Total processed: {processed_count}")
    print(f"Samples kept: {kept_count}")
    print(f"Samples filtered: {filtered_count}")
    print(f"Keep rate: {100*kept_count/(kept_count+filtered_count):.1f}%")
    print(f"Checkpoint saved to: {CHECKPOINT_FILE}")
    print("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Filter TACO training dataset")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Index to start from (for resuming)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to process")
    parser.add_argument("--batch_size", type=int, default=100,
                        help="Checkpoint save frequency")
    parser.add_argument("--api_delay", type=float, default=0.1,
                        help="Delay between API calls (seconds)")
    parser.add_argument("--model", type=str, default="gpt-5-nano",
                        help="OpenAI model to use")
    
    args = parser.parse_args()
    
    filter_dataset(
        start_idx=args.start_idx,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        api_delay=args.api_delay,
        model_name=args.model
    )

