"""
Load filtered TACO training dataset based on stored checkpoint.
Supports loading with rewritten solutions.
"""
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from datasets import Dataset, concatenate_datasets, DatasetDict

root = Path(".")
CHECKPOINT_FILE = root / "filter_checkpoint.json"
REWRITE_CHECKPOINT_FILE = root / "rewrite_checkpoint.json"


def load_checkpoint():
    """Load checkpoint file."""
    if not CHECKPOINT_FILE.exists():
        raise FileNotFoundError(
            f"Checkpoint file not found: {CHECKPOINT_FILE}\n"
            "Please run filter_dataset.py first to create the checkpoint."
        )
    
    with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_kept_indices(checkpoint=None):
    """
    Get list of indices to keep from checkpoint.
    
    Args:
        checkpoint: Optional checkpoint dict. If None, loads from file.
    
    Returns:
        List of integer indices to keep, sorted.
    """
    if checkpoint is None:
        checkpoint = load_checkpoint()
    
    kept_indices = checkpoint.get("kept_indices", [])
    
    # Also check processed_indices to ensure we have all kept samples
    # (in case kept_indices list is incomplete)
    if "processed_indices" in checkpoint:
        for idx_str, data in checkpoint["processed_indices"].items():
            if data.get("decision") == "keep":
                idx = int(idx_str)
                if idx not in kept_indices:
                    kept_indices.append(idx)
    
    return sorted(kept_indices)


def load_rewrite_checkpoint():
    """Load rewrite checkpoint if it exists."""
    if not REWRITE_CHECKPOINT_FILE.exists():
        return None
    
    with open(REWRITE_CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_rewritten_solutions_for_sample(rewrite_checkpoint: Dict, original_idx: int) -> Optional[List[Dict[str, Any]]]:
    """
    Get rewritten solutions for a sample by its original index.
    
    Args:
        rewrite_checkpoint: Rewrite checkpoint dictionary
        original_idx: Original index in the filtered dataset
    
    Returns:
        List of rewritten solutions or None if not found
    """
    if rewrite_checkpoint is None:
        return None
    
    processed_samples = rewrite_checkpoint.get("processed_samples", {})
    sample_data = processed_samples.get(str(original_idx))
    
    if sample_data is None or sample_data.get("status") != "completed":
        return None
    
    rewritten_solutions = sample_data.get("rewritten_solutions", [])
    
    # Filter to only verified solutions if requested
    verified_solutions = [
        sol for sol in rewritten_solutions
        if sol.get("verification_passed") is True and sol.get("rewritten_code") is not None
    ]
    
    return verified_solutions if verified_solutions else None


def add_rewritten_solutions_to_dataset(dataset: Dataset, rewrite_checkpoint: Optional[Dict] = None) -> Dataset:
    """
    Add rewritten solutions as a new field to the dataset.
    
    Args:
        dataset: The dataset to enhance
        rewrite_checkpoint: Rewrite checkpoint dictionary (loaded if None)
    
    Returns:
        Dataset with 'rewritten_solutions' field added
    """
    if rewrite_checkpoint is None:
        rewrite_checkpoint = load_rewrite_checkpoint()
    
    if rewrite_checkpoint is None:
        print("Warning: No rewrite checkpoint found. Returning dataset without rewritten solutions.")
        return dataset
    
    # Create mapping from dataset index to rewritten solutions
    rewritten_solutions_list = []
    
    for idx in range(len(dataset)):
        solutions = get_rewritten_solutions_for_sample(rewrite_checkpoint, idx)
        if solutions:
            # Extract just the code
            rewritten_solutions_list.append([sol["rewritten_code"] for sol in solutions])
        else:
            rewritten_solutions_list.append([])
    
    # Add as new column
    dataset = dataset.add_column("rewritten_solutions", rewritten_solutions_list)
    
    return dataset


def load_filtered_dataset(include_test=True, include_rewritten_solutions: bool = False):
    """
    Load filtered training dataset (and optionally test dataset).
    
    Args:
        split: Which split to filter ("train" or "all")
        include_test: Whether to include test dataset in the result
        include_rewritten_solutions: Whether to include rewritten solutions in the dataset
    
    Returns:
        DatasetDict with filtered train (and optionally test) datasets
    """
    # Load checkpoint
    checkpoint = load_checkpoint()
    
    # Get kept indices
    kept_indices = get_kept_indices(checkpoint)
    
    print(f"Loading dataset with {len(kept_indices)} kept samples "
          f"(out of {checkpoint.get('total_samples', 'unknown')} total)")
    
    # Load training dataset
    train_files = sorted((root / "train").glob("data-*.arrow"))
    train_datasets = [Dataset.from_file(str(f)) for f in train_files]
    train_ds = concatenate_datasets(train_datasets)
    
    # Filter training dataset
    filtered_train = train_ds.select(kept_indices)
    
    # Add rewritten solutions if requested
    if include_rewritten_solutions:
        print("Loading rewritten solutions...")
        filtered_train = add_rewritten_solutions_to_dataset(filtered_train)
        verified_count = sum(1 for i in range(len(filtered_train)) if len(filtered_train[i].get("rewritten_solutions", [])) > 0)
        print(f"Found rewritten solutions for {verified_count} samples")
    
    result = {"train": filtered_train}
    
    # Optionally include test dataset
    if include_test:
        test_file = root / "test" / "data-00000-of-00001.arrow"
        if test_file.exists():
            test_ds = Dataset.from_file(str(test_file))
            if include_rewritten_solutions:
                # Note: test dataset typically doesn't have rewritten solutions
                test_ds = add_rewritten_solutions_to_dataset(test_ds)
            result["test"] = test_ds
        else:
            print("Warning: Test dataset file not found, skipping test split")
    
    return DatasetDict(result)


def get_filter_statistics():
    """Print statistics about the filtering process."""
    checkpoint = load_checkpoint()
    
    kept_indices = get_kept_indices(checkpoint)
    total_processed = len(checkpoint.get("processed_indices", {}))
    total_samples = checkpoint.get("total_samples", 0)
    
    print("="*60)
    print("Filter Statistics")
    print("="*60)
    print(f"Total samples in dataset: {total_samples}")
    print(f"Samples processed: {total_processed}")
    print(f"Samples kept: {len(kept_indices)}")
    print(f"Samples filtered out: {total_processed - len(kept_indices)}")
    if total_processed > 0:
        print(f"Keep rate: {100*len(kept_indices)/total_processed:.1f}%")
    print(f"Checkpoint file: {CHECKPOINT_FILE}")
    print("="*60)


def get_rewrite_statistics():
    """Print statistics about the rewriting process."""
    rewrite_checkpoint = load_rewrite_checkpoint()
    
    if rewrite_checkpoint is None:
        print("No rewrite checkpoint found.")
        return
    
    processed_samples = rewrite_checkpoint.get("processed_samples", {})
    total_samples = rewrite_checkpoint.get("total_samples", 0)
    
    completed_count = sum(1 for s in processed_samples.values() if s.get("status") == "completed")
    verified_count = sum(1 for s in processed_samples.values() 
                         if s.get("status") == "completed" and s.get("verified_count", 0) > 0)
    skipped_count = sum(1 for s in processed_samples.values() if s.get("status") == "skipped")
    
    total_rewritten = sum(s.get("total_count", 0) for s in processed_samples.values() 
                          if s.get("status") == "completed")
    total_verified = sum(s.get("verified_count", 0) for s in processed_samples.values() 
                         if s.get("status") == "completed")
    
    print("="*60)
    print("Rewrite Statistics")
    print("="*60)
    print(f"Total samples in dataset: {total_samples}")
    print(f"Samples processed: {len(processed_samples)}")
    print(f"Samples completed: {completed_count}")
    print(f"Samples with verified solutions: {verified_count}")
    print(f"Samples skipped: {skipped_count}")
    print(f"Total solutions rewritten: {total_rewritten}")
    print(f"Total solutions verified: {total_verified}")
    if total_rewritten > 0:
        print(f"Verification rate: {100*total_verified/total_rewritten:.1f}%")
    print(f"Checkpoint file: {REWRITE_CHECKPOINT_FILE}")
    print("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Load filtered TACO dataset")
    parser.add_argument("--stats", action="store_true",
                        help="Show filtering statistics")
    parser.add_argument("--rewrite-stats", action="store_true",
                        help="Show rewriting statistics")
    parser.add_argument("--no-test", action="store_true",
                        help="Exclude test dataset")
    parser.add_argument("--include-rewritten", action="store_true",
                        help="Include rewritten solutions in the dataset")
    
    args = parser.parse_args()
    
    if args.stats:
        get_filter_statistics()
    elif args.rewrite_stats:
        get_rewrite_statistics()
    else:
        ds = load_filtered_dataset(
            include_test=not args.no_test,
            include_rewritten_solutions=args.include_rewritten
        )
        print(f"\nLoaded dataset:")
        print(f"  Train: {len(ds['train'])} samples")
        if "test" in ds:
            print(f"  Test: {len(ds['test'])} samples")
        
        # Example: print first sample
        print("\nFirst sample:")
        sample = ds["train"][0]
        print(f"  Question: {sample['question'][:100]}...")
        print(f"  Tags: {sample.get('tags', 'N/A')}")
        if args.include_rewritten and "rewritten_solutions" in sample:
            print(f"  Rewritten solutions: {len(sample['rewritten_solutions'])}")

