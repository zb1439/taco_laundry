"""
Load filtered TACO training dataset based on stored checkpoint.
Supports loading with rewritten solutions.
"""
import json
from pathlib import Path
from typing import Dict, Any, Union
from datasets import Dataset, concatenate_datasets, DatasetDict

root = Path(".")


def load_checkpoint(checkpoint_file: Union[Path, str]):
    """Load checkpoint file."""
    if isinstance(checkpoint_file, str) and checkpoint_file.endswith(".json"):
        checkpoint_file = root / checkpoint_file

    if not checkpoint_file.exists():
        raise FileNotFoundError(
            f"Checkpoint file not found: {checkpoint_file}\n"
            "Please run filter_dataset.py first to create the checkpoint."
        )
    
    with open(checkpoint_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_kept_indices(checkpoint: Union[Path, Dict[str, Any]]):
    """
    Get list of indices to keep from checkpoint.
    
    Args:
        checkpoint: Optional checkpoint dict. If None, loads from file.
    
    Returns:
        List of integer indices to keep, sorted.
    """
    if isinstance(checkpoint, Path):
        checkpoint = load_checkpoint(checkpoint)
    
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


def load_filtered_dataset(checkpoint: Path, include_test=True):
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
    checkpoint = load_checkpoint(checkpoint)
    
    # Get kept indices
    kept_indices = get_kept_indices(checkpoint)
    
    print(f"Loading dataset with {len(kept_indices)} kept samples "
          f"(out of {checkpoint.get('total_samples', 'unknown')} total)")
    
    # Load training dataset
    train_files = sorted((root / "train").glob("data-*.arrow"))
    train_datasets = [Dataset.from_file(str(f)) for f in train_files]
    train_ds = concatenate_datasets(train_datasets)
    train_ds = train_ds.add_column("original_idx", range(len(train_ds)))
    
    # Filter training dataset
    filtered_train = train_ds.select(kept_indices)
    result = {"train": filtered_train}
    
    # Optionally include test dataset
    if include_test:
        test_file = root / "test" / "data-00000-of-00001.arrow"
        if test_file.exists():
            test_ds = Dataset.from_file(str(test_file))
            test_ds = test_ds.add_column("original_idx", range(len(test_ds)))
            result["test"] = test_ds
        else:
            print("Warning: Test dataset file not found, skipping test split")
    
    return DatasetDict(result)


def load_rewritten_results(
    rewrite_checkpoint_file: Union[Path, str] = "rewrite_dataset_checkpoint.json",
    only_completed: bool = True
) -> Dict[int, Dict[str, Any]]:
    """
    Load rewritten results from rewrite_dataset.py checkpoint.
    
    Args:
        rewrite_checkpoint_file: Path to rewrite checkpoint file (default: rewrite_dataset_checkpoint.json)
        only_completed: If True, only return results with status "completed"
    
    Returns:
        Dictionary mapping original_idx to result dictionaries
    """
    if isinstance(rewrite_checkpoint_file, str):
        rewrite_checkpoint_file = root / rewrite_checkpoint_file
    
    if not rewrite_checkpoint_file.exists():
        raise FileNotFoundError(
            f"Rewrite checkpoint file not found: {rewrite_checkpoint_file}\n"
            "Please run rewrite_dataset.py first to create the checkpoint."
        )
    
    checkpoint = load_checkpoint(rewrite_checkpoint_file)
    
    results_by_original_idx = {}
    
    if only_completed:
        # Only return completed results
        for idx_str, result in checkpoint.get("results_by_original_idx", {}).items():
            original_idx = int(idx_str)
            results_by_original_idx[original_idx] = result
    else:
        # Return all processed samples with their status
        for idx_str, data in checkpoint.get("processed_original_indices", {}).items():
            original_idx = int(idx_str)
            if data.get("status") == "completed" and "result" in data:
                results_by_original_idx[original_idx] = data["result"]
    
    return results_by_original_idx


def get_rewrite_statistics(rewrite_checkpoint_file: Union[Path, str] = "rewrite_dataset_checkpoint.json"):
    """Print statistics about the rewriting process."""
    if isinstance(rewrite_checkpoint_file, str):
        rewrite_checkpoint_file = root / rewrite_checkpoint_file
    
    checkpoint = load_checkpoint(rewrite_checkpoint_file)
    
    processed = checkpoint.get("processed_original_indices", {})
    completed = sum(1 for data in processed.values() if data.get("status") == "completed")
    skipped = sum(1 for data in processed.values() if data.get("status") == "skipped")
    total_results = len(checkpoint.get("results_by_original_idx", {}))
    total_samples = checkpoint.get("total_samples", 0)
    
    print("="*60)
    print("Rewrite Statistics")
    print("="*60)
    print(f"Total samples in filtered dataset: {total_samples}")
    print(f"Samples processed: {len(processed)}")
    print(f"Samples completed: {completed}")
    print(f"Samples skipped: {skipped}")
    print(f"Total results stored: {total_results}")
    if len(processed) > 0:
        print(f"Completion rate: {100*completed/len(processed):.1f}%")
    print(f"Checkpoint file: {rewrite_checkpoint_file}")
    print("="*60)


def get_filter_statistics(checkpoint_file):
    """Print statistics about the filtering process."""
    checkpoint = load_checkpoint(checkpoint_file)
    
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
    print(f"Checkpoint file: {checkpoint_file}")
    print("="*60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load filtered TACO dataset")
    parser.add_argument("--checkpoint-file", type=str, default="filter_checkpoint.json",
                        help="Checkpoint file to use")
    parser.add_argument("--stats", action="store_true",
                        help="Show filtering statistics")
    parser.add_argument("--no-test", action="store_true",
                        help="Exclude test dataset")

    args = parser.parse_args()

    checkpoint_file = root / args.checkpoint_file

    if args.stats:
        get_filter_statistics(checkpoint_file)
    else:
        ds = load_filtered_dataset(
            checkpoint=checkpoint_file,
            include_test=not args.no_test,
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

