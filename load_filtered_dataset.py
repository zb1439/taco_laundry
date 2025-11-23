"""
Load filtered TACO training dataset based on stored checkpoint.
"""
import json
from pathlib import Path
from datasets import Dataset, concatenate_datasets, DatasetDict

root = Path(".")
CHECKPOINT_FILE = root / "filter_checkpoint.json"


def load_checkpoint():
    """Load checkpoint file."""
    if not CHECKPOINT_FILE.exists():
        raise FileNotFoundError(
            f"Checkpoint file not found: {CHECKPOINT_FILE}\n"
            "Please run filter_dataset.py first to create the checkpoint."
        )
    
    with open(CHECKPOINT_FILE, 'r') as f:
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


def load_filtered_dataset(split="train", include_test=True):
    """
    Load filtered training dataset (and optionally test dataset).
    
    Args:
        split: Which split to filter ("train" or "all")
        include_test: Whether to include test dataset in the result
    
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
    
    result = {"train": filtered_train}
    
    # Optionally include test dataset
    if include_test:
        test_file = root / "test" / "data-00000-of-00001.arrow"
        if test_file.exists():
            test_ds = Dataset.from_file(str(test_file))
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


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Load filtered TACO dataset")
    parser.add_argument("--stats", action="store_true",
                        help="Show filtering statistics")
    parser.add_argument("--no-test", action="store_true",
                        help="Exclude test dataset")
    
    args = parser.parse_args()
    
    if args.stats:
        get_filter_statistics()
    else:
        ds = load_filtered_dataset(include_test=not args.no_test)
        print(f"\nLoaded dataset:")
        print(f"  Train: {len(ds['train'])} samples")
        if "test" in ds:
            print(f"  Test: {len(ds['test'])} samples")
        
        # Example: print first sample
        print("\nFirst sample:")
        sample = ds["train"][0]
        print(f"  Question: {sample['question'][:100]}...")
        print(f"  Tags: {sample.get('tags', 'N/A')}")

