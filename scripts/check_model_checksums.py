#!/usr/bin/env python3
"""
Model Cache Integrity Checker

This script verifies the integrity of cached AudioCraft models by computing
SHA256 checksums of model files and comparing them to expected values.

Usage:
    python check_model_checksums.py [--update] [--cache-dir CACHE_DIR]

Options:
    --update: Update the stored checksums with current values
    --cache-dir: Path to the model cache directory (default: /workspace/cache)

Models checked:
- facebook/musicgen-small
- facebook/musicgen-medium
- facebook/musicgen-large
"""

import os
import sys
import hashlib
import json
import argparse
from pathlib import Path
from typing import Dict, List

# Expected models and their cache subdirectories
MODELS = {
    'facebook/musicgen-small': 'models--facebook--musicgen-small',
    'facebook/musicgen-medium': 'models--facebook--musicgen-medium',
    'facebook/musicgen-large': 'models--facebook--musicgen-large',
}

# File to store checksums
CHECKSUM_FILE = 'model_checksums.json'

def compute_file_sha256(file_path: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    try:
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    except Exception as e:
        print(f"Error computing hash for {file_path}: {e}")
        return None

def compute_model_checksum(model_dir: Path) -> Dict[str, str]:
    """Compute checksums for all files in a model directory."""
    checksums = {}
    if not model_dir.exists():
        return checksums

    for file_path in model_dir.rglob('*'):
        if file_path.is_file():
            rel_path = file_path.relative_to(model_dir)
            checksum = compute_file_sha256(file_path)
            if checksum:
                checksums[str(rel_path)] = checksum

    return checksums

def load_stored_checksums(checksum_file: Path) -> Dict[str, Dict[str, str]]:
    """Load stored checksums from file."""
    if not checksum_file.exists():
        return {}

    try:
        with open(checksum_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading checksums: {e}")
        return {}

def save_checksums(checksum_file: Path, checksums: Dict[str, Dict[str, str]]):
    """Save checksums to file."""
    try:
        checksum_file.parent.mkdir(parents=True, exist_ok=True)
        with open(checksum_file, 'w') as f:
            json.dump(checksums, f, indent=2)
        print(f"Checksums saved to {checksum_file}")
    except Exception as e:
        print(f"Error saving checksums: {e}")

def check_model_integrity(cache_dir: Path, model_name: str, model_cache_name: str,
                         stored_checksums: Dict[str, str]) -> bool:
    """Check if model files match stored checksums."""
    model_dir = cache_dir / 'huggingface' / 'hub' / model_cache_name
    if not model_dir.exists():
        print(f"Model directory not found: {model_dir}")
        return False

    current_checksums = compute_model_checksum(model_dir)
    if not current_checksums:
        print(f"No files found in {model_dir}")
        return False

    mismatches = []
    missing_files = []
    extra_files = []

    # Check existing files
    for rel_path, stored_hash in stored_checksums.items():
        if rel_path in current_checksums:
            if current_checksums[rel_path] != stored_hash:
                mismatches.append(rel_path)
        else:
            missing_files.append(rel_path)

    # Check for extra files
    for rel_path in current_checksums:
        if rel_path not in stored_checksums:
            extra_files.append(rel_path)

    if mismatches or missing_files or extra_files:
        print(f"Integrity check FAILED for {model_name}:")
        if mismatches:
            print(f"  Modified files: {len(mismatches)}")
            for f in mismatches[:5]:  # Show first 5
                print(f"    {f}")
            if len(mismatches) > 5:
                print(f"    ... and {len(mismatches) - 5} more")
        if missing_files:
            print(f"  Missing files: {len(missing_files)}")
        if extra_files:
            print(f"  Extra files: {len(extra_files)}")
        return False
    else:
        print(f"Integrity check PASSED for {model_name}")
        return True

def main():
    parser = argparse.ArgumentParser(description='Check model cache integrity')
    parser.add_argument('--update', action='store_true',
                       help='Update stored checksums with current values')
    parser.add_argument('--cache-dir', type=str, default='/workspace/cache',
                       help='Path to model cache directory')
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    checksum_file = cache_dir / CHECKSUM_FILE

    print(f"Checking models in cache directory: {cache_dir}")

    if args.update:
        print("Updating checksums...")
        checksums = {}
        for model_name, model_cache_name in MODELS.items():
            model_dir = cache_dir / 'huggingface' / 'hub' / model_cache_name
            if model_dir.exists():
                print(f"Computing checksums for {model_name}...")
                checksums[model_name] = compute_model_checksum(model_dir)
            else:
                print(f"Model directory not found for {model_name}: {model_dir}")
                checksums[model_name] = {}
        save_checksums(checksum_file, checksums)
        print("Update complete.")
        return

    # Load stored checksums
    stored_checksums = load_stored_checksums(checksum_file)
    if not stored_checksums:
        print(f"No stored checksums found at {checksum_file}")
        print("Run with --update to create initial checksums.")
        return

    # Check each model
    all_passed = True
    for model_name, model_cache_name in MODELS.items():
        if model_name in stored_checksums:
            passed = check_model_integrity(cache_dir, model_name, model_cache_name,
                                          stored_checksums[model_name])
            if not passed:
                all_passed = False
        else:
            print(f"No stored checksums for {model_name}")
            all_passed = False

    if all_passed:
        print("\nAll models passed integrity check.")
    else:
        print("\nSome models failed integrity check.")
        sys.exit(1)

if __name__ == '__main__':
    main()