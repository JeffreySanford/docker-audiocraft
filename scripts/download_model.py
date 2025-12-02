#!/usr/bin/env python3
"""Download a Hugging Face repo snapshot into the repository `model-cache` folder.
Usage:
  python scripts/download_model.py facebook/musicgen-style
This will place files under ./model-cache/models--<owner>--<repo>/
"""
import sys
import os
from huggingface_hub import snapshot_download

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: download_model.py <hf_repo_id> [--cache-dir path]')
        sys.exit(1)
    repo_id = sys.argv[1]
    cache_dir = 'model-cache'
    if '--cache-dir' in sys.argv:
        i = sys.argv.index('--cache-dir')
        cache_dir = sys.argv[i+1]
    os.makedirs(cache_dir, exist_ok=True)
    print(f'Downloading {repo_id} into {cache_dir}...')
    path = snapshot_download(repo_id=repo_id, cache_dir=cache_dir, repo_type='model')
    print('Downloaded to:', path)
