#!/usr/bin/env python3
import os
import time
import shutil
from pathlib import Path
import hashlib

cache_dir = os.environ.get('AUDIOCRAFT_CACHE_DIR', '/workspace/cache')
hf_home = os.environ.get('HF_HOME', '/workspace/cache')

def is_old(path, days=30):
    if not path.exists():
        return False
    mtime = path.stat().st_mtime
    now = time.time()
    age = now - mtime
    return age > days * 24 * 3600

def compute_hash(file_path):
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()

# Check and refresh caches
hf_cache_dir = Path(hf_home) / 'huggingface'
t5_cache = hf_cache_dir / 'models--t5-base'
musicgen_small_cache = hf_cache_dir / 'models--facebook--musicgen-small'
musicgen_medium_cache = hf_cache_dir / 'models--facebook--musicgen-medium'
musicgen_large_cache = hf_cache_dir / 'models--facebook--musicgen-large'

caches_to_check = [t5_cache, musicgen_small_cache, musicgen_medium_cache, musicgen_large_cache]

for cache in caches_to_check:
    if cache.exists():
        hash_file = cache / '.hash'
        if hash_file.exists():
            with open(hash_file, 'r') as f:
                stored_hash = f.read().strip()
            # Find the model file, assume safetensors or bin
            model_file = None
            for f in cache.rglob('*.safetensors'):
                model_file = f
                break
            if not model_file:
                for f in cache.rglob('*.bin'):
                    model_file = f
                    break
            if model_file:
                current_hash = compute_hash(model_file)
                if current_hash != stored_hash:
                    print(f"Hash mismatch for {cache}, removing to refresh")
                    shutil.rmtree(cache, ignore_errors=True)
                elif is_old(cache):
                    print(f"Cache {cache} is older than 30 days, removing to force refresh")
                    shutil.rmtree(cache, ignore_errors=True)
            else:
                if is_old(cache):
                    print(f"Cache {cache} old, removing")
                    shutil.rmtree(cache, ignore_errors=True)
        else:
            if is_old(cache):
                print(f"Cache {cache} old, removing")
                shutil.rmtree(cache, ignore_errors=True)

# Also check AudioCraft cache
audiocraft_cache = Path(cache_dir) / 'audiocraft'
if audiocraft_cache.exists() and is_old(audiocraft_cache, days=30):
    print("AudioCraft cache old, removing")
    shutil.rmtree(audiocraft_cache, ignore_errors=True)

# Now preload the models
print("Preloading models...")
from audiocraft.models.loaders import load_lm_model_ckpt, load_compression_model_ckpt
from transformers import T5EncoderModel, AutoTokenizer

models = ['facebook/musicgen-small', 'facebook/musicgen-medium', 'facebook/musicgen-large']

for model in models:
    print(f"Loading LM and compression for {model}")
    lm = load_lm_model_ckpt(model, cache_dir=cache_dir)
    comp = load_compression_model_ckpt(model, cache_dir=cache_dir)

print("Loading T5 model")
t5 = T5EncoderModel.from_pretrained('t5-base')
tokenizer = AutoTokenizer.from_pretrained('t5-base')

# After loading, save hashes
print("Saving checksums...")
for cache in caches_to_check:
    if cache.exists():
        hash_file = cache / '.hash'
        model_file = None
        for f in cache.rglob('*.safetensors'):
            model_file = f
            break
        if not model_file:
            for f in cache.rglob('*.bin'):
                model_file = f
                break
        if model_file:
            current_hash = compute_hash(model_file)
            with open(hash_file, 'w') as f:
                f.write(current_hash)

print("Preload completed successfully")