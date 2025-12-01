#!/usr/bin/env python3
"""
Demonstrate how to load a very large model (facebook/musicgen-large) using
accelerate and FSDP/CPU offload. This script attempts to use HF accelerate helpers
(e.g. init_empty_weights, load_checkpoint_and_dispatch) if available to avoid
loading the full model into GPU memory at once.

Usage:
  # Ensure ACCELERATE_CONFIG_FILE points to /workspace/accelerate_config.yaml
  accelerate launch --config_file /workspace/accelerate_config.yaml ./generate_fsdp.py --model large --prompt "a moody synthwave loop"

Notes:
  - This script uses best-effort utilities and falls back to a standard load if FSDP is not available.
  - You may still need plenty of CPU RAM to hold the model; `large` (3.3B) will not fit on a single 10GB GPU without offload.
"""

import argparse
import os
import sys
import time

from accelerate import Accelerator
from accelerate import init_empty_weights
try:
    from accelerate.utils import load_checkpoint_and_dispatch
except Exception:
    load_checkpoint_and_dispatch = None
import torch

# Try to import HF accelerate dispatch utilities if available
try:
    from accelerate import init_empty_weights
    from accelerate.utils import load_checkpoint_and_dispatch as load_and_dispatch
    HAS_DISPATCH = True
except Exception:
    HAS_DISPATCH = False

from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

parser = argparse.ArgumentParser(description='Generate using accelerate FSDP/offload techniques')
parser.add_argument('--model', choices=['small', 'medium', 'large'], default='large')
parser.add_argument('--prompt', default='a dreamy synth loop')
parser.add_argument('--duration', type=int, default=6)
parser.add_argument('--output', default='/workspace/out_fsdp.wav')
parser.add_argument('--offload_folder', default='/workspace/offload')
args = parser.parse_args()


def main():
    print('Accelerate FSDP example - model:', args.model)
    accelerator = Accelerator()
    device = accelerator.device
    print('Accelerator device:', device)

    mname = {
        'small': 'facebook/musicgen-small',
        'medium': 'facebook/musicgen-medium',
        'large': 'facebook/musicgen-large'
    }[args.model]

    # If accelerate helper functions are available we try a layout that avoids full GPU allocation
    if HAS_DISPATCH and args.model == 'large' and load_checkpoint_and_dispatch is not None:
        print('Using accelerate dispatch utilities to load model with offload/FSDP if available')
        try:
            print('Initializing empty weights context...')
            with init_empty_weights():
                model = MusicGen.get_pretrained(mname)

            print('Loading model weights with dispatch (device_map=auto) and offload folder:', args.offload_folder)
            # dispatch and load weights, relying on accelerate to place parameters
            load_checkpoint_and_dispatch(model, mname, device_map='auto', offload_folder=args.offload_folder)
        except Exception as e:
            print('Dispatch load failed:', e)
            print('Falling back to regular get_pretrained()')
            model = MusicGen.get_pretrained(mname)
    else:
        if load_checkpoint_and_dispatch is None and args.model == 'large':
            print('Warning: accelerate version does not have `load_checkpoint_and_dispatch` API; cannot dispatch/auto-offload large models reliably. Falling back to normal load (may OOM)')
        else:
            print('Accelerate dispatch helpers are unavailable, performing standard load.')
        model = MusicGen.get_pretrained(mname)

    # switch to half precision if device is CUDA
    if torch.cuda.is_available():
        try:
            model = model.half().to(device)
        except Exception:
            # If model is sharded/dispatch-managed, model.to(device) might fail; rely on accelerate to manage devices
            print('Could not `.to(device)` model directly after dispatch; relying on accelerate-managed devices')

    model.set_generation_params(duration=args.duration)

    print('Generating...')
    start = time.time()
    with torch.no_grad():
        wavs = model.generate([args.prompt])
    took = time.time() - start
    print(f"Generated {args.duration}s in {took:.2f}s")

    # Save
    audio_write(args.output.replace('.wav', ''), wavs[0], model.sample_rate, strategy='loudness')
    print('Saved to', args.output)


if __name__ == '__main__':
    main()
