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
    try:
        # Some accelerate versions export load_checkpoint_and_dispatch in big_modeling
        from accelerate.big_modeling import load_checkpoint_and_dispatch
    except Exception:
        load_checkpoint_and_dispatch = None

# set HAS_DISPATCH from whatever import succeeded
HAS_DISPATCH = load_checkpoint_and_dispatch is not None
import torch

# Try to import HF accelerate dispatch utilities if available
from accelerate import init_empty_weights

from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import importlib.util
_utils_path = os.path.join(os.path.dirname(__file__), 'utils.py')
spec = importlib.util.spec_from_file_location('local_utils', _utils_path)
_local_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_local_utils)
parse_song_file = _local_utils.parse_song_file
slugify = _local_utils.slugify

parser = argparse.ArgumentParser(description='Generate using accelerate FSDP/offload techniques')
parser.add_argument('--model', choices=['small', 'medium', 'large'], default='large')
parser.add_argument('--prompt', default='a dreamy synth loop')
parser.add_argument('--duration', type=int, default=6)
parser.add_argument('--output', default=None, help='Output wav path. If not provided and --song-file is set, filename uses the song title')
parser.add_argument('--song-file', default='/workspace/lyrics_terraform_my_heart.txt', help='Path to song text file containing title, lyrics, and style sections')
parser.add_argument('--offload_folder', default='/workspace/offload')
args = parser.parse_args()


def main():
    print('Accelerate FSDP example - model:', args.model)
    accelerator = Accelerator()
    device = accelerator.device
    print('HAS_DISPATCH', HAS_DISPATCH, 'load_checkpoint_and_dispatch present', load_checkpoint_and_dispatch is not None)
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
        # For large: warn clearly if dispatch helpers are missing.
        if args.model == 'large':
            if load_checkpoint_and_dispatch is None:
                print('Warning: accelerate version does not have `load_checkpoint_and_dispatch` API; '
                      'cannot dispatch/auto-offload large models reliably. Falling back to normal load (may OOM).')
            else:
                # HAS_DISPATCH is False or some other condition prevented using the dispatch path.
                print('Warning: accelerate dispatch helpers could not be used for large; '
                      'falling back to standard get_pretrained() load (may OOM).')
        else:
            # For small/medium this is expected: we always use the standard load path.
            print('Using standard get_pretrained() load for model', args.model)

        model = MusicGen.get_pretrained(mname)

    # If a song file is provided, parse it to build prompt and output filename
    if args.song_file and os.path.exists(args.song_file):
        try:
            title, lyrics, styles = parse_song_file(args.song_file)
            # choose style for model
            style_section = ''
            if args.model == 'small':
                style_section = styles.get('small', '')
            elif args.model == 'medium':
                style_section = styles.get('medium', '')
            else:
                style_section = styles.get('large', '')
            # Compose prompt: lyrics + style hint (if both present)
            prompt_parts = []
            if lyrics:
                prompt_parts.append(lyrics)
            if style_section:
                prompt_parts.append('\nStyle: ' + style_section)
            if prompt_parts:
                args.prompt = '\n\n'.join(prompt_parts)
            # default output path using title if not explicitly set
            if args.output is None:
                slug = slugify(title)
                args.output = f'/workspace/output/{slug}_{args.model}.wav'
        except Exception:
            pass

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
    out_path = args.output if args.output is not None else '/workspace/out_fsdp.wav'
    audio_write(out_path.replace('.wav', ''), wavs[0], model.sample_rate, strategy='loudness')
    print('Saved to', out_path)


if __name__ == '__main__':
    main()
