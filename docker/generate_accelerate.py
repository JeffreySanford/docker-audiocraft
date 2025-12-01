#!/usr/bin/env python3
"""
Example script to run with `accelerate launch` and an accelerate `config` that enables
CPU offload for large models. This script is intentionally minimal and demonstrates how
an accelerator is used to manage device placement and mixed precision.

Usage:
accelerate launch generate_accelerate.py --model large --prompt "a jazz loop"
"""
from accelerate import Accelerator
import argparse
import torch
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

parser = argparse.ArgumentParser(description='Generate with accelerate')
parser.add_argument('--model', default='medium', choices=['small', 'medium', 'large'])
parser.add_argument('--prompt', default='a dreamy synth loop')
parser.add_argument('--duration', type=int, default=6)
args = parser.parse_args()

if __name__ == '__main__':
    # Accelerator will read the config from the environment / accelerate_config.yaml
    accelerator = Accelerator()
    device = accelerator.device
    print('Device managed by accelerator:', device)

    mname = {
        'small': 'facebook/musicgen-small',
        'medium': 'facebook/musicgen-medium',
        'large': 'facebook/musicgen-large'
    }[args.model]

    # Just demonstrate model loading and generation with accelerator device placement
    print('Loading model (this may use CPU offload as configured by accelerate)')
    model = MusicGen.get_pretrained(mname)

    # prepare for the accelerator
    model = accelerator.prepare(model)

    if torch.cuda.is_available():
        model = model.half()

    model.set_generation_params(duration=args.duration)

    with torch.no_grad():
        wav = model.generate([args.prompt])[0]

    path = f"/workspace/out_accelerate_{args.model}.wav"
    audio_write(path.replace('.wav', ''), wav, model.sample_rate)
    print('Saved', path)
