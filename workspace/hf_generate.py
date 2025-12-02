#!/usr/bin/env python3
"""Simple wrapper to run Hugging Face / AudioCraft style models (MusicGen-Style).

Usage:
  python workspace/hf_generate.py --model-id style --prompt "text" --duration 20 --output /workspace/out.wav [--style-file /path/to/sample.wav]

This script uses the `audiocraft` library per the MusicGen-Style README.
It supports using a short style audio snippet (1.5-4.5s) to condition generation.
"""
import argparse
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-id', default=None, help='HuggingFace model id (e.g. "facebook/musicgen-style")')
    parser.add_argument('--model-key', default=None, help='Optional key into workspace/models_manifest.json to select a cached model')
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--duration', type=float, default=10.0)
    parser.add_argument('--output', required=True)
    parser.add_argument('--style-file', default=None, help='Optional short audio file (1.5-4.5s) for style conditioning')
    parser.add_argument('--cfg_coef', type=float, default=None, help='Optional CFG coefficient for style conditioning')
    parser.add_argument('--cfg_coef_2', type=float, default=None, help='Optional CFG coefficient for text conditioning')
    args = parser.parse_args()

    try:
        import json
        import os
        import torchaudio
        from audiocraft.models import MusicGen
        from audiocraft.data.audio import audio_write
    except Exception as e:
        print('Error importing audiocraft/torchaudio. Please install audiocraft per the README: pip install git+https://github.com/facebookresearch/audiocraft.git', file=sys.stderr)
        raise

    # Determine model source: manifest key -> local path -> hf id
    model_source = None
    manifest_path = os.path.join(os.path.dirname(__file__), 'models_manifest.json')
    if args.model_key and os.path.exists(manifest_path):
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
            entry = manifest.get(args.model_key)
            if entry:
                # prefer local_path if it exists
                lp = entry.get('local_path')
                if lp:
                    # try absolute path as-is
                    if os.path.isabs(lp) and os.path.exists(lp):
                        model_source = lp
                    else:
                        # common container mount for caches is /model-cache
                        alt = os.path.join('/model-cache', os.path.basename(lp))
                        if os.path.exists(alt):
                            model_source = alt
                # fallback to hf_id
                if model_source is None and entry.get('hf_id'):
                    model_source = entry.get('hf_id')
        except Exception:
            model_source = None

    # If model_source still None, use explicit model_id if provided
    if model_source is None:
        if args.model_id:
            model_source = args.model_id
        else:
            # default to 'style' if nothing provided (backwards compat)
            model_source = 'style'

    print(f'Loading model source: {model_source}')
    model = MusicGen.get_pretrained(model_source)
    model.set_generation_params(duration=args.duration)

    prompt = args.prompt
    wavs = None

    if args.style_file:
        print(f'Loading style file: {args.style_file}')
        melody, sr = torchaudio.load(args.style_file)
        # model.generate_with_chroma expects shape (channels, samples); README used melody[None].expand(3, -1, -1)
        # Expand to 3 channels if needed
        if melody.dim() == 1:
            melody = melody.unsqueeze(0)
        try:
            cond = melody[None].expand(3, -1, -1)
        except Exception:
            # fallback: replicate channels
            cond = melody.repeat(3, 1, 1)
        print('Generating with style conditioning...')
        # generate_with_chroma returns a batch of wavs
        wavs = model.generate_with_chroma([prompt], cond, sr)
    else:
        print('Generating from text only...')
        wavs = model.generate([prompt])

    # Save first result
    if isinstance(wavs, (list, tuple)):
        one_wav = wavs[0]
    else:
        one_wav = wavs

    print(f'Writing output to {args.output}')
    audio_write(args.output, one_wav.cpu(), model.sample_rate, strategy='loudness')

if __name__ == '__main__':
    main()
