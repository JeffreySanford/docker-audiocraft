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
    parser.add_argument('--model-id', default='style', help='Model id for MusicGen (e.g. "style")')
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--duration', type=float, default=10.0)
    parser.add_argument('--output', required=True)
    parser.add_argument('--style-file', default=None, help='Optional short audio file (1.5-4.5s) for style conditioning')
    parser.add_argument('--cfg_coef', type=float, default=None, help='Optional CFG coefficient for style conditioning')
    parser.add_argument('--cfg_coef_2', type=float, default=None, help='Optional CFG coefficient for text conditioning')
    args = parser.parse_args()

    try:
        import torchaudio
        from audiocraft.models import MusicGen
        from audiocraft.data.audio import audio_write
    except Exception as e:
        print('Error importing audiocraft/torchaudio. Please install audiocraft per the README: pip install git+https://github.com/facebookresearch/audiocraft.git', file=sys.stderr)
        raise

    print(f'Loading model: {args.model_id}')
    model = MusicGen.get_pretrained(args.model_id)
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
