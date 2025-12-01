#!/usr/bin/env python3
"""
Simple handler to run quick generations inside the Docker image.
This is intentionally minimal: load a small pretrained MusicGen and generate a short clip.
"""
import sys
from audiocraft.models import MusicGen
import torch
from audiocraft.data.audio import audio_write

PROMPT = "a short piano loop" if len(sys.argv) < 2 else " ".join(sys.argv[1:])

if __name__ == '__main__':
    print('Loading model...')
    model = MusicGen.get_pretrained('facebook/musicgen-small')
    if torch.cuda.is_available():
        model = model.cuda().half()
    model.set_generation_params(duration=4)
    wavs = model.generate([PROMPT])
    audio_write('out_handler', wavs[0], model.sample_rate, strategy='loudness')
    print('Saved out_handler.wav')
