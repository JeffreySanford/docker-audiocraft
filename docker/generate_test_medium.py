#!/usr/bin/env python3
"""
Simple test script: generate a short 4-6s audio with the medium model and save to out_medium.wav
"""
import time
import torch
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write


def generate_medium(prompt: str = "a relaxed piano loop", duration: int = 6):
    print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')
    model = MusicGen.get_pretrained('facebook/musicgen-medium')
    # MusicGen handles device placement internally; do not call .cuda/.half directly on wrapper
    # If you need to force fp16 on underlying torch models, you can use accelerate or internal APIs
    model.set_generation_params(duration=duration)
    start = time.time()
    with torch.no_grad():
        wav = model.generate([prompt])[0]
    took = time.time() - start
    print(f'Generated {duration}s in {took:.2f}s')
    audio_write('out_medium', wav, model.sample_rate, strategy='loudness')
    print('Saved out_medium.wav')


if __name__ == '__main__':
    generate_medium()
