"""
Minimal MusicGen example for inference using audiocraft
This script loads a pretrained musicgen model and generates audio.
"""
import torch
import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write


def generate_text_to_music(prompt: str = "a short piano loop"):
    # medium or small depending on GPU memory
    model = MusicGen.get_pretrained("facebook/musicgen-small")
    model.set_generation_params(duration=6)  # seconds
    wavs = model.generate([prompt])

    # Save first sample
    audio_write("out", wavs[0], model.sample_rate, strategy="loudness", loudness_compressor=True)
    print("Saved to out.wav")


if __name__ == "__main__":
    import sys
    prompt = "".join(sys.argv[1:]) if len(sys.argv) > 1 else "a short piano loop"
    generate_text_to_music(prompt)
