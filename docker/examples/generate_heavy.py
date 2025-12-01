"""
Generate with medium/large MusicGen and safe memory optimizations.
This example tries to use fp16 and minimal settings to run on 10 GB GPUs.
"""
import time
import torch
from audiocraft.models import MusicGen


def generate_medium(prompt="a jazz loop", duration=8, model_size="medium"):
    print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')
    mname = {
        'small': 'facebook/musicgen-small',
        'medium': 'facebook/musicgen-medium',
        'large': 'facebook/musicgen-large'
    }[model_size]

    # load model
    model = MusicGen.get_pretrained(mname)

    # run on half precision if cuda
    if torch.cuda.is_available():
        model = model.cuda().half()
        torch.backends.cudnn.benchmark = True

    model.set_generation_params(duration=duration)

    start = time.time()
    with torch.no_grad():
        wav = model.generate([prompt])
    took = time.time() - start
    print(f"Generated {duration}s in {took:.2f}s")

    return wav, model.sample_rate


if __name__ == '__main__':
    wav, sr = generate_medium('a moody synthwave loop', 4, 'medium')
    import torchaudio
    torchaudio.save('out_heavy.wav', wav[0].cpu(), sr)
    print('Saved out_heavy.wav')
