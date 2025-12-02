#!/usr/bin/env python3
import time
import torch
from accelerate.big_modeling import init_empty_weights, load_checkpoint_and_dispatch
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

mname = 'facebook/musicgen-large'
prompt = "a rock ballad with guitars, drum backbeat and bass riffs"

print('Init empty weights context and get skeleton model')
with init_empty_weights():
    model = MusicGen.get_pretrained(mname)

print('Dispatching weights to device_map=auto with offload to /workspace/offload')
load_checkpoint_and_dispatch(model, mname, device_map='auto', offload_folder='/workspace/offload')

print('Model ready; possibly offloaded. Generating...')
model.set_generation_params(duration=30)
start = time.time()
with torch.no_grad():
    wavs = model.generate([prompt])
print('Generated in', time.time() - start)
audio_write('/workspace/out_large_rock', wavs[0], model.sample_rate, strategy='loudness')
print('Saved to /workspace/out_large_rock.wav')
