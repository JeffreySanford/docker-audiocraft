#!/usr/bin/env python3
"""
Minimal Gradio UI for AudioCraft MusicGen.
"""
import os
import torch
import gradio as gr
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import random
import time

MODEL_SIZES = ["small", "medium", "large"]


def generate(prompt: str, model_size: str, duration: int, allow_large: bool, temperature: float, seed: int, fp16: bool, preset: str):
    if model_size not in MODEL_SIZES:
        return (None, "Invalid model selected")
    if model_size == "large" and not allow_large:
        return (None, "Large model is disabled. Toggle 'Allow large' to enable; it may not fit on 10GB GPUs")

    mname = {
        "small": "facebook/musicgen-small",
        "medium": "facebook/musicgen-medium",
        "large": "facebook/musicgen-large",
    }[model_size]

    # Use half precision if GPU is available and requested
    model = MusicGen.get_pretrained(mname)
    if torch.cuda.is_available() and fp16:
        model = model.cuda().half()

    # Set generation params; MusicGen supports some options (duration, temperature)
    model.set_generation_params(duration=duration, temperature=temperature)
    # Seed handling for deterministic runs (best-effort)
    if seed is None or seed == 0:
        seed = random.randint(1, 2**31 - 1)
    torch.manual_seed(seed)

    # Generation
    start = time.time()
    wav = model.generate([prompt])[0]
    took = time.time() - start

    # Keep log info for the user
    info = f"Saved to {fname} (seed={seed} time={took:.2f}s)"

    # Use a fixed output file path in the mounted workspace
    fname = "/workspace/out_gradio.wav"
    audio_write("/workspace/out_gradio", wav, model.sample_rate, strategy="loudness")
    return (fname, info)


with gr.Blocks() as demo:
    gr.Markdown("# AudioCraft MusicGen — Gradio demo (small / medium / large - large requires toggle)")
    with gr.Row():
        prompt = gr.Textbox(label="Prompt", value="A short piano loop")
        model_size = gr.Dropdown(choices=MODEL_SIZES, value="medium", label="Model Size")
    with gr.Row():
        duration = gr.Slider(minimum=2, maximum=30, step=1, value=6, label="Duration (seconds)")
        temperature = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.95, label="Temperature")
    with gr.Row():
        seed = gr.Number(value=0, label="Seed (0 for random)")
        fp16 = gr.Checkbox(label='Use FP16 (GPU only)', value=True)
        allow_large = gr.Checkbox(label='Allow large model (may require CPU offload or FSDP)', value=False)
    with gr.Row():
        preset = gr.Radio(choices=["balanced", "high_quality", "fast"], value='balanced', label='Preset')
    output_audio = gr.Audio(label="Generated audio", type='filepath')
    message = gr.Textbox(label="Status")

    generate_btn = gr.Button("Generate")
    generate_btn.click(fn=generate, inputs=[prompt, model_size, duration, allow_large, temperature, seed, fp16, preset], outputs=[output_audio, message])


if __name__ == '__main__':
    # Set address to 0.0.0.0 so it is reachable from the host
    demo.launch(server_name='0.0.0.0', server_port=7860, share=False)
