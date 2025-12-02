# GPU Usage and Cost Considerations

This document summarizes how GPU resources and cloud costs relate to generating AI music and cartoon videos based on our experiments and rough estimates.

## Local GPU: RTX 3080 (10 GB)

Capabilities:

- Runs `musicgen-small` and `musicgen-medium` reliably in Docker.
- Can generate 30–60 second tracks with stable prompts.
- Can handle multiple 3-minute tracks, though decoding and memory spikes must be managed.
- Struggles with long, complex runs of `musicgen-large`, especially with T5Conditioner/meta issues and higher VRAM demands.

For video:

- Comfortable for generating diffusion images (character designs, backgrounds).
- Can generate short diffusion-based clips (e.g. 10–15 seconds, low resolution).
- Full 3-minute, 1080p diffusion videos are theoretically possible but slow and impractical at scale.

## Cloud GPU Options (e.g. DigitalOcean)

Cloud providers offer GPU droplets with more VRAM and compute power. Typical examples (numbers are illustrative only):

- Mid-tier GPU (e.g. A10/T4-class): $0.50–$1.50 per hour.
- High-end GPU (e.g. A100/4090-class): $2–$4+ per hour.

With such instances, you can:

- Run the same Docker image (`audiocraft:large.community`).
- Generate longer tracks or larger batches of audio more comfortably.
- Experiment with heavier video/animation pipelines.

## Audio Generation Costs (100 Tracks)

Assumptions:

- 100 tracks, each ~3 minutes.
- Per track (small/medium, including overhead): ~5 minutes of GPU time.

Total GPU time:

- 100 × 5 minutes = 500 minutes ≈ 8.5 hours.

Estimated cost:

- On a $1/hour GPU: ~ $9.
- On a $3/hour GPU: ~ $25.

So generating 100 three-minute tracks is likely in the $10–$30 range, assuming efficient batching in a small number of sessions.

## Video Generation Costs (20 Cartoon Videos)

Two broad approaches:

### 1. Efficient Cartoon Pipeline

- Use AI mostly for images: characters, backgrounds, key art.
- Use conventional tools for 2D animation and editing.
- Optionally use lightweight AI for upscaling or small visual effects.

GPU usage:

- Roughly 0.25–0.5 GPU hours per 3-minute video.
- For 20 videos: 5–10 GPU hours.

Estimated cost:

- On a $1/hour GPU: ~$5–$10.
- On a $3/hour GPU: ~$15–$30.

This is the most cost-effective way to get full cartoon videos.

### 2. Heavy Diffusion Video Pipeline

- Use diffusion models to create many frames or longer clips.
- Aim for higher resolution and continuous AI-generated motion.

GPU usage:

- Approximately 1–3 GPU hours per 3-minute video, depending on settings.
- For 20 videos: 20–60 GPU hours.

Estimated cost:

- On a $3/hour GPU: ~$60–$180.
- On a $4/hour GPU: ~$80–$240.

This is viable but more expensive and time-consuming, best reserved for special projects.

## Recommended Strategy

- Use the local 3080 as much as possible for:
  - MusicGen small/medium.
  - Diffusion images (characters, backgrounds, cover art).
- Use cloud GPUs for:
  - Large batches (e.g. 100+ tracks in one session).
  - Heavier video or large-model experiments.

By combining local GPU for experimentation and cloud GPUs for scaled rendering, you can keep overall costs in the tens of dollars for substantial amounts of content.
