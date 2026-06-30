[DEPLOY.md](https://github.com/user-attachments/files/29526032/DEPLOY.md)
# Deploying the DiffSinger Cloud Engine

This is the part I genuinely can't do from this sandbox — no outbound network access, no GPU, and no ability to SSH into RunPod/Vast.ai/GCP on your behalf. Everything below is real, runnable, and matches the architecture you sent, with a few corrections. You'll need to run it yourself on a machine you control.

## What this actually buys you over the local prototype

The `harmony-deck.html` app I built earlier generates harmonies by pitch-shifting your *existing* recorded vocal — same words, same phrasing, transposed. DiffSinger is a different paradigm entirely: it's a **score-to-singing** model. You give it lyrics + note names + durations, and it synthesizes a new vocal performance from scratch. That's a genuinely AI-rendered voice, closer to what Suno/Kits actually do — but it means a teacher has to supply a melody as notes, not just "harmonize this audio."

## Step 1: Get a GPU box

| Provider | Realistic cost | Notes |
|---|---|---|
| RunPod (Community Cloud) | ~$0.20–0.40/hr for a T4 | Easiest on-ramp, has a built-in HTTPS proxy for exposing port 8000 |
| Vast.ai | ~$0.10–0.30/hr for a T4 | Cheaper, more variable host quality — check uptime reviews |
| GCP Compute Engine | ~$0.35/hr for `n1-standard-4` + T4, plus the GPU itself | More setup (you manage the VM, firewall rules, static IP) |

Pick one, launch an Ubuntu 22.04 image with CUDA drivers preinstalled (RunPod and Vast.ai both offer CUDA-ready templates — use those rather than installing drivers yourself).

## Step 2: SSH in and set up the engine

```bash
git clone https://github.com/openvpi/DiffSingerMiniEngine.git
cd DiffSingerMiniEngine
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Then copy `server.py`, `requirements.txt`, and the `assets/` layout from this delivery into that same directory, and install the wrapper layer on top:

```bash
pip install -r /path/to/wrapper-requirements.txt
```

## Step 3: Get model weights

DiffSinger needs two separate downloads — the engine code doesn't ship with them:

1. **Acoustic model** — a pretrained checkpoint from the OpenCpop or OpenVPI model releases (check their GitHub Releases / linked drive folders — these move around, so search "DiffSinger pretrained checkpoint" for the current link rather than trusting an old one). Drop it in `assets/acoustic/`.
2. **NSF-HiFiGAN vocoder** — also from OpenVPI's releases. Drop it in `assets/vocoder/`.
3. Point both paths in `configs/default.yaml`.

**Important caveat:** most publicly available DiffSinger checkpoints are trained on Mandarin Chinese vocal datasets (OpenCpop). English-language singing checkpoints exist but are less mature and scattered across community forks — confirm a checkpoint actually supports English lyrics before building a product around it, or you'll get phonetically garbled output on "Hallelujah."

## Step 4: Run it

```bash
python3 -m uvicorn server:app --host 0.0.0.0 --port 8000
```

Or build the Docker image instead (more reproducible, easier to redeploy):

```bash
docker build -t harmony-diffsinger .
docker run --gpus all -p 8000:8000 -v $(pwd)/assets:/app/assets harmony-diffsinger
```

Open the port on your provider's firewall/proxy settings (RunPod gives you a proxied HTTPS URL automatically; on GCP/Vast.ai you'll set a firewall rule yourself).

## Step 5: Point the mixer at it

In `harmony-deck.html`, open the **Cloud AI Voice Engine** panel and paste your server's URL (e.g. `https://your-pod-id-8000.proxy.runpod.net`). Test with the `/health` endpoint first — `curl https://your-url/health` should return `{"status":"ok","model_loaded":true}`.

## Corrections to the original plan, worth knowing before you build on it

- **Latency:** "under two seconds" is optimistic for a diffusion model on a single T4. Diffusion synthesis runs multiple denoising steps per inference; depending on step count, expect somewhere in the 2–10 second range per phrase on a T4, not a hard guarantee under 2s. Benchmark on your actual hardware and step count before promising real-time-feeling response in the UI — show a "rendering…" state rather than implying instant turnaround.
- **CORS is wide open (`allow_origins=["*"]`) in the code as written** — fine for development, not fine to leave that way once this is reachable from the public internet. Lock it to your app's real domain, and add at minimum an API key check before this goes anywhere production-facing — there's no auth in this wrapper at all right now.
- **Standard free hosting (Render/Heroku free tier) won't work** — that part of the original plan is correct. Diffusion inference needs a GPU and more RAM than free tiers offer; you do need one of the paid GPU options above.
