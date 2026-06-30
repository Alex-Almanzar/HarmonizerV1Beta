# Harmony Deck

A skeuomorphic vocal-harmony workstation for choir directors and music teachers. Split a vocal out of a mix, generate SATB-style harmony parts, mix them on a tactile fader rack, and export stems — all running client-side in the browser, with an optional cloud AI singing-synthesis backend for fully AI-rendered parts.

## What's in this repo

```
harmony-deck/
├── app/
│   └── index.html        ← the whole frontend: open this in a browser, no build step
├── backend/
│   ├── server.py          ← FastAPI wrapper around DiffSingerMiniEngine
│   ├── requirements.txt   ← wrapper-layer Python deps
│   ├── Dockerfile          ← reproducible GPU deploy
│   ├── DEPLOY.md           ← step-by-step cloud GPU deployment guide
│   └── assets/             ← put downloaded model checkpoints here (gitignored)
├── .gitignore
└── LICENSE
```

## Quickstart — local app only (no backend needed)

The frontend is fully self-contained and works without any server:

```bash
git clone https://github.com/<your-username>/harmony-deck.git
cd harmony-deck
open app/index.html      # macOS
# or just double-click app/index.html, or `python3 -m http.server` and visit it in a browser
```

Drop in a song or recording, hit **Split**, generate harmony parts on the **Soprano / Alto / Tenor** channels, and mix. Everything runs in-browser using the Web Audio API:
- Vocal isolation: mid/side (center-channel) processing
- Harmony generation: granular pitch-shifting of your actual recorded vocal at a chosen musical interval
- Key detection: autocorrelation pitch tracking + Krumhansl-Schmuckler key matching
- Export: real WAV stems and a mixed-down master, via a zero-dependency in-browser WAV encoder

## Optional — Cloud AI Voice Engine (DiffSinger)

For a different, fully AI-synthesized vocal (lyrics + notes → a new sung performance, not a transform of your recording), the **Cloud AI Voice Engine** panel in the app can call a self-hosted DiffSinger model.

This needs your own GPU server — it is not bundled or hosted for you. Follow [`backend/DEPLOY.md`](backend/DEPLOY.md) for the full walkthrough (cloud GPU provider options, model checkpoint downloads, Docker build, and known latency/CORS/licensing caveats). Once it's running, paste your server's URL into the app's Cloud AI panel.

> Model checkpoints are not included in this repo (they're multi-GB binaries and don't belong in git — see `.gitignore`). They also carry their own license terms from whichever source you download them from (e.g. OpenCpop/OpenVPI releases) — check those before using generated audio commercially; that's separate from this repo's own license below.

## License

This repo's code is MIT-licensed — see [`LICENSE`](LICENSE). Swap in your own name/year before publishing. Any model checkpoints you download separately are governed by their own upstream licenses, not this one.

## Status

V1 prototype. The local pitch-shift engine is a real, working DSP approximation — not a trained neural source-separation or voice-cloning model. See the in-app notes under each panel for exactly what technique is doing the work.
