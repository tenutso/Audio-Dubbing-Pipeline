# French Dubbing Pipeline

Turn English webinar MP4s into French audio tracks with synchronised SRT subtitles. Designed to run end-to-end on a single RTX 4090 (24 GB) RunPod instance.

The pipeline does everything from source separation through TTS and subtitle alignment, and exposes both a CLI and a small **web UI** for non-technical users to submit jobs, monitor progress, and download results.

---

## What's in the box

| Stage | Component(s) | Notes |
|---|---|---|
| Source separation | [Demucs htdemucs](https://github.com/facebookresearch/demucs) | Splits vocals from background music so the dub can be re-mixed cleanly |
| Transcription | [faster-whisper large-v3](https://github.com/SYSTRAN/faster-whisper) | CTranslate2, float16, word timestamps + VAD |
| Speaker diarization | [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) | Optional; per-speaker voice cloning when enabled |
| Translation | **EuroLLM-9B-Instruct**, **Qwen3:14b via Ollama**, or **Gemini API** | Pick at runtime with `--translator` |
| Translation review | Qwen3:14b (or Gemini) | Second-pass naturalness/idiom fix; CPS-budgeted |
| TTS | **VoxCPM2**, **Coqui XTTS v2**, **edge-tts**, **Qwen3-TTS (cloud)**, **Qwen3-TTS-local**, or **Gemini-TTS** | Pick at runtime with `--tts` |
| Speaker denoising | DeepFilterNet → noisereduce → FFmpeg `anlmdn` | Layered fallback chain for the voice-clone reference |
| Assembly | FFmpeg `atempo` + crossfade | Length-fits French into original timing windows |
| SRT alignment | [WhisperX](https://github.com/m-bain/whisperX) | Force-aligns French text to the dubbed audio |
| Output | AAC 192 kbps 48 kHz stereo (+ optional full-mix with background) + UTF-8 SRT | Vimeo-ready |

### Localisation

- **Languages**: 15 target languages out of the box (CPS-budgeted), set via `translation.target_lang`. French is the default.
- **Canadian French**: `--locale fr-ca` injects a glossary into the translation prompt, runs a deterministic post-processing substitution for "always"-mode terms, and tells the reviewer to use Québécois register. Glossary lives in [canadian_glossary.yaml](canadian_glossary.yaml).

---

## TTS engines — when to use which

| Engine | Where it runs | Voice cloning | API key | Best for |
|---|---|---|---|---|
| `voxcpm2` (default) | On-GPU, 48 kHz | Yes (3–25 s reference) | None | Highest fidelity. Default. |
| `xtts2` | On-GPU, 24 kHz | Yes | None | Fallback if VoxCPM2 fails to load |
| `qwen3-tts-local` | On-GPU, ~3.5 GB bf16, 24 kHz | Yes | None | Open-weights alternative (Apache 2.0, Qwen3-TTS 1.7B) |
| `edge-tts` | Cloud (Microsoft), 24 kHz | No (fixed voice) | None | Fast preview / no-VRAM path |
| `gemini-tts` | Cloud (Google), 24 kHz | No (preset voice) | `GEMINI_API_KEY` | Google's expressive preview voices |
| `qwen3-tts` | Cloud (Alibaba DashScope) | No (preset voice) | `DASHSCOPE_API_KEY` | Qwen3-TTS-Flash hosted variant |

Cloud engines (`edge-tts`, `gemini-tts`, `qwen3-tts`) don't compete for VRAM, so they're a good escape hatch when other models are loaded.

---

## Translation backends

| Backend | Runs | Setup | Notes |
|---|---|---|---|
| `eurollm` (default) | On-GPU via HF transformers | HuggingFace token + license acceptance for [EuroLLM-9B-Instruct](https://huggingface.co/utter-project/EuroLLM-9B-Instruct) | ~9 GB VRAM in 8-bit, ~18 GB in bfloat16. Best raw quality for European target languages. |
| `qwen` | Local Ollama | `ollama pull qwen3:14b` | No HF token needed; CPU/GPU hybrid. Good for comparison runs. |
| `gemini` | Cloud (Google) | `GEMINI_API_KEY` | No local GPU consumed. Uses the unified `google-genai` SDK; defaults to `gemini-2.5-flash`. |

All three back-ends share the same three-pass logic (fitted → batched overflow retry with tightening budgets → optional unconstrained "natural" pass for SRT readability).

---

## Quick start on RunPod

### 1. Launch a pod

1. Sign in at <https://runpod.io> → **Pods → Deploy**.
2. **GPU**: RTX 4090 (24 GB) — recommended. RTX A5000 / A6000 / H100 also work.
3. **Template**: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (PyTorch 2.8.0 / CUDA 12.8.1 pre-installed).
4. **Storage**: at least 100 GB container disk; persist a 50 GB+ volume at `/workspace` if you'll come back to the same pod.
5. **Expose HTTP port** `7860` (for the web UI) — set in the pod's **HTTP Service Ports**.
6. Hit **Deploy**, then **Connect → Web Terminal** (or grab the SSH command).

### 2. Clone and install

```bash
cd /workspace
git clone https://github.com/tenutso/french-dubbing.git
cd french-dubbing
bash 04_setup.sh
```

`04_setup.sh` is idempotent and self-recovering: each step prints a `[hh:mm:ss] >>> …` header, errors are collected and shown in a final summary instead of aborting. It will:

- install system packages (`ffmpeg`, `sox`, audio libs)
- create `/workspace/{videos/input,outputs,models,scripts,logs,temp}`
- pip-install the full stack (faster-whisper, demucs, DeepFilterNet, VoxCPM2, coqui-tts, WhisperX, EuroLLM deps, **fastapi + uvicorn for the web UI**, **edge-tts + dashscope + google-genai for cloud TTS**, **qwen-tts for local Qwen3-TTS**)
- prompt for **HF_TOKEN** (required for EuroLLM + pyannote), and optionally **GEMINI_API_KEY** / **DASHSCOPE_API_KEY** — all persisted to `/workspace/.env`
- install + start `ollama` and pull `qwen3:14b`
- pre-download Whisper large-v3 + VoxCPM2 weights
- copy `02_pipeline.py`, `03_batch_runner.py`, `verify_setup.py`, `05_web.sh`, and the `web/` package into `/workspace/scripts/`

Expect 25–45 minutes the first time (most of it is model downloads — `hf_transfer` is enabled).

### 3. Verify

```bash
python /workspace/scripts/verify_setup.py
```

GO/NO-GO summary with all the package + GPU + token checks.

### 4. Run it

**Option A — Web UI** (drag-drop, no terminal needed)

```bash
bash /workspace/scripts/05_web.sh
```

Open the RunPod-proxied URL for port 7860 (RunPod gives you something like `https://<pod-id>-7860.proxy.runpod.net`). Drop in a video, pick translator / TTS / locale / volume boost, hit submit. Live log streams in the page, downloads appear when the job finishes.

**Option B — Single video on the CLI**

```bash
python /workspace/scripts/02_pipeline.py \
    --video /workspace/videos/input/webinar.mp4 \
    --translator gemini \
    --tts voxcpm2 \
    --locale fr-ca \
    --volume-boost 15
```

All CLI flags:

| Flag | Choices / Type | Effect |
|---|---|---|
| `--video` | path (required) | Input MP4 |
| `--output-dir` | path | Default `/workspace/outputs` |
| `--config` | path | Default `/workspace/config.yaml` |
| `--translator` | `eurollm` \| `qwen` \| `gemini` | Overrides `translation.backend` |
| `--tts` | `voxcpm2` \| `xtts2` \| `edge-tts` \| `qwen3-tts` \| `qwen3-tts-local` \| `gemini-tts` | Overrides `tts.engine` |
| `--locale` | `fr` \| `fr-ca` | Overrides `translation.locale`; `fr-ca` triggers glossary |
| `--volume-boost` | float, % | Boost output loudness after peak-normalise; clipped at ±1.0 |
| `--force` | flag | Re-process even if outputs exist |

**Option C — Batch a folder**

```bash
cp ~/incoming/*.mp4 /workspace/videos/input/
python /workspace/scripts/03_batch_runner.py
```

One job at a time (VRAM-safe). Reports to `/workspace/logs/batch_report.json`.

### 5. Stop the pod

When you're done, **Stop** the pod from the RunPod console — that's what stops billing. Resume later; everything in `/workspace` persists if you allocated a volume.

---

## Web UI

The UI is a single FastAPI app at [web/app.py](web/app.py) with a vanilla-JS frontend in [web/static/](web/static/). Launch it with `bash 05_web.sh` (binds `0.0.0.0:7860`).

**Features**

- Drag-drop MP4 upload with progress bar (5 GB cap).
- Pre-fills translator / TTS / locale / volume-boost from `config.yaml` defaults so you only override what you want.
- Single-job FIFO queue — submit while a job is running, the next one starts automatically when VRAM frees.
- Live log panel streams the pipeline's `[N/7] PHASE …` lines via Server-Sent Events.
- Download buttons for `_french.m4a`, `_french.srt`, and the optional `_french_full.m4a` (background remix).
- Crash-safe: if `uvicorn` is killed mid-job, the running job is recovered as `failed` on restart and queued jobs resume automatically.
- Cancel sends SIGTERM to the whole process group, so ffmpeg children get reaped.
- Footer shows live GPU, VRAM-free, disk-free, Ollama status, and which API keys are present.

**Customise the port:**

```bash
DUBBING_WEB_PORT=8080 bash /workspace/scripts/05_web.sh
```

**No auth.** Per the original design choice — the RunPod proxy URL is unguessable for outsiders, but **don't post the URL publicly**. If you need protection, terminate it behind an authenticated reverse proxy (cloudflared / nginx + Basic auth).

---

## Configuration

Everything that isn't a runtime override lives in [config.yaml](config.yaml). The full file is heavily commented; the most useful knobs:

```yaml
diarization:
  enabled: true
  min_speakers: 2          # set to 1 only for known-single-speaker recordings
  max_speakers: 10

translation:
  backend: eurollm         # or qwen, gemini
  target_lang: fr          # fr es de it pt nl pl ru ja ko zh ar tr hi vi
  locale: fr               # or fr-ca
  cps_safety: 1.1          # bump toward 1.2 if dub feels too terse
  natural_pass: true       # 2nd unconstrained pass for SRT readability

tts:
  engine: voxcpm2          # voxcpm2 | xtts2 | edge-tts | qwen3-tts | qwen3-tts-local | gemini-tts
  cfg_value: 2.5
  inference_timesteps: 24  # VoxCPM2 quality knob; 32 for max quality

audio:
  volume_boost_pct: 0      # default; override at runtime with --volume-boost N

subtitles:
  whisperx_alignment: true
  sync_offset_ms: 0
```

---

## Outputs

For each `webinar.mp4` you'll get, in `/workspace/outputs/`:

- `webinar_french.m4a` — French dub only (vocals + silence in the original gaps), AAC 192 kbps / 48 kHz stereo
- `webinar_french.srt` — UTF-8 SRT, WhisperX-aligned to the French audio
- `webinar_french_full.m4a` — full mix: French vocals + the original background bed at −3 dB (only when Demucs separation succeeded and `source_separation.preserve_background: true`)

For Vimeo: upload `_full.m4a` as the alternate audio track and `.srt` as the French subtitle file.

---

## Running outside RunPod

Anywhere you have an NVIDIA GPU with ≥ 24 GB VRAM and CUDA 12.x:

- Use the [`Dockerfile`](Dockerfile) (PyTorch 2.8 / CUDA 12.8 base) — it mirrors what `04_setup.sh` does on the base RunPod image.
- Or run the setup script directly on Ubuntu 24.04 with PyTorch 2.8 already installed; everything else is pip-installable.

The `tts.engine: edge-tts` (or `gemini-tts`) path doesn't require any GPU at all — useful for a laptop sanity check before paying for cloud GPU time.

---

## Troubleshooting

| Symptom | Most likely cause | Fix |
|---|---|---|
| `Diarization failed: 'DiarizeOutput' object has no attribute 'itertracks'` | pyannote returned a newer wrapped namedtuple shape | Already handled by the extractor in [02_pipeline.py:`diarize_audio`](02_pipeline.py); upgrade pyannote (`pip install -U pyannote.audio`) and re-run. The log now prints the actual field names so any future shape change is one-line to patch. |
| Only `SPEAKER_00` detected on a multi-speaker recording | `min_speakers: 1` in config — pyannote collapses to 1 | Set `diarization.min_speakers: 2` (the new default). Verify HF token has accepted the model license at <https://huggingface.co/pyannote/speaker-diarization-community-1>. |
| Ollama `Read timed out (read timeout=180)` during Qwen translation | First-batch cold load (~20 s) + ~2000-token generation on `qwen3:14b` can spill over the old 180 s ceiling | Already raised to 600 s + `keep_alive: 30m` in [_ollama_call](02_pipeline.py). Also check `ollama ps` — if the model is offloaded to CPU, free VRAM and `ollama run qwen3:14b ""` to warm it. |
| "google-generativeai not installed" | Old SDK; we migrated to `google-genai` | `pip install -U google-genai`; or just re-run `04_setup.sh`. |
| Output too quiet | Hard-coded 0.95 peak-normalise | Add `--volume-boost 20` (or set `audio.volume_boost_pct` in config). |
| Coqui XTTS won't import: missing `isin_mps_friendly` / `is_torch_greater_or_equal` | Newer transformers removed these symbols | Already patched at import-time in [02_pipeline.py:`_patch_transformers_for_coqui`](02_pipeline.py). |
| Web UI shows "missing FastAPI" / `ModuleNotFoundError` | `04_setup.sh` didn't reach the web-deps step | `pip install fastapi 'uvicorn[standard]' python-multipart`, then re-run `bash /workspace/scripts/05_web.sh`. |
| Web UI restarts mid-job → job stuck `running` forever | Server crashed before the SSE could close | Already auto-recovered: `running` → `failed (server restarted mid-job)` on next launch, queued jobs replayed. |
| Cancel left zombie `ffmpeg` processes | Subprocess group kill failed | The web app uses `start_new_session=True` + `os.killpg(SIGTERM)`; if you see leftovers, `pkill -f ffmpeg` and report. |

For deeper guidance see `05_QUICK_REFERENCE.md`, `06_ARCHITECTURE.md`, and the comments inside `02_pipeline.py`.

---

## File layout

```
french-dubbing/
├── 02_pipeline.py            # main CLI (single video)
├── 03_batch_runner.py        # batch folder of MP4s
├── 04_setup.sh               # one-shot installer for RunPod / Ubuntu 24.04
├── 05_web.sh                 # uvicorn launcher (web UI)
├── verify_setup.py           # GO/NO-GO post-install sanity check
├── config.yaml               # all non-runtime knobs (commented)
├── canadian_glossary.yaml    # fr-ca vocabulary + formatting rules
├── requirements.txt          # pip deps (lower bounds; setup.sh auto-upgrades)
├── Dockerfile                # PyTorch 2.8 / CUDA 12.8 reproducible image
├── web/
│   ├── app.py                # FastAPI app + queue worker + SSE
│   ├── jobs.py               # Job dataclass + atomic jobs.json
│   └── static/
│       ├── index.html
│       ├── app.js
│       └── style.css
└── README.md                 # this file
```

Runtime layout (created by `04_setup.sh`):

```
/workspace/
├── videos/input/             # drop MP4s here for CLI / batch runs
├── outputs/                  # CLI / batch outputs land here
├── scripts/                  # installed copy of the pipeline + web/
├── models/                   # auto-populated (Whisper, VoxCPM2, EuroLLM, …)
├── logs/                     # per-video + batch + ollama logs
├── temp/                     # intermediate stems (auto-cleaned)
├── web/
│   ├── uploads/              # web-UI uploads, per-job
│   ├── outputs/<job_id>/     # web-UI per-job output dirs
│   └── jobs.json             # persisted queue + history
├── config.yaml
└── .env                      # HF_TOKEN, GEMINI_API_KEY, DASHSCOPE_API_KEY
```

---

## Costs

Rough numbers on RunPod (May 2026 pricing; subject to change):

| GPU | $/hr | Time/video (1 h source) | Cost/video |
|---|---|---|---|
| RTX 4090 | ~$0.36 | 25–35 min | **$0.15 – $0.21** |
| RTX A5000 | ~$0.45 | 30–40 min | **$0.23 – $0.30** |

Add the cloud-API call cost if you use `--translator gemini` or `--tts gemini-tts` / `--tts qwen3-tts`. The free tier of Gemini and the very low DashScope per-character rate are usually negligible at this volume.

---

## License & attribution

- Pipeline code, web UI, glue: **MIT**
- faster-whisper (CTranslate2): MIT
- pyannote.audio: MIT
- VoxCPM2 / Qwen3-TTS: Apache 2.0
- Coqui XTTS v2 (`coqui-tts` fork): MPL 2.0
- DeepFilterNet: MIT / Apache
- WhisperX: BSD
- Demucs: MIT
- EuroLLM-9B-Instruct: Apache 2.0 (gated — license acceptance required)
- FFmpeg: LGPL/GPL (depending on build)

No proprietary APIs are required to run the pipeline end-to-end; the Gemini / DashScope backends are opt-in.

---

## Contributing / debugging

- Logs per video: `/workspace/logs/<stem>.log`
- Web UI logs: stdout of `05_web.sh`
- Batch summary: `/workspace/logs/batch_report.json`
- Job state across web-UI restarts: `/workspace/web/jobs.json`

Issues and PRs welcome at <https://github.com/tenutso/french-dubbing>.
