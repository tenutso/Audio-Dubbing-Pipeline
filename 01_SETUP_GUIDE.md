# RunPod French Dubbing Pipeline - Setup Guide

## Architecture Overview

This pipeline automates converting English webinar MP4s into French audio tracks and subtitles using open-source LLMs and voice cloning. It's designed to be reproducible and not tie up compute indefinitely.

```
Input (English MP4)
    ↓
Audio Extraction (ffmpeg)
    ↓
Speech-to-Text (Whisper)
    ↓
Translation + Adaptation (Ollama + LLM)
    ↓
Text-to-Speech with Voice Cloning (F5-TTS + Speaker Embedding)
    ↓
Subtitle Generation (Timing Sync)
    ↓
Output: French Audio + SRT Files (ready for Vimeo)
```

## Hardware Requirements

- **GPU**: NVIDIA A5000 or RTX 4090 (both supported)
- **VRAM**: 24GB minimum (A5000: 24GB, RTX 4090: 24GB)
- **System RAM**: 32GB+ recommended
- **Storage**: 500GB+ for workspace (videos, models, outputs)
- **Pre-installed**: PyTorch 2.8

## Open-Source Components

| Component | Tool | Purpose | Model |
|-----------|------|---------|-------|
| Audio Extraction | FFmpeg | MP4 → WAV | N/A |
| Speech-to-Text | faster-whisper | English → Transcript (w/ VAD) | large-v3 (CTranslate2 float16) |
| Translation | Ollama + Qwen2.5 | English → French (batch + review) | qwen2.5:14b |
| TTS + Voice Cloning | Coqui XTTS v2 | French synthesis + speaker voice | xtts_v2 (multilingual) |
| Time-fitting | FFmpeg atempo | Fit French audio into timing windows | N/A |
| Subtitle Export | pysrt | Timestamped SRT generation | N/A |

## Cost Optimization (RunPod On-Demand Billing)

- **A5000**: ~$0.45/hour
- **RTX 4090**: ~$0.36/hour
- **Estimated per 1-hour video**: 20-30 minutes GPU time
- **Monthly cost for 20 videos**: ~$6-12

By using this setup, you can:
1. Start a RunPod instance
2. Run batch jobs
3. Download results
4. Stop instance immediately (hourly billing, no idle fees)

---

## Installation & Setup

### Step 1: Launch RunPod Template

Use RunPod's "PyTorch 2.8" template with:
- GPU: A5000 or RTX 4090
- Storage: 500GB network volume recommended
- Template should include CUDA 12.x, Python 3.10+

### Step 2: Initial System Setup (run these commands)

```bash
# Update system
apt-get update && apt-get upgrade -y

# Install system dependencies
apt-get install -y \
  ffmpeg \
  git \
  wget \
  curl \
  sox \
  libsndfile1 \
  libportaudio2

# Create workspace
mkdir -p /workspace/{videos,models,outputs,scripts,logs}
cd /workspace
```

### Step 3: Install Python Dependencies

Create `/workspace/requirements.txt`:

```
# Core Audio/Video
librosa==0.10.0
soundfile==0.12.1
pydub==0.25.1
moviepy==1.0.3
pysrt==1.1.2
scipy==1.11.4

# Speech Recognition
openai-whisper==20231117
faster-whisper==0.10.0

# TTS & Voice Cloning
# F5-TTS (if pre-built wheels available)
numpy==1.24.3
torch==2.0.0  # Already installed, but included for reference
torchaudio==2.0.0

# LLM + Translation
ollama==0.0.1  # Command-line tool
langchain==0.1.0
python-dotenv==1.0.0

# Utilities
tqdm==4.66.1
click==8.1.7
pyyaml==6.0.1
```

Install dependencies:

```bash
cd /workspace
pip install -r requirements.txt --no-cache-dir
```

### Step 4: Install Ollama (for LLM translation)

```bash
# Download and install Ollama
curl https://ollama.ai/install.sh | sh

# Start Ollama as background service
ollama serve &

# Wait for startup, then pull models
# Lightweight French-capable models:
ollama pull mistral:latest  # or llama2:13b
```

### Step 5: Install/Setup F5-TTS (Voice Cloning)

F5-TTS is a recent open-source TTS with voice cloning. Set it up:

```bash
cd /workspace
git clone https://github.com/SWivlEj/F5-TTS.git
cd F5-TTS
pip install -e .

# Download pretrained model (auto-downloads on first use)
# Model: ~2GB
```

**Alternative**: If F5-TTS isn't available, use **Coqui TTS** (more stable):

```bash
pip install TTS

# This will auto-download the model (~1GB)
# Supports French out of box
```

### Step 6: Create Configuration File

Create `/workspace/config.yaml`:

```yaml
# Pipeline Configuration
pipeline:
  input_folder: /workspace/videos/input
  output_folder: /workspace/outputs
  models_folder: /workspace/models
  logs_folder: /workspace/logs
  temp_folder: /workspace/temp
  
# Audio Settings
audio:
  sample_rate: 24000
  mono: true
  normalize: true
  
# Speech Recognition
whisper:
  model: large-v3
  language: en
  device: cuda
  batch_size: 1
  
# Translation LLM
translation:
  engine: ollama  # or: groq (API), local
  model: mistral:latest
  temperature: 0.7
  language_target: french
  context: "You are a professional video dubbing specialist. Translate this English dialogue into natural French suitable for video dubbing. Keep it conversational and adapt idioms naturally."
  
# Voice Cloning & TTS
tts:
  engine: f5tts  # or: coqui
  language: fr
  speaker_profile_duration: 10  # seconds of reference audio for cloning
  speed: 1.0
  
# Subtitle Settings
subtitles:
  format: srt
  encoding: utf-8
  sync_offset_ms: 0
  
# Processing
processing:
  max_workers: 2  # For batch processing
  chunk_duration_seconds: 30  # Process audio in 30-sec chunks
  timeout_seconds: 3600
```

---

## Python Pipeline Script

See `02_pipeline.py` for the main processing script. This is the heart of the automation.

Key features:
- Modular functions for each step
- Error handling and resume capability
- Progress tracking
- Logging to files

Run with:
```bash
python /workspace/scripts/02_pipeline.py --input-video /workspace/videos/webinar_1.mp4
```

---

## Docker Setup (Optional but Recommended)

Use Docker to make this 100% reproducible. Create `/workspace/Dockerfile`:

```dockerfile
FROM nvidia/cuda:12.2.0-cudnn8-runtime-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3-pip \
    ffmpeg git wget curl sox libsndfile1 libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy requirements
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# Setup Ollama
RUN curl https://ollama.ai/install.sh | sh

# Copy scripts
COPY scripts/ ./scripts/
COPY config.yaml .

VOLUME ["/workspace/videos", "/workspace/outputs", "/workspace/models"]

CMD ["python", "scripts/02_pipeline.py"]
```

Build and run:
```bash
docker build -t french-dubbing:latest .
docker run --gpus all -v /workspace/videos:/workspace/videos -v /workspace/outputs:/workspace/outputs french-dubbing:latest
```

---

## Memory/VRAM Management

For A5000/RTX 4090 (24GB VRAM):

- **Whisper (large-v3)**: ~6GB
- **F5-TTS**: ~8GB
- **Ollama LLM (13B)**: ~8GB
- **Headroom**: ~2GB

If OOM errors occur:
1. Use smaller Whisper model (`small` or `medium`)
2. Reduce Ollama model size (`mistral` instead of `llama2:13b`)
3. Process audio in smaller chunks

---

## Running the Pipeline

### Single Video

```bash
python /workspace/scripts/02_pipeline.py \
  --video /workspace/videos/webinar_1.mp4 \
  --output-dir /workspace/outputs
```

### Batch Processing (Multiple Videos)

Create `/workspace/scripts/batch_run.py` (see `03_batch_runner.py`)

```bash
python /workspace/scripts/batch_run.py \
  --input-dir /workspace/videos/input \
  --output-dir /workspace/outputs
```

### Resume Capability

The pipeline tracks progress with `.checkpoint` files. If interrupted, rerun the same command to resume.

---

## Output Format for Vimeo

The pipeline outputs:
1. **Audio Track**: `webinar_1_french.m4a` (AAC, 128kbps)
2. **Subtitles**: `webinar_1_french.srt` (UTF-8 encoded)

To upload to Vimeo:
1. Go to video settings → Audio & Tracks
2. Add `webinar_1_french.m4a` as additional audio track
3. Add `webinar_1_french.srt` as subtitle track (French)

---

## Troubleshooting

### CUDA Out of Memory
```bash
# Check GPU memory
nvidia-smi

# If OOM, reduce batch sizes in config.yaml
# Whisper batch_size: 1 → use small model
# TTS: process 15-sec chunks instead of 30-sec
```

### Ollama Model Loading Issues
```bash
# Check Ollama service
ps aux | grep ollama

# Restart if needed
pkill ollama
ollama serve &
ollama pull mistral:latest
```

### F5-TTS Not Available
Use Coqui TTS instead:
```bash
pip uninstall f5tts
pip install TTS
# Update config.yaml: tts.engine: coqui
```

---

## Cleanup & Shutdown

When finished:

```bash
# Remove temporary files
rm -rf /workspace/temp/*

# Stop Ollama
pkill ollama

# Compress outputs for download
cd /workspace/outputs
tar -czf results.tar.gz *.m4a *.srt
# Download results.tar.gz

# Stop RunPod instance (this stops billing immediately)
```

---

## Performance Expectations

- **1-hour webinar processing time**: 20-30 minutes (GPU-bound on TTS)
- **Cost per 1-hour video**: ~$0.25-0.50 (on-demand)
- **Audio quality**: Professional (24kHz, 128kbps AAC)
- **Subtitle sync accuracy**: ±100ms (industry standard)

---

## Next Steps

1. Launch RunPod instance with PyTorch 2.8 template
2. Follow Step 1-6 in "Installation & Setup"
3. Test with single video first
4. Scale to batch processing
5. Download outputs and stop instance

Estimated setup time: **30-45 minutes**
Estimated first video processing: **~25 minutes**
