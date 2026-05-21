#!/bin/bash
# French Dubbing Pipeline - Setup v2.0
# Target: RunPod pytorch:1.0.2-cu1281-torch280-ubuntu2404
#         (RTX 4090, PyTorch 2.8.0, CUDA 12.8.1, Ubuntu 24.04)
#
# Design: each step handles its own errors.
# One failing step does NOT abort the whole setup.
# Critical prerequisites (GPU, Python, PyTorch) DO abort on failure.
# A final summary lists every error and warning.

set -uo pipefail
set PYTHONUTF8=1
# ── Colour helpers ────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
ERRORS=(); WARNINGS=()

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LOGFILE=/tmp/dubbing_setup.log
mkdir -p "$(dirname "$LOGFILE")"
: > "$LOGFILE"

ts()          { date +%H:%M:%S; }
log_step()    { echo -e "\n${YELLOW}[$(ts)] >>> $1${NC}" | tee -a "$LOGFILE"; }
log_success() { echo -e "${GREEN}[$(ts)] ✓ $1${NC}"      | tee -a "$LOGFILE"; }
log_warn()    { echo -e "${YELLOW}[$(ts)] ⚠ $1${NC}"     | tee -a "$LOGFILE"; WARNINGS+=("$1"); }
log_error()   { echo -e "${RED}[$(ts)] ✗ $1${NC}"        | tee -a "$LOGFILE"; ERRORS+=("$1"); }
die()         { echo -e "${RED}[$(ts)] FATAL: $1${NC}"    | tee -a "$LOGFILE"; exit 1; }

echo "==========================================" | tee -a "$LOGFILE"
echo "  French Dubbing Pipeline — Setup v3.0"    | tee -a "$LOGFILE"
echo "  $(date)"                                  | tee -a "$LOGFILE"
echo "==========================================" | tee -a "$LOGFILE"

# ── CRITICAL PREREQUISITES (abort on failure) ─────────────────────────────

log_step "Checking critical prerequisites …"

command -v nvidia-smi &>/dev/null \
    || die "nvidia-smi not found. This pipeline requires an NVIDIA GPU."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$LOGFILE"

PYTHON=$(command -v python3 2>/dev/null || command -v python 2>/dev/null)
[[ -z "$PYTHON" ]] && die "Python not found in PATH."
PYTHON_VER=$($PYTHON --version 2>&1 | awk '{print $2}')
log_success "Python $PYTHON_VER at $PYTHON"

$PYTHON - <<'PYCHECK' || die "PyTorch 2.8.x not found. Use the pytorch:1.0.2-cu1281-torch280-ubuntu2404 RunPod image."
import sys, torch
v = torch.__version__
assert v.startswith("2.8"), f"Expected PyTorch 2.8.x, got {v}"
assert torch.cuda.is_available(), "torch.cuda.is_available() returned False"
print(f"PyTorch {v}, CUDA {torch.version.cuda}")
PYCHECK

log_success "Prerequisites OK"

# ── Step 1: System packages ────────────────────────────────────────────────

log_step "Installing system packages …"

if apt-get update -qq 2>&1 | tail -2 | tee -a "$LOGFILE" && \
   DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
       ffmpeg git wget curl sox \
       libsndfile1 libsndfile1-dev \
       libportaudio2 portaudio19-dev \
       build-essential python3-dev 2>&1 | tail -5 | tee -a "$LOGFILE"; then
    log_success "System packages installed"
else
    log_error "Some system packages failed — ffmpeg is required for this pipeline"
fi

# ── Step 2: Workspace structure ────────────────────────────────────────────

log_step "Creating workspace at /workspace …"

if mkdir -p /workspace/{videos/input,models/whisper,outputs,scripts,logs,temp} && \
   chmod -R 755 /workspace; then
    log_success "Workspace ready"
else
    log_error "Workspace creation failed"
fi

# Relocate log to workspace
LOGFILE=/workspace/logs/setup.log
cat /tmp/dubbing_setup.log >> "$LOGFILE" 2>/dev/null || true

# ── Step 3: Python package upgrades ───────────────────────────────────────

log_step "Upgrading pip / setuptools / wheel …"

if $PYTHON -m pip install --upgrade --no-cache-dir pip setuptools wheel packaging \
       2>&1 | tail -3 | tee -a "$LOGFILE"; then
    log_success "pip/setuptools/wheel upgraded"
else
    log_warn "pip upgrade had issues — continuing anyway"
fi

# ── Step 4: Core scientific stack ─────────────────────────────────────────

log_step "Installing numpy / scipy (pinned for TTS compatibility) …"

if $PYTHON -m pip install --no-cache-dir \
       "numpy>=1.26.4,<2.0.0" "scipy>=1.13.1" \
       2>&1 | tail -3 | tee -a "$LOGFILE"; then
    log_success "numpy / scipy installed"
else
    log_error "numpy/scipy installation failed"
fi

# ── Step 5: torchaudio 2.8.0 ──────────────────────────────────────────────

log_step "Installing torchaudio 2.8.0 (CUDA 12.8) …"

# Check if already present at the right version
if $PYTHON -c "import torchaudio; assert torchaudio.__version__.startswith('2.8')" 2>/dev/null; then
    log_success "torchaudio 2.8.x already present"
else
    if $PYTHON -m pip install --no-cache-dir \
           torchaudio==2.8.0 \
           --index-url https://download.pytorch.org/whl/cu128 \
           2>&1 | tail -3 | tee -a "$LOGFILE"; then
        log_success "torchaudio 2.8.0 installed"
    else
        log_warn "torchaudio install failed — audio processing may be limited"
    fi
fi

# ── Step 6: Audio libraries ────────────────────────────────────────────────

log_step "Installing audio libraries …"

if $PYTHON -m pip install --no-cache-dir \
       "librosa>=0.10.2" "soundfile>=0.12.1" "pydub>=0.25.1" \
       2>&1 | tail -3 | tee -a "$LOGFILE"; then
    log_success "Audio libraries installed"
else
    log_error "Audio library installation failed"
fi

# ── Step 7: faster-whisper ────────────────────────────────────────────────

log_step "Installing faster-whisper …"

if $PYTHON -m pip install --no-cache-dir "faster-whisper>=1.0.0" \
       2>&1 | tail -3 | tee -a "$LOGFILE"; then
    log_success "faster-whisper installed"
else
    log_error "faster-whisper installation failed"
fi

# ── Step 8: Source separation — Demucs ────────────────────────────────────────

log_step "Installing Demucs (source separation) …"

if $PYTHON -m pip install --no-cache-dir "demucs>=4.0.0" \
       2>&1 | tail -3 | tee -a "$LOGFILE"; then
    log_success "Demucs installed"
else
    log_warn "Demucs install failed — pipeline falls back to raw audio (lower voice-clone quality)"
fi

# ── Step 8b: Speaker denoising — DeepFilterNet (+ noisereduce fallback) ──────

log_step "Installing DeepFilterNet (speaker reference denoising) …"

if $PYTHON -m pip install --no-cache-dir "deepfilternet>=0.5.6" \
       2>&1 | tail -3 | tee -a "$LOGFILE"; then
    log_success "DeepFilterNet installed"
else
    log_warn "DeepFilterNet install failed — will rely on noisereduce fallback"
fi

# noisereduce is the pure-Python fallback denoiser used if DeepFilterNet is
# missing or errors at runtime. Install regardless so the fallback always works.
log_step "Installing noisereduce (denoise fallback) …"

if $PYTHON -m pip install --no-cache-dir "noisereduce>=3.0.0" \
       2>&1 | tail -3 | tee -a "$LOGFILE"; then
    log_success "noisereduce installed"
else
    log_warn "noisereduce install failed — pipeline will fall back to FFmpeg anlmdn"
fi

# ── Step 8c: EuroLLM dependencies (bitsandbytes + accelerate) ─────────────────

log_step "Installing EuroLLM dependencies (bitsandbytes, accelerate) …"

if $PYTHON -m pip install --no-cache-dir \
       "bitsandbytes>=0.41.0" "accelerate>=0.27.0" \
       2>&1 | tail -3 | tee -a "$LOGFILE"; then
    log_success "bitsandbytes + accelerate installed"
else
    log_warn "bitsandbytes/accelerate install failed — EuroLLM will run bfloat16 (~18 GB VRAM)"
fi

# ── Step 8d: VoxCPM2 TTS ──────────────────────────────────────────────────────

log_step "Installing VoxCPM2 (primary TTS) …"

if $PYTHON -m pip install --no-cache-dir voxcpm \
       2>&1 | tail -3 | tee -a "$LOGFILE"; then
    log_success "VoxCPM2 installed"
else
    log_warn "VoxCPM2 install failed — pipeline will fall back to Coqui XTTS v2"
fi

# ── Step 8e: Coqui TTS / XTTS v2 (fallback TTS) ──────────────────────────────

log_step "Installing Coqui TTS / XTTS v2 (fallback) …"
# The original 'TTS' package was removed from PyPI after Coqui AI dissolved.
# 'coqui-tts' is the community-maintained fork; same TTS.* import namespace.

TTS_INSTALLED=0

if COQUI_TOS_AGREED=1 $PYTHON -m pip install --no-cache-dir coqui-tts \
       2>&1 | tail -5 | tee -a "$LOGFILE"; then
    log_success "coqui-tts installed"
    TTS_INSTALLED=1
else
    log_warn "coqui-tts from PyPI failed — trying GitHub source …"
    if COQUI_TOS_AGREED=1 $PYTHON -m pip install --no-cache-dir \
           "git+https://github.com/coqui-ai/TTS.git" \
           2>&1 | tail -5 | tee -a "$LOGFILE"; then
        log_success "Coqui TTS installed from GitHub"
        TTS_INSTALLED=1
    else
        log_warn "Coqui TTS install failed — only VoxCPM2 will be available"
    fi
fi

# ── Step 8f: WhisperX (SRT forced alignment) ──────────────────────────────────

log_step "Installing WhisperX (SRT force-alignment) …"

if $PYTHON -m pip install --no-cache-dir whisperx \
       2>&1 | tail -3 | tee -a "$LOGFILE"; then
    log_success "WhisperX installed"
else
    log_warn "WhisperX install failed — SRT will use Whisper timestamps (slightly less accurate)"
fi

# NOTE: transformers downgrade is NOT attempted here — coqui-tts re-upgrades it
# during dep resolution and wins every time.  02_pipeline.py patches the
# missing symbols at runtime (isin_mps_friendly + is_torch_greater_or_equal).

# ── Step 9: Utilities ─────────────────────────────────────────────────────

log_step "Installing utility packages …"

if $PYTHON -m pip install --no-cache-dir \
       "pysrt>=1.1.2" "requests>=2.32.0" \
       "tqdm>=4.66.0" "click>=8.1.7" "pyyaml>=6.0.1" \
       "python-dotenv>=1.0.0" "hf_transfer>=0.1.6" \
       2>&1 | tail -3 | tee -a "$LOGFILE"; then
    log_success "Utilities installed"
else
    log_error "Utility package installation failed"
fi

# hf_transfer is opt-in via env var; export now and persist in /workspace/.env
# so it's active for all subsequent pre-downloads and pipeline runs.
export HF_HUB_ENABLE_HF_TRANSFER=1
grep -q "HF_HUB_ENABLE_HF_TRANSFER" /workspace/.env 2>/dev/null \
    || echo "HF_HUB_ENABLE_HF_TRANSFER=1" >> /workspace/.env

# ── Step 10: Ollama ───────────────────────────────────────────────────────

log_step "Setting up Ollama …"

if command -v ollama &>/dev/null; then
    log_success "Ollama already installed: $(ollama --version 2>&1 | head -1)"
else
    if curl -fsSL https://ollama.ai/install.sh | sh 2>&1 | tail -5 | tee -a "$LOGFILE"; then
        log_success "Ollama installed"
    else
        log_error "Ollama installation failed — run manually: curl -fsSL https://ollama.ai/install.sh | sh"
    fi
fi

# ── Step 11: Start Ollama service ─────────────────────────────────────────

log_step "Starting Ollama service …"

if pgrep -x ollama > /dev/null 2>&1; then
    log_success "Ollama already running"
elif command -v ollama &>/dev/null; then
    nohup ollama serve > /workspace/logs/ollama.log 2>&1 &
    # Wait up to 30 s for readiness
    READY=0
    for i in $(seq 1 15); do
        sleep 2
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            log_success "Ollama started ($(ollama --version 2>&1 | head -1))"
            READY=1
            break
        fi
    done
    [[ $READY -eq 0 ]] && log_error "Ollama did not start in 30 s — check /workspace/logs/ollama.log"
else
    log_warn "Ollama not installed — skipping service start"
fi

# ── Step 12: Pull Qwen2.5:14b ─────────────────────────────────────────────

pull_with_retry() {
    local model="$1"
    local max_tries=3
    local attempt
    for attempt in $(seq 1 $max_tries); do
        log_step "Pulling $model (attempt $attempt/$max_tries) …"
        if ollama pull "$model" 2>&1 | tee -a "$LOGFILE"; then
            log_success "$model ready"
            return 0
        fi
        [[ $attempt -lt $max_tries ]] && sleep 10
    done
    log_error "Could not pull $model — run manually: ollama pull $model"
    return 1
}

if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    pull_with_retry "qwen3:14b"
else
    log_warn "Ollama not reachable — skipping model download. Run: ollama pull qwen3:14b"
fi

# ── Step 12b: HuggingFace token (EuroLLM-9B is a gated model) ────────────────

log_step "Configuring HuggingFace access for EuroLLM-9B-Instruct …"

# Load from project-root .env (where user keeps it) and /workspace/.env if present.
for envfile in "$SCRIPT_DIR/.env" /workspace/.env; do
    [[ -f "$envfile" ]] && set -a && source "$envfile" 2>/dev/null && set +a || true
done

# Accept any of the canonical HF env-var names; normalize to HF_TOKEN.
HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}}"

if [[ -z "$HF_TOKEN" ]]; then
    echo ""
    echo -e "${YELLOW}EuroLLM-9B-Instruct is a gated HuggingFace model.${NC}"
    echo "  1. Request access: https://huggingface.co/utter-project/EuroLLM-9B-Instruct"
    echo "  2. Get your token: https://huggingface.co/settings/tokens (read permission)"
    echo ""
    read -rp "  Enter HuggingFace token (or press Enter to skip): " HF_TOKEN
    echo ""
fi

if [[ -n "$HF_TOKEN" ]]; then
    # Strip non-ASCII/non-token chars (e.g. Unicode ellipsis from copy-paste truncation)
    HF_TOKEN=$(echo "$HF_TOKEN" | LC_ALL=C tr -cd 'A-Za-z0-9_-')
    export HF_TOKEN="$HF_TOKEN"
    # Persist so future setup runs and pipeline runs can find it
    grep -q "HF_TOKEN" /workspace/.env 2>/dev/null \
        || echo "HF_TOKEN=\"$HF_TOKEN\"" >> /workspace/.env

    if $PYTHON -c "
from huggingface_hub import HfApi
try:
    info = HfApi().whoami(token='$HF_TOKEN')
    print(f\"Authenticated as: {info.get('name', 'unknown')}\")
except Exception as e:
    print(f'Auth failed: {e}')
    raise
" 2>&1 | tee -a "$LOGFILE"; then
        log_success "HuggingFace authenticated — token saved to /workspace/.env"
    else
        log_warn "HuggingFace token invalid — EuroLLM model download will fail"
    fi
else
    log_warn "No HuggingFace token provided — set HF_TOKEN env var before first pipeline run"
fi

# ── Step 13: Pre-download Whisper large-v3 ────────────────────────────────

log_step "Pre-downloading Whisper large-v3 (~3 GB) …"

if $PYTHON - <<'PYEOF' 2>&1 | tee -a "$LOGFILE"; then
from faster_whisper import WhisperModel
print("Downloading Whisper large-v3 to /workspace/models/whisper …")
m = WhisperModel("large-v3", device="cpu", compute_type="int8",
                 download_root="/workspace/models/whisper")
del m
print("✓ Whisper large-v3 cached")
PYEOF
    log_success "Whisper large-v3 cached"
else
    log_warn "Whisper download failed — it will auto-download on first use"
fi

# ── Step 14: Pre-download VoxCPM2 (~4 GB) and EuroLLM tokenizer ──────────────

log_step "Pre-downloading VoxCPM2 model (~4 GB) …"

if $PYTHON - <<'PYEOF' 2>&1 | tee -a "$LOGFILE"; then
try:
    from voxcpm import VoxCPM
    print("Downloading VoxCPM2 weights from HuggingFace …")
    m = VoxCPM.from_pretrained("openbmb/VoxCPM2")
    del m
    print("✓ VoxCPM2 cached")
except ImportError:
    print("voxcpm not installed — skipping pre-download")
except Exception as e:
    print(f"VoxCPM2 download error: {e}")
    raise
PYEOF
    log_success "VoxCPM2 cached (or not installed)"
else
    log_warn "VoxCPM2 pre-download failed — it will download on first use"
fi

log_step "Pre-downloading EuroLLM-9B tokenizer …"

if $PYTHON - <<'PYEOF' 2>&1 | tee -a "$LOGFILE"; then
try:
    from transformers import AutoTokenizer
    print("Caching EuroLLM tokenizer …")
    AutoTokenizer.from_pretrained("utter-project/EuroLLM-9B-Instruct")
    print("✓ EuroLLM tokenizer cached (weights download on first translate run)")
except Exception as e:
    print(f"EuroLLM tokenizer download error: {e}")
    raise
PYEOF
    log_success "EuroLLM tokenizer cached"
else
    log_warn "EuroLLM tokenizer pre-download failed — will download on first run"
fi

# ── Step 14b: Pre-download XTTS v2 fallback (~2 GB) ──────────────────────────

log_step "Pre-downloading XTTS v2 fallback (~2 GB) …"

if [[ $TTS_INSTALLED -eq 0 ]]; then
    log_warn "Skipping XTTS v2 download — TTS package not installed"
elif COQUI_TOS_AGREED=1 $PYTHON - <<'PYEOF' 2>&1 | tee -a "$LOGFILE"; then
import os, importlib
os.environ["COQUI_TOS_AGREED"] = "1"

import torch
from packaging.version import Version as _V

# Patch all symbols removed from transformers in 4.41+ that coqui-tts still imports
def _patch():
    # transformers.pytorch_utils.isin_mps_friendly
    try:
        m = importlib.import_module("transformers.pytorch_utils")
        if not hasattr(m, "isin_mps_friendly"):
            m.isin_mps_friendly = torch.isin
            print("Patched: transformers.pytorch_utils.isin_mps_friendly")
    except Exception as e:
        print(f"  skip pytorch_utils patch: {e}")

    # transformers.utils.import_utils.is_torch_greater_or_equal
    try:
        m = importlib.import_module("transformers.utils.import_utils")
        if not hasattr(m, "is_torch_greater_or_equal"):
            def _gte(version, revision=None):
                v = f"{version}.{revision}" if revision else version
                return _V(torch.__version__.split("+")[0]) >= _V(v)
            m.is_torch_greater_or_equal = _gte
            tu = importlib.import_module("transformers.utils")
            if not hasattr(tu, "is_torch_greater_or_equal"):
                tu.is_torch_greater_or_equal = _gte
            print("Patched: transformers.utils.import_utils.is_torch_greater_or_equal")
    except Exception as e:
        print(f"  skip import_utils patch: {e}")

_patch()

try:
    from TTS.api import TTS
    print("Downloading XTTS v2 …")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
    del tts
    print("✓ XTTS v2 cached")
except Exception as e:
    print(f"XTTS download error: {e}")
    raise
PYEOF
    log_success "XTTS v2 cached"
else
    log_warn "XTTS v2 download failed — it will auto-download on first use"
fi

# ── Step 15: Pin numpy (must run LAST — coqui-tts dep chain upgrades it) ──

log_step "Force-pinning numpy<2.0.0 (coqui-tts compatibility) …"
# coqui-tts pulls numpy 2.x as a transitive dep; force it back after everything
# else is installed so it wins regardless of dep resolution order.
$PYTHON -m pip install --no-cache-dir --force-reinstall "numpy>=1.26.4,<2.0.0" \
    2>&1 | tail -3 | tee -a "$LOGFILE" \
    && log_success "numpy pinned to 1.26.x" \
    || log_warn "numpy pin failed — coqui-tts audio ops may behave unexpectedly"

# ── Step 16: Install scripts & config ─────────────────────────────────────

log_step "Installing pipeline scripts …"

for script in 02_pipeline.py 03_batch_runner.py verify_setup.py; do
    if [[ -f "$SCRIPT_DIR/$script" ]]; then
        cp "$SCRIPT_DIR/$script" /workspace/scripts/
        chmod +x "/workspace/scripts/$script"
        log_success "$script → /workspace/scripts/"
    else
        log_warn "$script not found in $SCRIPT_DIR — copy manually to /workspace/scripts/"
    fi
done

# Install config only if one doesn't already exist
if [[ -f "$SCRIPT_DIR/config.yaml" ]]; then
    if [[ ! -f /workspace/config.yaml ]]; then
        cp "$SCRIPT_DIR/config.yaml" /workspace/config.yaml
        log_success "config.yaml → /workspace/config.yaml"
    else
        log_success "config.yaml already at /workspace/config.yaml (not overwritten)"
    fi
else
    log_warn "config.yaml not found in $SCRIPT_DIR"
fi

# ── Final Summary ──────────────────────────────────────────────────────────

echo ""
echo "==========================================" | tee -a "$LOGFILE"
if [[ ${#ERRORS[@]} -eq 0 ]]; then
    echo -e "${GREEN}Setup complete — no errors!${NC}" | tee -a "$LOGFILE"
else
    echo -e "${YELLOW}Setup complete with ${#ERRORS[@]} error(s):${NC}" | tee -a "$LOGFILE"
    for err in "${ERRORS[@]}"; do
        echo -e "  ${RED}✗ $err${NC}" | tee -a "$LOGFILE"
    done
fi
if [[ ${#WARNINGS[@]} -gt 0 ]]; then
    echo -e "${YELLOW}Warnings (${#WARNINGS[@]}):${NC}" | tee -a "$LOGFILE"
    for warn in "${WARNINGS[@]}"; do
        echo -e "  ${YELLOW}⚠ $warn${NC}" | tee -a "$LOGFILE"
    done
fi
echo "==========================================" | tee -a "$LOGFILE"
echo ""
echo "Next steps:"
echo "  1. Verify:  python /workspace/scripts/verify_setup.py"
echo "  2. Copy:    cp *.mp4 /workspace/videos/input/"
echo "  3. Single:  python /workspace/scripts/02_pipeline.py \\"
echo "                --video /workspace/videos/input/webinar.mp4"
echo "  4. Batch:   python /workspace/scripts/03_batch_runner.py"
echo ""
echo "Setup log: $LOGFILE"
echo ""

[[ ${#ERRORS[@]} -gt 0 ]] && exit 1 || exit 0
