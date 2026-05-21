#!/usr/bin/env python3
"""
French Dubbing Pipeline v3.0

English webinar → French audio track + SRT subtitles

Stack:
  Source separation : Demucs htdemucs         (vocals + background split)
  Transcription     : faster-whisper large-v3  (CTranslate2, float16, word timestamps)
  Translation       : EuroLLM-9B-Instruct      (European language specialist, on-GPU)
  Review            : Qwen2.5:14b via Ollama   (French naturalness / idiom pass)
  Speaker denoising : DeepFilterNet            (clean voice reference for cloning)
  TTS               : VoxCPM2 (2B, 48 kHz)    (diffusion AR, Ultimate Cloning mode)
  Assembly          : FFmpeg atempo            (fit French into original timing windows)
  SRT alignment     : WhisperX                (force-align French text to French audio)
  Output            : AAC 192 kbps 48000 Hz stereo  +  UTF-8 SRT
"""

import gc
import logging
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

# Enable accelerated HF downloads. MUST be set before huggingface_hub is imported.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ["COQUI_TOS_AGREED"] = "1"

# Load .env from project root (and /workspace/.env on RunPod) so HF_TOKEN is
# available without requiring the user to `source` it in their shell.
def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    for candidate in (
        Path(__file__).resolve().parent / ".env",
        Path("/workspace/.env"),
    ):
        if candidate.exists():
            load_dotenv(candidate, override=False)

_load_dotenv()

# Normalize the various HF token env var names HF libraries accept.
_hf_tok = (
    os.environ.get("HF_TOKEN")
    or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    or ""
)
if _hf_tok:
    os.environ["HF_TOKEN"] = _hf_tok
    os.environ["HUGGING_FACE_HUB_TOKEN"] = _hf_tok

import click
import librosa
import numpy as np
import pysrt
import requests
import soundfile as sf
import torch
import yaml
from pysrt import SubRipItem, SubRipTime
from tqdm import tqdm

from faster_whisper import WhisperModel

# ── Transformers compatibility shim ───────────────────────────────────────────
# Coqui XTTS v2 (fallback) imports symbols removed in transformers 4.41+.
# Patch before any TTS import so the import chain succeeds on any version.
def _patch_transformers_for_coqui():
    import importlib
    from packaging.version import Version as _V

    try:
        m = importlib.import_module("transformers.pytorch_utils")
        if not hasattr(m, "isin_mps_friendly"):
            m.isin_mps_friendly = torch.isin
    except Exception:
        pass

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
    except Exception:
        pass

_patch_transformers_for_coqui()

# Optional XTTS v2 — used only if voxcpm is not installed
try:
    from TTS.api import TTS as CoquiTTS
    HAS_XTTS = True
except ImportError:
    HAS_XTTS = False


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class PipelineConfig:
    input_folder: str
    output_folder: str
    models_folder: str
    logs_folder: str
    temp_folder: str

    # Transcription
    whisper_model: str = "large-v3"
    whisper_device: str = "cuda"
    whisper_compute_type: str = "float16"

    # Source separation
    use_demucs: bool = True
    demucs_model: str = "htdemucs"
    preserve_background: bool = True

    # Diarization — pyannote.audio (optional, requires HF token)
    use_diarization: bool = False
    diarization_model: str = "pyannote/speaker-diarization-3.1"
    diarization_min_speakers: int = 1
    diarization_max_speakers: int = 10
    diarization_profile_duration: float = 25.0   # seconds of audio to collect per speaker

    # Translation — EuroLLM primary, Qwen review (or Qwen primary via --translator)
    eurollm_model: str = "utter-project/EuroLLM-9B-Instruct"
    eurollm_quantize: bool = True          # 8-bit to save VRAM (~9 GB vs ~18 GB)
    translation_model: str = "qwen3:14b"   # Ollama — review pass or primary backend
    translation_backend: str = "eurollm"   # "eurollm" | "qwen" | "gemini"
    gemini_model: str = "gemini-2.5-flash"
    gemini_api_key: str = ""
    translation_temperature: float = 0.3
    translation_batch_size: int = 10
    translation_review: bool = True
    # Target language for the dubbed output (drives CPS budget + TTS language).
    target_lang: str = "fr"
    # Safety multiplier on the per-segment character budget (1.1 = +10% headroom).
    cps_safety: float = 1.1
    # Also produce an unconstrained "natural" translation for use in SRT subtitles.
    translate_natural_pass: bool = True

    # Speaker reference
    use_deepfilter: bool = True
    tts_speaker_duration: float = 25.0
    tts_speaker_skip: float = 20.0         # skip past intro/title cards

    # TTS — VoxCPM2 primary, XTTS v2 fallback
    tts_model: str = "openbmb/VoxCPM2"
    tts_max_stretch: float = 1.10
    tts_cfg_value: float = 2.5
    tts_inference_timesteps: int = 24
    # Ultimate Cloning passes the reference transcript. For cross-lingual dubbing
    # (English reference → French output) it tends to bleed English phonemes.
    tts_use_prompt_text: bool = False

    # HuggingFace token — required for gated models (EuroLLM-9B-Instruct)
    huggingface_token: str = ""

    # Locale and vocabulary enforcement
    # "fr"    → standard European French (no glossary applied)
    # "fr-ca" → Canadian/Québécois French (glossary injected + post-processed)
    locale: str = "fr"
    glossary_path: str = ""

    # Segment merging
    segment_merge_gap: float = 0.8
    segment_merge_max_duration: float = 8.0

    # SRT
    use_whisperx_alignment: bool = True
    subtitle_offset_ms: int = 0
    # Use the natural (unconstrained) translation for SRT text instead of the
    # length-fitted one used for audio. Better readability when both are available.
    subtitles_use_natural: bool = True

    synthesis_sample_rate: int = 48000     # VoxCPM2 native
    output_sample_rate: int = 48000        # Vimeo accepts 48 kHz

    timeout_seconds: int = 7200


@dataclass
class GlossaryEntry:
    en: str
    fr_ca: str
    fr_std: str = ""
    mode: str = "suggest"   # "always" | "suggest"
    category: str = ""
    note: str = ""


@dataclass
class Glossary:
    entries: List[GlossaryEntry]
    formatting_rules: List[str]
    inclusive_language: List[str]

    @property
    def has_content(self) -> bool:
        return bool(self.entries or self.formatting_rules or self.inclusive_language)


def load_config(path: str) -> PipelineConfig:
    with open(path) as f:
        c = yaml.safe_load(f)
    p    = c.get("pipeline", {})
    w    = c.get("whisper", {})
    t    = c.get("translation", {})
    tts  = c.get("tts", {})
    proc = c.get("processing", {})
    aud  = c.get("audio", {})
    sep  = c.get("source_separation", {})
    sub  = c.get("subtitles", {})
    return PipelineConfig(
        input_folder=p.get("input_folder", "/workspace/videos/input"),
        output_folder=p.get("output_folder", "/workspace/outputs"),
        models_folder=p.get("models_folder", "/workspace/models"),
        logs_folder=p.get("logs_folder", "/workspace/logs"),
        temp_folder=p.get("temp_folder", "/workspace/temp"),
        whisper_model=w.get("model", "large-v3"),
        whisper_device=w.get("device", "cuda"),
        whisper_compute_type=w.get("compute_type", "float16"),
        use_demucs=sep.get("enabled", True),
        demucs_model=sep.get("model", "htdemucs"),
        preserve_background=sep.get("preserve_background", True),
        use_diarization=c.get("diarization", {}).get("enabled", False),
        diarization_model=c.get("diarization", {}).get("model", "pyannote/speaker-diarization-3.1"),
        diarization_min_speakers=c.get("diarization", {}).get("min_speakers", 1),
        diarization_max_speakers=c.get("diarization", {}).get("max_speakers", 10),
        diarization_profile_duration=c.get("diarization", {}).get("profile_duration", 25.0),
        eurollm_model=t.get("eurollm_model", "utter-project/EuroLLM-9B-Instruct"),
        eurollm_quantize=t.get("eurollm_quantize", True),
        translation_model=t.get("model", "qwen3:14b"),
        translation_backend=t.get("backend", "eurollm"),
        translation_temperature=t.get("temperature", 0.3),
        translation_batch_size=t.get("batch_size", 10),
        translation_review=t.get("review_pass", True),
        target_lang=t.get("target_lang", "fr"),
        cps_safety=t.get("cps_safety", 1.1),
        translate_natural_pass=t.get("natural_pass", True),
        use_deepfilter=tts.get("use_deepfilter", True),
        tts_speaker_duration=tts.get("speaker_profile_duration", 25.0),
        tts_speaker_skip=tts.get("speaker_profile_skip", 20.0),
        tts_model=tts.get("model", "openbmb/VoxCPM2"),
        tts_max_stretch=tts.get("max_stretch", 1.10),
        tts_cfg_value=tts.get("cfg_value", 2.5),
        tts_inference_timesteps=tts.get("inference_timesteps", 24),
        tts_use_prompt_text=tts.get("use_prompt_text", False),
        huggingface_token=(
            t.get("huggingface_token", "")
            or os.environ.get("HF_TOKEN", "")
        ),
        gemini_model=t.get("gemini_model", "gemini-2.5-flash"),
        gemini_api_key=(
            t.get("gemini_api_key", "")
            or os.environ.get("GEMINI_API_KEY", "")
        ),
        locale=t.get("locale", "fr"),
        glossary_path=t.get("glossary_path", ""),
        segment_merge_gap=tts.get("segment_merge_gap", 0.8),
        segment_merge_max_duration=tts.get("segment_merge_max_duration", 8.0),
        use_whisperx_alignment=sub.get("whisperx_alignment", True),
        subtitle_offset_ms=sub.get("sync_offset_ms", 0),
        subtitles_use_natural=sub.get("use_natural_translation", True),
        synthesis_sample_rate=aud.get("synthesis_sample_rate", 48000),
        output_sample_rate=aud.get("output_sample_rate", 48000),
        timeout_seconds=proc.get("timeout_seconds", 7200),
    )


def setup_logging(log_dir: str, name: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def free_vram(log: Optional[logging.Logger] = None) -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if log:
            used_gb = torch.cuda.memory_allocated() / 1e9
            log.debug(f"VRAM after cleanup: {used_gb:.1f} GB")


# ============================================================================
# Glossary — Canadian French vocabulary enforcement
# ============================================================================

def load_glossary(path: str, log: logging.Logger) -> "Glossary":
    """Load a YAML glossary file and return a Glossary with terms, formatting rules,
    and inclusive language rules."""
    empty = Glossary(entries=[], formatting_rules=[], inclusive_language=[])
    if not path:
        return empty
    if not os.path.exists(path):
        log.warning(f"Glossary file not found: {path} — continuing without glossary")
        return empty
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        entries = [
            GlossaryEntry(
                en=str(t.get("en", "")),
                fr_ca=str(t.get("fr_ca", "")),
                fr_std=str(t.get("fr_std", "")),
                mode=str(t.get("mode", "suggest")),
                category=str(t.get("category", "")),
                note=str(t.get("note", "")),
            )
            for t in (data.get("terms") or [])
            if t.get("fr_ca")
        ]
        formatting_rules  = [str(r) for r in (data.get("formatting_rules")  or [])]
        inclusive_language = [str(r) for r in (data.get("inclusive_language") or [])]
        glossary = Glossary(
            entries=entries,
            formatting_rules=formatting_rules,
            inclusive_language=inclusive_language,
        )
        log.info(
            f"✓ Glossary loaded: {len(entries)} terms, "
            f"{len(formatting_rules)} formatting rules, "
            f"{len(inclusive_language)} inclusive language rules ({path})"
        )
        return glossary
    except Exception as e:
        log.warning(f"Glossary load failed ({e}) — continuing without glossary")
        return empty


def _build_glossary_section(glossary: "Glossary", locale: str) -> str:
    """Build the full prompt injection block from all three glossary sections."""
    if locale != "fr-ca" or not glossary.has_content:
        return ""

    blocks: List[str] = []

    if glossary.entries:
        lines = ["MANDATORY VOCABULARY — use Québécois/Canadian French forms:"]
        for e in glossary.entries:
            line = f"  {e.en} → {e.fr_ca}"
            if e.fr_std and e.fr_std.lower() != e.fr_ca.lower():
                line += f"  (NOT: {e.fr_std})"
            if e.note:
                line += f"  [{e.note}]"
            lines.append(line)
        blocks.append("\n".join(lines))

    if glossary.formatting_rules:
        lines = ["FORMATTING RULES (Canadian French / CAPS standards):"]
        for rule in glossary.formatting_rules:
            lines.append(f"  • {rule}")
        blocks.append("\n".join(lines))

    if glossary.inclusive_language:
        lines = ["INCLUSIVE LANGUAGE (CAPS standard — apply to every translation):"]
        for rule in glossary.inclusive_language:
            lines.append(f"  • {rule}")
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks) + "\n"


def _match_case(original: str, replacement: str) -> str:
    """Preserve ALL-CAPS or Title-Case of the matched token in the replacement."""
    if original.isupper():
        return replacement.upper()
    if original and original[0].isupper():
        return replacement[0].upper() + replacement[1:] if replacement else replacement
    return replacement


def apply_glossary(
    segments: List[dict],
    entries: List[GlossaryEntry],
    log: logging.Logger,
    text_keys: Tuple[str, ...] = ("text_fr", "text_fr_natural"),
) -> List[dict]:
    """Post-process translated segments applying 'always' glossary substitutions.

    'mode: suggest' entries are handled entirely via prompt injection.
    'mode: always' entries are substituted here deterministically.

    Per entry, substitution order (first match at each position wins):
      1. fr_std form found in the translation → replaced with fr_ca
      2. English term left untranslated        → replaced with fr_ca
    Original token case (Title, ALL-CAPS, lowercase) is preserved.
    """
    always = [e for e in entries if e.mode == "always"]
    if not always:
        return segments

    out = [dict(s) for s in segments]
    total_subs = 0

    for seg in out:
        for key in text_keys:
            text = seg.get(key)
            if not text:
                continue
            for e in always:
                if e.fr_std:
                    new = re.sub(
                        r"\b" + re.escape(e.fr_std) + r"\b",
                        lambda m, rep=e.fr_ca: _match_case(m.group(), rep),
                        text,
                        flags=re.IGNORECASE,
                    )
                    if new != text:
                        total_subs += 1
                        text = new
                if e.en:
                    new = re.sub(
                        r"\b" + re.escape(e.en) + r"\b",
                        lambda m, rep=e.fr_ca: _match_case(m.group(), rep),
                        text,
                        flags=re.IGNORECASE,
                    )
                    if new != text:
                        total_subs += 1
                        text = new
            seg[key] = text

    log.info(f"✓ Glossary: {total_subs} substitution(s) across {len(out)} segments")
    return out


# ============================================================================
# Step 0: Source Separation — Demucs
# ============================================================================

def separate_vocals(
    video_path: str,
    temp_dir: str,
    model_name: str,
    log: logging.Logger,
) -> Tuple[Optional[str], Optional[str]]:
    """Separate speaker vocals from background using Demucs.

    Cleaner vocals → better Whisper accuracy and better voice-clone reference.
    The background stem can optionally be re-mixed back with the French audio.
    Returns (vocals_path, no_vocals_path); both None on failure.
    """
    log.info(f"[Demucs] Separating vocals — model: {model_name} …")
    try:
        subprocess.run(
            [
                sys.executable, "-m", "demucs",
                "--two-stems", "vocals",
                "-n", model_name,
                "--out", temp_dir,
                "--device", "cuda" if torch.cuda.is_available() else "cpu",
                video_path,
            ],
            check=True, capture_output=True, text=True, timeout=1800,
        )
        vocals_files    = sorted(Path(temp_dir).rglob("vocals.wav"))
        no_vocals_files = sorted(Path(temp_dir).rglob("no_vocals.wav"))

        if not vocals_files:
            log.error("Demucs output vocals.wav not found")
            return None, None

        vocals_path    = str(vocals_files[-1])
        no_vocals_path = str(no_vocals_files[-1]) if no_vocals_files else None
        log.info(f"✓ Vocals separated: {os.path.getsize(vocals_path) / 1e6:.1f} MB")
        return vocals_path, no_vocals_path

    except subprocess.TimeoutExpired:
        log.error("Demucs timed out (30 min) — falling back to raw audio")
    except Exception as e:
        log.error(f"Demucs failed ({e}) — falling back to raw audio")
    return None, None


# ============================================================================
# Step 1: Audio Extraction (fallback / duration probe)
# ============================================================================

def extract_audio(
    video_path: str,
    wav_path: str,
    sample_rate: int,
    log: logging.Logger,
) -> bool:
    log.info(f"Extracting audio → {Path(wav_path).name}")
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", video_path,
                "-vn", "-ac", "1",
                "-ar", str(sample_rate),
                "-acodec", "pcm_s16le",
                wav_path,
            ],
            check=True, capture_output=True, timeout=600,
        )
        log.info(f"✓ Audio extracted: {os.path.getsize(wav_path) / 1e6:.1f} MB")
        return True
    except Exception as e:
        log.error(f"Audio extraction failed: {e}")
        return False


def get_duration(video_path: str, log: logging.Logger) -> float:
    try:
        r = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            capture_output=True, text=True, timeout=30,
        )
        return float(r.stdout.strip())
    except Exception as e:
        log.warning(f"Could not read duration ({e}), defaulting to 3600 s")
        return 3600.0


# ============================================================================
# Step 2: Transcription — faster-whisper
# ============================================================================

def transcribe_audio(
    wav_path: str,
    model_name: str,
    device: str,
    compute_type: str,
    models_dir: str,
    log: logging.Logger,
) -> Optional[List[dict]]:
    log.info(f"Loading faster-whisper {model_name} [{compute_type}] …")
    try:
        model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
            download_root=os.path.join(models_dir, "whisper"),
        )
        log.info("Transcribing with VAD filter + word timestamps …")
        segments_gen, info = model.transcribe(
            wav_path,
            language="en",
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
            word_timestamps=True,
        )
        segments = [
            {"id": i, "start": s.start, "end": s.end, "text": s.text.strip()}
            for i, s in enumerate(segments_gen)
            if s.text.strip()
        ]
        log.info(f"✓ {len(segments)} segments ({info.language} detected, {info.duration:.0f} s)")
        del model
        free_vram(log)
        return segments
    except Exception as e:
        log.error(f"Transcription failed: {e}")
        return None


# ============================================================================
# Step 2b: Segment merging
# ============================================================================

def merge_segments(
    segments: List[dict],
    max_gap: float,
    max_duration: float,
    log: logging.Logger,
) -> List[dict]:
    """Merge short Whisper fragments into sentence-level chunks.

    Synthesizing sub-second fragments individually produces robotic, unnatural
    output. Sentence-level chunks give VoxCPM2 the context needed for proper
    intonation and natural rhythm.
    """
    if not segments:
        return segments

    merged: List[dict] = []
    current = dict(segments[0])

    for seg in segments[1:]:
        gap          = seg["start"] - current["end"]
        combined_dur = seg["end"] - current["start"]
        ends_sent    = bool(re.search(r"[.!?]\s*$", current["text"].rstrip()))

        if gap <= max_gap and combined_dur <= max_duration and not ends_sent:
            current["end"]  = seg["end"]
            current["text"] = current["text"].rstrip() + " " + seg["text"].lstrip()
        else:
            merged.append(current)
            current = dict(seg)

    merged.append(current)
    for i, s in enumerate(merged):
        s["id"] = i

    log.info(f"✓ Merged {len(segments)} Whisper segments → {len(merged)} sentence-level chunks")
    return merged


# ============================================================================
# Step 2c: Speaker Diarization — pyannote.audio (optional)
# ============================================================================

def diarize_audio(
    wav_path: str,
    model_name: str,
    hf_token: str,
    min_speakers: int,
    max_speakers: int,
    log: logging.Logger,
) -> Optional[List[Tuple[float, float, str]]]:
    """Run pyannote.audio speaker diarization on a mono WAV file.

    Returns a list of (start_s, end_s, speaker_label) tuples, or None on failure.
    Requires: pip install pyannote.audio
    The HF token must have accepted the model license at:
      https://huggingface.co/pyannote/speaker-diarization-3.1
    """
    try:
        from pyannote.audio import Pipeline as PyannotePipeline
    except ImportError:
        log.error(
            "pyannote.audio not installed.\n"
            "  Fix: pip install pyannote.audio"
        )
        return None

    try:
        log.info(f"Loading diarization model: {model_name} …")
        pipeline = PyannotePipeline.from_pretrained(model_name, use_auth_token=hf_token)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipeline.to(device)

        diarize_kwargs: dict = {}
        if min_speakers > 1:
            diarize_kwargs["min_speakers"] = min_speakers
        if max_speakers < 50:
            diarize_kwargs["max_speakers"] = max_speakers

        log.info("Running speaker diarization …")
        annotation = pipeline(wav_path, **diarize_kwargs)

        turns = [
            (turn.start, turn.end, speaker)
            for turn, _, speaker in annotation.itertracks(yield_label=True)
        ]
        speaker_ids = sorted({t[2] for t in turns})
        log.info(f"✓ Diarization complete — {len(speaker_ids)} speaker(s): {speaker_ids}")
        del pipeline
        free_vram(log)
        return turns

    except Exception as e:
        log.error(f"Diarization failed: {e}")
        return None


def assign_speakers(
    segments: List[dict],
    turns: List[Tuple[float, float, str]],
) -> List[dict]:
    """Tag each segment with the speaker that occupies the most of its duration."""
    out = [dict(s) for s in segments]
    for seg in out:
        seg_start, seg_end = seg["start"], seg["end"]
        best_spk, best_overlap = "SPEAKER_00", 0.0
        for (t_start, t_end, speaker) in turns:
            if t_end <= seg_start or t_start >= seg_end:
                continue
            overlap = min(t_end, seg_end) - max(t_start, seg_start)
            if overlap > best_overlap:
                best_overlap, best_spk = overlap, speaker
        seg["speaker"] = best_spk
    return out


def build_speaker_profiles(
    vocals_wav: str,
    segments: List[dict],
    temp_dir: str,
    profile_duration: float,
    log: logging.Logger,
    use_deepfilter: bool = True,
) -> dict:
    """Build a per-speaker voice-clone reference clip.

    Collects each speaker's longest utterances (up to profile_duration seconds),
    concatenates them, resamples to 16 kHz (VoxCPM2 input spec), and optionally
    denoises. Returns {speaker_id: wav_path} — value is None when a speaker has
    fewer than 3 s of usable audio.
    """
    from collections import defaultdict

    TARGET_SR = 16000
    MIN_PROFILE_S = 3.0

    by_speaker: dict = defaultdict(list)
    for seg in segments:
        spk = seg.get("speaker", "SPEAKER_00")
        by_speaker[spk].append(seg)

    log.info(f"  Loading vocals at {TARGET_SR} Hz for profile extraction …")
    try:
        full_audio, _ = librosa.load(vocals_wav, sr=TARGET_SR, mono=True)
    except Exception as e:
        log.error(f"Cannot load vocals for speaker profiles: {e}")
        return {}

    total_s = len(full_audio) / TARGET_SR
    profiles: dict = {}

    for speaker, spk_segs in by_speaker.items():
        # Longest segments first — maximises voice fidelity per second collected
        spk_segs = sorted(spk_segs, key=lambda s: s["end"] - s["start"], reverse=True)
        chunks: List[np.ndarray] = []
        collected = 0.0

        for seg in spk_segs:
            if collected >= profile_duration:
                break
            s_start = max(0.0, float(seg["start"]))
            s_end   = min(total_s, float(seg["end"]))
            dur     = s_end - s_start
            if dur < 0.5:
                continue
            want      = min(dur, profile_duration - collected)
            idx_start = int(s_start * TARGET_SR)
            idx_end   = int((s_start + want) * TARGET_SR)
            chunks.append(full_audio[idx_start:idx_end])
            collected += want

        if not chunks or collected < MIN_PROFILE_S:
            log.warning(f"  {speaker}: only {collected:.1f}s available — skipping profile (need ≥{MIN_PROFILE_S}s)")
            profiles[speaker] = None
            continue

        combined = np.concatenate(chunks)
        raw_path = os.path.join(temp_dir, f"profile_{speaker}_raw.wav")
        sf.write(raw_path, combined, TARGET_SR)

        if use_deepfilter:
            denoised_path = os.path.join(temp_dir, f"profile_{speaker}_denoised.wav")
            profiles[speaker] = denoise_audio(raw_path, denoised_path, log)
        else:
            profiles[speaker] = raw_path

        log.info(f"  {speaker}: {collected:.1f}s profile built → {profiles[speaker]}")

    return profiles


# ============================================================================
# Step 3: Translation — EuroLLM-9B primary + Qwen2.5 review
# ============================================================================

# Measured average TTS characters-per-second by language. Source: ZastTranslate
# (fitted_cps_config.py). Used to compute a per-segment character budget so the
# LLM produces a translation that fits the original audio window.
LANG_CPS = {
    # Latin (European) — same family, similar VoxCPM2 speaking rate
    "fr": 9.0, "es": 9.0, "it": 9.0, "pt": 9.0,
    "en": 14.0, "de": 9.0, "nl": 9.0, "da": 9.0, "sv": 9.0, "no": 9.0, "fi": 9.0,
    "pl": 9.0, "tr": 9.0, "id": 9.0, "ms": 9.0, "tl": 9.0, "sw": 9.0,
    # Cyrillic
    "ru": 9.0, "el": 9.0,
    # CJK — fewer characters per second of speech
    "zh": 5.0, "ja": 5.5, "ko": 6.0,
    # Other scripts
    "ar": 8.0, "he": 8.0, "hi": 7.5, "th": 7.0, "vi": 9.0,
    "my": 7.0, "km": 7.0, "lo": 7.0,
}
_DEFAULT_CPS = 7.5

def _budget_chars(seg: dict, lang: str, safety: float) -> int:
    """Per-segment max character count for a length-fitted translation."""
    duration = max(0.5, float(seg["end"]) - float(seg["start"]))
    cps = LANG_CPS.get(lang, _DEFAULT_CPS)
    return max(20, int(duration * cps * safety))


_LANG_NAMES = {
    "fr": "French", "es": "Spanish", "de": "German", "it": "Italian",
    "pt": "Portuguese", "nl": "Dutch", "pl": "Polish", "ru": "Russian",
    "ja": "Japanese", "ko": "Korean", "zh": "Chinese", "ar": "Arabic",
    "tr": "Turkish", "hi": "Hindi", "vi": "Vietnamese",
}

_TRANSLATE_PROMPT = """\
You are a professional {language} dubbing translator.
Translate each numbered English segment into natural, conversational {language}
that fits a strict character budget for lip-sync.

CRITICAL RULES:
- Each segment has a "(MAX N chars)" budget. Your translation MUST fit within it.
- Be BRUTALLY concise: remove fillers, redundancies, and secondary details.
- Use contractions, spoken-language forms, and shorter synonyms.
- Adapt idioms naturally; do not translate literally.
- Preserve key technical terms and proper nouns.
- Output ONLY the numbered translations, one per line, same numbering as input.
- No explanations, notes, or extra commentary.
{glossary_section}
English segments with budgets:
{segments}

{language} translations:"""

_TRANSLATE_NATURAL_PROMPT = """\
You are a professional {language} translator for video subtitles.
Translate each numbered English segment into fluent, natural {language}.
Preserve meaning, tone, and nuance — readability is the priority, no length limit.

- Output ONLY the numbered translations, one per line, same numbering as input.
- No explanations, notes, or extra commentary.
{glossary_section}
English segments:
{segments}

{language} translations:"""

_REVIEW_PROMPT = """\
You are a native {language} expert reviewing dubbed video subtitles.{locale_note}
Correct any unnatural phrasing, Anglicisms, grammar errors, or register issues.

CRITICAL CONSTRAINTS:
- Output must be the SAME LENGTH OR SHORTER than the input (character count).
  This is for video dubbing; longer text breaks lip-sync.
- Prefer shorter, conversational {language} over formal/literary phrasing.
- Keep changes minimal — only fix what is actually incorrect.
- Do NOT add filler words or expand abbreviations.
- Output only the corrected numbered list, same numbering as input.
{glossary_section}
{language} subtitles to review:
{segments}

Corrected {language} subtitles:"""


def check_ollama(model: str, log: logging.Logger) -> bool:
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
    except requests.exceptions.ConnectionError:
        log.error(
            "Ollama is NOT running.\n"
            "  Start it: nohup ollama serve > /workspace/logs/ollama.log 2>&1 &\n"
            "  Wait 5 s, then re-run."
        )
        return False

    if r.status_code != 200:
        log.error(f"Ollama returned HTTP {r.status_code}")
        return False

    available  = [m["name"] for m in r.json().get("models", [])]
    model_base = model.split(":")[0]
    if not any(model_base in m for m in available):
        log.error(
            f"Model '{model}' not found.\n"
            f"  Available: {available or ['(none)']}\n"
            f"  Fix: ollama pull {model}"
        )
        return False

    log.info(f"✓ Ollama ready — '{model}' available")
    return True


def _verify_translation_quality(segments: List[dict], log: logging.Logger) -> None:
    unchanged = sum(1 for s in segments if s.get("text_fr") == s["text"])
    pct = 100 * unchanged / max(len(segments), 1)
    if pct > 50:
        log.error(
            f"TRANSLATION FAILURE: {unchanged}/{len(segments)} segments ({pct:.0f}%) "
            f"still in English. Check EuroLLM download and VRAM availability."
        )
    elif unchanged > 0:
        log.warning(f"{unchanged} segment(s) could not be translated — kept in English")


def _ollama_call(prompt: str, model: str, temperature: float, log: logging.Logger) -> Optional[str]:
    try:
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": 4096},
            },
            timeout=180,
        )
        if r.status_code == 200:
            return r.json().get("response", "").strip()
        log.warning(f"Ollama HTTP {r.status_code}")
        return None
    except requests.exceptions.ConnectionError:
        log.error("Cannot reach Ollama at localhost:11434.")
        return None
    except Exception as e:
        log.error(f"Ollama call failed: {e}")
        return None


def _gemini_call(
    prompt: str,
    model_name: str,
    temperature: float,
    api_key: str,
    log: logging.Logger,
) -> Optional[str]:
    try:
        import google.generativeai as genai
    except ImportError:
        log.error("google-generativeai not installed. Run: pip install google-generativeai")
        return None
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        resp = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=4096,
            ),
        )
        return resp.text.strip() if resp.text else None
    except Exception as e:
        log.error(f"Gemini API call failed: {e}")
        return None


def _parse_numbered(text: str, count: int) -> List[str]:
    # Strip Qwen3 chain-of-thought blocks emitted when /no_think isn't honoured
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    result: dict = {}
    for line in text.splitlines():
        m = re.match(r"^[\(\[]?(\d+)[\.\)\]]\s+(.*)", line.strip())
        if m:
            idx, content = int(m.group(1)), m.group(2).strip()
            if 1 <= idx <= count and content:
                result[idx] = content
    return [result.get(i + 1, "") for i in range(count)]


def _format_fitted_segments(batch: List[dict], lang: str, safety: float) -> Tuple[str, List[int]]:
    """Format a batch for the fitted-translation prompt and return per-segment budgets."""
    budgets = [_budget_chars(s, lang, safety) for s in batch]
    numbered = "\n".join(
        f"{i + 1}. (MAX {budgets[i]} chars) {s['text']}"
        for i, s in enumerate(batch)
    )
    return numbered, budgets


def _llm_generate(
    llm,
    tokenizer,
    prompt: str,
    temperature: float,
    max_new_tokens: int = 1024,
) -> str:
    """Single chat-template generation. Returns the decoded assistant response."""
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
        return_dict=True,
    ).to("cuda")

    with torch.no_grad():
        output_ids = llm.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            # Pass both eos and pad so the model can actually stop on EOS instead
            # of running to max_new_tokens when it would otherwise emit EOS.
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    ).strip()


def translate_segments(
    segments: List[dict],
    eurollm_model_name: str,
    use_quantize: bool,
    temperature: float,
    batch_size: int,
    log: logging.Logger,
    hf_token: str = os.environ.get("HF_TOKEN", ""),
    target_lang: str = "fr",
    cps_safety: float = 1.1,
    natural_pass: bool = True,
    glossary_section: str = "",
) -> List[dict]:
    """Translate with EuroLLM-9B-Instruct using a length-fitted prompt.

    Each segment gets a character budget from its time slot × LANG_CPS[target_lang].
    Overflowing segments are retried per-item with a 30% tighter budget (up to 3
    iterations). If natural_pass is True, a second unconstrained pass is run and
    stored in text_fr_natural for use in subtitles (and as a final fallback when
    fitted retries exhaust).

    Runs on-GPU via HuggingFace transformers (8-bit quantized by default, ~9 GB).
    """
    log.info(f"Loading EuroLLM: {eurollm_model_name} …")
    llm = None
    tokenizer = None

    if not hf_token:
        log.error(
            "No HuggingFace token found for EuroLLM (gated model).\n"
            "  Set HF_TOKEN env var or add to config.yaml: translation.huggingface_token"
        )
        return [dict(s) for s in segments]

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer   = AutoTokenizer.from_pretrained(eurollm_model_name, token=hf_token)
        load_kwargs: dict = {"device_map": "cuda", "token": hf_token}
        if use_quantize:
            load_kwargs["load_in_8bit"] = True
        else:
            load_kwargs["torch_dtype"] = torch.bfloat16

        llm = AutoModelForCausalLM.from_pretrained(eurollm_model_name, **load_kwargs)
        llm.eval()
        log.info("✓ EuroLLM ready")

    except ImportError:
        log.error("transformers not installed — segments left untranslated")
        return [dict(s) for s in segments]
    except Exception as e:
        log.error(f"EuroLLM load failed: {e}")
        return [dict(s) for s in segments]

    language = _LANG_NAMES.get(target_lang, target_lang.upper())
    out = [dict(s) for s in segments]

    # ── Pass 1: fitted batched translation ────────────────────────────────────
    for start in tqdm(range(0, len(segments), batch_size), desc=f"Translating fitted ({target_lang})"):
        batch = segments[start : start + batch_size]
        numbered, budgets = _format_fitted_segments(batch, target_lang, cps_safety)
        prompt = _TRANSLATE_PROMPT.format(language=language, segments=numbered, glossary_section=glossary_section)

        try:
            response = _llm_generate(llm, tokenizer, prompt, temperature)
            translations = _parse_numbered(response, len(batch))
            for i, seg in enumerate(batch):
                out[start + i]["text_fr"] = translations[i] or seg["text"]
                out[start + i]["_budget"] = budgets[i]
        except Exception as e:
            log.warning(f"EuroLLM batch {start // batch_size + 1} failed: {e}")
            for i, seg in enumerate(batch):
                out[start + i]["text_fr"] = seg["text"]
                out[start + i]["_budget"] = budgets[i]

    # ── Pass 2: batched overflow retry with progressively tighter budgets ────
    # Up to MAX_RETRIES sweeps; each sweep batches all still-overflowing items
    # in groups of `batch_size` and shrinks every remaining budget by 30%.
    OVERFLOW_FACTOR = 1.4
    MAX_RETRIES = 3

    def _overflow(seg: dict) -> bool:
        budget = seg.get("_budget", _budget_chars(seg, target_lang, cps_safety))
        return len(seg.get("text_fr", "")) > budget * OVERFLOW_FACTOR

    remaining = [i for i, s in enumerate(out) if _overflow(s)]

    if not remaining:
        log.info("  No segments overflow the CPS budget — skipping retry pass")
    else:
        log.info(f"  {len(remaining)}/{len(out)} segments overflow — batched retry with tighter budgets")
        # Track each segment's current (shrinking) budget across attempts.
        budgets_now = {i: out[i].get("_budget", _budget_chars(out[i], target_lang, cps_safety)) for i in remaining}

        for attempt in range(1, MAX_RETRIES + 1):
            if not remaining:
                break
            for i in remaining:
                budgets_now[i] = max(20, int(budgets_now[i] * 0.7))

            for start in tqdm(
                range(0, len(remaining), batch_size),
                desc=f"Retry pass {attempt}/{MAX_RETRIES} ({len(remaining)} segs)",
            ):
                batch_ids = remaining[start : start + batch_size]
                numbered = "\n".join(
                    f"{k + 1}. (MAX {budgets_now[i]} chars) {out[i]['text']}"
                    for k, i in enumerate(batch_ids)
                )
                prompt = _TRANSLATE_PROMPT.format(language=language, segments=numbered, glossary_section=glossary_section)
                try:
                    response = _llm_generate(llm, tokenizer, prompt, temperature)
                    candidates = _parse_numbered(response, len(batch_ids))
                    for k, i in enumerate(batch_ids):
                        if candidates[k]:
                            out[i]["text_fr"] = candidates[k]
                except Exception as e:
                    log.debug(f"Retry pass {attempt} batch starting at {start} failed: {e}")

            remaining = [i for i in remaining if _overflow(out[i])]
            log.info(f"  After retry pass {attempt}: {len(remaining)} segments still overflow")

        if remaining:
            log.info(f"  {len(remaining)} segments did not converge — natural-pass fallback will handle them")

    # ── Pass 3: optional natural (unconstrained) translation for SRT ─────────
    if natural_pass:
        for start in tqdm(range(0, len(segments), batch_size), desc=f"Translating natural ({target_lang})"):
            batch = segments[start : start + batch_size]
            numbered = "\n".join(f"{i + 1}. {s['text']}" for i, s in enumerate(batch))
            prompt = _TRANSLATE_NATURAL_PROMPT.format(language=language, segments=numbered, glossary_section=glossary_section)
            try:
                response = _llm_generate(llm, tokenizer, prompt, 0.0)
                translations = _parse_numbered(response, len(batch))
                for i, seg in enumerate(batch):
                    out[start + i]["text_fr_natural"] = translations[i] or out[start + i].get("text_fr", seg["text"])
            except Exception as e:
                log.warning(f"Natural-pass batch {start // batch_size + 1} failed: {e}")
                for i in range(len(batch)):
                    out[start + i]["text_fr_natural"] = out[start + i].get("text_fr", batch[i]["text"])

        # Final fallback: if fitted text is still way over budget AND we have a
        # natural translation, prefer the natural one (better than nothing).
        # In practice this rarely triggers after the retry loop above.
        salvaged = 0
        for seg in out:
            budget = seg.get("_budget", _budget_chars(seg, target_lang, cps_safety))
            if (
                len(seg.get("text_fr", "")) > budget * OVERFLOW_FACTOR
                and seg.get("text_fr_natural")
                and len(seg["text_fr_natural"]) < len(seg["text_fr"])
            ):
                seg["text_fr"] = seg["text_fr_natural"]
                salvaged += 1
        if salvaged:
            log.info(f"  Used natural fallback for {salvaged} stubborn segment(s)")

    del llm, tokenizer
    free_vram(log)
    log.info(f"✓ EuroLLM translation complete ({target_lang})")
    return out


def review_translations(
    segments: List[dict],
    model: str,
    temperature: float,
    log: logging.Logger,
    batch_size: int = 50,
    target_lang: str = "fr",
    locale: str = "fr",
    glossary_section: str = "",
) -> List[dict]:
    log.info(f"Qwen review pass — {len(segments)} segments …")
    language = _LANG_NAMES.get(target_lang, target_lang.upper())
    locale_note = (
        "\nUse Québécois/Canadian French register throughout "
        "(e.g. courriel, fin de semaine, dîner for lunch, souper for supper)."
        if locale == "fr-ca" else ""
    )
    out = [dict(s) for s in segments]

    for start in tqdm(range(0, len(segments), batch_size), desc=f"Reviewing ({target_lang})"):
        batch   = segments[start : start + batch_size]
        numbered = "\n".join(f"{i + 1}. {s.get('text_fr', '')}" for i, s in enumerate(batch))
        prompt = _REVIEW_PROMPT.format(
            language=language,
            segments=numbered,
            locale_note=locale_note,
            glossary_section=glossary_section,
        )
        response = _ollama_call(prompt, model, temperature, log)
        if response:
            corrected = _parse_numbered(response, len(batch))
            for i in range(len(batch)):
                if corrected[i]:
                    out[start + i]["text_fr"] = corrected[i]

    log.info("✓ Review complete")
    return out


def translate_segments_qwen(
    segments: List[dict],
    model: str,
    temperature: float,
    batch_size: int,
    log: logging.Logger,
    target_lang: str = "fr",
    cps_safety: float = 1.1,
    natural_pass: bool = True,
    glossary_section: str = "",
) -> List[dict]:
    """Translate using Qwen via Ollama as the primary translation engine.

    Mirrors the EuroLLM translate_segments logic but uses Ollama instead of an
    on-GPU HuggingFace model — no VRAM consumed during translation.
    Qwen3's thinking mode is suppressed via /no_think prefix; any residual
    <think> blocks are stripped in _parse_numbered before parsing.
    """
    language = _LANG_NAMES.get(target_lang, target_lang.upper())
    out = [dict(s) for s in segments]
    # Qwen3 enables chain-of-thought by default; /no_think turns it off so
    # we get clean numbered output without <think>...</think> wrapping
    think_prefix = "/no_think\n" if "qwen3" in model.lower() else ""

    log.info(f"Translating with {model} (Ollama) as primary engine …")

    # ── Pass 1: fitted batched translation ────────────────────────────────────
    for start in tqdm(range(0, len(segments), batch_size), desc=f"Translating fitted ({target_lang})"):
        batch = segments[start : start + batch_size]
        numbered, budgets = _format_fitted_segments(batch, target_lang, cps_safety)
        prompt = think_prefix + _TRANSLATE_PROMPT.format(language=language, segments=numbered, glossary_section=glossary_section)
        response = _ollama_call(prompt, model, temperature, log)
        if response:
            translations = _parse_numbered(response, len(batch))
            for i, seg in enumerate(batch):
                out[start + i]["text_fr"] = translations[i] or seg["text"]
                out[start + i]["_budget"] = budgets[i]
        else:
            for i, seg in enumerate(batch):
                out[start + i]["text_fr"] = seg["text"]
                out[start + i]["_budget"] = budgets[i]

    # ── Pass 2: batched overflow retry with progressively tighter budgets ────
    OVERFLOW_FACTOR = 1.4
    MAX_RETRIES = 3

    def _overflow(seg: dict) -> bool:
        budget = seg.get("_budget", _budget_chars(seg, target_lang, cps_safety))
        return len(seg.get("text_fr", "")) > budget * OVERFLOW_FACTOR

    remaining = [i for i, s in enumerate(out) if _overflow(s)]

    if not remaining:
        log.info("  No segments overflow the CPS budget — skipping retry pass")
    else:
        log.info(f"  {len(remaining)}/{len(out)} segments overflow — batched retry with tighter budgets")
        budgets_now = {i: out[i].get("_budget", _budget_chars(out[i], target_lang, cps_safety)) for i in remaining}

        for attempt in range(1, MAX_RETRIES + 1):
            if not remaining:
                break
            for i in remaining:
                budgets_now[i] = max(20, int(budgets_now[i] * 0.7))

            for start in tqdm(
                range(0, len(remaining), batch_size),
                desc=f"Retry pass {attempt}/{MAX_RETRIES} ({len(remaining)} segs)",
            ):
                batch_ids = remaining[start : start + batch_size]
                numbered = "\n".join(
                    f"{k + 1}. (MAX {budgets_now[i]} chars) {out[i]['text']}"
                    for k, i in enumerate(batch_ids)
                )
                prompt = think_prefix + _TRANSLATE_PROMPT.format(language=language, segments=numbered, glossary_section=glossary_section)
                response = _ollama_call(prompt, model, temperature, log)
                if response:
                    candidates = _parse_numbered(response, len(batch_ids))
                    for k, i in enumerate(batch_ids):
                        if candidates[k]:
                            out[i]["text_fr"] = candidates[k]

            remaining = [i for i in remaining if _overflow(out[i])]
            log.info(f"  After retry pass {attempt}: {len(remaining)} segments still overflow")

        if remaining:
            log.info(f"  {len(remaining)} segments did not converge — natural-pass fallback will handle them")

    # ── Pass 3: optional natural (unconstrained) translation for SRT ─────────
    if natural_pass:
        for start in tqdm(range(0, len(segments), batch_size), desc=f"Translating natural ({target_lang})"):
            batch = segments[start : start + batch_size]
            numbered = "\n".join(f"{i + 1}. {s['text']}" for i, s in enumerate(batch))
            prompt = think_prefix + _TRANSLATE_NATURAL_PROMPT.format(language=language, segments=numbered, glossary_section=glossary_section)
            response = _ollama_call(prompt, model, 0.0, log)
            if response:
                translations = _parse_numbered(response, len(batch))
                for i, seg in enumerate(batch):
                    out[start + i]["text_fr_natural"] = translations[i] or out[start + i].get("text_fr", seg["text"])
            else:
                for i in range(len(batch)):
                    out[start + i]["text_fr_natural"] = out[start + i].get("text_fr", batch[i]["text"])

        salvaged = 0
        for seg in out:
            budget = seg.get("_budget", _budget_chars(seg, target_lang, cps_safety))
            if (
                len(seg.get("text_fr", "")) > budget * OVERFLOW_FACTOR
                and seg.get("text_fr_natural")
                and len(seg["text_fr_natural"]) < len(seg["text_fr"])
            ):
                seg["text_fr"] = seg["text_fr_natural"]
                salvaged += 1
        if salvaged:
            log.info(f"  Used natural fallback for {salvaged} stubborn segment(s)")

    log.info(f"✓ Qwen translation complete ({target_lang})")
    return out


def translate_segments_gemini(
    segments: List[dict],
    model_name: str,
    api_key: str,
    temperature: float,
    batch_size: int,
    log: logging.Logger,
    target_lang: str = "fr",
    cps_safety: float = 1.1,
    natural_pass: bool = True,
    glossary_section: str = "",
) -> List[dict]:
    """Translate using the Gemini API (google-generativeai SDK).

    Mirrors the three-pass EuroLLM/Qwen translation logic:
      Pass 1 — fitted batched translation with CPS character budgets
      Pass 2 — batched overflow retry with progressively tighter budgets
      Pass 3 — optional unconstrained natural pass for SRT readability
    """
    language = _LANG_NAMES.get(target_lang, target_lang.upper())
    out = [dict(s) for s in segments]
    log.info(f"Translating with {model_name} (Gemini API) as primary engine …")

    # ── Pass 1: fitted batched translation ────────────────────────────────────
    for start in tqdm(range(0, len(segments), batch_size), desc=f"Translating fitted ({target_lang})"):
        batch = segments[start : start + batch_size]
        numbered, budgets = _format_fitted_segments(batch, target_lang, cps_safety)
        prompt = _TRANSLATE_PROMPT.format(language=language, segments=numbered, glossary_section=glossary_section)
        response = _gemini_call(prompt, model_name, temperature, api_key, log)
        if response:
            translations = _parse_numbered(response, len(batch))
            for i, seg in enumerate(batch):
                out[start + i]["text_fr"] = translations[i] or seg["text"]
                out[start + i]["_budget"] = budgets[i]
        else:
            for i, seg in enumerate(batch):
                out[start + i]["text_fr"] = seg["text"]
                out[start + i]["_budget"] = budgets[i]

    # ── Pass 2: batched overflow retry with progressively tighter budgets ────
    OVERFLOW_FACTOR = 1.4
    MAX_RETRIES = 3

    def _overflow(seg: dict) -> bool:
        budget = seg.get("_budget", _budget_chars(seg, target_lang, cps_safety))
        return len(seg.get("text_fr", "")) > budget * OVERFLOW_FACTOR

    remaining = [i for i, s in enumerate(out) if _overflow(s)]

    if not remaining:
        log.info("  No segments overflow the CPS budget — skipping retry pass")
    else:
        log.info(f"  {len(remaining)}/{len(out)} segments overflow — batched retry with tighter budgets")
        budgets_now = {i: out[i].get("_budget", _budget_chars(out[i], target_lang, cps_safety)) for i in remaining}

        for attempt in range(1, MAX_RETRIES + 1):
            if not remaining:
                break
            for i in remaining:
                budgets_now[i] = max(20, int(budgets_now[i] * 0.7))

            for start in tqdm(
                range(0, len(remaining), batch_size),
                desc=f"Retry pass {attempt}/{MAX_RETRIES} ({len(remaining)} segs)",
            ):
                batch_ids = remaining[start : start + batch_size]
                numbered = "\n".join(
                    f"{k + 1}. (MAX {budgets_now[i]} chars) {out[i]['text']}"
                    for k, i in enumerate(batch_ids)
                )
                prompt = _TRANSLATE_PROMPT.format(language=language, segments=numbered, glossary_section=glossary_section)
                response = _gemini_call(prompt, model_name, temperature, api_key, log)
                if response:
                    candidates = _parse_numbered(response, len(batch_ids))
                    for k, i in enumerate(batch_ids):
                        if candidates[k]:
                            out[i]["text_fr"] = candidates[k]

            remaining = [i for i in remaining if _overflow(out[i])]
            log.info(f"  After retry pass {attempt}: {len(remaining)} segments still overflow")

        if remaining:
            log.info(f"  {len(remaining)} segments did not converge — natural-pass fallback will handle them")

    # ── Pass 3: optional natural (unconstrained) translation for SRT ─────────
    if natural_pass:
        for start in tqdm(range(0, len(segments), batch_size), desc=f"Translating natural ({target_lang})"):
            batch = segments[start : start + batch_size]
            numbered = "\n".join(f"{i + 1}. {s['text']}" for i, s in enumerate(batch))
            prompt = _TRANSLATE_NATURAL_PROMPT.format(language=language, segments=numbered, glossary_section=glossary_section)
            response = _gemini_call(prompt, model_name, 0.0, api_key, log)
            if response:
                translations = _parse_numbered(response, len(batch))
                for i, seg in enumerate(batch):
                    out[start + i]["text_fr_natural"] = translations[i] or out[start + i].get("text_fr", seg["text"])
            else:
                for i in range(len(batch)):
                    out[start + i]["text_fr_natural"] = out[start + i].get("text_fr", batch[i]["text"])

        salvaged = 0
        for seg in out:
            budget = seg.get("_budget", _budget_chars(seg, target_lang, cps_safety))
            if (
                len(seg.get("text_fr", "")) > budget * OVERFLOW_FACTOR
                and seg.get("text_fr_natural")
                and len(seg["text_fr_natural"]) < len(seg["text_fr"])
            ):
                seg["text_fr"] = seg["text_fr_natural"]
                salvaged += 1
        if salvaged:
            log.info(f"  Used natural fallback for {salvaged} stubborn segment(s)")

    log.info(f"✓ Gemini translation complete ({target_lang})")
    return out


def review_translations_gemini(
    segments: List[dict],
    model_name: str,
    api_key: str,
    temperature: float,
    log: logging.Logger,
    batch_size: int = 50,
    target_lang: str = "fr",
    locale: str = "fr",
    glossary_section: str = "",
) -> List[dict]:
    log.info(f"Gemini review pass — {len(segments)} segments …")
    language = _LANG_NAMES.get(target_lang, target_lang.upper())
    locale_note = (
        "\nUse Québécois/Canadian French register throughout "
        "(e.g. courriel, fin de semaine, dîner for lunch, souper for supper)."
        if locale == "fr-ca" else ""
    )
    out = [dict(s) for s in segments]

    for start in tqdm(range(0, len(segments), batch_size), desc=f"Reviewing ({target_lang})"):
        batch = segments[start : start + batch_size]
        numbered = "\n".join(f"{i + 1}. {s.get('text_fr', '')}" for i, s in enumerate(batch))
        prompt = _REVIEW_PROMPT.format(
            language=language,
            segments=numbered,
            locale_note=locale_note,
            glossary_section=glossary_section,
        )
        response = _gemini_call(prompt, model_name, temperature, api_key, log)
        if response:
            corrected = _parse_numbered(response, len(batch))
            for i in range(len(batch)):
                if corrected[i]:
                    out[start + i]["text_fr"] = corrected[i]

    log.info("✓ Gemini review complete")
    return out


# ============================================================================
# Step 4: Speaker Reference — extraction + denoising
# ============================================================================

def extract_speaker_sample(
    wav_path: str,
    duration: float,
    output_path: str,
    log: logging.Logger,
    skip_seconds: float = 20.0,
) -> bool:
    """Extract a speaker reference clip for voice cloning.

    Skips 20 s to avoid intro music and title cards common in webinars.
    A 25 s reference gives VoxCPM2 substantially more voice data than 15 s.
    16 kHz is VoxCPM2's spec'd input rate (AudioVAE V2 upsamples to 48 kHz).
    """
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", wav_path,
                "-ss", str(skip_seconds),
                "-t",  str(duration),
                "-ar", "16000",
                "-ac", "1",
                output_path,
            ],
            check=True, capture_output=True, timeout=60,
        )
        log.info(f"✓ Speaker sample: {duration:.0f} s at {skip_seconds:.0f} s offset")
        return True
    except Exception as e:
        log.error(f"Speaker sample extraction failed: {e}")
        return False


def denoise_audio(
    audio_path: str,
    output_path: str,
    log: logging.Logger,
) -> str:
    """Denoise the speaker reference. Tries three methods in order:
      1. DeepFilterNet  — best quality (48 kHz neural model)
      2. noisereduce    — good quality, easy install (spectral gating)
      3. FFmpeg anlmdn  — built-in, no extra package needed
    Returns the denoised path on success, the original path if all methods fail.
    """
    # ── 1. DeepFilterNet ─────────────────────────────────────────────────────
    try:
        from df.enhance import enhance, init_df
        try:
            from df.enhance import load_audio, save_audio
        except ImportError:
            from df.io import load_audio, save_audio

        log.info("Denoising with DeepFilterNet …")
        model, df_state, _ = init_df()
        audio, _  = load_audio(audio_path, sr=df_state.sr())
        enhanced  = enhance(model, df_state, audio)
        save_audio(output_path, enhanced, df_state.sr())
        log.info("✓ Speaker reference denoised (DeepFilterNet)")
        return output_path
    except ImportError:
        log.debug("DeepFilterNet not available — trying noisereduce")
    except Exception as e:
        log.warning(f"DeepFilterNet failed ({e}) — trying noisereduce")

    # ── 2. noisereduce ───────────────────────────────────────────────────────
    try:
        import noisereduce as nr
        import soundfile as _sf

        log.info("Denoising with noisereduce …")
        data, rate = _sf.read(audio_path)
        reduced    = nr.reduce_noise(y=data, sr=rate, prop_decrease=0.75)
        _sf.write(output_path, reduced, rate)
        log.info("✓ Speaker reference denoised (noisereduce)")
        return output_path
    except ImportError:
        log.debug("noisereduce not available — trying FFmpeg anlmdn")
    except Exception as e:
        log.warning(f"noisereduce failed ({e}) — trying FFmpeg anlmdn")

    # ── 3. FFmpeg anlmdn (built-in, no extra packages) ───────────────────────
    try:
        log.info("Denoising with FFmpeg anlmdn …")
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", audio_path,
                "-af", "anlmdn=s=7:p=0.002:r=0.002:m=15",
                output_path,
            ],
            check=True, capture_output=True, timeout=60,
        )
        log.info("✓ Speaker reference denoised (FFmpeg anlmdn)")
        return output_path
    except Exception as e:
        log.warning(f"FFmpeg anlmdn failed ({e}) — using raw speaker reference")
        return audio_path


def _get_reference_transcript(
    segments: List[dict],
    skip_seconds: float,
    duration: float,
) -> str:
    """Collect the English transcript for the speaker reference window.

    VoxCPM2 Ultimate Cloning takes both reference audio AND its transcript,
    giving it more signal for voice fidelity. We already have the transcript
    from Whisper, so this costs nothing extra.
    """
    ref_end = skip_seconds + duration + 5.0   # +5 s tolerance
    texts = [
        s["text"]
        for s in segments
        if s["start"] >= skip_seconds - 2.0 and s["end"] <= ref_end
    ]
    return " ".join(texts).strip()


# ============================================================================
# Step 5: TTS Synthesis — VoxCPM2 (XTTS v2 fallback)
# ============================================================================

def synthesize_all_segments(
    segments: List[dict],
    speaker_wav: Optional[str],
    reference_transcript: str,
    tts_model_id: str,
    log: logging.Logger,
    cfg_value: float = 2.5,
    inference_timesteps: int = 24,
    use_prompt_text: bool = False,
    speaker_profiles: Optional[dict] = None,
) -> Tuple[List[Tuple[np.ndarray, float, float]], int]:
    """Synthesize French audio using VoxCPM2 Ultimate Cloning.

    When speaker_profiles is provided (multi-speaker diarization mode), each
    segment is synthesized with its assigned speaker's voice clone. Falls back
    to speaker_wav for segments whose speaker has no usable profile.

    Returns (synthesized_segments, sample_rate_hz).
    Falls back to Coqui XTTS v2 if voxcpm is not installed.
    """
    def _pick_wav(seg: dict) -> Optional[str]:
        if speaker_profiles:
            spk = seg.get("speaker", "SPEAKER_00")
            profile = speaker_profiles.get(spk)
            if profile and os.path.exists(profile):
                return profile
        return speaker_wav

    # ── VoxCPM2 ──────────────────────────────────────────────────────────────
    try:
        from voxcpm import VoxCPM

        log.info(f"Loading VoxCPM2: {tts_model_id} …")
        model = VoxCPM.from_pretrained(tts_model_id)
        sr    = model.tts_model.sample_rate   # 48000
        log.info(f"✓ VoxCPM2 ready (output: {sr} Hz)")

        synthesized: List[Tuple[np.ndarray, float, float]] = []

        with tqdm(total=len(segments), desc="Synthesizing (VoxCPM2)") as pbar:
            for seg in segments:
                text = seg.get("text_fr") or seg["text"]
                if not text.strip():
                    pbar.update(1)
                    continue
                try:
                    ref_wav = _pick_wav(seg)
                    kwargs: dict = {
                        "text": text,
                        "cfg_value": cfg_value,
                        "inference_timesteps": inference_timesteps,
                        # normalize=False protects French diacritics/digits from
                        # VoxCPM2's English-centric text normalizer.
                        "normalize": False,
                        "retry_badcase": True,
                        "retry_badcase_max_times": 3,
                    }
                    if ref_wav and os.path.exists(ref_wav):
                        kwargs["reference_wav_path"] = ref_wav
                        if use_prompt_text and reference_transcript:
                            kwargs["prompt_text"] = reference_transcript
                    wav = model.generate(**kwargs)
                    synthesized.append((np.array(wav, dtype=np.float32), seg["start"], seg["end"]))
                except Exception as e:
                    log.warning(f"Segment {seg['id']} VoxCPM2 failed: {e}")
                pbar.update(1)

        log.info(f"✓ Synthesized {len(synthesized)} segments at {sr} Hz")
        del model
        free_vram(log)
        return synthesized, sr

    except ImportError:
        log.warning("voxcpm not installed — falling back to Coqui XTTS v2. Install: pip install voxcpm")

    # ── XTTS v2 fallback ─────────────────────────────────────────────────────
    if not HAS_XTTS:
        log.error("Neither voxcpm nor Coqui TTS is installed. Cannot synthesize audio.")
        return [], 24000

    xtts_model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
    log.info(f"Loading XTTS v2 fallback …")
    sr = 24000
    try:
        tts = CoquiTTS(xtts_model_name).to("cuda")
    except Exception as e:
        log.error(f"XTTS v2 load failed: {e}")
        return [], sr

    synthesized = []
    with tqdm(total=len(segments), desc="Synthesizing (XTTS v2)") as pbar:
        for seg in segments:
            text = seg.get("text_fr") or seg["text"]
            if not text.strip():
                pbar.update(1)
                continue
            try:
                ref_wav = _pick_wav(seg)
                kwargs = {"text": text[:220], "language": "fr", "speed": 1.0}
                if ref_wav and os.path.exists(ref_wav):
                    kwargs["speaker_wav"] = ref_wav
                wav = np.array(tts.tts(**kwargs), dtype=np.float32)
                synthesized.append((wav, seg["start"], seg["end"]))
            except Exception as e:
                log.warning(f"Segment {seg['id']} XTTS failed: {e}")
            pbar.update(1)

    del tts
    free_vram(log)
    return synthesized, sr


# ============================================================================
# Step 6: Audio Assembly, Encoding & Background Re-mix
# ============================================================================

_CROSSFADE_MS = 50.0   # equal-power crossfade between segments that overlap
_FADE_OUT_MS  = 80.0   # cosine fade-out applied when an overflowing segment is truncated


def _atempo_stretch(
    audio: np.ndarray,
    target_samples: int,
    src_rate: int,
    max_ratio: float,
    temp_dir: str,
    log: logging.Logger,
) -> np.ndarray:
    ratio = len(audio) / max(target_samples, 1)
    if ratio <= 1.02:
        return audio

    ratio   = min(ratio, max_ratio)
    tmp_in  = os.path.join(temp_dir, "_at_in.wav")
    tmp_out = os.path.join(temp_dir, "_at_out.wav")
    sf.write(tmp_in, audio, src_rate)

    af = f"atempo={ratio:.4f}" if ratio <= 2.0 else f"atempo=2.0,atempo={ratio / 2:.4f}"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_in, "-af", af, tmp_out],
            check=True, capture_output=True, timeout=60,
        )
        stretched, _ = librosa.load(tmp_out, sr=src_rate, mono=True)
        return stretched
    except Exception as e:
        log.debug(f"atempo failed ({e}), truncating instead")
        return audio[:target_samples]
    finally:
        for f in (tmp_in, tmp_out):
            try:
                os.unlink(f)
            except OSError:
                pass


def _apply_fade_out(audio: np.ndarray, fade_samples: int) -> np.ndarray:
    """Cosine fade-out over the last fade_samples. Modifies a copy in place."""
    n = min(fade_samples, len(audio))
    if n <= 0:
        return audio
    out = audio.copy()
    # cosine ramp from 1 → 0
    ramp = 0.5 * (1.0 + np.cos(np.linspace(0, np.pi, n, dtype=np.float32)))
    out[-n:] *= ramp
    return out


def _equal_power_crossfade(buf: np.ndarray, start: int, new_audio: np.ndarray, xfade_samples: int) -> int:
    """Mix new_audio into buf starting at `start`, using an equal-power crossfade
    over xfade_samples for any overlap with existing non-zero content.
    Returns the number of samples written (clamped to buf length)."""
    end = min(start + len(new_audio), len(buf))
    n   = end - start
    if n <= 0:
        return 0

    # Determine the actual overlap region: where buf already has audio
    # (use a small absolute threshold to detect prior content).
    xfade = max(0, min(xfade_samples, n))
    existing = buf[start : start + xfade]
    has_overlap = xfade > 0 and float(np.max(np.abs(existing))) > 1e-4

    if has_overlap:
        # Equal-power (sin/cos) crossfade
        t       = np.linspace(0, 1, xfade, dtype=np.float32)
        fade_out = np.cos(0.5 * np.pi * t)   # existing buf side
        fade_in  = np.sin(0.5 * np.pi * t)   # new audio side
        buf[start : start + xfade] = existing * fade_out + new_audio[:xfade] * fade_in
        if n > xfade:
            buf[start + xfade : end] = new_audio[xfade : n]
    else:
        buf[start : end] = new_audio[:n]
    return n


def assemble_and_encode(
    synthesized: List[Tuple[np.ndarray, float, float]],
    total_duration: float,
    wav_path: str,
    aac_path: str,
    src_rate: int,
    out_rate: int,
    max_stretch: float,
    temp_dir: str,
    log: logging.Logger,
) -> bool:
    """Place each synthesized segment into the timeline.

    Per-segment policy (in order):
      1. Try to fit in the original window + 1 borrowed gap from the next segment.
      2. If audio still overflows by ≤ max_stretch (default 1.10×): atempo stretch.
      3. If audio overflows by > max_stretch: truncate with an 80ms cosine fade-out.
    Adjacent segments are joined with a 50ms equal-power crossfade when they
    physically overlap in the buffer.
    """
    log.info(
        f"Assembling {len(synthesized)} segments at {src_rate} Hz "
        f"(stretch ≤ {max_stretch:.2f}, crossfade {_CROSSFADE_MS:.0f}ms) …"
    )

    total_samples = int((total_duration + 2) * src_rate)
    assembled     = np.zeros(total_samples, dtype=np.float32)
    ordered       = sorted(synthesized, key=lambda x: x[1])

    xfade_samples = int(_CROSSFADE_MS / 1000.0 * src_rate)
    fade_samples  = int(_FADE_OUT_MS / 1000.0 * src_rate)

    stretched_count = 0
    truncated_count = 0

    for i, (audio, start, end) in enumerate(ordered):
        start_s  = int(start * src_rate)
        window_s = int((end - start) * src_rate)

        # Gap-borrow: include silence up to the next segment's start (minus a
        # 50ms breathing room). The final segment can extend to the file end.
        if i + 1 < len(ordered):
            next_start_s = int(ordered[i + 1][1] * src_rate)
            available    = max(window_s, next_start_s - start_s - xfade_samples)
        else:
            available = max(window_s, total_samples - start_s)

        # Decide what to do with overflow.
        if len(audio) > available:
            ratio = len(audio) / max(available, 1)
            if ratio <= max_stretch:
                audio = _atempo_stretch(audio, available, src_rate, max_stretch, temp_dir, log)
                stretched_count += 1
            else:
                # Truncate with a cosine fade-out at the cut.
                audio = _apply_fade_out(audio[:available], fade_samples)
                truncated_count += 1
                log.debug(
                    f"  segment {i}: truncated ({ratio:.2f}× over budget, "
                    f"available={available / src_rate:.2f}s)"
                )

        # Crossfade-aware placement into the buffer.
        _equal_power_crossfade(assembled, start_s, audio, xfade_samples)

    if stretched_count or truncated_count:
        log.info(
            f"  Overflow handling: {stretched_count} stretched (≤{max_stretch:.2f}×), "
            f"{truncated_count} truncated with fade-out"
        )

    peak = np.max(np.abs(assembled))
    if peak > 0:
        assembled *= 0.95 / peak

    sf.write(wav_path, assembled, src_rate)
    log.info(f"✓ WAV assembled: {os.path.getsize(wav_path) / 1e6:.1f} MB")

    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", wav_path,
                "-ar", str(out_rate), "-ac", "2",
                "-c:a", "aac", "-b:a", "192k",
                aac_path,
            ],
            check=True, capture_output=True, timeout=300,
        )
        log.info(
            f"✓ AAC encoded: {os.path.getsize(aac_path) / 1e6:.1f} MB "
            f"@ 192 kbps {out_rate} Hz stereo"
        )
        return True
    except Exception as e:
        log.error(f"AAC encoding failed: {e}")
        return False


def remix_with_background(
    french_wav: str,
    no_vocals_wav: str,
    output_aac: str,
    log: logging.Logger,
    bg_gain_db: float = -3.0,
) -> bool:
    """Mix French dubbed vocals with the original background (music, ambient sound).

    The background is attenuated by 3 dB so dialogue stays intelligible.
    Produces a second output file (_french_full.m4a) alongside the dry dub.
    """
    log.info("Re-mixing French vocals with original background …")
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", french_wav,
                "-i", no_vocals_wav,
                "-filter_complex",
                (
                    f"[1:a]volume={bg_gain_db}dB[bg];"
                    "[0:a][bg]amix=inputs=2:duration=first[out]"
                ),
                "-map", "[out]",
                "-ar", "48000", "-ac", "2",
                "-c:a", "aac", "-b:a", "192k",
                output_aac,
            ],
            check=True, capture_output=True, timeout=600,
        )
        log.info(f"✓ Background re-mixed: {os.path.getsize(output_aac) / 1e6:.1f} MB")
        return True
    except Exception as e:
        log.error(f"Background re-mix failed: {e}")
        return False


# ============================================================================
# Step 7: Subtitle Generation + WhisperX Alignment
# ============================================================================

def _wrap_subtitle(text: str, max_chars: int = 42) -> str:
    if len(text) <= max_chars:
        return text
    words  = text.split()
    lines: List[str] = []
    line:  List[str] = []
    for word in words:
        if line and sum(len(w) for w in line) + len(line) - 1 + 1 + len(word) > max_chars:
            lines.append(" ".join(line))
            line = [word]
        else:
            line.append(word)
    if line:
        lines.append(" ".join(line))
    return "\n".join(lines[:2])


def create_srt(
    segments: List[dict],
    output_path: str,
    log: logging.Logger,
    offset_ms: int = 0,
    text_key: Optional[str] = None,
) -> bool:
    """Write segments to an SRT file.

    text_key selects which field to use as subtitle text. Lookup order per seg:
      1. seg[text_key] (if text_key is provided and present)
      2. seg["text_fr_natural"] (natural translation, best readability)
      3. seg["text_fr"]          (length-fitted translation)
      4. seg["text"]             (original English fallback)
    """
    def _pick(seg: dict) -> str:
        if text_key and seg.get(text_key):
            return seg[text_key]
        return seg.get("text_fr_natural") or seg.get("text_fr") or seg["text"]

    try:
        offset_s = offset_ms / 1000.0
        subs     = pysrt.SubRipFile()
        for idx, seg in enumerate(segments, 1):
            text  = _wrap_subtitle(_pick(seg))
            start = max(0.0, seg["start"] + offset_s)
            end   = max(start + 0.1, seg["end"] + offset_s)
            subs.append(SubRipItem(
                index=idx,
                start=SubRipTime(seconds=start),
                end=SubRipTime(seconds=end),
                text=text,
            ))
        subs.save(output_path, encoding="utf-8")
        log.info(f"✓ SRT: {len(subs)} entries" + (f" (offset {offset_ms:+d} ms)" if offset_ms else ""))
        return True
    except Exception as e:
        log.error(f"SRT creation failed: {e}")
        return False


def align_srt_with_whisperx(
    french_wav: str,
    segments: List[dict],
    output_srt: str,
    log: logging.Logger,
    offset_ms: int = 0,
    target_lang: str = "fr",
    use_natural: bool = True,
) -> bool:
    """Force-align the dubbed text against the dubbed audio using WhisperX.

    The resulting SRT timestamps match the actual TTS speech positions, not the
    original English timing windows.  When use_natural is True and a natural
    (unconstrained) translation exists on each segment, it is used for the SRT
    text — better readability than the length-fitted version that drives audio.

    Falls back to Whisper segment timestamps if WhisperX is unavailable.
    """
    def _srt_text(s: dict) -> str:
        if use_natural and s.get("text_fr_natural"):
            return s["text_fr_natural"]
        return s.get("text_fr") or s["text"]

    try:
        import whisperx

        log.info(f"WhisperX: force-aligning {target_lang} text to dubbed audio …")
        device   = "cuda" if torch.cuda.is_available() else "cpu"
        # WhisperX aligns audio to the fitted text (matches what was synthesized);
        # the natural text is only swapped in for the final SRT output below.
        wx_input = [
            {"start": s["start"], "end": s["end"], "text": s.get("text_fr") or s["text"]}
            for s in segments
        ]

        model_a, metadata = whisperx.load_align_model(language_code=target_lang, device=device)
        aligned            = whisperx.align(wx_input, model_a, metadata, french_wav, device)

        aligned_segs = aligned.get("segments", wx_input)
        updated: List[dict] = []
        for orig, aln in zip(segments, aligned_segs):
            s          = dict(orig)
            s["start"] = aln.get("start", orig["start"])
            s["end"]   = aln.get("end",   orig["end"])
            s["_srt_text"] = _srt_text(orig)
            updated.append(s)

        del model_a
        free_vram(log)
        log.info("✓ WhisperX alignment complete")
        return create_srt(updated, output_srt, log, offset_ms=offset_ms, text_key="_srt_text")

    except ImportError:
        log.warning("whisperx not installed (pip install whisperx) — using Whisper timestamps")
    except Exception as e:
        log.warning(f"WhisperX alignment failed ({e}) — using Whisper timestamps")

    # No alignment: stamp the SRT text on each segment before creating the file.
    annotated = [dict(s, _srt_text=_srt_text(s)) for s in segments]
    return create_srt(annotated, output_srt, log, offset_ms=offset_ms, text_key="_srt_text")


# ============================================================================
# Main Pipeline
# ============================================================================

def process_video(
    video_path: str,
    output_dir: str,
    config: PipelineConfig,
    log: logging.Logger,
    force: bool = False,
) -> bool:
    name = Path(video_path).stem
    os.makedirs(output_dir, exist_ok=True)
    temp_dir = os.path.join(config.temp_folder, name)
    os.makedirs(temp_dir, exist_ok=True)

    final_aac = os.path.join(output_dir, f"{name}_french.m4a")
    final_srt = os.path.join(output_dir, f"{name}_french.srt")

    if not force and os.path.exists(final_aac) and os.path.exists(final_srt):
        log.info(f"SKIP {name} — outputs exist (use --force to reprocess)")
        return True

    log.info(f"\n{'=' * 60}\nPipeline v3.0: {name}\n{'=' * 60}")

    # Pre-flight: Ollama needed for Qwen backend or the Qwen review pass.
    # Gemini path is fully self-contained — Ollama is not required.
    needs_ollama = (
        config.translation_backend == "qwen"
        or (config.translation_review and config.translation_backend != "gemini")
    )
    if needs_ollama:
        if not check_ollama(config.translation_model, log):
            return False

    if config.translation_backend == "gemini":
        if not config.gemini_api_key:
            log.error(
                "GEMINI_API_KEY not set.\n"
                "  Get a key at https://aistudio.google.com/app/apikey\n"
                "  Then: export GEMINI_API_KEY=your_key  or set translation.gemini_api_key in config.yaml"
            )
            return False

    # Load glossary (Canadian French vocabulary + formatting + inclusive language)
    glossary = load_glossary(config.glossary_path, log) if config.locale == "fr-ca" else Glossary([], [], [])
    glossary_section = _build_glossary_section(glossary, config.locale)
    if glossary.has_content:
        always_n  = sum(1 for e in glossary.entries if e.mode == "always")
        suggest_n = sum(1 for e in glossary.entries if e.mode == "suggest")
        log.info(
            f"  Locale: {config.locale} — {always_n} always-substitute, "
            f"{suggest_n} suggest-only terms, "
            f"{len(glossary.formatting_rules)} formatting rules, "
            f"{len(glossary.inclusive_language)} inclusive language rules"
        )

    total_duration = get_duration(video_path, log)

    # ── 0. Source separation (Demucs) ─────────────────────────────────────────
    vocals_wav:    Optional[str] = None
    no_vocals_wav: Optional[str] = None

    if config.use_demucs:
        log.info("\n[0/7] SOURCE SEPARATION (Demucs)")
        vocals_wav, no_vocals_wav = separate_vocals(
            video_path, temp_dir, config.demucs_model, log
        )

    # Fallback: extract raw audio if Demucs was disabled or failed
    if not vocals_wav:
        log.info("\n[1/7] EXTRACTING AUDIO")
        raw_wav = os.path.join(temp_dir, f"{name}.wav")
        if not extract_audio(video_path, raw_wav, config.synthesis_sample_rate, log):
            return False
        vocals_wav = raw_wav

    # ── 2. Transcribe ─────────────────────────────────────────────────────────
    log.info("\n[2/7] TRANSCRIBING (faster-whisper)")
    segments = transcribe_audio(
        vocals_wav,
        config.whisper_model,
        config.whisper_device,
        config.whisper_compute_type,
        config.models_folder,
        log,
    )
    if not segments:
        return False
    free_vram(log)

    # ── 2b. Merge into sentence-level chunks ──────────────────────────────────
    segments = merge_segments(
        segments,
        max_gap=config.segment_merge_gap,
        max_duration=config.segment_merge_max_duration,
        log=log,
    )

    # ── 2c. Speaker diarization (optional) ────────────────────────────────────
    if config.use_diarization:
        log.info("\n[2c/7] SPEAKER DIARIZATION (pyannote.audio)")
        turns = diarize_audio(
            vocals_wav,
            config.diarization_model,
            config.huggingface_token,
            config.diarization_min_speakers,
            config.diarization_max_speakers,
            log,
        )
        if turns:
            segments = assign_speakers(segments, turns)
            speaker_counts = {}
            for seg in segments:
                spk = seg.get("speaker", "?")
                speaker_counts[spk] = speaker_counts.get(spk, 0) + 1
            for spk, n in sorted(speaker_counts.items()):
                log.info(f"  {spk}: {n} segment(s)")
        else:
            log.warning("  Diarization failed — all segments assigned to SPEAKER_00")
            for seg in segments:
                seg["speaker"] = "SPEAKER_00"

    # ── 3. Translate ──────────────────────────────────────────────────────────
    if config.translation_backend == "qwen":
        log.info(f"\n[3/7] TRANSLATING ({config.translation_model} via Ollama)")
        segments = translate_segments_qwen(
            segments,
            config.translation_model,
            config.translation_temperature,
            config.translation_batch_size,
            log,
            target_lang=config.target_lang,
            cps_safety=config.cps_safety,
            natural_pass=config.translate_natural_pass,
            glossary_section=glossary_section,
        )
    elif config.translation_backend == "gemini":
        log.info(f"\n[3/7] TRANSLATING ({config.gemini_model} via Gemini API)")
        segments = translate_segments_gemini(
            segments,
            config.gemini_model,
            config.gemini_api_key,
            config.translation_temperature,
            config.translation_batch_size,
            log,
            target_lang=config.target_lang,
            cps_safety=config.cps_safety,
            natural_pass=config.translate_natural_pass,
            glossary_section=glossary_section,
        )
    else:
        log.info("\n[3/7] TRANSLATING (EuroLLM-9B)")
        segments = translate_segments(
            segments,
            config.eurollm_model,
            config.eurollm_quantize,
            config.translation_temperature,
            config.translation_batch_size,
            log,
            hf_token=config.huggingface_token,
            target_lang=config.target_lang,
            cps_safety=config.cps_safety,
            natural_pass=config.translate_natural_pass,
            glossary_section=glossary_section,
        )
    _verify_translation_quality(segments, log)

    # ── 3b. Review pass ───────────────────────────────────────────────────────
    if config.translation_review:
        if config.translation_backend == "gemini":
            log.info(f"\n[3b/7] REVIEWING TRANSLATIONS ({config.gemini_model} — Gemini)")
            segments = review_translations_gemini(
                segments,
                config.gemini_model,
                config.gemini_api_key,
                config.translation_temperature,
                log,
                batch_size=config.translation_batch_size,
                target_lang=config.target_lang,
                locale=config.locale,
                glossary_section=glossary_section,
            )
        else:
            if config.translation_backend == "qwen":
                log.info(f"\n[3b/7] REVIEWING TRANSLATIONS ({config.translation_model} — self-review)")
            else:
                log.info(f"\n[3b/7] REVIEWING TRANSLATIONS ({config.translation_model})")
            segments = review_translations(
                segments,
                config.translation_model,
                config.translation_temperature,
                log,
                target_lang=config.target_lang,
                locale=config.locale,
                glossary_section=glossary_section,
            )

    # ── 3c. Glossary post-processing (Canadian French enforcement) ─────────────
    if glossary.entries:
        log.info("\n[3c/7] APPLYING GLOSSARY (deterministic substitution)")
        segments = apply_glossary(segments, glossary.entries, log)

    # ── 4. Prepare speaker voice reference(s) ─────────────────────────────────
    log.info("\n[4/7] PREPARING SPEAKER REFERENCE(S)")
    speaker_wav: Optional[str] = None
    speaker_profiles: Optional[dict] = None

    if config.use_diarization and any("speaker" in s for s in segments):
        # Multi-speaker: build a profile clip for every detected speaker
        speaker_profiles = build_speaker_profiles(
            vocals_wav,
            segments,
            temp_dir,
            config.diarization_profile_duration,
            log,
            use_deepfilter=config.use_deepfilter,
        )
        valid = sum(1 for v in speaker_profiles.values() if v)
        log.info(f"  Built {valid}/{len(speaker_profiles)} speaker profile(s)")
        # Also build a single fallback wav for speakers with insufficient audio
        raw_speaker_wav = os.path.join(temp_dir, "speaker_raw.wav")
        if extract_speaker_sample(
            vocals_wav, config.tts_speaker_duration, raw_speaker_wav, log,
            skip_seconds=config.tts_speaker_skip,
        ):
            speaker_wav = denoise_audio(
                raw_speaker_wav, os.path.join(temp_dir, "speaker_denoised.wav"), log
            ) if config.use_deepfilter else raw_speaker_wav
    else:
        # Single-speaker (original path)
        raw_speaker_wav = os.path.join(temp_dir, "speaker_raw.wav")
        if extract_speaker_sample(
            vocals_wav,
            config.tts_speaker_duration,
            raw_speaker_wav,
            log,
            skip_seconds=config.tts_speaker_skip,
        ):
            if config.use_deepfilter:
                denoised_wav = os.path.join(temp_dir, "speaker_denoised.wav")
                speaker_wav  = denoise_audio(raw_speaker_wav, denoised_wav, log)
            else:
                speaker_wav = raw_speaker_wav
        else:
            log.warning("Voice cloning disabled — using default VoxCPM2 voice")

    # Reference transcript for VoxCPM2 Ultimate Cloning
    reference_transcript = _get_reference_transcript(
        segments,
        skip_seconds=config.tts_speaker_skip,
        duration=config.tts_speaker_duration,
    )
    if reference_transcript:
        log.info(f"  Reference transcript ({len(reference_transcript)} chars): "
                 f"{reference_transcript[:80]}…")

    # ── 5. TTS synthesis (VoxCPM2) ────────────────────────────────────────────
    log.info("\n[5/7] SYNTHESIZING FRENCH AUDIO (VoxCPM2)")
    synthesized, actual_sr = synthesize_all_segments(
        segments,
        speaker_wav,
        reference_transcript,
        config.tts_model,
        log,
        cfg_value=config.tts_cfg_value,
        inference_timesteps=config.tts_inference_timesteps,
        use_prompt_text=config.tts_use_prompt_text,
        speaker_profiles=speaker_profiles,
    )
    if not synthesized:
        return False
    free_vram(log)

    # ── 6. Assemble & encode ──────────────────────────────────────────────────
    log.info("\n[6/7] ASSEMBLING & ENCODING")
    interim_wav = os.path.join(temp_dir, f"{name}_french.wav")

    if not assemble_and_encode(
        synthesized,
        total_duration,
        interim_wav,
        final_aac,
        src_rate=actual_sr,
        out_rate=config.output_sample_rate,
        max_stretch=config.tts_max_stretch,
        temp_dir=temp_dir,
        log=log,
    ):
        return False

    # Optional background re-mix (French vocals + original music/ambience)
    if config.preserve_background and no_vocals_wav and os.path.exists(no_vocals_wav):
        remixed_aac = os.path.join(output_dir, f"{name}_french_full.m4a")
        if remix_with_background(interim_wav, no_vocals_wav, remixed_aac, log):
            log.info(f"  Full mix (vocals + background): {Path(remixed_aac).name}")

    # ── 7. Subtitles ──────────────────────────────────────────────────────────
    log.info("\n[7/7] GENERATING SUBTITLES")
    if config.use_whisperx_alignment:
        align_srt_with_whisperx(
            interim_wav,
            segments,
            final_srt,
            log,
            offset_ms=config.subtitle_offset_ms,
            target_lang=config.target_lang,
            use_natural=config.subtitles_use_natural,
        )
    else:
        create_srt(
            segments,
            final_srt,
            log,
            offset_ms=config.subtitle_offset_ms,
            text_key="text_fr_natural" if config.subtitles_use_natural else "text_fr",
        )

    shutil.rmtree(temp_dir, ignore_errors=True)

    log.info(f"\n{'=' * 60}")
    log.info(f"DONE: {name}")
    log.info(f"  Audio : {final_aac}")
    log.info(f"  Subs  : {final_srt}")
    log.info(f"{'=' * 60}\n")
    return True


# ============================================================================
# CLI
# ============================================================================

@click.command()
@click.option("--video",      type=click.Path(exists=True), required=True, help="Input MP4")
@click.option("--output-dir", default="/workspace/outputs",  help="Output directory")
@click.option("--config", "config_path", default="/workspace/config.yaml",
              type=click.Path(exists=True), help="Path to config.yaml")
@click.option("--force", is_flag=True, default=False, help="Overwrite existing outputs")
@click.option(
    "--translator",
    type=click.Choice(["eurollm", "qwen", "gemini"], case_sensitive=False),
    default=None,
    help=(
        "Translation backend override. "
        "'eurollm' = EuroLLM-9B-Instruct on-GPU (gated HF model, best quality). "
        "'qwen' = Qwen3:14b via Ollama (no HF token needed, good for comparison). "
        "'gemini' = Gemini API (requires GEMINI_API_KEY, no local GPU needed). "
        "Defaults to translation.backend in config.yaml."
    ),
)
@click.option(
    "--locale",
    type=click.Choice(["fr", "fr-ca"], case_sensitive=False),
    default=None,
    help=(
        "Output language locale. "
        "'fr-ca' loads the Canadian French glossary and enforces Québécois vocabulary "
        "via prompt injection and post-processing substitution. "
        "Defaults to translation.locale in config.yaml."
    ),
)
def main(video: str, output_dir: str, config_path: str, force: bool, translator: Optional[str], locale: Optional[str]) -> None:
    """Dub a single video to French (audio track + SRT subtitles).

    Compare backends:
      --translator eurollm --output-dir /workspace/outputs/eurollm
      --translator qwen    --output-dir /workspace/outputs/qwen
      --translator gemini  --output-dir /workspace/outputs/gemini

    Canadian French:
      --locale fr-ca
    """
    try:
        config = load_config(config_path)
    except Exception as e:
        print(f"Config error: {e}", file=sys.stderr)
        sys.exit(1)
    if translator:
        config.translation_backend = translator.lower()
    if locale:
        config.locale = locale.lower()
    log     = setup_logging(config.logs_folder, Path(video).stem)
    success = process_video(video, output_dir, config, log, force=force)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
