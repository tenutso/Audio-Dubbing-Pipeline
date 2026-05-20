# French Dubbing Pipeline - Quick Reference & Troubleshooting

## Quick Start (30 seconds)

```bash
# 1. SSH into RunPod
ssh runpod@your-runpod-ip

# 2. Run setup (one-time)
bash /workspace/04_setup.sh

# 3. Copy your video
cp your_webinar.mp4 /workspace/videos/input/

# 4. Process it
python /workspace/scripts/02_pipeline.py --video /workspace/videos/input/your_webinar.mp4

# 5. Download outputs
# Audio: /workspace/outputs/your_webinar_french.m4a
# Subtitles: /workspace/outputs/your_webinar_french.srt
```

---

## Performance & Cost

### Time Estimates (per 1-hour webinar)
| Stage | Duration | Notes |
|-------|----------|-------|
| Audio Extraction | 2-3 min | Fast, I/O bound |
| Transcription (Whisper) | 3-5 min | GPU accelerated |
| Translation (Ollama) | 5-8 min | CPU bound, LLM inference |
| Voice Synthesis (TTS) | 8-15 min | GPU accelerated (bottleneck) |
| Audio Assembly | 1-2 min | Fast, I/O bound |
| Subtitle Generation | 1 min | Very fast |
| **Total** | **20-35 min** | 1.33x real-time |

### Cost on RunPod (On-Demand)
- **A5000**: $0.45/hour × 0.5 hours = **~$0.23/video**
- **RTX 4090**: $0.36/hour × 0.5 hours = **~$0.18/video**
- **20 videos/month**: **$3.60-4.60**

Key: Stop the instance immediately after processing (hourly billing).

---

## Common Issues & Solutions

### Issue 1: CUDA Out of Memory

**Symptom**: `RuntimeError: CUDA out of memory`

**Solution**:
```bash
# In /workspace/config.yaml, change:
whisper:
  model: small  # Instead of large-v3

translation:
  model: mistral:latest  # Instead of llama2:13b

processing:
  chunk_duration_seconds: 15  # Instead of 30
```

### Issue 2: Ollama Won't Start

**Symptom**: `Cannot connect to Ollama. Is it running?`

**Solution**:
```bash
# Check if Ollama is running
ps aux | grep ollama

# If not running, start it manually
ollama serve &

# Wait 10 seconds, then verify
curl http://localhost:11434/api/tags

# If that fails, check permissions
sudo systemctl restart ollama
```

### Issue 3: Whisper Model Download Fails

**Symptom**: `Error downloading model weights`

**Solution**:
```bash
# Download manually to cache
python3 -c "import whisper; whisper.load_model('large-v3')"

# Or use smaller model
# Edit config.yaml: whisper.model: medium

# Check disk space
df -h /workspace
# Need ~10GB free for models
```

### Issue 4: Video File Not Found

**Symptom**: `No such file or directory: /workspace/videos/...`

**Solution**:
```bash
# Verify file exists
ls -lh /workspace/videos/input/

# Ensure correct path in command
# Use absolute paths, not relative

# Check file is readable
file /workspace/videos/input/webinar.mp4
# Should output: video/mp4
```

### Issue 5: Translation Output is Gibberish

**Symptom**: French output doesn't make sense

**Solution**:
```bash
# Restart Ollama with more context
pkill ollama
ollama serve &

# Wait for startup
sleep 10

# Pull a better model
ollama pull mistral:latest

# Edit config.yaml to increase temperature:
translation:
  temperature: 0.5  # Lower = more deterministic
```

### Issue 6: TTS Audio Sounds Robotic

**Symptom**: French audio lacks naturalness

**Solution**:
```bash
# Switch TTS engine in config.yaml:
tts:
  engine: coqui  # Instead of f5tts

# Or adjust speaker voice settings
# Note: Voice cloning improves naturalness significantly
```

### Issue 7: Processing Hangs / Timeout

**Symptom**: Process runs for >1 hour and dies

**Solution**:
```bash
# Check GPU status during processing
watch -n 1 nvidia-smi

# If GPU idle: LLM (Ollama) is the bottleneck
# Solutions:
# 1. Use smaller model: mistral instead of llama2:13b
# 2. Split videos into 30-min segments before processing
# 3. Increase RunPod timeout in config.yaml

processing:
  timeout_seconds: 7200  # 2 hours instead of 1
```

---

## Monitoring During Processing

### Watch GPU Usage
```bash
watch -n 1 nvidia-smi
```
Expected pattern:
- **Whisper stage**: High GPU memory (6GB), 90% utilization
- **Translation stage**: Low GPU usage (<1GB), CPU busy
- **TTS stage**: High GPU memory (8GB), 90% utilization

### Check Progress
```bash
# View real-time log
tail -f /workspace/logs/batch.log

# View specific video log
cat /workspace/logs/webinar_1.log
```

### Monitor Disk Space
```bash
df -h /workspace/
# Needs ~2-5GB free during processing
```

### Monitor Memory
```bash
free -h
# Should have 32GB+ available system RAM
```

---

## Advanced Configuration

### Multiple Worker Threads (Careful!)
```bash
# In config.yaml
processing:
  max_workers: 2  # NOT RECOMMENDED for RTX 4090/A5000

# With 24GB VRAM, running 2 TTS jobs causes OOM
# Stick with max_workers: 1
```

### Custom Speaker Voice (F5-TTS Only)
```python
# Edit 02_pipeline.py to use custom speaker embedding
speaker_path = "/path/to/speaker_audio.wav"
speaker_sample = np.load(speaker_path)

# Pass to synthesize_speech_f5tts()
audio = synthesize_speech_f5tts(
    text,
    speaker_audio=speaker_sample
)
```

### Batch Processing with Checkpoints
```bash
# Resume interrupted batch
python /workspace/scripts/03_batch_runner.py \
  --input-dir /workspace/videos/input \
  --workers 1

# If interrupted, rerun same command - it resumes from checkpoint
```

### Testing with Smaller Video
```bash
# Extract first 5 minutes for testing
ffmpeg -i webinar.mp4 -t 300 webinar_5min.mp4

# Process test version (5 minutes ≈ 2-3 minutes GPU time)
python /workspace/scripts/02_pipeline.py --video webinar_5min.mp4
```

---

## Vimeo Upload Instructions

### Upload French Audio Track
1. Go to Vimeo video page
2. **Settings** → **Audio & Tracks**
3. Click **Add additional audio track**
4. Upload: `webinar_french.m4a` (128 kbps AAC)
5. Set language: **French**
6. Make default: **No** (keep English as default)

### Upload Subtitles
1. In same **Audio & Tracks** section
2. Click **Add subtitles**
3. Upload: `webinar_french.srt` (UTF-8)
4. Set language: **French**
5. Make default: **No**

### Verify Quality
1. Play video on Vimeo
2. Click **CC** button to enable subtitles
3. Click speaker icon → select **Français** audio
4. Check sync and quality

---

## Cleanup & Resource Management

### Stop Instance (Critical for Cost Control)
```bash
# After getting outputs...

# Option 1: Use RunPod UI
# Click "Stop Pod" button (stops billing immediately)

# Option 2: Via SSH
sudo poweroff

# Option 3: Keep for batch processing, then stop
# Don't leave idle - costs accumulate per hour!
```

### Compress Outputs for Download
```bash
cd /workspace/outputs
tar -czf dubbing_results.tar.gz *.m4a *.srt
# Download dubbing_results.tar.gz from RunPod file browser
```

### Clear Temp Files
```bash
rm -rf /workspace/temp/*
du -sh /workspace  # Check total size before download
```

---

## Performance Tuning

### For Slower Videos (Lots of Narration)
```yaml
# config.yaml
processing:
  chunk_duration_seconds: 15  # Smaller chunks
  
whisper:
  model: medium  # Faster transcription
  
translation:
  model: mistral:latest  # Fast & accurate
```

### For Faster Processing (Standard Presentations)
```yaml
# config.yaml
whisper:
  model: large-v3  # Better accuracy for technical terms
  
tts:
  engine: coqui  # Faster than F5-TTS
```

### Maximum Quality (Slower, Higher Cost)
```yaml
# config.yaml
whisper:
  model: large-v3
  
translation:
  model: neural-chat:latest  # Better comprehension
  temperature: 0.3  # More consistent
  
tts:
  engine: f5tts  # Better naturalness with voice cloning
```

---

## Debugging Commands

```bash
# Check if all tools are available
ffmpeg -version  # Should show FFmpeg version
ollama list      # Should show downloaded models
nvidia-smi       # Should show GPU info

# Test individual components
# Test Whisper
python3 -c "import whisper; print('Whisper OK')"

# Test Ollama API
curl http://localhost:11434/api/tags

# Test TTS
python3 -c "from TTS.api import TTS; print('TTS OK')"

# Monitor system while processing
# In another terminal:
watch -n 1 'nvidia-smi && echo "---" && free -h'
```

---

## Support Resources

- **Whisper Issues**: https://github.com/openai/whisper
- **Ollama Issues**: https://github.com/jmorganca/ollama
- **Coqui TTS**: https://github.com/coqui-ai/TTS
- **FFmpeg Help**: https://ffmpeg.org/documentation.html
- **RunPod Docs**: https://docs.runpod.io

---

## File Structure Reference

```
/workspace/
├── videos/
│   └── input/                    # PUT YOUR MP4 FILES HERE
├── outputs/                      # DOWNLOAD RESULTS FROM HERE
│   ├── video_name_french.m4a    # Audio track for Vimeo
│   └── video_name_french.srt    # Subtitles
├── models/                       # Model cache (auto-populated)
├── scripts/
│   ├── 02_pipeline.py           # Single video processor
│   └── 03_batch_runner.py       # Batch processor
├── logs/                         # Processing logs
├── temp/                         # Temporary files (auto-cleaned)
├── config.yaml                  # Configuration file
└── requirements.txt             # Python dependencies
```

---

## Estimated Costs for Common Workflows

### Dubbing 20 one-hour webinars monthly
- **RTX 4090**: 20 videos × 30 min × $0.36/hr = **$3.60/month**
- **A5000**: 20 videos × 30 min × $0.45/hr = **$4.50/month**

### Dubbing 100 webinars quarterly
- **On-demand**: 100 × 30 min × $0.36/hr = **$18/quarter** (4090)
- **Spot instances**: ~60% savings = **$7.20/quarter**

### Don't pay for idle time!
- 1 hour idle at $0.36/hr = **$0.36 wasted**
- Always stop the pod after processing

---

## FAQ

**Q: Can I process multiple videos at once?**
A: Yes, use `03_batch_runner.py`. But limit `max_workers: 1` on A5000/4090 (VRAM constraint).

**Q: Will voice cloning work without reference audio?**
A: F5-TTS works best with reference audio, but has fallback voices. Coqui TTS has built-in French voices.

**Q: How long do models take to download?**
A: First run takes 10-15 min (models, Whisper, TTS). Subsequent runs use cached models (instant).

**Q: What if translation is inaccurate?**
A: Try different LLM (llama2:13b, neural-chat) or adjust temperature in config.yaml.

**Q: Can I use my own TTS voice?**
A: Not easily with Coqui. F5-TTS supports voice cloning with 10-sec reference clip.

**Q: Does the setup work with different GPU models?**
A: Yes! Tested on RTX 4090, A5000, A100. Adjust VRAM estimates accordingly.

**Q: How do I add English audio track alongside French?**
A: Extract original audio with `ffmpeg -i webinar.mp4 audio_en.m4a`, upload both to Vimeo.

---

Last updated: 2025-05-18
