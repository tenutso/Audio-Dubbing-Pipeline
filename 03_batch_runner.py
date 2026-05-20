#!/usr/bin/env python3
"""
Batch Runner for French Dubbing Pipeline

Process multiple videos with:
- Sequential processing (VRAM-safe on 24 GB GPU)
- Progress tracking and per-video timeout
- Error recovery and retry
- JSON summary report
"""

import glob
import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import argparse
import yaml
from tqdm import tqdm


@dataclass
class JobStatus:
    video_file: str
    status: str  # pending | running | completed | failed
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0
    error: str = ""
    output_audio: str = ""
    output_srt: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class BatchProcessor:
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        config_file: str = "/workspace/config.yaml",
        max_workers: int = 1,
        log_dir: str = "/workspace/logs",
        timeout: int = 7200,
    ):
        self.input_dir   = input_dir
        self.output_dir  = output_dir
        self.config_file = config_file
        self.max_workers = max_workers
        self.log_dir     = log_dir
        self.timeout     = timeout
        self.force       = False  # set by CLI

        self.logger  = self._setup_logger()
        self.config  = self._load_config()
        self.jobs: Dict[str, JobStatus] = {}
        self.results = {
            "start_time":   datetime.now().isoformat(),
            "end_time":     None,
            "total_videos": 0,
            "completed":    0,
            "failed":       0,
            "jobs":         [],
        }

    def _setup_logger(self) -> logging.Logger:
        os.makedirs(self.log_dir, exist_ok=True)
        logger = logging.getLogger("BatchProcessor")
        if logger.handlers:
            return logger
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
        fh = logging.FileHandler(os.path.join(self.log_dir, "batch.log"))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)
        return logger

    def _load_config(self) -> dict:
        try:
            with open(self.config_file) as f:
                cfg = yaml.safe_load(f)
            # Use per-video timeout from config if not overridden
            if self.timeout == 7200:
                self.timeout = cfg.get("processing", {}).get("timeout_seconds", 7200)
            return cfg
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            return {}

    def discover_videos(self) -> List[str]:
        files = sorted(glob.glob(os.path.join(self.input_dir, "*.mp4")))
        if not files:
            self.logger.warning(f"No MP4 files found in {self.input_dir}")
            return []
        self.logger.info(f"Found {len(files)} video(s):")
        for f in files:
            # Skip files already processed (pipeline-level checkpoint)
            name = Path(f).stem
            aac = os.path.join(self.output_dir, f"{name}_french.m4a")
            srt = os.path.join(self.output_dir, f"{name}_french.srt")
            status = " [DONE — will skip]" if os.path.exists(aac) and os.path.exists(srt) else ""
            self.logger.info(f"  {Path(f).name}{status}")
        return files

    def process_single(self, video_path: str, force: bool = False) -> JobStatus:
        name = Path(video_path).stem
        job  = JobStatus(video_file=video_path, status="running")
        job.start_time = time.time()

        self.logger.info(f"\n{'=' * 60}\nProcessing: {name}\n{'=' * 60}")

        try:
            # Use the same Python interpreter running this script
            cmd = [
                sys.executable,
                "/workspace/scripts/02_pipeline.py",
                "--video",      video_path,
                "--output-dir", self.output_dir,
                "--config",     self.config_file,
            ]
            if self.force:
                cmd.append("--force")
            self.logger.info(f"Command: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                timeout=self.timeout,
                capture_output=True,
                text=True,
            )
            job.end_time  = time.time()
            job.duration  = job.end_time - job.start_time

            if result.returncode == 0:
                job.status = "completed"
                aac = os.path.join(self.output_dir, f"{name}_french.m4a")
                srt = os.path.join(self.output_dir, f"{name}_french.srt")
                if os.path.exists(aac):
                    job.output_audio = aac
                if os.path.exists(srt):
                    job.output_srt = srt
                self.logger.info(f"✓ COMPLETED in {job.duration / 60:.1f} min")
            else:
                job.status = "failed"
                job.error  = (result.stderr or result.stdout)[-2000:]  # last 2 KB
                self.logger.error(f"✗ FAILED (exit {result.returncode})")
                self.logger.error(f"  {job.error[-500:]}")

        except subprocess.TimeoutExpired:
            job.status   = "failed"
            job.error    = f"Timeout after {self.timeout // 60} min"
            job.end_time = time.time()
            job.duration = job.end_time - job.start_time
            self.logger.error(f"✗ TIMEOUT after {job.duration / 60:.1f} min")

        except Exception as e:
            job.status   = "failed"
            job.error    = str(e)
            job.end_time = time.time()
            job.duration = job.end_time - job.start_time
            self.logger.error(f"✗ ERROR: {e}")

        self.jobs[video_path] = job
        return job

    def process_batch(self) -> None:
        files = self.discover_videos()
        if not files:
            self.logger.error("No videos to process — exiting")
            return

        self.results["total_videos"] = len(files)
        self.logger.info(
            f"\nStarting batch: {len(files)} video(s), "
            f"{self.max_workers} worker(s), "
            f"timeout={self.timeout // 60} min/video"
        )

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(self.process_single, f, self.force): f for f in files}
            with tqdm(total=len(files), desc="Batch progress") as pbar:
                for future in as_completed(futures):
                    try:
                        job = future.result()
                        if job.status == "completed":
                            self.results["completed"] += 1
                        else:
                            self.results["failed"] += 1
                        self.results["jobs"].append(job.to_dict())
                    except Exception as e:
                        self.logger.error(f"Unexpected error: {e}")
                        self.results["failed"] += 1
                    pbar.update(1)

        self.results["end_time"] = datetime.now().isoformat()
        self._write_report()

    def _write_report(self) -> None:
        report = os.path.join(self.log_dir, "batch_report.json")
        try:
            with open(report, "w") as f:
                json.dump(self.results, f, indent=2)
        except Exception as e:
            self.logger.error(f"Could not write report: {e}")

        total = self.results["total_videos"]
        done  = self.results["completed"]
        fail  = self.results["failed"]
        pct   = 100 * done / max(1, total)

        self.logger.info(f"\n{'=' * 60}")
        self.logger.info("BATCH COMPLETE")
        self.logger.info(f"{'=' * 60}")
        self.logger.info(f"Total    : {total}")
        self.logger.info(f"Completed: {done}  ({pct:.0f}%)")
        self.logger.info(f"Failed   : {fail}")
        self.logger.info(f"Report   : {report}")
        self.logger.info(f"{'=' * 60}")

        # Per-video table
        self.logger.info(
            f"\n{'Video':<40} {'Status':<12} {'Duration':<10} {'Output'}"
        )
        self.logger.info("-" * 100)
        for j in self.results["jobs"]:
            dur = f"{j['duration'] / 60:.1f} min" if j["duration"] else "N/A"
            out = Path(j["output_audio"]).name if j["output_audio"] else j.get("error", "")[:40]
            self.logger.info(
                f"{Path(j['video_file']).name:<40} {j['status']:<12} {dur:<10} {out}"
            )
        self.logger.info("-" * 100)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-process videos through the French dubbing pipeline"
    )
    parser.add_argument("--input-dir",  default="/workspace/videos/input",
                        help="Directory containing input MP4 files")
    parser.add_argument("--output-dir", default="/workspace/outputs",
                        help="Output directory for audio and subtitles")
    parser.add_argument("--config",     default="/workspace/config.yaml",
                        help="Path to config.yaml")
    parser.add_argument("--workers",    type=int, default=1,
                        help="Parallel workers (keep at 1 for 24 GB VRAM)")
    parser.add_argument("--timeout",    type=int, default=7200,
                        help="Per-video timeout in seconds (default 7200)")
    parser.add_argument("--log-dir",    default="/workspace/logs",
                        help="Log directory")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing outputs (skip checkpoint)")
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"Error: input directory does not exist: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir,    exist_ok=True)

    processor = BatchProcessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        config_file=args.config,
        max_workers=args.workers,
        log_dir=args.log_dir,
        timeout=args.timeout,
    )
    processor.process_batch()


if __name__ == "__main__":
    main()
