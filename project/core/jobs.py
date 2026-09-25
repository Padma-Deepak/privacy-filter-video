"""Bounded, session-owned local review jobs with cooperative cancellation and expiry.

Run one application process: job metadata and thumbnails intentionally stay in RAM.
Only source uploads and redacted exports exist on disk, in private job directories.
"""
from __future__ import annotations

import atexit
import json
import logging
import os
from pathlib import Path
import secrets
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from core import review

logger = logging.getLogger(__name__)


@dataclass
class Job:
    id: str
    owner: str
    directory: Path
    source: Path
    video: bool
    profile: dict
    state: str = "queued"
    done: int = 0
    total: int = 0
    touched: float = field(default_factory=time.monotonic)
    started: float = field(default_factory=time.monotonic)
    cancel: threading.Event = field(default_factory=threading.Event)
    lock: threading.RLock = field(default_factory=threading.RLock)
    analysis: dict | None = None
    output: Path | None = None
    report: dict | None = None
    error: str | None = None

    def check_cancel(self) -> None:
        if self.cancel.is_set() or time.monotonic() - self.started > 3600:
            raise review.Cancelled()

    def progress(self, done: int, total: int) -> None:
        with self.lock:
            self.done, self.total = done, total
            if self.directory.exists():
                os.utime(self.directory, None)

    def status(self) -> dict:
        with self.lock:
            return {"id": self.id, "state": self.state, "done": self.done,
                    "total": self.total, "error": self.error,
                    "has_report": self.report is not None}


class JobManager:
    def __init__(self, root: Path, models_dir: str, ttl: int = 1800):
        self.root, self.models_dir, self.ttl = root, models_dir, ttl
        self.jobs: dict[str, Job] = {}
        self.lock = threading.RLock()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="privacy-job")
        self.stopping = threading.Event()
        self.reaper = None

    def _start(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.reaper is None:
            self.reaper = threading.Thread(target=self._reap_loop, daemon=True)
            self.reaper.start()
            atexit.register(self.close)

    def create(self, owner: str, upload, extension: str, video: bool, profile: dict) -> Job:
        with self.lock:
            self._start()
            if len(self.jobs) >= 8:
                raise ValueError("Eight review jobs are open. Delete one before uploading again.")
            key = secrets.token_hex(24)
            directory = self.root / key
            directory.mkdir(mode=0o700)
            source = directory / f"source.{extension}"
            try:
                upload.save(source)
                source.chmod(0o600)
            except BaseException:
                shutil.rmtree(directory, ignore_errors=True)
                raise
            job = Job(key, owner, directory, source, video, profile)
            self.jobs[key] = job
            self.pool.submit(self._analyse, job)
            return job

    def get(self, key: str, owner: str) -> Job | None:
        with self.lock:
            job = self.jobs.get(key)
            if job is None or job.owner != owner:
                return None
            with job.lock:
                job.touched = time.monotonic()
                if job.directory.exists():
                    os.utime(job.directory, None)
            return job

    def _analyse(self, job: Job) -> None:
        try:
            job.check_cancel()
            with job.lock:
                job.state = "analysing"
                job.started = time.monotonic()
            data = review.analyse(str(job.source), job.video, self.models_dir, job.profile,
                                  job.progress, job.check_cancel)
            job.check_cancel()
            with job.lock:
                job.analysis = data
                job.state = "ready"
                job.touched = time.monotonic()
        except review.Cancelled:
            self._discard(job)
        except Exception as exc:
            self._fail(job, exc)

    def start_export(self, job: Job, choices: dict, manual: list, audio: str) -> None:
        with job.lock:
            if job.state != "ready":
                raise ValueError("Job is not ready for export")
            job.state, job.done = "exporting", 0
            job.started = time.monotonic()
            self.pool.submit(self._export, job, choices, manual, audio)

    def _export(self, job: Job, choices: dict, manual: list, audio: str) -> None:
        started = time.monotonic()
        try:
            job.check_cancel()
            output = review.export(str(job.source), job.directory, job.analysis, job.profile,
                                   choices, manual, audio, job.progress, job.check_cancel)
            job.check_cancel()
            report = review.build_report(str(job.source), job.analysis, job.profile, choices,
                                         manual, audio, time.monotonic() - started) if job.profile["report"] else None
            with job.lock:
                job.check_cancel()
                job.source.unlink(missing_ok=True)
                # No original crops survive a completed export, even in memory.
                job.analysis = None
                job.output, job.report, job.state = output, report, "complete"
                job.touched = time.monotonic()
        except review.Cancelled:
            self._discard(job)
        except Exception as exc:
            self._fail(job, exc)

    def _fail(self, job: Job, exc: Exception) -> None:
        logger.error("Review job failed (%s)", type(exc).__name__)
        with job.lock:
            shutil.rmtree(job.directory, ignore_errors=True)
            job.analysis = None
            job.state = "error"
            job.error = str(exc) if isinstance(exc, ValueError) else "Processing failed. Check dependencies and try another file."

    def _discard(self, job: Job) -> None:
        with job.lock:
            shutil.rmtree(job.directory, ignore_errors=True)
            job.analysis = job.report = None
            job.output = None
            job.state = "cancelled"

    def delete(self, job: Job) -> None:
        job.cancel.set()
        with job.lock:
            if job.state not in {"queued", "analysing", "exporting"}:
                self._discard(job)
        # Running worker owns final deletion; do not unlink a file it is reading.

    def reap(self) -> None:
        now = time.monotonic()
        with self.lock:
            for key, job in list(self.jobs.items()):
                with job.lock:
                    active = job.state in {"queued", "analysing", "exporting"}
                    expired = (now - job.started > 3600) if active else (now - job.touched > self.ttl)
                    if expired:
                        self.delete(job)
                    if job.state == "cancelled" or (expired and not active):
                        self.jobs.pop(key, None)
            owned = set(self.jobs)
            # Recover directories left by a crash, without touching arbitrary files.
            for directory in self.root.iterdir():
                if (directory.is_dir() and not directory.is_symlink() and len(directory.name) == 48
                        and all(c in "0123456789abcdef" for c in directory.name)
                        and directory.name not in owned and time.time() - directory.stat().st_mtime > self.ttl):
                    shutil.rmtree(directory, ignore_errors=True)

    def _reap_loop(self) -> None:
        while not self.stopping.wait(30):
            self.reap()

    def close(self) -> None:
        self.stopping.set()
        with self.lock:
            jobs = list(self.jobs.values())
        for job in jobs:
            self.delete(job)
        self.pool.shutdown(wait=True)
        for job in jobs:
            self._discard(job)
