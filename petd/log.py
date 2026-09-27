"""
Logging: the console at the level asked for, and a log folder per run.

Each run gets runtime/logs/<start time>[-fake|-echo]/ (runtime/logs/latest
points at the newest), holding:

- petd.log, at DEBUG whatever the console shows: dropped transcripts and
  why, subprocess stderr, retries. The things nobody was watching for when
  they happened.
- run.yaml: when, the command line, the git commit (and whether the tree
  had uncommitted changes), and the full effective config, so a log can be
  read against the settings it ran with.

Only the newest `log.keep_runs` folders are kept, and of those the older
runs together take at most `log.max_old_runs_mb` (the oldest go first).
"""

from __future__ import annotations

import dataclasses
import logging
import logging.handlers
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import yaml

from .config import PROJECT_ROOT, Config

FORMAT = "%(asctime)s.%(msecs)03d %(levelname)-7s %(name)s: %(message)s"
RUN_DIR_NAME = re.compile(r"^\d{8}-\d{6}(-[a-z0-9]+)*$")
# Chatter from the HTTP and websocket libraries isn't useful, even in a file.
NOISY = ("urllib3", "websocket", "websockets", "uvicorn.access", "httpx", "httpcore",
         "multipart", "PIL")


def setup_logging(level: str = "INFO", cfg: Optional[Config] = None, tags: tuple = (),
                  argv: Optional[list] = None) -> Optional[Path]:
    """Console logging, plus this run's log folder if cfg asks for one. Returns the folder."""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(getattr(logging, level.upper(), logging.INFO))
    console.setFormatter(logging.Formatter(FORMAT, datefmt="%H:%M:%S"))
    root.addHandler(console)
    for noisy in NOISY:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if cfg is None or not cfg.log.enabled:
        return None
    run_dir, handler = start_run_log(cfg, tags, argv if argv is not None else sys.argv)
    root.addHandler(handler)
    logging.getLogger("petd").info("logging this run to %s", run_dir)
    return run_dir


def start_run_log(cfg: Config, tags: tuple = (), argv: Optional[list] = None,
                  now: Optional[float] = None) -> tuple[Path, logging.Handler]:
    """Creates the run's folder and its log file handler, and prunes old runs."""
    base = cfg.path(cfg.log.dir)
    started = time.time() if now is None else now
    name = "-".join([time.strftime("%Y%m%d-%H%M%S", time.localtime(started)), *tags])
    run_dir = base / name
    suffix = 1
    while run_dir.exists():          # two starts within a second
        suffix += 1
        run_dir = base / f"{name}-{suffix}"
    run_dir.mkdir(parents=True)

    handler = logging.handlers.RotatingFileHandler(
        run_dir / "petd.log", maxBytes=int(cfg.log.max_file_mb * 1024 * 1024),
        backupCount=cfg.log.keep_files, encoding="utf-8")
    handler.setLevel(getattr(logging, cfg.log.file_level.upper(), logging.DEBUG))
    handler.setFormatter(logging.Formatter(FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))

    meta = {
        "started": time.strftime("%Y-%m-%d %H:%M:%S %z", time.localtime(started)),
        "argv": list(argv or []),
        "python": sys.version.split()[0],
        "git": _git_state(),
        "config": dataclasses.asdict(cfg),
    }
    (run_dir / "run.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
    _point_latest(base, run_dir)
    prune_runs(base, cfg.log.keep_runs, cfg.log.max_old_runs_mb, current=run_dir)
    return run_dir, handler


def prune_runs(base: Path, keep: int, max_old_mb: Optional[float] = None,
               current: Optional[Path] = None) -> list[Path]:
    """
    Deletes all but the newest `keep` run folders, then the oldest of the rest
    until the runs other than `current` take at most `max_old_mb`. Only run
    folders (by name) directly in `base` are ever touched, never `current`.
    """
    runs = sorted(p for p in base.iterdir()
                  if p.is_dir() and not p.is_symlink() and RUN_DIR_NAME.match(p.name))
    doomed = runs[:max(0, len(runs) - keep)]
    if max_old_mb is not None:
        older = [p for p in runs[len(doomed):] if current is None or p.resolve() != current.resolve()]
        sizes = {p: _size(p) for p in older}
        total, budget = sum(sizes.values()), max_old_mb * 1024 * 1024
        for path in older:                      # oldest first
            if total <= budget:
                break
            doomed.append(path)
            total -= sizes[path]
    for path in doomed:
        shutil.rmtree(path, ignore_errors=True)
    return doomed


def _size(folder: Path) -> int:
    return sum(f.stat().st_size for f in folder.rglob("*") if f.is_file() and not f.is_symlink())


def _point_latest(base: Path, run_dir: Path) -> None:
    latest = base / "latest"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(run_dir.name)
    except OSError:
        pass                          # a convenience; the run folder is what matters


def _git_state() -> dict:
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True,
                              text=True, timeout=5).stdout.strip()
    try:
        return {"commit": git("rev-parse", "--short", "HEAD"),
                "uncommitted": git("status", "--porcelain").splitlines()}
    except (OSError, subprocess.SubprocessError):
        return {}

