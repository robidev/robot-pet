"""
python -m petd [--config config.yaml] [--fake] [--echo] [--log-level DEBUG]

--fake   in-memory hardware: no robot, face, mic or speaker needed.
--echo   repeat back whatever is heard (tests mic -> STT -> TTS -> robot speaker
         and the self-hearing gate before the brain exists).

Every run also logs, at DEBUG, to runtime/logs/<start time>/petd.log, next to
a run.yaml with its config and events.jsonl with every bus event (for
scripts/latency.py); runtime/logs/latest is the newest (petd/log.py).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
from pathlib import Path
from typing import Optional

from .app import App
from .config import Config, load_config
from .log import setup_logging

log = logging.getLogger("petd")


async def main_async(args: argparse.Namespace, cfg: Config, run_dir: Optional[Path] = None) -> None:
    app = App(cfg, fake=args.fake, echo=args.echo, run_dir=run_dir)
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, app.request_shutdown)
    try:
        await app.start()
        await app.run_until_stopped()
    finally:
        log.info("shutting down")
        await app.close()


def main() -> None:
    parser = argparse.ArgumentParser(prog="petd", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="YAML config (default: ./config.yaml if present)")
    parser.add_argument("--fake", action="store_true", help="use in-memory fake hardware")
    parser.add_argument("--echo", action="store_true", help="say back what is heard (audio path test)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    cfg = load_config(Path(args.config) if args.config else None)
    tags = tuple(tag for tag, on in (("fake", args.fake), ("echo", args.echo)) if on)
    run_dir = setup_logging(args.log_level, cfg, tags)
    try:
        asyncio.run(main_async(args, cfg, run_dir))
    except Exception:
        log.exception("petd crashed")
        raise
    log.info("petd stopped")


if __name__ == "__main__":
    main()
