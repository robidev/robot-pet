from __future__ import annotations

import logging


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s.%(msecs)03d %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Per-request chatter from the HTTP clients and server isn't useful here.
    for noisy in ("urllib3", "websocket", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
