"""
Firmware stress test / crash diagnostic for the RobotFace.

Reproduces the load petd puts on the device: a /ws event stream (with
pings) plus /api/status polling at 2 Hz over a keep-alive connection,
with audio streaming to this PC (and, unless --no-listen, a UDP socket
draining it so the device isn't streaming into a closed port).

Reports reboots (audio_packets_sent or uptime_s going backwards), and on
firmware with the diagnostics fields also prints why the device restarted
(`reset_reason`: panic / task_wdt / brownout / ...) and tracks free heap.

    .venv/bin/python scripts/face_stress.py [--host ...] [--seconds 180] [--no-audio] [--no-listen]
"""

from __future__ import annotations

import argparse
import socket
import sys
import threading
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "face-api"))
sys.path.insert(0, str(ROOT))
from face_client import FaceApiClient, FaceEventStream  # noqa: E402
from petd.net import local_ip_towards                   # noqa: E402


def drain_udp(port: int, stop: threading.Event) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.settimeout(0.5)
    sock.bind(("0.0.0.0", port))
    while not stop.is_set():
        try:
            sock.recv(2048)
        except socket.timeout:
            pass
    sock.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="192.168.101.40")
    parser.add_argument("--seconds", type=float, default=180)
    parser.add_argument("--audio-port", type=int, default=5000)
    parser.add_argument("--no-audio", action="store_true", help="don't (re)point the device's audio stream here")
    parser.add_argument("--no-listen", action="store_true", help="don't drain the UDP audio on this PC")
    args = parser.parse_args()

    t0 = time.time()
    stamp = lambda: f"+{time.time() - t0:6.1f}s"  # noqa: E731
    counts = {"face": 0, "motion": 0, "ws_errors": 0, "fails": 0, "reboots": 0, "polls": 0}
    client = FaceApiClient(args.host)

    stop = threading.Event()
    if not (args.no_audio or args.no_listen):
        threading.Thread(target=drain_udp, args=(args.audio_port, stop), daemon=True).start()

    events = FaceEventStream(client)
    events.on_face = lambda e: counts.__setitem__("face", counts["face"] + 1)
    events.on_motion = lambda e: counts.__setitem__("motion", counts["motion"] + 1)

    def on_error(exc: Exception) -> None:
        counts["ws_errors"] += 1
        print(f"{stamp()} ws error: {exc!r}", flush=True)

    events.on_error = on_error
    events.start()

    session = requests.Session()
    url = f"http://{args.host}/api/status"
    last_packets = last_uptime = None
    min_heap = None
    try:
        while time.time() - t0 < args.seconds:
            try:
                status = session.get(url, timeout=3).json()
                counts["polls"] += 1
                packets, uptime = status["audio_packets_sent"], status.get("uptime_s")
                heap = status.get("heap", {})
                if heap:
                    min_heap = min(min_heap or heap["free"], heap["free"])
                rebooted = (last_packets is not None and packets < last_packets) or \
                           (last_uptime is not None and uptime is not None and uptime < last_uptime)
                if rebooted:
                    counts["reboots"] += 1
                    print(f"{stamp()} REBOOT  reason={status.get('reset_reason', '?')} "
                          f"uptime={uptime}s heap_free={heap.get('free')} "
                          f"min_free_since_boot={heap.get('min_free')}", flush=True)
                if rebooted or last_packets is None:
                    if not args.no_audio:
                        client.set_audio_destination(local_ip_towards(args.host), args.audio_port)
                        client.set_face_detection(True)
                        client.set_recognition(True)
                last_packets, last_uptime = packets, uptime
            except Exception as exc:  # noqa: BLE001
                counts["fails"] += 1
                print(f"{stamp()} status failed: {type(exc).__name__}", flush=True)
                session = requests.Session()
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        events.stop()
    try:
        final = session.get(url, timeout=3).json()
        print(f"{stamp()} final: reset_reason={final.get('reset_reason', '?')} "
              f"uptime={final.get('uptime_s')}s heap={final.get('heap')} lowest_seen={min_heap}")
    except Exception:  # noqa: BLE001
        pass
    print(f"{stamp()} {counts}")
    return 1 if counts["reboots"] or counts["fails"] else 0


if __name__ == "__main__":
    sys.exit(main())
