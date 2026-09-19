#!/usr/bin/env python3
"""
Read-only status dump. Safe to run at any time -- never enrolls, clears
faces, moves servos, or otherwise changes device state.

Usage: python3 status.py [ip-address]  (defaults to 192.168.4.1, the
device's default AP-mode address)
"""

import sys

from face_client import FaceApiClient


def main() -> None:
    host = sys.argv[1] if len(sys.argv) > 1 else "192.168.4.1"
    face = FaceApiClient(host)

    status = face.get_status()

    print(f"device:  {status['ip']}")
    print(f"camera:  {'ready' if status['camera'] else 'not ready'}")
    print(f"audio:   {'ready' if status['audio'] else 'not ready'}"
          f" (configured={status['audio_configured']},"
          f" packets_sent={status['audio_packets_sent']})")
    print(f"motion:  {'active' if status['motion'] else 'idle'}")

    print(f"face:    enabled={status['face_enabled']} detected={status['face_detected']}"
          f" id={status['face_id']} confidence={status['face_confidence']:.3f}")
    print(f"         enrolling={status['face_enrolling']}"
          f" recognition={status['face_recognition']} enrolled_faces={status['enrolled_faces']}")

    eye = status["eye"]
    print(f"eye:     x={eye['x']:.3f} y={eye['y']:.3f} aperture={eye['aperture']:.3f}"
          f" mode={eye['mode']}")

    servo = status["servo"]
    print(f"servo:   ready={servo['ready']} mode={servo['mode']}"
          f" pan={servo['pan']:.1f} tilt={servo['tilt']:.1f}"
          f" tracking_gain={servo['tracking_gain']:.1f} tracking_rate={servo['tracking_rate']:.1f}")

    cpu = status["cpu_utilization"]
    print(f"cpu:     core0={cpu['core0']:.1f}% core1={cpu['core1']:.1f}%")

    time_info = status["time"]
    print(f"time:    synced={time_info['synced']} utc={time_info['utc']}"
          f" ntp_server={time_info['ntp_server']}")


if __name__ == "__main__":
    main()
