from __future__ import annotations

import socket


def local_ip_towards(host: str, port: int = 80) -> str:
    """
    The local address the OS would use to reach `host`. A UDP connect
    sends nothing; it only selects a route. Works with WSL2 mirrored
    networking, where this is the Windows host's LAN address.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect((host, port))
        return s.getsockname()[0]


def tcp_port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False
