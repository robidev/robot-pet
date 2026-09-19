"""
Minimal pure-stdlib Python client for the Player robot server running on
the vacuum at TCP port 6665 (the `playerc` section of
~/robot-pet/todo.txt -- "timestamped by sensors").

This talks the raw Player wire protocol directly (socket + struct only --
no libplayerc / playerc_python bindings, no xdrlib). It only implements
enough of the protocol to subscribe to a device and read its DATA
messages, which is all that's needed for timestamped odometry/pose.

Protocol details below were read directly out of the actual Player source
checked out at ~/player-build/player (not reconstructed from memory), and
cross-checked against ~/xiaomi_bridge (a working C++ client for this exact
robot) and a live handshake against the robot's own Player server:

  - Wire format: every message is a 40-byte header followed by a payload.
    All scalar fields (u_int, u_short, u_char alike) are encoded as plain
    XDR integers -- i.e. 4 bytes, big-endian -- confirmed in
    ~/player-build/player/replace/xdr.c (xdr_u_short/xdr_u_char delegate to
    XDR_PUTLONG/GETLONG, a 4-byte unit). Doubles are 8-byte big-endian IEEE754
    (xdr_double). Variable-length byte arrays are a 4-byte length prefix
    followed by the data, zero-padded to a 4-byte boundary (xdr_bytes).
    See ~/player-build/player/libplayerinterface/player.h (player_msghdr_t)
    and ~/player-build/player/build/libplayerinterface/playerxdr.c
    (xdr_player_msghdr_t, xdr_player_device_req_t, xdr_player_position2d_data_t).

  - On connect, the server immediately sends a 32-byte ASCII banner
    (PLAYER_IDENT_STRLEN), e.g. b"Player v.3.1.0-svn\\x00...". Confirmed live
    against 192.168.101.43:6665.

  - Data delivery defaults to PUSH mode (player.h: "mode = PLAYER_DATAMODE_PUSH"),
    so once subscribed you just keep reading messages off the socket --
    no PULL/DATA-request/SYNCH dance required.

  - To receive data from a device you first send a REQ to the special
    "player" meta-device (interface code 1, index 0) with subtype
    PLAYER_PLAYER_REQ_DEV (3), carrying a player_device_req_t
    {addr: {interf, index}, access: PLAYER_OPEN_MODE}. The server replies
    RESP_ACK with the same struct (access echoed back) if it succeeded.
    See ~/player-build/player/libplayerinterface/interfaces/001_player.def
    and .../client_libs/libplayerc/client.c (playerc_client_subscribe).

  - ~/xiaomi_bridge/src/xiaomi_player_interface.cpp confirms which
    interfaces/indices this robot's Player server actually exposes:
    position2d@0 (odometry/pose+velocity), ir@0 (wall) / ir@1 (cliff),
    sonar@0 (front ultrasonic), laser@0 (lidar scan), power@0 (battery).
    Only position2d is implemented here since that's what was asked for;
    the others follow the exact same subscribe+read pattern with a
    different interface code and payload layout (see the .def files under
    ~/player-build/player/libplayerinterface/interfaces/ for their struct
    definitions if you need them later).

  - position2d DATA (subtype PLAYER_POSITION2D_DATA_STATE) payload is
    player_position2d_data_t: pos{px,py,pa} + vel{px,py,pa} (doubles,
    meters/radians, meters-per-sec/radians-per-sec) + stall (bool, XDR-padded
    to 4 bytes). Crucially, the *message header* carries a proper
    `timestamp` (double, seconds) -- "Time associated with message
    contents" per player_msghdr_t's own doc comment. That's the per-sample
    timestamp Valetudo doesn't give you.

    IMPORTANT (observed live, not just read from docs): despite player.h's
    comment saying "seconds since epoch", the actual values streamed by
    this robot's driver are NOT Unix epoch time -- a live test returned
    timestamps around 42477 seconds (~11.8 hours), consistent with the
    driver stamping messages using system uptime or its own free-running
    clock rather than wall-clock time. Treat these timestamps as accurate
    and monotonically increasing for computing relative timing (e.g. dt
    between two poses, ordering samples), but do NOT assume they line up
    with time.time() / datetime.now() on your PC -- if you need pose data
    correlated to wall-clock time, stamp it yourself at receipt time
    instead (e.g. `local_recv_time = time.time()` right after each
    read_position2d() yield) and only use the robot's own timestamp for
    fine-grained relative ordering/delta-t within the stream.

This was tested read-only against the real robot (subscribing and reading
position2d data does not move it). Driving the robot via
PLAYER_POSITION2D_CMD_VEL is a different, separate message type (CMD, not
REQ) and is NOT implemented here -- deliberately out of scope for this
read-only pose client. See PositionCommandVel in
~/player-build/player/libplayerinterface/interfaces/004_position2d.def if
you need it later; note it's a genuinely better fit for the
{omega, velocity, duration, seqnum} drive shape from the original wishlist
than Valetudo's manual-control API is (vel.px = velocity m/s, vel.pa =
omega rad/s) -- worth considering instead of the Valetudo-side drive()
approximation if driving is going to happen over this channel.
"""

from __future__ import annotations

import socket
import struct
from dataclasses import dataclass
from typing import Optional

PLAYER_IDENT_STRLEN = 32

PLAYER_MSGTYPE_DATA = 1
PLAYER_MSGTYPE_CMD = 2
PLAYER_MSGTYPE_REQ = 3
PLAYER_MSGTYPE_RESP_ACK = 4
PLAYER_MSGTYPE_SYNCH = 5
PLAYER_MSGTYPE_RESP_NACK = 6

PLAYER_PLAYER_CODE = 1
PLAYER_POSITION2D_CODE = 4

PLAYER_PLAYER_REQ_DEV = 3

PLAYER_POSITION2D_DATA_STATE = 1

PLAYER_OPEN_MODE = 1
PLAYER_CLOSE_MODE = 2

HEADER_SIZE = 40


class PlayerProtocolError(Exception):
    pass


@dataclass
class Pose2D:
    timestamp: float  # seconds, robot's own clock (NOT Unix epoch -- see module docstring)
    x: float           # meters
    y: float           # meters
    yaw: float          # radians
    vx: float           # meters/sec
    vy: float           # meters/sec
    vyaw: float          # radians/sec
    stalled: bool


def _pack_u32(v: int) -> bytes:
    return struct.pack(">I", v & 0xFFFFFFFF)


def _pack_double(v: float) -> bytes:
    return struct.pack(">d", v)


def _pack_bytes(data: bytes) -> bytes:
    """XDR variable-length opaque: 4-byte length prefix + data, padded to 4 bytes."""
    n = len(data)
    pad = (-n) % 4
    return _pack_u32(n) + data + b"\x00" * pad


def _pack_devaddr(interf: int, index: int, host: int = 0, robot: int = 0) -> bytes:
    return _pack_u32(host) + _pack_u32(robot) + _pack_u32(interf) + _pack_u32(index)


def _pack_header(interf: int, index: int, msg_type: int, subtype: int,
                  timestamp: float, seq: int, size: int) -> bytes:
    return (
        _pack_devaddr(interf, index)
        + _pack_u32(msg_type)
        + _pack_u32(subtype)
        + _pack_double(timestamp)
        + _pack_u32(seq)
        + _pack_u32(size)
    )


def _unpack_header(buf: bytes) -> dict:
    host, robot, interf, index, msg_type, subtype = struct.unpack(">6I", buf[0:24])
    (timestamp,) = struct.unpack(">d", buf[24:32])
    seq, size = struct.unpack(">2I", buf[32:40])
    return {
        "host": host, "robot": robot, "interf": interf, "index": index,
        "type": msg_type, "subtype": subtype, "timestamp": timestamp,
        "seq": seq, "size": size,
    }


class PlayerClient:
    """
    Bare-bones Player protocol client. Subscribes to devices and reads
    their DATA messages. Does not send CMD (drive) messages.

    Example:
        client = PlayerClient("192.168.101.43")
        client.connect()
        client.subscribe(PLAYER_POSITION2D_CODE, index=0)
        for pose in client.read_position2d():
            print(pose.timestamp, pose.x, pose.y, pose.yaw)
    """

    def __init__(self, host: str, port: int = 6665, timeout: float = 5.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock: Optional[socket.socket] = None
        self._seq = 0

    def connect(self) -> bytes:
        self.sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        banner = self._recv_exact(PLAYER_IDENT_STRLEN)
        return banner

    def close(self) -> None:
        if self.sock is not None:
            self.sock.close()
            self.sock = None

    def _recv_exact(self, n: int) -> bytes:
        chunks = []
        remaining = n
        while remaining > 0:
            chunk = self.sock.recv(remaining)
            if not chunk:
                raise PlayerProtocolError("connection closed while reading")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _read_message(self) -> tuple[dict, bytes]:
        hdr = _unpack_header(self._recv_exact(HEADER_SIZE))
        payload = self._recv_exact(hdr["size"]) if hdr["size"] else b""
        return hdr, payload

    def subscribe(self, interface_code: int, index: int = 0,
                   access: int = PLAYER_OPEN_MODE) -> None:
        """
        MOVES NOTHING. Subscribes to a device so its DATA messages start
        arriving on subsequent read_* calls (server defaults to push mode).
        """
        payload = (
            _pack_devaddr(interface_code, index)
            + _pack_u32(access)
            + _pack_u32(0)       # driver_name_count
            + _pack_bytes(b"")   # driver_name (empty on a subscribe request)
        )
        header = _pack_header(
            interf=PLAYER_PLAYER_CODE, index=0,
            msg_type=PLAYER_MSGTYPE_REQ, subtype=PLAYER_PLAYER_REQ_DEV,
            timestamp=0.0, seq=self._next_seq(), size=len(payload),
        )
        self.sock.sendall(header + payload)

        hdr, payload = self._read_message()
        if hdr["type"] != PLAYER_MSGTYPE_RESP_ACK:
            raise PlayerProtocolError(
                f"subscribe to interface {interface_code}/{index} failed "
                f"(response type {hdr['type']}, expected RESP_ACK)"
            )
        granted_access = struct.unpack(">I", payload[16:20])[0]
        if granted_access != access:
            raise PlayerProtocolError(
                f"requested access {access}, server granted {granted_access}"
            )

    def read_position2d(self):
        """
        Yields Pose2D for every position2d DATA message received.
        Blocks waiting on the socket; run in a loop / thread as needed.
        Requires subscribe(PLAYER_POSITION2D_CODE, index) first.
        Silently skips messages from other interfaces (in case you've
        subscribed to more than one device on this connection).
        """
        while True:
            hdr, payload = self._read_message()
            if (
                hdr["type"] == PLAYER_MSGTYPE_DATA
                and hdr["interf"] == PLAYER_POSITION2D_CODE
                and hdr["subtype"] == PLAYER_POSITION2D_DATA_STATE
            ):
                px, py, pa, vx, vy, va = struct.unpack(">6d", payload[0:48])
                (stall,) = struct.unpack(">I", payload[48:52])
                yield Pose2D(
                    timestamp=hdr["timestamp"],
                    x=px, y=py, yaw=pa,
                    vx=vx, vy=vy, vyaw=va,
                    stalled=bool(stall),
                )


if __name__ == "__main__":
    import sys

    host = sys.argv[1] if len(sys.argv) > 1 else "192.168.101.43"
    client = PlayerClient(host)
    banner = client.connect()
    print(f"connected, server banner: {banner!r}")

    client.subscribe(PLAYER_POSITION2D_CODE, index=0)
    print("subscribed to position2d@0, reading pose samples (Ctrl-C to stop)...")

    count = 0
    prev_t = None
    for pose in client.read_position2d():
        dt = "" if prev_t is None else f" (dt {pose.timestamp - prev_t:+.4f}s)"
        prev_t = pose.timestamp
        print(
            f"t={pose.timestamp:.4f}{dt}  "
            f"x={pose.x:.3f} y={pose.y:.3f} yaw={pose.yaw:.3f}  "
            f"vx={pose.vx:.3f} vy={pose.vy:.3f} vyaw={pose.vyaw:.3f}  "
            f"stalled={pose.stalled}"
        )
        count += 1
        if count >= 10:
            break
    client.close()
