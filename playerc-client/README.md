# playerc-client

Minimal pure-stdlib Python client for the Player robot server running on
the vacuum at TCP port 6665 (the `playerc` section of
`~/robot-pet/todo.txt`, chosen specifically because it gives
**timestamped** pose data, unlike the Valetudo HTTP API -- see
`~/robot-pet/vacuum-api/README.md`).

No dependencies beyond the standard library (`socket`, `struct`) -- no
`libplayerc`/`playerc_python` C bindings needed. The wire protocol
(XDR-encoded messages over TCP) was read directly out of the real Player
source at `~/player-build/player` and cross-checked against
`~/xiaomi_bridge` (a working C++ client for this exact robot), then
verified live against the robot at `192.168.101.43:6665`.

## Usage

```
python3 playerc_client.py [ip]   # defaults to 192.168.101.43
```

```python
from playerc_client import PlayerClient, PLAYER_POSITION2D_CODE

client = PlayerClient("192.168.101.43")
client.connect()
client.subscribe(PLAYER_POSITION2D_CODE, index=0)
for pose in client.read_position2d():
    print(pose.timestamp, pose.x, pose.y, pose.yaw)
```

`Pose2D` fields: `timestamp` (seconds, robot's own clock), `x`/`y` (m),
`yaw` (rad), `vx`/`vy` (m/s), `vyaw` (rad/s), `stalled` (bool).

## Important: the timestamp is not wall-clock time

`player.h` documents the header's `timestamp` field as "seconds since
epoch", but a live test against this robot returned values around 42513
seconds (~11.8 hours) -- clearly the driver's own uptime/free-running
clock, not Unix epoch time. Use it for relative timing (delta between
samples, ordering) within the stream; if you need pose correlated to your
PC's wall-clock time, stamp it yourself at receipt (`time.time()`
immediately after each yield).

## What's implemented vs. not

- Implemented: connect, subscribe, read `position2d` DATA (odometry pose +
  velocity, at ~50Hz on this robot).
- Not implemented: driving (`PLAYER_POSITION2D_CMD_VEL`), and the other
  interfaces `~/xiaomi_bridge` exposes (`ir@0`/`ir@1` for wall/cliff,
  `sonar@0`, `laser@0`, `power@0` for battery) -- only pose was asked for.
  They follow the same subscribe+read pattern with a different interface
  code; see the `.def` files under
  `~/player-build/player/libplayerinterface/interfaces/` for their struct
  layouts if/when those are needed.
- Worth knowing for later: `PLAYER_POSITION2D_CMD_VEL`'s fields
  (`vel.px` = forward velocity m/s, `vel.pa` = angular velocity rad/s) are
  actually a much closer match to the original
  `{omega, velocity, duration, seqnum}` drive wishlist than Valetudo's
  manual-control API is. If driving ends up going over this channel
  instead of Valetudo, that's the natural place for it.

## Safety

This client only subscribes and reads (`PLAYER_OPEN_MODE`) -- it never
sends a CMD message, so it cannot move the robot. Tested live, read-only,
against the real robot while building this.
