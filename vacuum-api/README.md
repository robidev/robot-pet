# vacuum-api

Python client for the Valetudo HTTP API running on the vacuum robot
(`~/robot-pet/todo.txt` low-level-glue-logic MVP, "python HTTP API call to
valetudo" section).

Built by reading the actual Valetudo source (`~/Valetudo/backend`), then
verified read-only against the real robot at `192.168.101.43`.

## Install

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Usage

```python
from valetudo_client import ValetudoClient

vac = ValetudoClient("192.168.101.43")

# measurements (all read-only / safe)
vac.get_status()            # {"value": "docked", "flag": "none", "error": None}
vac.get_battery()           # {"level": 100, "flag": "charged"}
vac.get_position()          # Pose(x=2560, y=2549, angle=342)
vac.get_charger_position()  # Pose(x=2560, y=2530, angle=None)
vac.get_map()                # raw ValetudoMap JSON
vac.get_map_image()          # PIL.Image bitmap rendered from the map JSON
vac.get_map_image(markers=[(2699, 2558, "kitchen")])   # plus labeled points, in map cm

# commands (THESE MOVE THE ROBOT)
vac.start_cleaning()
vac.pause_cleaning()
vac.stop_cleaning()
vac.go_to_dock()
vac.go_to(x, y)
vac.locate()                          # just beeps, doesn't move
vac.drive(omega=-0.8, velocity=0.3, duration=1500, seqnum=1)
```

Run `python3 status.py [ip]` for a quick read-only status dump (defaults to
`192.168.101.43`).

## Notes / deviations from the original wishlist

- **No bitmap map endpoint exists in Valetudo.** `/api/v2/robot/state/map`
  returns JSON (RLE-compressed pixel layers + entities); Valetudo's own web
  UI rasterizes this client-side. `get_map_image()` does the same locally
  with Pillow, and by default crops to the bounding box of the floor/wall/
  segment layers (+ padding) before scaling up -- otherwise the mapped room
  is a tiny speck in the middle of Valetudo's full (e.g. 5120x5120) canvas.
  The crop intentionally ignores the `path` entity (cleaning trail), which
  can extend well outside the room. Pass `crop_to_content=False` for the
  raw full-canvas image.
- **No native `{omega, velocity, duration, seqnum}` drive command exists.**
  This robot has `HighResolutionManualControlCapability`, which only offers
  a one-shot `{velocity: -1..1, angle: -180..180}` vector that must be kept
  alive by resending it (dead-man's switch) while manual control is
  enabled. `drive()` emulates the requested interface on top of that:
  `omega` is passed straight through as `angle` in degrees (not an actual
  angular velocity), and the vector is resent every ~150ms for `duration`
  ms. `seqnum` is accepted but unused by Valetudo -- it's yours for
  bookkeeping. See the docstring in `valetudo_client.py` for details, and
  `TeleopSession` for a background-thread variant suited to a
  joystick/live-control UI instead of a fixed duration.
- Position/angle come from the `robot_position` map entity
  (`get_map()["entities"]`), since Valetudo doesn't have a separate
  position-only endpoint.

## Safety

Nothing in this module calls the robot automatically on import. All
state-changing calls (`start_cleaning`, `stop_cleaning`, `go_to`,
`drive*`, `enable_manual_control`, ...) are documented as such and were
**not** exercised against the real robot while building this -- only GET
endpoints were used to verify response shapes.
