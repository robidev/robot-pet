# face-api

Python client for the LilyGo-Cam-RobotFace HTTP + WebSocket API
(`~/robot-pet/todo.txt` low-level-glue-logic MVP, "python HTTP API call to
face" section).

Built by reading the actual firmware (`~/LilyGo-Cam-RobotFace/src/control_server.cpp`,
`face_service.cpp`, `motion_service.cpp`), not a generic spec.

## Install

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Usage

```python
from face_client import FaceApiClient, FaceEventStream

face = FaceApiClient("192.168.4.1")

# measurements
face.get_status()             # full JSON: face/eye/servo/motion/time/...
face.get_snapshot()           # Snapshot(jpeg=b'...', utc=..., servo_pan_deg=..., servo_tilt_deg=...)
for jpeg in face.iter_stream_frames():
    ...                       # live MJPEG frames, no per-frame metadata

# commands
face.set_face_detection(True)
face.set_face_detector(resize_scale=1.0)  # detector's first stage: 0.1-1.0 of the frame
face.set_recognition(True)
face.enroll_face(True)              # arms enrollment of the next face seen
face.get_current_faces()            # CurrentFaces: fresh boxes + servo + ms UTC + age_ms
face.list_enrolled_faces()          # [0, 1, 3]
face.delete_enrolled_face(3)        # DELETES one enrolled ID, returns count remaining
face.clear_enrolled_faces()         # DELETES all enrolled IDs
face.set_servo(mode="track")
face.set_servo(pan_deg=90, tilt_deg=90)
face.set_eye_mode("manual")         # pause idle/tracking so eye target sticks
face.set_eye_target(x=0.3, y=-0.1, aperture=1.0)
face.set_audio_gain(1.5)
face.set_audio_destination("192.168.4.50", 5000)
face.set_camera(brightness=1, vflip=1)
face.set_ntp_server("pool.ntp.org")
face.set_wifi_credentials("ssid", "password")   # REBOOTS THE DEVICE

# events (background thread, subscribes to /ws)
events = FaceEventStream(face)
events.on_face = lambda evt: print("face:", evt)
events.on_motion = lambda evt: print("motion:", evt)
events.start()
...
events.stop()
```

Run `python3 status.py [ip]` for a quick read-only status dump (defaults to
`192.168.4.1`, the device's AP-mode address).

## Notes / deviations from the original wishlist

- **Update (2026-09-19): firmware now has `GET /api/face/current`**
  (`face.get_current_faces()`), which returns the latest processed frame's
  boxes, ids, servo pose, ms-resolution UTC and `age_ms`. That resolves the
  point below on firmware with that endpoint. The same firmware adds
  `face.list_enrolled_faces()` and `face.delete_enrolled_face(id)`
  (a single delete, so "forget one person" no longer means clear-all), a
  `seq` on every `/ws` event, and millisecond UTC timestamps everywhere
  (`parse_utc()` converts both old and new formats to epoch seconds).
- **(Older firmware) "get face position+size+id ... with servo coordinates -> timestamped"
  has no matching GET endpoint.** The device only computes the full
  per-face box/landmarks list while the vision task processes a frame, and
  only pushes it over `/ws` when the detected face set actually changes
  (`FaceService::consumeFrameEvent` in `face_service.cpp`) -- there's no
  polling endpoint that returns it on demand. `/api/status`'s
  `face_detected`/`face_id`/`face_confidence` fields are a *summary* of just
  the single primary (first) face, with no box and no per-face timestamp.
  `FaceEventStream.last_face` is the closest equivalent to a "get" call:
  it's the most recent pushed event, cached and timestamped, but can be
  stale (or `None`) if nothing has changed recently. If you need a
  guaranteed-fresh face read, request `get_snapshot()` instead and run your
  own detection on the PC side.
- **The PIR motion event didn't originally carry servo-pos.** The wishlist
  asks for "movement detected -> timestamped, servo-pos" but the firmware's
  first cut of the motion broadcast only sent `{event, active, utc}`. Fixed
  in `control_server.cpp`'s `broadcast_motion_event()` to also sample
  `servo_service->state()` at broadcast time (same pattern the face event
  already used), so `MotionEvent.servo` is now populated. Requires that
  firmware change to be flashed; older firmware builds will send motion
  events without a `"servo"` key and `FaceEventStream._handle_motion` will
  raise a `KeyError` (surfaced via `on_error`, not silently dropped).
- **No native MJPEG frame metadata.** `/stream` (used by `iter_stream_frames()`)
  has no per-frame servo pose or timestamp, unlike `/api/snapshot`. Use
  `get_snapshot()` when you need a single frame tied to a known pose/time,
  and the stream only for smooth live viewing.
- **Every command endpoint is a GET, not a PUT/POST.** Unlike
  `../vacuum-api`'s Valetudo client (which uses PUT for state-changing
  calls), the RobotFace firmware exposes commands as plain `GET
  /api/...?param=value` query endpoints (see `control_server.cpp`). This
  client follows that -- there's no semantic distinction in the HTTP verb
  here, only in which calls change device state (documented per-method).

## Safety

Nothing in this module talks to the device automatically on import.
State-changing calls (`set_face_detection`, `set_servo`, `set_eye_target`,
`enroll_face`, `clear_enrolled_faces`, `set_wifi_credentials`, ...) are
documented as such. `set_wifi_credentials()` reboots the device and can
leave it unreachable at its current address if the credentials are wrong
(it falls back to AP mode). `clear_enrolled_faces()` cannot be undone.
