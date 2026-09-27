# robot-pet

A robot pet with GLaDOS's personality, built from a Roborock V1 vacuum running
Valetudo, a LilyGo ESP32-S3 camera board as its face (eye, pan/tilt head,
camera, microphone), and `petd` on a PC: hearing, speech, a Claude brain,
memory of people and places, face recognition, and driving.

`PLAN.md` is the design and the running log of what was found and decided;
this file is how to set it up, run it, and fix it when it misbehaves.

## How it fits together

```
      face (ESP32-S3, 192.168.101.40)            robot (Roborock V1, 192.168.101.43)
  eye, pan/tilt, camera, mic, face detection    Valetudo, Player, speaker (aplay)
     |  HTTP :80  status, commands, /ws events        |  HTTP :80  Valetudo (state, map, go_to, manual control)
     |  HTTP :81  snapshots                            |  TCP :6665 Player (odometry)
     |  UDP  -> PC :5000  microphone (16 kHz PCM)      |  TCP :6000 speech PCM in, :6001 stop
     v                                                 v
  ------------------------------ petd (PC, WSL2) ------------------------------
   stt: whisper-udp-stream (whisper.cpp + Silero VAD)   speech: piper (GLaDOS voice) :5001
   brain: Claude CLI (Haiku 4.5 by default) + tools over MCP
   memory: runtime/pet.db (people, faces, places, conversations)
   face recognition: YuNet + SFace (onnxruntime) on head snapshots
   dashboard and local API: http://127.0.0.1:8765
```

The addresses are this household's; they're all in `config.yaml`.

## Setup

### The PC (WSL2)

1. **Mirrored networking.** The face streams its microphone to the PC over
   UDP, which WSL2's default NAT networking doesn't forward inbound. In
   `%UserProfile%\.wslconfig` on Windows:

   ```ini
   [wsl2]
   networkingMode=mirrored
   ```

   then `wsl --shutdown` and start WSL again.

2. **Python.**

   ```sh
   python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
   ```

3. **Speech recognition:** build `whisper-udp-stream` and fetch its models,
   as in [`stt/README.md`](stt/README.md).
4. **The voice:** set up piper and the GLaDOS model, as in
   [`piper-tts/README.md`](piper-tts/README.md). petd starts piper itself
   when nothing listens on port 5001.
5. **Face recognition models** (~40 MB, into `models/face/`, git-ignored):

   ```sh
   .venv/bin/python scripts/fetch_face_models.py
   ```

6. **The brain** runs through the Claude Code CLI (`claude`, on the
   subscription): install it and log in once. petd starts it with its own
   MCP config (`runtime/brain/`), so it needs nothing else.
7. **Config:** `cp config.example.yaml config.yaml` and set the addresses.
   Every key is optional (defaults in `petd/config.py`), and any key can be
   overridden from the environment: `PETD__FACE__HOST=192.168.101.41`.

### The robot (Roborock V1)

Rooted, with Valetudo and root SSH. What petd relies on there, started by the
robot's WatchDoge from `/opt/rockrobo/watchdog/ProcessList.conf` (read at boot
only) through small scripts in `/root/watchdog_scripts/`:

- **Player** on TCP 6665 (odometry while manual control is armed), with the
  vacuum motor and brushes on Player's null driver.
- **The speaker:** `socat -u TCP-LISTEN:6000,reuseaddr,fork EXEC:'aplay -f S16_LE -r 22050 -c 1 -t raw -'`
  (`speaker.sh`), and `killall aplay` on TCP 6001 for interrupting (`speaker_stop.sh`).
- `RoboController`, `AppProxy` and the Xiaomi cloud clients run once at boot
  and can then be killed without WatchDoge restarting them (`*_once.sh`).
- **`/root/wlanmgr_pause.sh stop | resume | status`**, run by hand only:
  pauses Xiaomi's `wlanmgr`, whose roaming scan every 30 s leaves the radio
  deaf for ~1.5 s. A reboot always starts it as stock. Don't remove
  `wlanmgr` or change how it starts: it brings the WiFi up at boot, and a
  robot without WiFi can only be recovered over its serial console.

The robot reboots itself every day at 21:54 local time (03:54 on its own
clock, which is Asia/Shanghai). Its map lives in `/mnt/data/rockrobo/`.

### The face (LilyGo ESP32-S3 camera)

The firmware is a separate repository, `~/LilyGo-Cam-RobotFace` (PlatformIO:
`~/.platformio/penv/bin/pio run -t upload` with the board on USB). On a board
without WiFi settings it opens an access point `LilyGo-Cam-…`; join it and
open `http://192.168.4.1/api/wifi?ssid=NETWORK&password=PASSWORD`. Its own page
at `http://<face>/` lists every endpoint. Snapshots come from port 81 (port 80
redirects); there is no video stream.

## Running

```sh
.venv/bin/python -m petd                 # the pet; Ctrl-C stops it cleanly
.venv/bin/python -m petd --fake          # no hardware at all: fakes for everything
.venv/bin/python -m petd --echo          # repeats what it hears: tests the audio path and the echo gate
.venv/bin/python -m petd --log-level DEBUG
```

- **Dashboard:** http://127.0.0.1:8765 (status, events, a snapshot, the
  tools). The same port is the local API the brain's tools go through.
- **Logs:** a folder per run in `runtime/logs/<start time>/` (`latest` points
  at the newest): `petd.log` at DEBUG, `run.yaml` (the config it ran with and
  the git commit), `events.jsonl` (every event, for `scripts/latency.py`).
- **Talking to it:** say its name ("Hey GLaDOS, …"); within a conversation
  it keeps listening for a while. "Remember my face, I'm …" enrolls you.
- Everything it keeps is under `runtime/` (git-ignored: it holds photos of
  people). It stays under ~300 MB: old runs, logs and face crops are pruned.

## Tools

| script | what for |
|---|---|
| `scripts/show_memory.py [conversation N \| map \| faces]` | what the pet stored: people, places, conversations; `map` draws places on the map; `faces` lists fingerprints and draws the face crops and recent recognition attempts |
| `scripts/fix_faces.py forget <id>… \| assign <attempt> <name>` | correct a stored face: drop a wrong fingerprint, or give a misread attempt to the right person |
| `scripts/latency.py [run]` | where each turn's time went, from a run's event trace |
| `scripts/smoke.py vacuum \| face \| …` | hardware smoke tests of each adapter |
| `scripts/fetch_face_models.py` | downloads the face recognition models |
| `scripts/echo_timing.py` | how long the robot's voice outlasts what was sent (sets the echo gate) |
| `scripts/tracking_log.py` | the head's face tracking, pass by pass |
| `scripts/face_stress.py` | the face firmware under petd-like load (crash hunting) |
| `scripts/calibrate_face.py`, `scripts/calibrate_motion.py` | geometry and motion calibration |

## Configuration

`config.example.yaml` has every section with comments; the most useful:

| section | what it sets |
|---|---|
| `face`, `vacuum`, `network` | the devices' addresses; `network.pc_ip` is where the face streams audio (`auto`: the PC's address towards the face) |
| `stt` | the whisper binary, models, threads, extra VAD arguments |
| `speaker` | piper's URL, the robot's ports, `volume`, `lead_s` (buffer against WiFi stalls) and the echo gate |
| `brain` | `backend` (`claude_cli`, or `ollama`), the model, the persona files in `memory/` |
| `memory` | the database, how often it greets someone, familiarity tiers |
| `recognition` | face recognition thresholds (from measurements: see PLAN.md 4.7), growth of the stored faces, the 10 s recheck |
| `log` | log folder, run count and size limits |

## Troubleshooting

**Everything is unreachable at once.** The PC sometimes sits on another
network (192.168.178.x). `ip -brief addr`: petd's devices are on 192.168.101.x.

**The face "didn't answer" / goes offline.** Usually the WiFi: from the PC,
`runtime/face-serial/netwatch.sh <log>` pings the face, the robot and the router
every 0.5 s and logs misses. The PC is a laptop on WiFi too, so misses to the
router as well point at the PC's side or the router. If the face stops
answering pings for good while its display runs, its link is stuck: the
firmware reconnects after 5 s of failed sends and restarts itself after 30 s
more (`wifi.stuck_resets` in `/api/status`).

**Why did the face reboot or hang?** `/api/status` only has a reset reason; the
serial log has the cause. With the board on the PC's USB (attached to WSL as
`/dev/ttyACM0`): `.venv/bin/python runtime/face-serial/capture.py <log>` keeps a
timestamped log. It has to be running *before* the problem: plugging the board
into the PC is itself a power cycle.

**It doesn't hear anything.** The face's `/api/status` should say
`"audio_configured": true` (petd sets it); the stt process in `petd.log`
should report `stt ready`. Without mirrored networking (setup, step 1) the
audio never reaches WSL.

**It hears "Gladys", "Clovis", "Class" instead of GLaDOS.** Whisper, primed
with the name; the name matcher accepts the common misses. Saying "Hey GLaDOS"
helps most.

**The voice cuts out mid-sentence.** The robot's WiFi: `wlanmgr`'s scan every
30 s (`/root/wlanmgr_pause.sh stop` on the robot), and `speaker.lead_s` buffers
2 s against other stalls.

**It calls someone by the wrong name.** `scripts/show_memory.py faces` shows
the crops behind each stored face and the recent attempts; `scripts/fix_faces.py`
corrects them. A running petd also looks again every 10 s while a named face is
in view.

**The robot ignores manual driving.** Arming manual control spins the lidar up
and moves are ignored for ~6 s; petd waits 7 s and keeps the session alive.

**The robot's Valetudo map pose is stale.** It freezes while manual control is
armed and catches up ~1 s after disarming.

More findings, with the measurements behind them, are in PLAN.md (0.2
Progress and the dated findings under it).

## Repository

| folder | what |
|---|---|
| `petd/` | the daemon: `io/` (devices), `brain/`, `behavior/`, `memory/`, `spatial/`, `vision/`, `api/` |
| `memory/` | the persona files the brain is prompted with |
| `face-api/`, `vacuum-api/`, `playerc-client/` | clients for the face, Valetudo and Player, each with its own README |
| `stt/`, `piper-tts/` | speech recognition and the voice |
| `scripts/` | the tools above |
| `tests/` | `.venv/bin/python -m pytest` (fakes only, no hardware) |
| `runtime/` | everything the pet keeps (git-ignored) |
