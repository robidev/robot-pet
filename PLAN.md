# Robot Pet: MVP design and implementation plan

Status: design, ready to implement. Date: 2026-09-19.
Scope: the glue ("petd") that turns vacuum-api + face-api + playerc-client + whisper-udp-stream + piper-tts into a pet.

---

## 0. TL;DR

- **One Python 3.12+ asyncio daemon (`petd`)** owns all the hardware connections, an event bus, a behavior state machine, a people/memory SQLite DB, and an LLM "brain" that can be swapped out.
- **The existing clients are reused as they are.** Blocking calls run in a thread executor. `whisper-udp-stream` and `piper.http_server` stay as separate processes.
- **The LLM is a pluggable backend.** `ClaudeCliBackend` (the MVP) keeps one persistent `claude -p` process in stream-json mode. `OllamaBackend` is designed now and built later. Tools are defined once in a registry. The Claude CLI reaches them through a tiny MCP shim, and Ollama calls them directly.
- **Reflexes run locally, not in the LLM:** "stop", low battery, bump/stuck, self-hearing, and quiet hours. The LLM decides *what* to do. Local behaviors decide *how* to do it safely.
- **"Drive to a person" starts as stop-and-look:** stop, let servo tracking center the face, then use the pan angle for bearing and the face box size for distance. Take the map pose while stationary, project the target, check it against the map, `go_to`, and re-acquire. This does not depend on timestamp fusion. Fusion (Player odometry and clock offset) is still built, for events seen while moving and for logging where people were seen.
- **Four milestones:** M1 *It talks*, M2 *It knows you*, M3 *It comes to you*, M4 *It feels alive*.

---

## 0.1 Decisions (answers from 2026-09-19)

| Topic | Decision | Effect on the plan |
|---|---|---|
| Personality | **GLaDOS**: dry, sardonic, passive-aggressive "science" framing, secretly attached to its humans | F1 writes a GLaDOS persona. The piper GLaDOS voice fits. Safety rules still apply: the sarcasm stays verbal. |
| Host | This WSL2 machine. `networkingMode=mirrored` is set and `whisper-udp-stream` already works | F7 is solved. B4 only needs to wrap the existing tool. |
| Firmware | Millisecond timestamps: **yes**. Regular face events: replaced by an **on-demand GET** (see A2) | A1 stays. A2 becomes `/api/face/current`. |
| LLM | Claude CLI on the subscription, **Haiku 4.5** by default and Sonnet 5 selectable | As designed. |
| People | Unknown faces are welcome. **Recognized faces get familiarity**: greeting by name, nicknames, running jokes, inside references | A `familiarity` tier on `people` (4.7). Tier-aware greeting and summon rules (4.6). |
| Photos | Stored on the PC **occasionally** (explore, `look()` on request, first sighting of a new person), never continuously | `store_photos: true` with a rate cap and a disk cap. |
| Language | English only | `base.en` stays. |
| Rooms | One room, part of it kitchen. **No Valetudo segments or no-go areas:** this V1's Valetudo has neither capability (2026-09-26) | `go_to_room` is low value. Add **named places** ("the couch", "the door") taught by voice (4.5). A kitchen area, if wanted, is our own zone (C1). |
| Geometry | Pan 90° = straight ahead. **Camera height 0.20 m** | Tilt elevation becomes a primary distance cue (4.2). Standoff raised to 1.0 m so standing faces stay within tilt range. |
| Household | ~~Adults only~~ **Corrected 2026-09-27: Robin's son Noah is a young child** and is around the robot. No animals | **Open:** speed caps and approach rules (E4/E5) were set for adults only; decide what a small child in the room changes before M3. |

---

## 0.2 Progress

| Cluster | Status (2026-09-27) |
|---|---|
| **Milestones** | **M1 (It talks) works on the robot. M2 (It knows you) is back** with face recognition on the PC (E6b; verified live with Robin and Claudia, in daylight, and a child guest stayed unknown; two at once and backlight still to check). **M3 and M4 not started.** |
| **A: firmware and C++** | **Done**, and the face firmware (`~/LilyGo-Cam-RobotFace`) has moved on since, all flashed: A1–A3 (`edc6f87`, `e6102eb`, `b3066da`); tilt clamp 58–105° (`6b6d3c7`); **detection-only head**: raw RGB565, VGA with stage one at 0.5, faces detected out to 2.7 m (`5bc6dca`); **stable tracking**: pose looked up at the frame's capture time, tilt's own gain (`95da934`); per-pass timing and a camera probe (`90eaf76`); WiFi reconnects whatever the reason, drops in `/api/status` (`0ec26d4`). A4 (`--json`) and `--prompt` are in `stt/udp-stream`. |
| **B: foundations and I/O** | **Done and verified on hardware.** Since: a **log folder per run** (`runtime/logs/<run>/`: `petd.log` at DEBUG, `run.yaml` with the config, `events.jsonl` with every bus event); short network stalls no longer count as offline (`offline_after_s`); face-board reboots and WiFi drops are logged with their reason. |
| **C: spatial** | **C4 done on hardware** (`spatial/motion.py`: `turn`/`move`, `remember_place`/`go_to_place`); a straight forward `move` leaves the dock. **Docking** (`spatial/dock.py`): `go_home` drives to a point 60 cm in front of the dock first, then docks, retrying once; verified live. **C1 built and wired in, not yet tried live** (2026-09-27): `spatial/mapgeo.py` (decode, `is_free`, `march_back`, `align`), `spatial/frame.py` (places kept in a reference map's frame); a `go_to` counts as arrived only within 20 cm of its goal. **Open:** C1's live check (Next up, item 1), **the face half of C3** (worth retrying now detection reaches 2.7 m), **C5**. C2 shrank to nothing (see 4.1). |
| **D: brain** | **D1–D4 done and verified on hardware.** The claude process now starts before anyone speaks (`brain.prestart`). D5 (ollama) written, never run against a real ollama. |
| **E: behaviors and memory** | **E1–E3 done** (E3's faces now come from E6). **E6a** measured in lamp light (`runtime/e6a/`); **E6b** verified live with Robin and Claudia, in daylight, and a guest (2026-09-27); a swap within the presence debounce kept the old name, fixed with rechecks (after a face leaves view, and every 10 s). **E6c** done: crops kept, shown (`show_memory.py faces`) and correctable (`scripts/fix_faces.py`). **Open:** E6's remaining live checks (Next up, item 0), **E4** (approach, search, the arbiter: M3), **E5** (drives, sleep, explore: M4). |
| **F: personality** | **F1 done.** F2 (games) open. |
| **G: polish and ops** | **G4 done** (turn timings on the bus, `scripts/latency.py`); real numbers still to take from a live run. G1–G3 open. Also `scripts/show_memory.py` (what the pet stored, places on the map). |

**Findings from the first hardware smoke tests (2026-09-20):**

- **The face firmware rebooted every 20–40 s under `petd`. Fixed** (firmware `e6102eb` and `b3066da`, flashed and verified). `scripts/face_stress.py` reproduces the load; `/api/status` now also reports `uptime_s`, `reset_reason` and `heap`, and a serial capture named the task. Causes, all pre-existing:
  - The device was **resetting itself** (`reset_reason=task_wdt`, "CPU 0: vision"): a detection pass is hundreds of ms of uninterrupted inference, and the vision loop's single 5 ms yield went to the higher-priority audio and WiFi tasks, so core 0's idle task never ran inside the 5 s watchdog window.
  - `/ws` frames were sent **straight from the vision and control tasks**, racing the httpd task on the same sockets. They are queued with `httpd_queue_work` now.
  - **The eye redrew the OLED every 5 ms tick over a 100 kHz I2C bus** (~90 ms per frame), so the control task rendered back-to-back: core 1 at 87 % even with vision off. Now 400 kHz, redrawn only when the eye's state changed.
  - `/api/status` waited on the recognizer mutex (held for a whole pass), which is why responses took ~780 ms. The enrolled count is cached.
  - Result under the same load: core 0 3 %, core 1 8 % (was 3 %/100 %), `/api/status` median 21 ms (was 781 ms), no reboots.
  - Worth knowing: the task watchdog only watches **core 0's** idle task, so starving core 1 fails silently. Vision now runs on core 1 and caps its own duty cycle.
- **"GLaDOS" is transcribed as "Gladys."** The wake-word matcher (E2) must accept it, along with "glados", "gladis" and "glad os". Consider passing whisper an initial prompt containing "GLaDOS" to bias it (a small `--prompt` option in udp-stream).
- **Recognition flickers.** It alternates between id=1 and id=-1 frame to frame (similarity about 0.55, close to the threshold), and detection drops out for more than 1.5 s at a time even with a face held still close to the camera (box h ≈ 0.5). Two consequences:
  - Identity needs to be **sticky per presence episode**: once recognized, keep the name until presence is lost. That goes in E3.
  - Presence debounce should probably be about 3 s.
- **The device clock was about 6.7 s behind the PC** before the reboot, and 0.17 s ahead after re-syncing. `device_utc` can't be trusted in absolute terms. Estimate a face-clock→PC offset (for example from `/api/status` round-trips, the same lower-envelope approach as Player) in C2.
- **STT latency:** about 1.9–2.2 s from the end of speech to text. The echo-filter `ignore list` caught a stray "you" as designed.
- **Barge-in is not possible yet, as expected:** while the pet speaks, the user's speech is dropped along with the pet's own (the gate can't tell them apart). Phase 2.

**Findings from issues.txt (2026-09-22):** the pet said it had stored Noah's face, and that it was on its way somewhere, when it had no tool for either. `body.md` describes the whole body, and the model filled the gap between that and its tools with claims. `style.md` now says a missing tool means "I can't yet", and it says so in testing. Both abilities still need their tools: the face one landed in E3; driving is cluster C.

**Findings from the first E2/E3 hardware test (2026-09-22):** enrolling two faces, recognition after a restart, the name gate and interrupting all worked. Four problems, fixed in software:

- **The pet's own echo started a conversation with itself.** The timing gate ends `playback_latency_s + gate_tail_s` (0.9 s) after the last audio was *due*; whatever the robot plays after that is heard as someone else, and the 20 s conversation window then passes it to the brain. Now also filtered by content (a transcript that mostly repeats the pet's last 20 s of speech is dropped). `scripts/echo_timing.py` measures the real lag, to set the timing gate from data instead of a guess.
- **A photo in a tool result crashed the turn** (`LimitOverrunError`): the CLI echoes it as one base64 line, over asyncio's 64 KiB readline limit. Raised to 64 MiB.
- **Whisper hears "GLaDOS" as "Clovis", "Gladys", "G let us".** udp-stream has `--prompt` now, and petd primes it with the pet's name. On piper-spoken test lines that turned "Hey G let us" into "Hey GLaDOS" and "Glottos" into "GLaDOS"; the misses keep the capitals ("GularDOS"), which the name matcher accepts.
- **"You're new" to someone it had just recognized:** the people list called familiarity tier 0 "new". It now says "I know their face", and tier 0 isn't named at all.

Presence still dropped a seated, recognized person three times in 45 s with the 3 s debounce; raised to 5 s.

**Audio cutting out mid-sentence (2026-09-22): the robot's WiFi, not the watchdog.** `socat`/`aplay` had run untouched since boot. The robot's link stalls now and then for 1.3–1.4 s (pings to the face and the router, on the same network, stayed under 20 ms), and aplay's buffer is 0.5 s while petd kept only 0.3 s ahead: every stall was an underrun (aplay: "at least 1000 ms"). `lead_s` is 2.0 now; a simulated 1.4 s stall played through smoothly (heard at the robot), and the same stall with 0.3 s cut out. A long lead would make interrupting slow (~2 s: socat notices the reset only after the pipe drains, then gives aplay a second), so `interrupt()` also connects to `speaker.stop_port` (6001) on the robot, which runs `killall aplay`: silent within ~30 ms of the stop reaching it. **Installed on the robot:** `/root/watchdog_scripts/speaker_stop.sh`, registered in `/opt/rockrobo/watchdog/ProcessList.conf` next to `speaker.sh` (the previous list is kept as `ProcessList.conf.before-speaker-stop`). Xiaomi's `wlanmgr` is still running on the robot and is a suspect for the stalls. **Confirmed 2026-09-27:** its roaming scans every 30 s are the stalls (below).

Second round: `echo_timing.py` measured the voice lasting up to 1.02 s past its last byte (start lag 0.14 s), so `gate_tail_s` is 1.0 (1.3 s in total). No segment *started* after the audio was due, so the timing gate alone may have been enough for these lines; long multi-sentence replies are still to be checked. Greetings after a restart work. The stranger note fired 4 s before recognition caught up ("Who are you?" to Robin), so it now waits 10 s.

**Findings from driving the robot (2026-09-22), which replace several assumptions in 4.1:**

- **"Doesn't drive when asked" had two causes.** No tool (fixed in C4), and manual control that seemed dead: on this Roborock V1, arming spins the lidar up and moves are *ignored* for ~6 s until it's ready; with no moves for a few seconds it spins down again, and the next move waits another 6 s. Valetudo's joystick hides this because people hold it. `motion.py` waits out a 7 s warm-up and keeps the session alive with zero-vectors every 200 ms, disarming after 20 s idle.
- **The drive vector, measured:** velocity 0.3 → 12.6 cm/s (Valetudo divides by 2.5 and the V1 ignores ≥0.3 after that, hence the 0.29 cap); angle *a* → spins in place at ~*a* deg/s, clockwise for positive *a* (omega = −*a* rad/s); it coasts ~0.3 s after a stop.
- **Player's `position2d` only streams while manual control is armed**, and not at all otherwise (a subscription just stays silent; that isn't a protocol bug). Under manual control it is live 50 Hz odometry. `position3d` is interface 30, not 7.
- **Valetudo's map pose freezes while manual control is armed** and catches up ~1 s after disarming. Reading it mid-session gives the pose from before, which cost several confusing readings.
- **Map heading convention:** angle *a* points along (sin *a*, −cos *a*) in map cm (x right, y down as stored): θ = *a* − 90° from +x toward +y, growing with a counter-clockwise turn like odometry. In `calibration:` in config.
- **`go_to` works** and plans its own path, but overshot a 30 cm target to ~90 cm before settling 12 cm past it.
- **Valetudo drops idle HTTP keep-alive connections**, and the robot's WiFi stalls (see the audio finding) outlast a 5 s timeout now and then: arming and disarming retry (disarm for up to 20 s: left armed, the lidar keeps spinning), moves use a 1 s timeout and are simply resent, and a motion stops and waits up to 3 s for odometry rather than driving blind.

**Findings from the face calibration attempt (2026-09-23, around midnight): stop-and-look can't see a standing adult.**

- **Tilt: lower looks up** (snapshots at 60 and 120). `look_direction`'s description and the `[look:up]` glance had it backwards; fixed.
- **The tilt mount only moves between 58 (up) and 105 (down)**, measured by hand; face tracking drove the servo into it following a standing face: the stall browned out the board and reset it. Firmware now clamps tilt (58–105, measured by hand; `~/LilyGo-Cam-RobotFace` HEAD, **built, not yet flashed**), and petd clamps what it asks for too. Until it's flashed, don't leave tracking on with a face near the top of the frame.
- **The head sees up to ~45° of elevation at most** (tilt 58 plus half the frame). A standing adult's eyes (1.70 m) are 45° up from 1.5 m away and 56° from 1.0 m.
- **Faces aren't detected much beyond ~1.5 m**, even in full room light: stage one of the detector (`HumanFaceDetectMSR01(..., 0.2F)`) runs on the VGA frame scaled to 20%, 128×96, where a face 2 m away is ~8 px. Robin was only detected at 1.5 m when bending their knees.
- **So there's no distance at which a standing adult is both high enough in the frame and big enough to detect.** The face-distance calibration wasn't possible; `scripts/calibrate_face.py` (fixed-tilt captures, no tracking) is ready for when it is. Options, roughly in order of payoff for "come here": (1) **a person/feet detector on the PC**, run on `/snapshot`: legs are in view at tilt 90, the bearing comes from the box, and the distance from where the feet meet the floor (camera 0.20 m up, known tilt) needs no face height at all; face recognition stays for *who*, up close; (2) the detector's resize scale 0.2 → 0.3–0.4 for range, at a CPU cost to measure against the vision duty cycle; (3) tilting the camera up on its mount by 15–20°.

**Detection-only head (2026-09-23).** The face firmware (`~/LilyGo-Cam-RobotFace`, `5bc6dca`) captures raw RGB565 and feeds it to the detector directly, with no JPEG decode on the detection path; JPEG is encoded only for `/api/snapshot` (and `/stream`, removed 2026-09-27). Stage one now gets a 320×240 input instead of 128×96. Measured on the device:

- **Range 2.7 m** (was ~1.5 m) in good indoor light; close range fine.
- **QVGA × 1.0: 2.17 passes/s** with nobody in view (was 1.9 at VGA × 0.2 with a face): the JPEG decode saved paid for the 6.25× larger stage-one input.
- **VGA × 1.0 rebooted the board:** stage one wanted one 2.4 MB buffer, the malloc failed. Stage one is now capped at 320×240 and the scale is lowered to fit, so no framesize can do that again.
- **Default VGA × 0.5:** the same 320×240 stage-one input at **1.89 passes/s**, and `/api/snapshot` (the brain's `look()`) stays at 640×480, 0.38 s, correctly exposed. QVGA × 1.0 is the faster alternative with QVGA photos (`/api/camera?framesize=5`, `/api/face/detector?resize_scale=1.0`).

`resize_scale` is settable at runtime (`/api/face/detector?resize_scale=`), and a `framesize` change re-initializes the camera, keeping every other setting. petd switches the head's own recognition off; recognition runs on the PC (4.7, E6).

**Face tracking (2026-09-23).** Tilt overshot into a nod up close. `scripts/tracking_log.py` showed each frame matching the previous pass's pose: frames reach detection ~0.3 s after capture (the camera driver queues them), but the pose was sampled at copy time. The firmware now looks the pose up at the frame's capture timestamp, keeps an axis's target inside the deadband (aiming at the older capture pose there caused a two-pass ping-pong on pan), and gives tilt its own gain (30 at VGA, pan 55). Stable on both axes at 0.6 and 2 m. Reaction time is still ~1 s: frame age ~280 ms + detection ~370 ms + up to one pass. Fresher frames (one buffer, or skipping stale frames) measured no better overall; the numbers are in the firmware's `docs/camera-settings.md`.

**Findings from the map (2026-09-26):**

- **This Valetudo has no room segmentation, no no-go areas or virtual walls, and no persistent-map control** (`/api/v2/robot/capabilities`: no `MapSegmentationCapability`, `CombinedVirtualRestrictionsCapability` or `PersistentMapControlCapability`). Rooms and off-limits areas can't come from Valetudo; the kitchen end of the room would be our own zone.
- **A reboot keeps the map, pixel for pixel;** only `vendorMapId` goes up (3578 → 3579). The robot reboots itself every day at 03:54 on its clock, which runs on Asia/Shanghai time: **21:54 here**.
- **A new map can have a different frame.** The full-room map made from the dock tonight (3578, `tests/fixtures/valetudo_map_2026-09-26.json`) matches 2026-09-19's closely (dock within 2 cm; the best wall fit still wants ~2° and a shift, unresolved). A partial map earlier tonight (3576, 15 m²) was rotated **~74°** against it. Every map's path starts at (2560, 2560): the frame is wherever the robot was when the map began.
- **A `go_to` from the dock and back keeps the map and its frame** (`runtime/map-test/`, 23:41): 96% of the wall pixels identical, the dock within 3 cm, the map a little larger; only the `path` entity starts over. Asked for (2562, 2601), 50 cm out, it stopped 8 cm short, facing away from the dock; 12 s out, 37 s back. **`vendorMapId` is no new-map signal:** it went 3579 → 3580 in between with the map unchanged.
- **A goal on or inside an obstacle is never an error** (23:52, a hollow box in the map ~1.2 m from the dock; `runtime/map-test/wall_*`). On its edge (2670, 2570): stopped 25 cm short, in front of it, 13 s. Inside it (2690, 2570): drove *around* the box for 49 s and stopped on its far side, 39 cm from the goal, with the bumper pressed (RoboController's next GoCheck: "error for: Bumper"). Both ended `moving` → `idle` and `RE_Nav_GotoTargetComplete` with ErrorCode 0, the same as a real arrival. So petd can't learn from the status whether it got there, only from the final pose: `await_arrival` now calls it arrived only within `motion.arrive_cm` (20) of the goal, and otherwise says how far off it stopped. Not yet tried: a goal outside the room.
- **Named places are raw map coordinates, so they move with the frame.** `kitchen` (saved 2026-09-23) is on the floor, 16 cm from a wall, in the 09-19 map, but 3 cm from a wall and off the floor in tonight's. Not yet known what starts a new map, or what made the 74° one.

**The face board's network (2026-09-27, the family session):** the face went "offline" twice. Firmware `657e2eb` and the snapshot server since, all flashed.

- **A stuck link, not a crash.** The station stayed "connected" while every UDP send failed with ENOMEM (12) and it answered neither ping nor HTTP, for minutes, until reset (10:49, then 11:05). No disconnect event, so the reconnect logic never ran. The ENOMEM means the WiFi driver's 32 dynamic TX buffers were full (Arduino's default, not the sdkconfig's 8 static ones): about 1 s of audio the radio couldn't get out. Internal RAM wasn't short (~139 KB free). **Fix:** 5 s of nothing but failed sends while connected forces a reconnect, 30 s more restarts the board (`wifi.stuck_resets` in `/api/status`); modem sleep is off. Not yet seen in action.
- **The air got lossy at 10:31 and stayed so.** Audio packets lost at the PC: ~0% a minute before 10:30, then 5-90% most minutes, across petd restarts and flashes; the robot had 7-17 s stalls in the same windows. A scan from the robot (`iw dev wlan0 scan`): the router is on channel 1, 20 MHz, alone there; neighbours are on 10-11 at -71 to -83 dBm. So the cause is in the house (traffic on the same router, a microwave, Bluetooth): **open**, as is whether it's still happening.
- **Snapshots have their own server now (port 81, port 80 redirects).** An httpd serves one request at a time: a snapshot crawling over the lossy link held up `/api/status` and `/ws`, which is most of what petd logged as "face didn't answer". Status now answers in 20-40 ms during back-to-back snapshots. Measured first: snapshots don't strain the network stack (29 KB JPEG in PSRAM, lwIP prefers PSRAM, at most 4 segments in flight, internal RAM's low point unchanged). **The MJPEG `/stream` is gone:** nothing used it, and it held whichever server it ran on for as long as anyone watched.
- `internal_min_free` was ~20 KB after boot, and once 14.8 KB with no snapshot from petd: unexplained, worth watching. Each boot prints "SHA-256 comparison failed… Attempting to boot anyway" from the ROM; boots fine.
- `runtime/face-serial/`: the serial captures (`capture.py`, PC timestamps, survives resets) and `netwatch.sh` (pings the face, robot and router every 0.5 s, logs misses).

**The robot's WiFi: a scan every 30 s (2026-09-27).** A PC watcher and one on the robot (both pinging the face every 0.5 s for 30 min) found the robot deaf for ~1 s **every 30 s exactly**: 59 of its misses, 55 of them at the same moments the PC couldn't reach the robot, none to the router. `iw event` shows why: a 14-channel scan every 30.1 s, 1.45 s each (the 1.3-1.4 s stalls of 2026-09-22). `wpa_supplicant` has no `bgscan`; the scans are **`wlanmgr`'s** roaming (`wlanmgr_scan_router`: "Found candidate router", for a stronger AP with the same SSID; there is only one here).

- `wlanmgr` also **brings the WiFi up at boot** (it runs `wifi_start.sh -cn`, which starts `wpa_supplicant` or the `…_miap…` access-point fallback), re-runs it when the link is lost, and drives the WiFi LED. Once up, `wpa_supplicant` and `dhclient` run on their own (parent init). WatchDoge restarts it if it dies (`ProcessList.conf`, read at boot).
- So it isn't removed or shimmed: a mistake in the boot path would leave the robot with no WiFi and no fallback AP, recoverable only over its serial console. **Paused instead** (SIGSTOP: alive, so WatchDoge leaves it be): no scans for 70 s and for 3 min, the link stayed up, Valetudo answered, and it resumed normally both times.
- **`/root/wlanmgr_pause.sh stop | resume | status`** on the robot, **run by hand only**: nothing starts it, and neither the boot process nor `wlanmgr` is changed. While paused, nothing re-runs `wifi_start.sh` if the link is lost for good; a reboot (the nightly one at 21:54 included) always starts the robot stock.
- Not yet measured: how much the pause helps the speaker's audio and Valetudo over a longer calm spell.

**Bad spells on the whole WiFi (2026-09-27):** heavy loss from ~10:31 to ~11:15 and again from ~12:30, hitting the face, the robot and even pings to the router from the PC. The PC is a laptop on WiFi too (Intel AX201, `Ziggo-gast679` on 5 GHz channel 44, -57 dBm), so every PC measurement crosses two radio links. The router's 2.4 GHz is on channel 1, alone there (neighbours on 10-11, -71 to -83 dBm). Breakfast and lunch time: a microwave is a suspect for the 2.4 GHz side; the 5 GHz router misses don't fit it as neatly (a busy router may just answer pings late). **Open.**

**The eye and the head stuck in petd's pose (2026-09-27, `af2995a`):**

- **The eye stayed parked on "thinking".** SpeechEnded sets it; only a turn's end or converse's own drop handed it back to the device's animation, so a transcript the stt adapter dropped ([BLANK_AUDIO] at 11:28, and any `[Music]` or echo drop) left it frozen until the next turn. Now any `HeardDropped` hands it back (unless a turn is under way; the turn does so when it ends), and "thinking" with nothing after it times out after 15 s.
- **Nods, shakes and glances left head tracking off** (set manual, never back). They now restore tracking if it was on; a glance holds its pose 1.5 s first, in the background.
- **Every face reboot switched tracking off:** the firmware starts in manual, and petd's re-initialization (after a reboot, when the device has forgotten its audio destination) didn't set the head. It now re-applies the mode petd last set, tracking by default. Not yet seen live (no board reboot since).
- The firmware keeps starting in manual (Robin, 2026-09-27: maybe later, if it proves useful); petd sets tracking.

**Next up (in order):**

0. **E6, face recognition on the PC: the remaining live checks** (backlog; the family session of 2026-09-27 got through the first half).
   - **Two people in view at once** (Robin and Claudia side by side, ~1.5 m): each named, neither as the other.
   - **A swap on purpose:** one walks out while the other stays or walks in, within the 5 s presence debounce; the right name within ~10 s, and speech credited to the right person. Built after it went wrong live (below), tested only with fakes.
   - **The 10 s recheck in a real conversation:** does someone sitting turned away lose their name too often? `recognition.recheck_every_s`.
   - **Backlit** (someone in front of the window), then revisit `unknown_sim` / `accept_sim` / `margin` with today's numbers.
   - **Enroll Noah**, then all three in view. A child's face changes quickly: expect growth to do more work, and a re-enrollment now and then.
   - **Growth:** a look with a hand over the chin was kept (0.60, `grow_sim` 0.55); consider raising `grow_sim` to ~0.65. Short visits each grow 2, so Robin's 20 grown were mostly replaced by daylight looks within minutes.
   - The head's "nobody in view" while Robin faced the camera (2026-09-23) and Whisper's "GLaDOS" misses ("Gladys", "GlaDOS" got through today) still stand.

   **Done 2026-09-27 (family session, daylight, runs `20260927-095950` to `-104303`):**
   - **Robin in daylight**, from lamp-light fingerprints: named after 2 attempts at 0.67, then 0.55-0.86 on every attempt.
   - **A guest stays unknown:** Noah (a young child, not enrolled), 38 attempts close up, far (48 px) and in profile, **-0.07 to 0.16** against Robin; never named. This also cleared two fingerprints grown at 10:03 when he might have been in view: they matched Robin's own at 0.73-0.76.
   - **Claudia enrolled** (5 close, 3 a step back; 197-292 px), then named at 0.81-0.91; Robin at 0.72-0.80 with her enrolled. Every stored fingerprint with a crop checked by eye: all the right person.
   - **Found and fixed: a swap kept the old name.** Robin walked out and Claudia in during a 1-2 s detection gap, under the 5 s presence debounce: one face, one named track, so no new visit, and Claudia was "Robin" for four minutes ("Robin says" on all her speech). Now a drop in the face count, and every `recheck_every_s` (10) while a named face is in view, starts a fresh look (~2 snapshots); its names replace the old ones, and a face it can't name loses its name (`128f459`, `95c7ad5`).
   - **E6c, first half:** every attempt's aligned crop in `runtime/faces/attempts/` (the last 200, named by time, verdict, best match and score) and every stored fingerprint's in `runtime/faces/fingerprints/<id>.jpg` (`da69dba`). Fingerprints from before have no crop.
   - A power glitch rebooted the face board (`poweron`); petd reconnected in 9 s. A greeting took 9 s from recognition to speech: for G4.
1. **C1, map geometry: try it live** (built 2026-09-27; how it works: the map findings above, and C1 under "Cluster C: spatial core").
   - **Teach the kitchen again** (the old place was deleted: it was saved in a frame we have no map of): park the robot there, "remember this as the kitchen". Drive it elsewhere, then "go to the kitchen". The log should show a `map frame:` line (current map vs `runtime/map/reference.json`) and the arrival message: "arrived at the kitchen", or "stopped N cm from the kitchen".
   - Then, if wanted: **our own zones** (a kitchen *area*, for "you're in the kitchen"), and `march_back` in use once E4 has person goals to check.
   - **The map backup's restore is untested** (`runtime/map-backups/2026-09-27_0001/`): copy the five files back to `/mnt/data/rockrobo/` and reboot at once (it rewrites `last_map` after every go_to). Worth trying once, on purpose, while the backup is fresh.
   - Unexplained: what makes a new map come in rotated (not a reboot, not a go_to; the user has seen it before).
2. **The face half of C3, with Robin in the room:** stop-and-look gave up because a standing adult was never both in view and detectable (faces then stopped at ~1.5 m). Detection now reaches 2.7 m, and the PC's YuNet found a face the head missed: measure again whether a standing person can be seen and their distance estimated. Then **C5** (person on the map) and **E4** (`approach_person`, search, "come here", the arbiter): **M3**.
3. **First real latency numbers (G4):** a few questions in a live session, then `scripts/latency.py`. Before E4, which adds model round trips per request.
4. Later: **E6c's attempt window** (see E6c: rechecks fill it). **Microphone levels (backlog, 2026-09-27):** the firmware keeps the top 16 of the MSM261's 24 bits (`raw >> 16`) and applies `gain_` after that. By the datasheet as recalled (sensitivity -26 dBFS at 94 dB SPL, SNR ~57 dB: self-noise ~-83 dBFS, above 16-bit's ~-101), the dropped bits are mostly mic hiss, but speech sits low: ~-60 dBFS at 1 m, ~5-6 bits. Measure before changing anything: the UDP stream's speech and silence levels at 0.5/1/2/3 m and a minute of room quiet, silero's speech probabilities and whisper's misses at 2-3 m. Then, if quiet speech is the weak point: gain applied to the 24-bit value before truncating, a fixed gain putting speech at 2 m near -30 dBFS with a limiter, or per-utterance normalization on the PC. No AGC or compression on the device (pumps room noise into the VAD, changes the echo's level); no 24-bit transport (more traffic on a lossy WiFi for mic hiss). **E6c** (kept attempts, which also gives `show_memory.py` images), **E5** + `emotions.yaml` (**M4**), **F2** games, **G3** dashboard extras; G2's reboot part if wanted (`scripts/start.sh install`, README.md).
5. Loose ends: barge-in on the real mic (E2's hardware check); D5 against a real ollama; the face-clock offset (see the findings below).

**Running it:**

- `.venv/bin/python -m petd`, with the dashboard at http://127.0.0.1:8765.
- `--fake` runs without any hardware.
- `--echo` repeats back what it hears, to test the audio path and the echo gate.
- `scripts/smoke.py {vacuum,face,stt,say,echo}` exercises each adapter on its own.
- `.venv/bin/python -m pytest` runs the tests.
- Each run logs to `runtime/logs/<start time>/` (`latest` is the newest): `petd.log`, `run.yaml`, `events.jsonl`.
- `scripts/latency.py [run]`: where each turn's time went. `scripts/show_memory.py [conversation N | map]`: what the pet has stored.
- `scripts/tracking_log.py`: the head's tracking, pass by pass. `scripts/fetch_face_models.py`: the face recognition models (git-ignored).
- `runtime/face-serial/`: the face board's serial log (`capture.py <log>`) and the ping watcher (`netwatch.sh <log>`); `20260927-robot-facewatch.log` is the robot-side watcher's. `runtime/faces/`: kept face crops (E6c).
- **`runtime/` stays under ~300 MB** (2026-09-27, `89b12bf`): a run's `petd.log` 20 MB + 3 and `events.jsonl` 20 MB + 1; older runs at most 30 and 120 MB together (`log.max_old_runs_mb`), oldest first; face attempts the last 200, fingerprint crops only while their fingerprint exists; the two `face-serial` logs 10 MB + 1 each. `pet.db` isn't capped (184 KB after 5 days).
- On the robot: `/root/wlanmgr_pause.sh stop | resume | status` pauses `wlanmgr`'s 30 s roaming scans, by hand only (see the robot's WiFi findings).
- `runtime/e6a/`: E6a's face captures of Robin and Claudia (photos of people: never into git) and the probe that took them.
- `runtime/map/reference.json`: the reference map places are kept in (spatial/frame.py). `runtime/map-backups/<date>/`: the robot's own map files (`last_map`, `ChargerPos.data`, `StartPos.data`, `slam_info.cfg`, `appproxy.map` from `/mnt/data/rockrobo/`, over root SSH, with the robot's md5sums) plus Valetudo's JSON of the same map; floor plans of the home, so never into git. `runtime/map-test/`: the go_to tests of 2026-09-26.

---

## 1. What exists, and what I found while reading it

| Component | State | Notes relevant to the glue |
|---|---|---|
| `vacuum-api/valetudo_client.py` | Done, GETs verified, commands untested on hardware | `go_to(x,y)` uses map units (cm, but check `pixelSize`). `drive()` **blocks** and `omega` is a *steering angle in degrees*, not rad/s. `get_position()` fetches the whole map JSON every time (heavy, so cache it). |
| `face-api/face_client.py` | Done | All commands are GETs. `FaceEventStream` runs on a background thread with auto-reconnect. |
| `playerc-client/playerc_client.py` | Done, read-only, ~50 Hz | See F1 and F2. |
| `stt/udp-stream/whisper-udp-stream` | Built, CLI | Writes `[Speech detected]`, `[Speech ended]` and `[YOU] <text>` to stdout, flushed. Can be parsed as it is. Uses `ggml-base.en` (**English only**) with Silero VAD. |
| `piper-tts` | GLaDOS medium voice, 22050 Hz | `POST :5001/synthesize` returns WAV. The robot side is `socat TCP-LISTEN:6000 … aplay -f S16_LE -r 22050 -c 1 -t raw`, so strip the WAV header and send raw PCM. |
| `claude` CLI 2.1.278 | Installed | Supports `--input-format/--output-format stream-json`, `--include-partial-messages`, `--system-prompt[-file]`, `--mcp-config` plus `--strict-mcp-config`, `--tools`, `--setting-sources`, `--model` and `--session-id/--resume`. That is everything needed for a persistent brain. |
| `ollama` | Not installed, no GPU visible in WSL | Plan for it, don't build it yet. |

**Findings that shape the design:**

- **F1: the Player pose is wheel odometry, not the SLAM map frame.** `xiaomi_bridge` publishes `position2d@0` as `/odom`. Units are meters and radians, it has its own origin, and it drifts. Valetudo's pose is in the SLAM map frame (cm, degrees), with no timestamp and a low update rate. **The glue has to keep an `odom→map` transform**, estimated when the robot is stationary, where both poses describe the same instant.
- **F2: the Player timestamp counts from boot.** Estimate `offset = min(recv_utc − player_ts)` over a sliding window. The lower envelope filters out network delay. Refresh it continuously.
- **F3: face UTC timestamps have 1-second resolution.** `time_service.cpp` uses `"%Y-%m-%dT%H:%M:%SZ"`. That is useless for fusing pose at the robot's turn rates. **Fix in firmware** (add milliseconds). Until then, use the PC receipt time (`received_monotonic` already exists) and accept ~50–150 ms of WebSocket latency.
- **F4: face events are pushed only when the *set* of faces changes.** The cached box is from the moment the face *appeared*, not from now. After the robot drives 1 m toward someone, or the person steps closer or sits down, the cached box still shows the old size and position, so the distance estimate is stale. The same person staying in view never triggers a new event. **Fix:** an on-demand `GET /api/face/current` that returns the current boxes, servo pose and ms timestamp from the latest processed frame (A2). A periodic push would also work, but a GET is simpler and costs nothing when unused. Fallback without firmware: run a PC-side face detector on `get_snapshot()`.
- **F5: the pet will hear itself.** The speaker is on the vacuum and the mic is on the face, so they sit on the same body. The glue must **gate STT while TTS is playing** (plus a ~600 ms tail). Motor and brush noise while driving will also produce junk transcripts.
- **F6: face memory is limited to 7 slots** (`face_id_save_number = 7`). **Single-ID delete now exists** (A3, firmware `edc6f87`): `GET /api/face/delete?id=N` returns the number remaining, and `GET /api/face/list` returns the current set. "Forget X" is one call, not a clear-and-re-enroll. The DB still owns the slot→person mapping, and now has to *reconcile* it: **ESP-DL ids are not guaranteed to be 0..6 or contiguous after a delete**, so never assume an id range — read `/api/face/list` (`FaceApiClient.list_enrolled_faces()`) and drop DB rows whose slot is gone.
- **F7: WSL2 does not forward inbound UDP under NAT networking.** Port 5000 audio from the face only reaches WSL with `networkingMode=mirrored` in `%UserProfile%\.wslconfig` (or run the glue on a Linux box or Raspberry Pi). Outbound HTTP, WebSocket and TCP to the robot work either way.
- **F8: the pan servo range is 0–180° with 90 = center.** Check the direction sign, and that 90° really means "robot forward", during calibration. The camera runs at QVGA (320×240) and the horizontal field of view has to be calibrated.
- **F9: Player `CMD_VEL` would be a better drive channel than Valetudo manual control.** It has real rad/s and m/s, but it may fight Valetudo's own control. The MVP uses Valetudo only. Closed-loop turns use Player yaw as *feedback*.

---

## 2. Architecture

```
                         ┌──────────────────────────── PC (petd) ─────────────────────────────┐
 Face (ESP32-S3)         │                                                                     │
  mic ──UDP:5000──────────┼─► whisper-udp-stream ─stdout─► STTAdapter ──┐                        │
  /ws events ─────────────┼─► FaceAdapter (FaceEventStream) ────────────┤                        │
  /api/* ◄────────────────┼── FaceAdapter (commands, snapshot)          │     ┌──────────────┐   │
                          │                                             ├──►  │  EventBus    │   │
 Vacuum (Valetudo+Player) │                                             │     └──────┬───────┘   │
  :80 Valetudo ◄──────────┼── VacuumAdapter (status/map cache, goto…) ──┤            │           │
  :6665 Player ───────────┼─► PoseTracker (odom ring buffer, clock      │     ┌──────▼───────┐   │
                          │    offset, odom→map transform) ─────────────┘     │ Behavior     │   │
  :6000 socat→aplay ◄─────┼── Speaker (TTS queue, speaking gate) ◄─────────── │ Arbiter +    │   │
                          │      ▲ POST :5001/synthesize (piper)              │ State Machine│   │
                          │      │                                            └──┬────────┬──┘   │
                          │      │  sentences + [emote:…] tags                  │        │      │
                          │   ┌──┴──────────────────┐   tool calls   ┌───────────▼──┐  ┌──▼────┐ │
                          │   │ Brain (LLMBackend)  │◄──────────────►│ ToolRegistry │  │Memory │ │
                          │   │  claude-cli | ollama│  (MCP shim for │ (+ local HTTP│  │SQLite │ │
                          │   └─────────────────────┘   claude-cli)  │  API :8765)  │  │+ .md  │ │
                          │                                          └──────────────┘  └───────┘ │
                          └─────────────────────────────────────────────────────────────────────┘
```

**Processes:**

- `petd`, the Python daemon.
- `whisper-udp-stream`, a child process of `petd`.
- `piper.http_server :5001`, a child process or separate service.
- `claude -p …`, a child process of the Brain.
- `robot-mcp`, a stdio MCP shim spawned *by* claude. It forwards to `petd`'s local HTTP API on `127.0.0.1:8765`.
- The `socat` listener on the robot, which is already set up.

**Why a local HTTP API plus an MCP shim?** The Claude CLI spawns MCP servers as its own children, so they cannot share memory with `petd`. A 50-line shim that forwards tool calls to `petd` over HTTP keeps one source of truth. The same HTTP API also serves a debug dashboard and a manual `curl` interface.

---

## 3. Key design decisions

1. **Python and asyncio**, with the existing sync clients wrapped through `asyncio.to_thread`. The Player reader and `FaceEventStream` keep their own threads and push into the loop with `loop.call_soon_threadsafe`. No rewrite of working code.
2. **Two ways for the LLM to act:**
   - **Inline tags in speech** for expressive, fire-and-forget actions that cost no extra round-trip: `[emote:happy]`, `[look:left|right|up|down|you]`, `[nod]`, `[shake]`. The streaming parser strips them out before TTS and runs them in sync with the sentence they are in. These work with any model, including small Ollama models.
   - **Real tools** for anything that returns data or takes time: `look()` (image), `where_am_i()`, `go_to_room()`, `approach_person()`, `remember_face()`, and so on.
3. **The LLM never drives motors directly.** Tools start *behaviors*, such as `approach_person` or `search_for_person`. The behaviors run the control loops and report back asynchronously as events ("I arrived", "I lost sight of Robin").
4. **A local reflex layer overrides the LLM.** Keyword stop, low battery, error or stuck state, quiet hours, and speaking-gate all live here.
5. **Conversation episodes.** A Claude process lives for one episode, ending after about 10 minutes idle. At the end, the brain writes a short summary into the journal (SQLite), and the next episode's system prompt includes recent journal entries. This keeps context small and gives the pet long-term memory without a huge context window.
6. **Runtime model:** Haiku 4.5 by default, for conversational latency. Sonnet 5 can be selected in config for richer personality. Everything is configured in `config.yaml`.
7. **Isolate the CLI from your dev setup.** Run `claude` from a dedicated cwd (`runtime/brain/`) with `--setting-sources ""` (or the minimum that works), `--strict-mcp-config`, `--tools ""` (no Bash, Edit or Read), and `--allowedTools "mcp__robot__*"`. The pet must not inherit your CLAUDE.md, memories or file access.

---

## 4. Subsystem designs

### 4.1 Time and pose (`petd/spatial/pose.py`)

- **`PoseTracker`** runs a thread that reads Player `position2d`, stamps each sample with `recv_utc`, and keeps a ring buffer of about 60 s: `(t_utc_est, x, y, yaw, vx, vyaw)`.
  - `t_utc_est = player_ts + offset`, where `offset` is the lower envelope of `recv_utc − player_ts` over the last ~30 s.
  - `pose_odom_at(t_utc)` interpolates (linearly, with a wrapped yaw lerp).
  - `is_stationary(window=0.5s)` returns true when |v| and |vyaw| are close to 0.
- **`odom→map` transform** (SE2). Whenever the robot has been stationary for more than 1 s:
  - Fetch the Valetudo pose, which is valid because nothing is moving.
  - Pair it with the current odom pose and solve `T = map_pose ⊕ odom_pose⁻¹`, converting m→cm and handling the axis and angle conventions (see calibration).
  - Smooth it (EMA on x, y and θ), and store the timestamp of the last fix.
  - `pose_map_at(t) = T ∘ pose_odom_at(t)`. Accuracy degrades with distance driven since the last fix. Expose that as `pose_confidence`.
- **Valetudo cache:** refresh map JSON and status every ~2 s in the background (plus on demand). Nothing else calls `get_map()` directly, because it is heavy.
- **Conventions calibration** (a script, run once, with the results saved in `config.yaml`):
  - Drive forward about 30 cm with `drive()` and log the Valetudo and odom deltas. This gives the map axis orientation, the angle zero and direction, and the odom→map scale sanity check.
  - Rotate in place and log both yaws. This gives the heading sign.
  - Look at a face straight ahead at 1, 2 and 3 m and log pan, box and tilt. This gives `pan_sign`, `pan_forward_deg`, `hfov_deg` and the distance constant `K`.

### 4.2 Person localization (`petd/spatial/person.py`)

Pure functions, unit-tested with synthetic numbers.

```
bearing_robot = (pan_deg − 90) * pan_sign + (box_cx − 0.5) * hfov_deg * cam_sign     # pan 90 = ahead (confirmed)
elev          = (tilt_deg − tilt_level_deg) * tilt_sign + (0.5 − box_cy) * vfov_deg  # camera at cam_z = 0.20 m
d_size        = K / box_h                                   # K calibrated
d_tilt        = (face_z − 0.20) / tan(elev)                 # face_z: 1.55 standing / 1.15 sitting / per-person learned
distance_m    = fuse(d_size, d_tilt)                         # weighted; if they disagree, pick the posture that fits d_size
target_map    = robot_map_xy + (distance_m − standoff_m) * unit(heading_map + bearing_robot)   # standoff 1.0 m
```

- **Why tilt matters with a 0.20 m camera:** the camera looks steeply up at faces. A standing person 2 m away sits at about 36° elevation, so the angle is a strong distance cue, and it doesn't depend on face size. The unknown is face height (standing or sitting). Box size resolves that, and for **recognized people** the DB learns their typical `face_z` over time.
- **Tilt range limit:** at a 1.0 m standoff, a standing face is at about 54° elevation. C3 checks that the tilt servo can reach it. If it can't, increase the standoff.

- **`validate_target(map, xy)`:** the target must fall on a floor or segment pixel, at least ~20 cm from a wall pixel. If not, march back along the ray toward the robot until it does. If nothing is valid, return "can't reach", and the pet says so.
- **Stop-and-look (the MVP approach behavior):**
  1. Stop, set servo `track`, and wait for the face to be centered and the pan stable (≤3 s).
  2. Collect about 1 s of measurements (face events and `/api/status` pan) and take the median.
  3. Compute the target and call `go_to`.
  4. Poll status until the robot is idle.
  5. Re-acquire. Loop at most 3 times, or stop early when `distance < standoff + 0.3 m`.
  Step 2 uses `GET /api/face/current` (A2) for fresh boxes. It never uses the cached event box after the robot has moved.
  6. At the end, rotate to face the person (closed-loop turn) and emote happy.
  7. If the face is lost, switch to `SearchBehavior`.
- **Sightings log:** every recognized face event is stored as `(person_id, t, x_map, y_map, confidence)` using `pose_map_at(t)`. That powers "where did you last see Robin?" and later "go to where Robin usually sits".

### 4.3 Hearing (`petd/io/stt.py`)

- Spawn `whisper-udp-stream` with its arguments from config. Parse stdout lines into `SpeechStart`, `SpeechEnd` and `Heard(text, t_end)` events. (An optional `--json` output mode is added in the C++ tool later, step A2.)
- **Filters:**
  - Drop anything heard while `Speaker.speaking` is true or within `tail_ms` after it (F5).
  - Drop known Whisper hallucinations: `[BLANK_AUDIO]`, "Thank you.", "you", and one-word noise.
  - Drop anything heard while the robot is driving, unless it matches a reflex keyword.
- **Attention gate:** the pet reacts only when:
  - (a) its **name** is heard (fuzzy match, for example Levenshtein ≤2 on each token, to tolerate Whisper misspellings), or
  - (b) it is inside an **open conversation window** (about 20 s after its last utterance or last addressed utterance), or
  - (c) a recognized person is **looking at it** (face present and centered) and speaks.
  
  Everything else is ignored, or at most logged.
- **Reflex keywords**, checked *before* the LLM and always active: "stop", "halt", "freeze", "go home", "go to sleep", "be quiet". They map to local actions immediately, and the LLM is told afterward.
- **Instant feedback:** on `SpeechStart`, set the eye to "listening" (aperture wide). On `SpeechEnd`, set it to "thinking" (look up-left, pulse). This covers the 2–4 s of latency and makes the pet feel alive.

### 4.4 Speaking (`petd/io/speaker.py`)

- An `asyncio.Queue` of sentences. For each utterance, open **one** TCP connection to `robot:6000`. Then, for each sentence, call `POST :5001/synthesize`, strip the WAV header (with the `wave` module), and write the raw PCM.
- **`speaking` flag:** set when the first bytes are written. It clears at `start + total_samples/22050 + playback_latency` (configurable, about 300 ms), plus the gate tail.
- **Streaming:** the Brain emits text deltas, a sentence splitter (`.?!…` followed by a space or the end, with a minimum length) cuts them, and the tag parser pulls out `[emote:…]` and similar. Each clean sentence goes to the queue while the LLM is still generating, so the first audio plays after about one sentence of latency.
- `interrupt()` closes the socket and clears the queue. It is used by the "be quiet" and "stop" reflexes.

### 4.5 Brain (`petd/brain/`)

```python
class LLMBackend(Protocol):
    async def start_episode(self, system_prompt: str) -> None: ...
    async def send(self, user_turn: str | list[ContentBlock]) -> AsyncIterator[BrainEvent]: ...
        # BrainEvent = TextDelta | ToolStarted | ToolFinished | TurnDone | Error
    async def end_episode(self) -> str | None: ...   # returns summary if produced
```

- **`ClaudeCliBackend`:** runs `claude -p --input-format stream-json --output-format stream-json --verbose --include-partial-messages --model <cfg> --system-prompt-file <runtime/brain/system.md> --mcp-config <runtime/brain/mcp.json> --strict-mcp-config --tools "" --allowedTools "mcp__robot__*" --setting-sources ""` in `runtime/brain/`.
  - It writes user turns as stream-json lines to stdin and parses `stream_event` text deltas and `result` messages from stdout.
  - Tool execution happens inside claude, through MCP to the shim and then to `petd`.
  - If the process dies, it restarts with `--resume <session-id>`.
  - **The step's first task is a spike** to confirm the exact stream-json message shapes and flag combination on 2.1.278.
- **`OllamaBackend` (designed, stub only):** `POST /api/chat` with `tools=[…]` generated from the same registry, plus its own tool loop, which calls `ToolRegistry.call()` directly. Vision needs a VL model (for example qwen2.5-vl). Keep the tag-based actions, because small models handle them better than tools.
- **Prompt assembly (`prompt.py`):**
  - The system prompt is `persona.md` + `backstory.md` + `body.md` (what I am, what I can do, my limits) + `style.md` (spoken, short, tags reference) + a summary of the people I know + the last N journal entries + learned facts (SQLite, from `remember_fact` and `note_about_person`).
  - Each user turn gets a compact **senses header**, included only for fields that changed since the last turn to save tokens. For example:
    `[t=19:42 | room=Living room | battery 64% docked=no | sees: Robin (1.6 m, slightly left), unknown person | mood: curious]`
    `Robin: "hey buddy, what are you up to?"`
- **Autonomous turns:** behaviors can also send turns that are not triggered by speech, such as `[event] Robin just walked in (recognized)` or `[event] You finished exploring the kitchen; photo attached`. The LLM decides whether to say anything, and an empty reply is allowed.
- **MVP tool registry (`tools.py`)**, each tool carrying a JSON schema, a description and an `allowed_when` guard:

| Tool | Kind | Notes |
|---|---|---|
| `get_senses()` | query | Full current state: battery, status, room, pose confidence, visible faces with names, time. |
| `look()` | query | Snapshot JPEG returned as an image block, plus servo pose and room. |
| `look_direction(pan, tilt)` / `track_faces(on)` | action | Servo control, clamped. |
| `turn(degrees)` / `move(cm)` | action | Closed-loop on Player yaw and odometry, max 180° and 100 cm. |
| `go_to_place(name)` / `remember_place(name)` | action | Named places stored in the DB. `remember_place` saves the current pose ("this is the couch"). Stored in raw map coordinates for now, so only valid while the map's frame holds (C1). |
| `approach_person(name?)` | behavior | Stop-and-look (4.2). Result arrives later as an event. |
| `search_for_person(name?)` | behavior | See 4.6. |
| `go_home()` / `stop()` | action | Dock, and stop everything. |
| `remember_face(name)` | action | Enrollment flow (4.7). Refuses when slots are full, and explains. |
| `forget_person(name)` | action | Single-ID delete (4.7). Explicit request only, and confirms who by name. |
| `who_do_i_know()` / `recall_person(name)` | query | From the DB: notes, last seen, where. |
| `note_about_person(name, note)` / `remember_fact(text)` | memory | Append-only, length-limited. |
| `set_mood(mood)` | state | Persistent baseline mood (idle eye style). |

  Not exposed to the LLM, ever: `clear_enrolled_faces` (wipes everyone at once — `forget_person` is the one-person alternative), `set_wifi_credentials`, `start_cleaning`, camera or NTP config.

### 4.6 Behaviors (`petd/behavior/`)

A small **priority arbiter**: the highest-priority active behavior owns the motors, the servos and the eye. Lower behaviors are suspended, not killed.

| Prio | Behavior | Trigger | Does |
|---|---|---|---|
| 100 | `Reflex` | Stop keyword, error or stuck status, bump storm | Stop everything, interrupt speech, emote "startled/sorry". |
| 90 | `LowBattery` | < 20 % (config) and not docked | Announce once, `go_to_dock`, ignore other motion. |
| 80 | `Sleep` | Quiet hours, or "go to sleep" | Dock. Eye mostly closed with slow "breathing" aperture. Only the name or a known face wakes it (logs only in quiet hours). |
| 60 | `Approach` / `Search` | Tool call, or "come here" | 4.2, and the search below. |
| 50 | `Converse` | Attention gate open | Servo `track`, eye auto, route speech to the Brain, keep the window open. |
| 30 | `AttentionSeeking` | Social need high, and PIR motion or a face seen | Turn toward the motion (PIR carries the servo pan), wake eye, short LLM-generated greeting via an `[event]` turn. |
| 20 | `Explore` | Schedule window and social need low | Visit a random room centroid, `look()`, send an `[event]` turn with the photo ("anything interesting?"), store the caption in `observations`, go on. |
| 10 | `Idle` | Default | Eye `auto` (the firmware idles and scans), occasional small head movements, drives tick over. |

- **Search behavior** (for "I call it but it can't see me"):
  1. Say "Hm? Who's there?" and emote curious.
  2. Pan sweep: servo `manual` over 30 → 150° in steps of about 30°, holding about 0.8 s at each step while watching face events. Use the PIR motion pan as a first guess if motion was seen in the last 5 s.
  3. If nothing is found, turn the body 120° (closed loop) and sweep again, up to 3×.
  4. If a face is found: switch servo to `track` and hand over to the Brain with an `[event]`:
     - **known face** → "Robin is in front of you; you were just called by name"
     - **unknown face** → "unknown person"

     The LLM then asks "Did you call me?"
  5. Config option `only_known_people_can_summon: true` limits who can summon it: when set, it ignores unknown faces and says "I'll wait for someone I know".
  6. If nothing is found, give up with a sad emote and go back to idle.
- **Drives** (simple floats from 0 to 1, updated every tick, persisted):
  - `social` rises over time and resets on interaction.
  - `curiosity` rises with time spent in the same room.
  - `energy` follows the battery.

  They pick between `Idle`, `AttentionSeeking` and `Explore` inside the allowed schedule windows (config: `active_hours`, `explore_windows`, `quiet_hours`, `max_autonomous_moves_per_hour`).
- **Emotion presets** (`emotions.yaml`) map each name to a short keyframe sequence of eye `(x, y, aperture, ms)` plus optional servo gestures:
  - happy, curious, sleepy, surprised, sad, thinking, listening, annoyed, love.
  - `nod` is a tilt dip, and `shake` is a pan wiggle.
  - The eye is switched to `manual` while a sequence plays, then set back to `auto`.

### 4.7 Memory (`petd/memory/`)

**SQLite schema (`pet.db`):**

```sql
people(id INTEGER PK, name TEXT UNIQUE, face_slot INTEGER UNIQUE NULL,  -- 0..6 device id
       created_at, last_seen_at, last_seen_x, last_seen_y, notes TEXT,
       nickname TEXT NULL,            -- what the pet calls them ("test subject #2")
       familiarity INTEGER DEFAULT 0, -- 0 new, 1 acquaintance, 2 regular, 3 favourite test subject
       interactions INTEGER DEFAULT 0,
       face_z_m REAL NULL)            -- learned typical face height, for tilt distance
places(id PK, name TEXT UNIQUE, x, y, created_at)                             -- "the couch", taught by voice
sightings(id PK, person_id NULL, t_utc, x, y, pose_conf, face_conf, source)   -- source: face|motion
conversations(id PK, started_at, ended_at, summary TEXT)
utterances(id PK, conversation_id, t_utc, speaker TEXT, text TEXT)            -- 'pet' | name | 'unknown'
facts(id PK, t_utc, about TEXT NULL, text TEXT)                               -- remember_fact / note_about_person
observations(id PK, t_utc, room, x, y, caption TEXT, jpeg_path TEXT NULL)     -- phase 2 heavy use
kv(key PK, value)                                                             -- drives, mood, calibration cache
```

**Enrollment flow (`remember_face(name)`):**

1. Require exactly one face that is visible and centered.
2. Arm `enroll_face(True)`.
3. Wait up to about 5 s for a face event with `id ≥ 0` that is not already in the DB. That is the new slot.
4. Store the name→slot mapping and confirm out loud.

Step 3 compares against `/api/face/list` rather than assuming the next id: after a delete, ids are neither contiguous nor reused in order.

If the slots are full, tell the LLM "slots full" — and, since F6 was lifted, it can offer to forget someone instead. **"Forget X"** is `forget_person(name)`: look the slot up in the DB, `delete_enrolled_face(slot)`, delete the person row, confirm out loud. It is the one destructive tool the LLM gets, so it names who it is forgetting and acts only on an explicit request (never on its own initiative, never on a bare "forget that").

**Familiarity:** `familiarity` rises with the interaction count and the days seen (thresholds in config). The LLM can also bump it through `note_about_person`. It shapes behavior:

| Tier | Greeting | Summon | Prompt hint |
|---|---|---|---|
| unknown | Notices them and may ask who they are and offer to remember them | Asks "Did you call me?" before approaching | "a stranger" |
| 0–1 | Calls out their name when they appear (once per N hours) | Approaches after they confirm | Name plus notes |
| 2–3 | Name or nickname, references past conversations and running jokes, GLaDOS-style "fondness" | Approaches directly when called | Name, nickname, notes, last 3 facts, last-seen summary |

**Face recognition on the PC (E6).** Since 2026-09-23 the head only
detects: recognizing every face on every pass cost the tracking loop time and
bought nothing (tracking needs only the box), and the head's 112x112 recognizer
flickered even up close (similarity ~0.55). Recognition runs on the PC instead
(E6b): built, and verified live with one person, but not yet fully trusted:
two people, daylight and a guest are still to check (Next up, item 0). The design below borrows from Frigate's face recognition
(the matching and voting) and Immich (grouping unknown faces); see "Prior art".

*Stack.* `onnxruntime` + `numpy`: 122 MB installed, measured from the wheels
(onnxruntime 64, numpy 56, protobuf/flatbuffers/packaging 2), no build step.
Pillow, already a dependency, decodes, crops and warps. Two OpenCV Zoo models,
downloaded by a setup script into a git-ignored folder (`*.onnx` already is):
about 161 MB in all.

| model | job | size | licence |
|---|---|---|---|
| YuNet 2023mar | faces + 5 landmarks in the snapshot | 0.23 MB | MIT |
| SFace 2021dec **fp32** | aligned 112x112 face -> 128-d fingerprint | 38.7 MB | Apache 2.0 |

fp32, not the 9.9 MB int8 export: on the PC's i7-1185G7 it measured 2.4x faster
(12.6 vs 30.4 ms; onnxruntime's int8 path isn't the fast one here) and scored
0.02-0.03 higher throughout E6a. Rejected: OpenCV itself (216 MB with numpy; it
would save ~100 lines), insightface (a Cython build, and its model packs are
non-commercial), dlib/face_recognition (a C++ build), DeepFace (TensorFlow, a
274 MB download alone). Speed is no concern, measured: YuNet 6.5 ms, alignment
0.8 ms, SFace 12.6 ms per face, against ~0.42 s to fetch the snapshot, and only
on arrivals, not per frame.

*One attempt* (`petd/vision/faces.py`):

1. `GET /api/snapshot`: a VGA JPEG, with the pose it was taken at.
2. YuNet on that image: boxes and landmarks from the same frame. The head's own
   landmarks come from a different frame, and alignment from those would suffer.
3. Gates, skipping rather than guessing: face at least `min_face_px` (45) tall,
   which is ~2.5 m at VGA, where E6a still told Robin and Claudia apart;
   detection score; too dark. No blur gate: the aligned crop's sharpness ranged
   from 18 (Claudia at 2.5 m, recognized fine) to 661 (Robin at 0.6 m), so no
   single cut-off tells a blurred face from a small one, and a motion-blurred
   frame at 0.6 m still matched its own person (0.47) over the other (0.29).
   The vote absorbs it; Frigate's mild score penalty is the fallback.
4. Align: fit all 5 landmarks to the standard 112 px template (least-squares
   similarity transform, ~20 lines of numpy), warp with Pillow. All 5 points,
   not the eyes alone, which slip on small faces.
5. SFace: a 128-d fingerprint, normalized.
6. Compare with each known person's centre (below): cosine similarity. Below
   `unknown_sim`, or within `margin` of the second-best person: "unknown".

*Deciding who it is*, per presence episode, as Frigate does per tracked person:

- Thresholds, from E6a (evening, lamp light; daylight still to measure):
  `unknown_sim` 0.35, just above the highest other-person score (0.29);
  `accept_sim` 0.45 for the vote's weighted mean, under the own-person medians
  even at 2.5 m (~0.50); `margin` 0.15 to the second-best person (the smallest
  gap seen was 0.28).
- Attempts start when a face appears and nobody in view is known yet: one
  snapshot every ~1.5 s while a face is in view and passes the gates, up to 12,
  and up to 6 more after a name is given, to confirm it.
- The name is a weighted vote over the episode's attempts: each counts by its
  face area (capped) times how far its similarity is above `unknown_sim`, so a
  close, clear look outweighs several distant ones. A name only when at least
  `min_agree` (2) attempts agree, their weighted mean reaches `accept_sim`, and
  no other name has as many votes. Otherwise nobody, not a guess: greeting a
  guest by someone else's name is the failure that matters, and "unknown" is
  the safe side of it.
- Several faces: each gets its own vote, matched between snapshots by position
  (people in a living room mostly sit still); a face that can't be matched
  starts afresh.
- Identity stays sticky for the episode, as now; greetings, sightings and
  familiarity work as they do.

*Fingerprints.* `face_embeddings(id, person_id, t_utc, vector BLOB, face_px,
blur, brightness, source)`: 512 bytes each, as many per person as useful, and
no 7-slot limit. Each person's centre is a trimmed mean (15%), after dropping
fingerprints whose similarity to the person's mean is below 0.30 once there are
5 or more, so one bad or mislabeled sample doesn't drag it off (Frigate's
`build_class_mean`). `people.face_slot` and the head's flash slots are retired;
people re-enroll once.

*Enrollment and growth.*

- `remember_face(name)`, as now, needs exactly one face in view. It collects ~5
  crops close up (~0.6 m) and ~5 at a step back (~1.5 m), from different frames,
  asking the person to step back in between, and says "come a bit closer" when
  crops don't pass the gates. Two distances because a face's fingerprint drifts
  with its size: enrolled at 0.6 m alone, Robin at 2.5 m scored as low as 0.43
  against his own centre; with 1.5 m added, 0.51.
- Variety matters more than count (Frigate: 20-30 varied images; no more than 4-6
  near-identical ones). So the set grows by itself: a confident recognition
  (well above `accept_sim`) adds its fingerprint unless it's a near-duplicate of
  one already kept, up to ~30 per person. Daylight, lamp light and new angles
  come with ordinary use. Uncertain attempts are never added: a wrong face in a
  person's set is what spoils it.
- `forget_person` deletes the fingerprints with the person.

*Kept attempts.* The last ~200 aligned crops, as small JPEGs under
`runtime/faces/attempts/` named by time, verdict and score. `show_memory.py`
shows them, which also gives it its first images, and a misread can be deleted
or given the right name. They stay on the PC.

*Night and light.* The models are trained on colour images; the OV2640 has no
IR mode, so night here means noise, blur and a colour cast from the camera
raising its gain, and by day a backlit person is a silhouette. The gates keep
the worst out, the varied fingerprints cover the rest, and the head's exposure
settings (the firmware's `docs/camera-settings.md`) matter as much as the model.

*Later, not E6: unknown faces.* Keep the fingerprints of unknown faces; once 3
or more from different episodes are close to each other (Immich's rule), they're
someone the pet keeps seeing, and it can ask their name the next time.

*Measured first (E6a, 2026-09-23 evening).* In a venv outside petd's, with a
probe that follows OpenCV's own decoding and alignment for these models: 8
snapshots each of Robin and Claudia at 0.6, 1.5 and 2.5 m, lamp light, head
tracking on. The captures, the probe and the full numbers are in `runtime/e6a/`
(git-ignored: they're photos of people). Faces were ~195, ~80 and ~47 px tall.
Against each person's centre (trimmed mean, outlier filter):

| enrolled at | lowest own | highest other | smallest margin |
|---|---|---|---|
| 0.6 m, recognized at 1.5 / 2.5 m | 0.43 | 0.22 | 0.34 |
| 0.6 + 1.5 m, recognized at 2.5 m | 0.51 | 0.18 | 0.37 |
| the enrolled distances themselves | 0.47 (a motion-blurred frame) | 0.29 | 0.28 |

Capture against capture, the same person never scored below 0.38 and different
people never above 0.30 (OpenCV's reference threshold for SFace is 0.363). So
two people are told apart out to 2.5 m, the edge of detection. The thinnest
margin is the 0.29 other-person score against `unknown_sim` 0.35, which is what
the vote, the margin rule and "unsure means unknown" are for. Still to measure
before E6b is called done: daylight, and a person backlit by the window; and
only two faces are in it, so guests and look-alikes are untested.

*Acceptance.* Greets a known person by name after a restart, day and evening,
within a couple of seconds of them facing it, out to the measured range. A
guest is never greeted as someone known. Recognition costs nothing while
nobody new is in view.

*Prior art.* [Frigate](https://docs.frigate.video/configuration/face_recognition/):
voting over attempts weighted by face area, `unknown_score`/`recognition_threshold`/
`min_faces`, `min_area` (750 px²), blur penalty, trimmed-mean class centres,
saved attempts, 20-30 varied training images, no IR. [Immich](https://docs.immich.app/features/facial-recognition/):
clustering unknown faces into new people once 3 are alike. Small YuNet + SFace
projects on GitHub do enrollment with exactly one face and cosine matching with
an unknown threshold, and stop there.

**Markdown files (`memory/`):**

- `persona.md`, `backstory.md` and `body.md` are written by a human (step F1).
- What the pet learns and its journal live in SQLite (`petd/memory/db.py`), not in markdown.

### 4.9 Latency budget and profiling

The felt latency is capture → STT → petd → LLM → petd → TTS → speaker. Most of it
is already recorded: every event carries `t`, the PC wall clock at creation
(`events.py`), the bus keeps the last 300 (`bus.py`), and `GET /events?n=` serves
them as JSON. So the timeline exists; what is missing is the inside of the brain
and the two legs that cross a hardware boundary.

| Leg | Known today | How it is (or would be) measured |
|---|---|---|
| Mic capture on the face → whisper on the PC | **No.** `Heard.t_start`/`t_end` are PC *receive* times, not capture times | Needs C2's face-clock offset. The device clock was 6.7 s out once, so absolute device time is not usable until then |
| End of speech → transcript | **Yes: 1.9–2.2 s** (B smoke tests) | `Heard.t − SpeechEnded.t_utc`, straight from the event history |
| Transcript → turn starts | No | Needs a brain event (G4) |
| Think → first sentence emitted | Only as a bound: `Heard.t → SpeakingStarted.t` | G4 splits thinking from synthesis |
| Piper synthesis, per sentence | No | One timer in `speaker._synthesize_all` (G4) |
| First PCM byte → audible in the room | **No.** `playback_latency_s: 0.3` is a configured guess the echo gate depends on | Loopback: speak with the gate off and time when our own STT hears it. One number for speaker → air → mic → STT, using hardware we already have |
| Whole turn | **Yes: 7–10 s on Haiku 4.5** (D, on hardware) | `TurnDone.duration_s`, currently only a log line |

**The brain is the one stage the event history cannot see.** `petd/brain/` publishes
nothing to the bus: `TurnDone`, `ToolStarted` and `ToolFinished` (`brain/backend.py`)
are internal to the backend protocol, and turn duration and cost end up in a log
line (`brain/brain.py`). Putting them on the bus is what makes the rest measurable.

**Watch time-to-first-word, not turn duration.** The brain already streams
sentence by sentence through `SpeechStreamParser`, and the speaker synthesizes one
sentence ahead, so first audio does not wait for the full reply. Turn duration is
therefore a misleading thing to optimize; `Heard.t → SpeakingStarted.t` is what
someone standing in the room actually experiences.

**Expect the LLM to dominate.** ~2 s of STT and 7–10 s of turn against in-process
async plumbing measured in microseconds: profiling is here to confirm that before
anyone optimizes the wrong thing, and to catch regressions as tools land on the
critical path. E4's `approach_person` turn is transcript → `get_senses` → `look`
→ answer, which is three model round trips where M1 had one. That is where this
will quietly get worse, so **G4 is worth doing before E4**, not after.

### 4.8 Safety

- Motion tools are clamped (turn ≤180°, move ≤100 cm, speed ≤0.3) and rate-limited.
- Every motion behavior has a timeout.
- The robot never leaves the dock while charging below `min_battery_to_leave` (default 40 %).
- `stop` is always reachable in four ways: the voice reflex, `POST /stop` on the local API, a dashboard button, and Ctrl-C (which disables manual control on shutdown).
- The LLM tool allowlist is enforced both in the registry and through `--allowedTools`.
- **Privacy:** snapshots stay in memory unless `store_photos: true`. The DB lives on the PC only.

---

## 5. Repository layout (new code under `robot-pet/petd/`)

```
robot-pet/
  PLAN.md                     ← this file
  config.example.yaml         ← IPs, ports, model, schedule, calibration constants
  petd/
    __main__.py               ← `python -m petd` : starts everything, graceful shutdown
    config.py  bus.py  log.py
    io/        vacuum.py face.py stt.py speaker.py procs.py (child-process supervisor)
    spatial/   pose.py person.py mapgeo.py (map decode, rooms, free-space) motion.py (closed-loop turn/move)
    brain/     backend.py claude_cli.py ollama.py prompt.py tags.py tools.py
    behavior/  arbiter.py reflex.py converse.py approach.py search.py explore.py attention.py sleep.py drives.py emotions.py
    memory/    db.py people.py journal.py
    api/       server.py (FastAPI :8765: tools, status, stop, tiny dashboard)
    mcp_shim/  robot_mcp.py (stdio MCP → HTTP :8765)
  memory/     persona.md backstory.md body.md style.md emotions.yaml
  runtime/    brain/ (claude cwd, generated system.md + mcp.json), pet.db, logs/
  scripts/    calibrate_conventions.py calibrate_face_distance.py smoke_*.py
  tests/      (pytest; spatial math, tag parser, sentence splitter, gate logic, fake hardware)
```

**Dependencies:** `requests`, `websocket-client`, `pillow` (already used), plus `fastapi`, `uvicorn`, `pyyaml`, `mcp` (official Python SDK), `numpy` (optional), `rapidfuzz` (fuzzy name matching), `pytest` and `pytest-asyncio`. Keep the existing `vacuum-api/`, `face-api/` and `playerc-client/` folders as they are and import them through `sys.path` or small `pyproject` packages.

**Fake hardware:** `io/*` gets `Fake*` twins (for example a fake face event generator, a fake Valetudo pose, and an STT fed from a text file) so the behaviors and the brain can be developed and tested without the robot. This is essential for agent-driven implementation.

---

## 6. Implementation plan

Steps are **clustered by model and effort**, so each cluster can run as one session without switching. Within a cluster the steps run in order. Clusters A and B can run in parallel. Everything after them depends on B.

**Model legend:**

- **Opus 5 / high:** tricky math, concurrency and architecture.
- **Sonnet 5 / medium:** well-specified plumbing.
- **Haiku 4.5 / low:** docs, config and boilerplate.

Every step ends with its **acceptance check**.

### Cluster A: firmware and C++ tweaks (Sonnet 5, medium). Can run in parallel with B.

Needs flashing and a hardware check by you.

| # | Step | Acceptance |
|---|---|---|
| A1 | Face firmware: **millisecond UTC** in all events, snapshot headers and status (`%Y-%m-%dT%H:%M:%S.mmmZ` via `gettimeofday`). Add a **monotonic `seq`** to events. | Events show ms, and the face-api parser still works (it treats `utc` as a string). |
| A2 | Face firmware: **`GET /api/face/current`**, which returns the latest processed frame's faces (boxes, ids, confidence), servo pan and tilt, ms UTC and the frame age. Add `FaceApiClient.get_current_faces()`. | Stand still in view, then step forward. Two GETs show different box sizes with fresh timestamps. |
| A3 | **Done** (`edc6f87`). Face firmware: **delete a single enrolled id** (`/api/face/delete?id=N`) plus `/api/face/list`; `face_client.py` and its README have `delete_enrolled_face()` / `list_enrolled_faces()`, and `petd/io/face.py` exposes `delete_enrolled()` / `list_enrolled()`. | Enroll 2 faces, delete one, and the other is still recognized. **Not yet run on hardware** (destructive); do it when E3 lands. |
| A4 | `whisper-udp-stream`: add `--json` (one JSON line per event: `speech_start`, `speech_end`, `text` with `t_start_utc`, `t_end_utc`, `no_speech_prob`). The text output stays the default. | `--json` output parses. The old output is unchanged. |

A1 and A2 are strongly recommended before M3. The MVP still works without them, with degraded accuracy.

### Cluster B: foundations and I/O (Sonnet 5, medium)

| # | Step | Acceptance |
|---|---|---|
| B1 | Skeleton: package layout, `config.py` (YAML plus env overrides), logging, `EventBus` (typed dataclass events, async pub/sub), child-process supervisor with restart and backoff, `python -m petd` with clean shutdown. | `python -m petd --fake` starts and stops cleanly. The tests run. |
| B2 | `io/vacuum.py`: async wrapper with a cached status/map poller (2 s), `goto`, `dock`, `stop`, `manual drive` running in executor threads, and status-change events. Add `FakeVacuum`. | The smoke script prints status and pose. The fake passes the same interface tests. |
| B3 | `io/face.py`: async wrapper that starts `FaceEventStream`, bridges events onto the bus (`FaceSeen`, `FacesChanged`, `FaceLost` via 1.5 s debounce, `Motion`), runs init (detection and recognition on, audio destination set to the PC IP, gain), polls status at 2 Hz for pan and tilt, and handles `snapshot`. Add `FakeFace`. | The smoke script shows events live with a face in view. |
| B4 | `io/stt.py`: spawn and parse `whisper-udp-stream` (text mode now, JSON when A4 lands), the hallucination filter, and the speaking gate hook. Add `FakeSTT` (reads lines from stdin or a file). | Speaking to the face logs `Heard("…")`. **Check the WSL2 mirrored networking here (F7).** |
| B5 | `io/speaker.py`: piper synthesis, WAV→raw, a persistent per-utterance TCP stream to `robot:6000`, sentence queue, `speaking` flag and tail, `interrupt()`. Supervise piper as a child process. | `scripts/smoke_say.py "hello"` plays on the robot. While it plays, STT drops self-speech. |
| B6 | Local API (`api/server.py`, FastAPI): `GET /status`, `POST /say`, `POST /stop`, `POST /tool/{name}` (wired to the registry in D2), and a minimal HTML dashboard (status, last events, stop button, last snapshot). | `curl -X POST :8765/stop` works. The dashboard loads. |

### Cluster C: spatial core (Opus 5, high)

The hardest cluster. It is math-heavy and needs careful unit tests.

| # | Step | Acceptance |
|---|---|---|
| C1 | `spatial/mapgeo.py`: decode the Valetudo map once per update into a numpy grid (floor, wall); align a new map to a stored reference map (rotation + shift from the walls) and keep places and our own zones in the reference frame, converted before each `go_to`; `zone_at(x, y)` over those zones (Valetudo has no segments here). `is_free(x, y, clearance)` and `march_back(ray)` for `validate_target` (the firmware accepts any goal: see the map findings). | Unit tests on `tests/fixtures/valetudo_map.json` (2026-09-19) and `valetudo_map_2026-09-26.json` (the full room, same frame to within ~2°), and `valetudo_map_2026-09-26_partial.json` (3576, 15 m², ~74° off). |
| C2 | `spatial/pose.py`: `PoseTracker` (Player thread, ring buffer, clock offset lower envelope, `pose_odom_at(t)`, `is_stationary`), and the odom→map SE2 fix while stationary, with confidence decaying with distance travelled. | Synthetic tests: known offset and transform recovered within tolerance. The live smoke prints `pose_map_at(now)` against Valetudo while pushing the robot by hand. |
| C3 | `scripts/calibrate_conventions.py` and `scripts/calibrate_face_distance.py`: interactive, careful, low speed. They write `calibration:` into `config.yaml` (`map_axis`, `angle_zero`, `angle_sign`, `pan_sign`, `pan_forward_deg`, `hfov_deg`, `K_face`, `cam_height_m`). | You run them once, and the values look sane. **This needs you with the robot.** |
| C4 | `spatial/motion.py`: closed-loop `turn_by(deg)` and `move_by(cm)` using the Valetudo manual vector as the actuator and Player yaw/odom as feedback. Includes timeouts, a speed cap, and a stop on stall. Tune the relation between Valetudo `angle` and turn rate here, and document it. | ±10° turn accuracy on 90° and 180°, tested on the robot. |
| C5 | `spatial/person.py`: bearing and distance estimation, target projection, validation, and the sightings logger hook. | Synthetic tests plus a hardware test: stand 2 m away at about 30° and check the printed target on the map image. |

### Cluster D: brain (Opus 5, high for D1 and D2; the rest Sonnet 5, medium)

| # | Step | Model | Acceptance |
|---|---|---|---|
| D1 | **Spike, then build `ClaudeCliBackend`.** First confirm the stream-json input/output shapes, text deltas, tool events, isolation flags, and resume behavior on the installed CLI (save the transcripts to `docs/claude-cli-notes.md`). Then implement the persistent-process backend with episode lifecycle and crash-resume. | Opus 5 / high | A scripted two-turn conversation streams text deltas. The pet cannot use Bash or Read (a negative test). |
| D2 | `tools.py` registry (schema generation, guards, async execution, results as text or image blocks), `mcp_shim/robot_mcp.py` (stdio MCP server that lists tools from `GET :8765/tools` and forwards calls), and generation of `runtime/brain/mcp.json`. | Opus 5 / high | Claude calls `get_senses` and `look` through MCP. The image is visible to the model (it describes the snapshot). |
| D3 | `tags.py` streaming tag parser and sentence splitter. Wire Brain → Speaker plus emote dispatch in sync with sentences. | Sonnet 5 / medium | Unit tests with tags split across deltas. On hardware, the eye changes as the sentence starts. |
| D4 | `prompt.py`: system prompt assembly from `memory/*.md` + people + journal, the senses header diffing, and the `[event]` turn format. | Sonnet 5 / medium | A golden-file test of the assembled prompt. |
| D5 | `OllamaBackend` stub: `/api/chat` with tools from the registry and its own tool loop, behind the same interface. It can be enabled in config. It is not required to pass the hardware tests. | Sonnet 5 / medium | Works against a local ollama if you install one. Otherwise tested with a mocked HTTP server. |

### Cluster E: behaviors and memory (Opus 5, high)

Concurrency, state machines, and where the "feel" lives.

| # | Step | Acceptance |
|---|---|---|
| E1 | `memory/db.py`: schema, migrations, and the people, sightings, conversations, utterances and facts APIs. Add journal writing and compaction. | Unit tests. |
| E2 | `behavior/arbiter.py` + `reflex.py` + `emotions.py` + `converse.py`: priorities, resource ownership (motors, servos, eye, speech), attention gate (name fuzzy match, window, gaze), reflex keywords, listening and thinking eye feedback. **This gives M1.** | With the fakes, and then with hardware: say its name and a question, get a spoken answer with emotes. "Stop" works while it is speaking or driving. |
| E3 | Enrollment flow and the people tools (`remember_face`, `who_do_i_know`, `recall_person`, `note_about_person`), plus the greeting-on-recognition behavior (once per person per N hours, via an `[event]` turn). **This gives M2.** | Introduce yourself. It enrolls you, then greets you by name after a restart. |
| E4 | `approach.py` (stop-and-look, 4.2) and `search.py` (4.6), plus the `approach_person` and `search_for_person` tools, the "come here" local shortcut, and the `only_known_people_can_summon` option. **This gives M3.** | From 3 m, "Come here, <name>" brings it to about 0.7 m, facing you. Calling from out of view triggers the search and "Did you call me?". |
| E5 | `drives.py`, `sleep.py`, `attention.py`, `explore.py`, `lowbattery`, and the schedule config. **This gives M4.** | A simulated day with the fakes (a fast clock) produces a sane behavior timeline. A live session: it explores within its window, greets you when you walk by, sleeps in quiet hours, and docks on low battery. |
| E6a | Measure first (4.7), in a venv outside petd's: YuNet + SFace on head snapshots of Robin and Claudia at 0.6/1.5/2.5 m, day and lamp light. **Lamp light done 2026-09-23** (`runtime/e6a/`); daylight and backlight to go. | Same-person vs other-person similarities, face sizes, blur and timings; `unknown_sim` 0.35, `accept_sim` 0.45, `margin` 0.15, `min_face_px` 45 and SFace fp32 chosen from them. |
| E6b | **Built 2026-09-23, with fakes and E6a's real snapshots; verified live with Robin** (enrollment, recognition on return and after a restart), **and on 2026-09-27 with Claudia, in daylight, and a child guest** (unknown throughout); rechecks added after a swap kept the old name. Two at once, a deliberate swap and backlight still to check. Face recognition on the PC (4.7): `onnxruntime` + `numpy`, the model setup script, `vision/faces.py` (detect, gate, align, embed), the per-episode vote, the `face_embeddings` table with trimmed-mean centres, `remember_face`/`forget_person` on top of it, growth from confident recognitions. The head stays detection-only. Brings back M2's "greets you by name". | Greets a known person by name after a restart, day and evening, out to the measured range. A guest is never greeted as someone known. |
| E6c | Kept attempts (4.7): the last ~200 aligned crops under `runtime/faces/attempts/`, shown by `show_memory.py`, a misread deletable or renamable. **Done 2026-09-27, tried on the day's real crops (on copies):** crops kept (attempts, and `runtime/faces/fingerprints/<id>.jpg` per stored fingerprint, gone with it); `show_memory.py faces` lists every fingerprint and draws contact sheets (`runtime/faces/sheets/`); `scripts/fix_faces.py forget <id>…` / `assign <attempt> <name>` (source `assigned`, kept like `enroll`; refuses a crop unlike the person or a look they have, and points at the same look grown under someone else). A fingerprint recomputed from its saved crop scores 0.99 to the stored one, neighbouring looks ≤ 0.91. **Note:** with someone in view the 10 s recheck keeps ~2 near-identical crops every 10 s, so the 200 reach back only ~1 h then; keep a recheck's crop only when it names someone else, or scores low, if that proves short. | A wrong or unsure attempt can be found and corrected after a session. |

### Cluster F: personality and content (Opus 5, medium). Creative writing, best done in one sitting.

| # | Step | Acceptance |
|---|---|---|
| F1 | Write `persona.md`, `backstory.md`, `body.md` (an honest description of its body: vacuum base, head on the lidar, one eye, can't pick things up, bad at stairs, and so on), `style.md` (1–3 spoken sentences, no markdown or emoji, tag reference, when to stay silent), and `emotions.yaml` keyframes. | You read them and like them. |
| F2 | Simple games that need no new hardware, as prompt modules plus light tool use: *20 questions*, *riddles*, *guess what I see* (uses `look()`), *hide and seek-lite* (uses `search_for_person`). | Each can be played end to end by voice. |

### Cluster G: polish and ops (Haiku 4.5, low; Sonnet 5, medium for G3)

| # | Step | Acceptance |
|---|---|---|
| G1 | `README.md` for petd: setup, the WSL `.wslconfig` note, how to run, config reference, and troubleshooting. Update `todo.txt`. **Done 2026-09-27:** `README.md` (how it fits together, setup of the PC, robot and face, running, the tools, the config sections, troubleshooting from the findings so far); `todo.txt` kept as the original plan, with a status note on top. | Someone else could set it up. |
| G2 | `scripts/start.sh` or a systemd user unit, log rotation, and `runtime/` layout creation. **Done 2026-09-27:** `scripts/start.sh` (`run`, `start`/`stop`/`restart`/`status` as the systemd user unit `petd`, transient unless installed; `log`; `check`; `install`/`uninstall`), `scripts/preflight.py` (what petd needs, from the config: errors stop the start, unreachable devices warn); log rotation and size limits since `89b12bf`. Tried: the checks on the real setup (15 ok), a background start and clean stop with `--fake`. **Not yet:** `install` (a user decision), linger, and the Windows side (README.md, Starting with WSL). | One command starts the pet. A reboot restores it. |
| G3 | Dashboard extras (Sonnet): live event log, map with robot, people and target overlay, drives, and a manual tool console. | Useful for debugging M3 and M4. |
| G4 | **Done 2026-09-23, with fakes; real numbers from the next live run.** The trace is always on, as `events.jsonl` in each run's log folder (`log.events`), rather than behind a `--trace` flag: the run folders came after this step was written. Latency instrumentation (4.9): publish the brain's turn and tool timings on the bus, time piper per sentence, a `--trace` flag appending every event to JSONL (the 300-event ring is for the last turn, not for a session), and `scripts/latency.py` to print a per-turn breakdown from `/events` or a trace. **Do this before E4.** | A spoken turn prints hear → think → synth → first word → done, with the numbers adding up to the wall clock. |

### Suggested run order

```
A (you + Sonnet, parallel) ──┐
B1–B6 (Sonnet) ──────────────┼─► D1–D2 (Opus) ─► F1 (Opus) ─► D3–D5 (Sonnet) ─► E1–E2 (Opus)  = M1
                             │                                                  └► E3 (Opus)   = M2
                             └─► C1–C5 (Opus, needs you for C3/C4) ─────────────► E4 (Opus)   = M3
                                                                                  E5, F2 (Opus)= M4
                                                                                  G1–G3 (Haiku/Sonnet)
```

Grouped by model to minimize switching:

1. **Sonnet 5 / medium:** A1–A4, B1–B6.
2. **Opus 5 / high:** D1, D2, C1–C5.
3. **Opus 5 / medium:** F1.
4. **Sonnet 5 / medium:** D3–D5.
5. **Opus 5 / high:** E1–E5.
6. **Opus 5 / medium:** F2.
7. **Haiku 4.5 / low:** G1, G2.
8. **Sonnet 5 / medium:** G3.

---

## 7. Milestones (MVP = M1 to M3; M4 is "nice MVP")

| Milestone | Demo |
|---|---|
| **M1: It talks** | Say its name and a question. It turns toward you (tracking), its eye listens and then thinks, it answers in the GLaDOS voice with matching eye emotes, and it stops when told. |
| **M2: It knows you** | "My name is Robin, remember me." It enrolls you. Tomorrow it greets you by name and remembers what you told it. |
| **M3: It comes to you** | "Come here!" It drives to about 0.7 m in front of you. Called from another angle, it searches, finds you, and asks "Did you call me?" |
| **M4: It feels alive** | It explores on a schedule, seeks attention when you walk by, sleeps at night on the dock, and docks on low battery. |

## 8. Out of scope for the MVP (phase 2 ideas)

- Photo memory with object search ("where did you see my keys?"). The table exists, but it is not heavily used yet.
- Continuous approach while driving (a visual-servoing loop over Player `CMD_VEL`), and person following.
- Sound direction. With a single mic there is no direction of arrival. The PIR plus a sweep is the substitute.
- Barge-in (talking over the pet). Needs echo cancellation, or at least a louder-than-self threshold.
- Multilingual STT (`ggml-base` instead of `.en`, or `small`).
- An Anthropic API backend that uses the SDK directly (lowest latency and cost). It can be added behind the same `LLMBackend` interface.

---

## 9. Open questions

Questions 1–9 were answered on 2026-09-19. See 0.1 for the decisions. The original list is kept below for reference.

**Still open:**

- The pet's **name**, which is also its wake word. "GLaDOS" transcribes reliably with Whisper and is the obvious default.
- Is the tilt servo's 90° level with the floor? What is the maximum upward tilt? (C3 measures this, but it decides the standoff.)

**Original questions:**

1. **Name and personality.** What is the pet called? That is its wake word, so pick something Whisper transcribes reliably (2 syllables, uncommon). With a GLaDOS voice, do you want a GLaDOS-ish sardonic personality, or a sweet and curious pet with a funny voice?
2. **Where does `petd` run?** On this WSL2 machine permanently (then mirrored networking is required for UDP, F7), or on a Pi or mini-PC near the robot? Does the face's audio already reach `whisper-udp-stream` in WSL today?
3. **Firmware changes A1–A3.** Are you OK changing and flashing the face firmware? A1 and A2 matter most for M3 accuracy.
4. **Claude via the CLI on your subscription.** Is that acceptable for always-on use (usage limits, and about 2–5 s of latency per turn)? Haiku 4.5 by default, or Sonnet 5?
5. **Strangers.** Should it talk to, approach and summon-respond to unknown faces, or only to enrolled people? Should it store photos (`store_photos`)?
6. **Language.** English only (current `base.en` model)?
7. **Home rules.** Rooms and no-go zones can't come from Valetudo on this robot (2026-09-26). Are there areas it must never enter? (Carpets aren't off-limits now that it doesn't vacuum.) What are the quiet hours, and which windows allow autonomous movement?
8. **Mechanical.** Does pan 90° point exactly at the robot's front, and is the camera mounted upright? What is the rough camera height? (C3 measures these, but knowing helps.)
9. **Pets and kids.** Are there real pets or small children around? That affects the default speed caps and autonomous exploring.
