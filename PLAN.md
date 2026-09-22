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
| Rooms | One Valetudo segment (living room). Carpets become no-go areas in Valetudo | `go_to_room` is low value. Add **named places** ("the couch", "the door") taught by voice (4.5). Rely on Valetudo `go_to` to respect no-go areas. |
| Geometry | Pan 90° = straight ahead. **Camera height 0.20 m** | Tilt elevation becomes a primary distance cue (4.2). Standoff raised to 1.0 m so standing faces stay within tilt range. |
| Household | Adults only, no animals | Default speed caps can stay as designed. No extra child safety needed. |

---

## 0.2 Progress

| Cluster | Status (2026-09-22) |
|---|---|
| **A: firmware and C++** | **Done.** A1–A3 are committed in `~/LilyGo-Cam-RobotFace` (`edc6f87`, then `e6102eb` and `b3066da`), flashed, and verified read-only on the device at 192.168.101.40. `delete` hasn't been exercised on hardware because it's destructive. A4 (`--json`) is built into `stt/udp-stream/whisper-udp-stream` and tested with `jfk.wav` over UDP. It also fixes a bug that was already there: SIGINT/SIGTERM were ignored while no audio arrived. |
| **D: brain** | **Done and verified on hardware** (`886de32`). Persistent `claude -p` per episode (stream-json, isolated: own cwd, no inherited settings, `--tools ""`, only `mcp__robot__*`), the tool registry behind the local API, the `robot_mcp` stdio shim, the streaming tag/sentence parser, expressions, prompt assembly, and an untested ollama backend. On the robot it called `look()`, saw the room through MCP and described it correctly. Turn latency 7–10 s on Haiku 4.5, plus ~2 s for STT. D5 (ollama) needs a real ollama to verify. |
| **F1: persona** | **Done** (`a816850`). `memory/{persona,backstory,body,style}.md`. Checked in conversation: in character, refuses what the body can't do, drops the act when someone is upset. `emotions.yaml` keyframes still belong to E. |
| **E1–E3: memory, listening, people** | **Done with fakes and the real model, not yet on hardware** (`5b74e4c`, `44356fd`). E1: `memory/db.py` (SQLite, migrations; people, sightings, conversations, utterances, facts, places, kv). E3: `memory/people.py` (sticky identity, greetings, a lingering stranger, enrollment that diffs `/api/face/list`, forget, slot reconciliation) and the people tools; idle episodes end with a journal line. E2: `behavior/converse.py` (reflexes, barge-in, attention gate). Deferred: the priority **arbiter** moves to E4, where the first competing behaviors (approach, search) arrive and there is something to arbitrate; **`emotions.yaml` keyframes** move to E5 (the single-pose emotes work). |
| **B: foundations and I/O** | **Done and verified on hardware**, apart from the `/tool/{name}` route, which moves to D2. All five smoke tests pass: vacuum polling, face events and presence, STT, TTS out of the robot's speaker, and `echo` — where the pet heard itself zero times (the gate caught its own utterances) and answered every real one. |

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

**Next up (in order):**

1. **Hardware check of E2 and E3 (you, ~15 min):** the gate and barge-in on the real mic, enrolling a second face, recognition after a restart, and A3's single delete (`forget_person`).
2. **Cluster C** (Opus 5 / high), which **needs you at the robot** for C3 and C4: map/heading conventions and the face-distance calibration, then `move`/`turn`/`go_to_place` (fixes "doesn't drive when asked") and `approach_person` (M3).
3. **G4, the latency instrumentation (4.9), before E4** puts extra model round trips on the critical path. Then E4 (with the arbiter) and E5 (with `emotions.yaml`).
4. Loose ends: D5 against a real ollama; a `--prompt` option for udp-stream to bias whisper toward "GLaDOS"; face-clock offset estimation (see the findings below).

**Running it:**

- `.venv/bin/python -m petd`, with the dashboard at http://127.0.0.1:8765.
- `--fake` runs without any hardware.
- `--echo` repeats back what it hears, to test the audio path and the echo gate.
- `scripts/smoke.py {vacuum,face,stt,say,echo}` exercises each adapter on its own.
- `.venv/bin/python -m pytest` runs the tests.

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
5. **Conversation episodes.** A Claude process lives for one episode, ending after about 10 minutes idle. At the end, the brain writes a short summary into the journal (SQLite plus `memory/journal.md`), and the next episode's system prompt includes recent journal entries. This keeps context small and gives the pet long-term memory without a huge context window.
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
  - The system prompt is `persona.md` + `backstory.md` + `body.md` (what I am, what I can do, my limits) + `style.md` (spoken, short, tags reference) + a summary of the people I know + the last N journal entries + `learned.md`.
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
| `go_to_place(name)` / `remember_place(name)` | action | Named places stored in the DB. `remember_place` saves the current pose ("this is the couch"). Valetudo segment centroids are fallback places. |
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

**Markdown files (`memory/`):**

- `persona.md`, `backstory.md` and `body.md` are written by a human (step F1).
- `learned.md` and `journal.md` are appended by the pet through tools, with a size cap. When they get too large, they are compacted by an LLM call at episode end.

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
  memory/     persona.md backstory.md body.md style.md learned.md journal.md emotions.yaml
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
| C1 | `spatial/mapgeo.py`: decode the Valetudo map once per update into a numpy grid (floor, wall, segment ids, names), `is_free(x, y, clearance)`, `room_at(x, y)`, `room_centroid(name)` (nearest free pixel to the centroid), and `march_back(ray)`. | Unit tests on `vacuum-api/map_snapshot`'s JSON (save a fixture from the live robot). |
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

### Cluster F: personality and content (Opus 5, medium). Creative writing, best done in one sitting.

| # | Step | Acceptance |
|---|---|---|
| F1 | Write `persona.md`, `backstory.md`, `body.md` (an honest description of its body: vacuum base, head on the lidar, one eye, can't pick things up, bad at stairs, and so on), `style.md` (1–3 spoken sentences, no markdown or emoji, tag reference, when to stay silent), and `emotions.yaml` keyframes. | You read them and like them. |
| F2 | Simple games that need no new hardware, as prompt modules plus light tool use: *20 questions*, *riddles*, *guess what I see* (uses `look()`), *hide and seek-lite* (uses `search_for_person`). | Each can be played end to end by voice. |

### Cluster G: polish and ops (Haiku 4.5, low; Sonnet 5, medium for G3)

| # | Step | Acceptance |
|---|---|---|
| G1 | `README.md` for petd: setup, the WSL `.wslconfig` note, how to run, config reference, and troubleshooting. Update `todo.txt`. | Someone else could set it up. |
| G2 | `scripts/start.sh` or a systemd user unit, log rotation, and `runtime/` layout creation. | One command starts the pet. A reboot restores it. |
| G3 | Dashboard extras (Sonnet): live event log, map with robot, people and target overlay, drives, and a manual tool console. | Useful for debugging M3 and M4. |
| G4 | Latency instrumentation (4.9): publish the brain's turn and tool timings on the bus, time piper per sentence, a `--trace` flag appending every event to JSONL (the 300-event ring is for the last turn, not for a session), and `scripts/latency.py` to print a per-turn breakdown from `/events` or a trace. **Do this before E4.** | A spoken turn prints hear → think → synth → first word → done, with the numbers adding up to the wall clock. |

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
7. **Home rules.** Are rooms named in Valetudo (used for `go_to_room`)? Are there rooms or areas it must never enter (Valetudo no-go zones are respected by `go_to`)? What are the quiet hours, and which windows allow autonomous movement?
8. **Mechanical.** Does pan 90° point exactly at the robot's front, and is the camera mounted upright? What is the rough camera height? (C3 measures these, but knowing helps.)
9. **Pets and kids.** Are there real pets or small children around? That affects the default speed caps and autonomous exploring.
