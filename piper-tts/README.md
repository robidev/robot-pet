# piper-tts

GLaDOS text-to-speech, served over HTTP on port 5001 (petd starts it when
nothing is listening there; `run.sh` starts it by hand).

## Setup

```sh
cd piper-tts
python3 -m venv .venv
.venv/bin/pip install piper-tts flask   # piper 1.8.0 was used
```

Put the voice model next to its config: `glados_piper_medium.onnx` (about 63 MB,
git-ignored). The matching `glados_piper_medium.onnx.json` is in the repo.
It is 22050 Hz mono, which matches the robot's `aplay -r 22050`.

## Playback on the robot

The robot plays raw PCM that it receives on TCP port 6000:

```sh
socat -u TCP-LISTEN:6000,reuseaddr,fork EXEC:'aplay -f S16_LE -r 22050 -c 1 -t raw -'
```
