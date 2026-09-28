# stt: whisper-udp-stream

Speech-to-text for the face's UDP microphone stream (16 kHz, 16-bit mono PCM,
`LGA1` packets on port 5000), using whisper.cpp with Silero VAD.
The source is `udp-stream/udp-stream.cpp`. It is built as an example inside a
whisper.cpp checkout, which is **not** in this repository (`stt/whisper.cpp/`
is git-ignored).

## Setup

```sh
cd stt
git clone https://github.com/ggml-org/whisper.cpp.git
git -C whisper.cpp checkout fd7d8abb          # v1.9.4-81, the version this was built against

# hook udp-stream into the whisper.cpp examples build
cp -r udp-stream whisper.cpp/examples/udp-stream
echo 'add_subdirectory(udp-stream)' >> whisper.cpp/examples/CMakeLists.txt

cmake -S whisper.cpp -B whisper.cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build whisper.cpp/build --target whisper-udp-stream -j
cp whisper.cpp/build/bin/whisper-udp-stream udp-stream/

# models (git-ignored)
mkdir -p udp-stream/models
sh whisper.cpp/models/download-ggml-model.sh base.en udp-stream/models
sh whisper.cpp/models/download-vad-model.sh silero-v6.2.0 udp-stream/models
```

The binary is linked against the shared libraries in `whisper.cpp/build`, so
keep that build directory around.

After editing `udp-stream/udp-stream.cpp`, copy it to
`whisper.cpp/examples/udp-stream/` and rebuild the target.

## Run

petd starts and supervises it in `--json` mode (see `petd/io/stt.py`). To run it by hand:

```sh
cd stt/udp-stream
./whisper-udp-stream --port 5000 --threads 4 \
  --model models/ggml-base.en.bin --vad-model models/ggml-silero-v6.2.0.bin [--json]
```

`--json` writes one object per line to stdout: `ready`, `speech_start`,
`speech_end`, `text` (with `t_start_utc`, `t_end_utc`, `no_speech_prob`) and
`stopped`. Times are Unix epoch seconds.
