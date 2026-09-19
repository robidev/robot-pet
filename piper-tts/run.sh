#!/bin/sh
python3 -m piper.http_server --port 5001 -m glados_piper_medium.onnx

# socat -u TCP-LISTEN:6000,reuseaddr,fork EXEC:'aplay -f S16_LE -r 22050 -c 1 -t raw -'
# echo "Would you like a cake?" | .venv/bin/piper -m glados_piper_medium.onnx --output-raw | socat -u - TCP:192.168.101.43:6000
