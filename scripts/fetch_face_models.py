"""
Downloads the face recognition models (PLAN.md 4.7, E6) into models/face/:

    .venv/bin/python scripts/fetch_face_models.py

YuNet finds faces and their 5 landmarks (0.23 MB, MIT); SFace turns an aligned
face into a 128-number fingerprint (38.7 MB, Apache 2.0). Both are from the
OpenCV Zoo. fp32 rather than SFace's int8 export: on this PC it measured 2.4x
faster (12.6 vs 30.4 ms) and more accurate (E6a). The files are git-ignored
(*.onnx); petd runs without them, with face recognition off.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "models" / "face"
BASE = "https://huggingface.co/opencv"
MODELS = {
    "face_detection_yunet_2023mar.onnx": (f"{BASE}/face_detection_yunet/resolve/main", 232_589),
    "face_recognition_sface_2021dec.onnx": (f"{BASE}/face_recognition_sface/resolve/main", 38_696_353),
}


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    for name, (url, size) in MODELS.items():
        path = TARGET / name
        if path.exists() and path.stat().st_size == size:
            print(f"{name}: already there")
            continue
        print(f"{name}: downloading {size / 1e6:.1f} MB ...")
        partial = path.with_suffix(".part")
        urllib.request.urlretrieve(f"{url}/{name}", partial)
        if partial.stat().st_size != size:
            partial.unlink()
            sys.exit(f"{name}: expected {size} bytes, got something else; not kept")
        partial.rename(path)
    print(f"models in {TARGET.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
