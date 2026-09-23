"""
Faces in a snapshot, on the PC (PLAN.md 4.7, E6): where they are, and a
fingerprint for each.

    jpeg -> YuNet (faces + 5 landmarks) -> align each face onto the 112x112
    template -> SFace (128 numbers, normalized)

The head only detects: its own recognizer flickered even up close, and it
took time from tracking. Here, with the models from scripts/fetch_face_models.py,
a VGA snapshot takes ~20 ms (YuNet 6.5, alignment 0.8, SFace 12.6 per face,
measured in E6a), against ~0.4 s to fetch it.

Decoding and alignment follow OpenCV's FaceDetectorYN and FaceRecognizerSF,
which these models were published with, so no OpenCV is needed: numpy for the
maths, Pillow for the images, onnxruntime for the models.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

# ArcFace's 112x112 template (eyes, nose tip, mouth corners), as in OpenCV's
# FaceRecognizerSF::alignCrop.
TEMPLATE = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                     [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float64)
DETECTOR_SIZE = 640                 # YuNet's fixed input


@dataclass
class FaceSample:
    """One face in one snapshot."""
    box: tuple                      # x, y, w, h in snapshot pixels
    score: float                    # YuNet's detection score
    embedding: np.ndarray           # SFace, 128 float32, unit length
    brightness: float               # mean of the aligned face, 0-255
    sharpness: float                # Laplacian variance of the aligned face
    image_size: tuple               # snapshot width, height
    crop: Optional[Image.Image] = field(default=None, repr=False)

    @property
    def height(self) -> float:
        return self.box[3]

    @property
    def area(self) -> float:
        return self.box[2] * self.box[3]

    @property
    def centre(self) -> tuple[float, float]:
        """Normalized 0..1 in the snapshot."""
        x, y, w, h = self.box
        return (x + w / 2) / self.image_size[0], (y + h / 2) / self.image_size[1]


class FaceEngine:
    """The two models, loaded once. analyse() is CPU work: run it in a thread."""

    def __init__(self, models_dir: Path, detector: str, recognizer: str):
        import onnxruntime as ort
        paths = [Path(models_dir) / detector, Path(models_dir) / recognizer]
        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"face models missing ({', '.join(missing)}): "
                                    "run scripts/fetch_face_models.py")
        options = ort.SessionOptions()
        options.log_severity_level = 3    # SFace's export warns about every initializer
        self._detector = ort.InferenceSession(str(paths[0]), options, providers=["CPUExecutionProvider"])
        self._recognizer = ort.InferenceSession(str(paths[1]), options, providers=["CPUExecutionProvider"])
        self._outputs = [o.name for o in self._detector.get_outputs()]

    def analyse(self, jpeg: bytes, min_score: float = 0.6) -> list[FaceSample]:
        image = Image.open(io.BytesIO(jpeg)).convert("RGB")
        faces = []
        for score, box, landmarks in self.detect(image, min_score):
            crop = align(image, landmarks)
            gray = np.asarray(crop.convert("L"), np.float64)
            faces.append(FaceSample(
                box=box, score=score, embedding=self.embed(crop), brightness=float(gray.mean()),
                sharpness=laplacian_variance(gray), image_size=image.size, crop=crop))
        return faces

    def detect(self, image: Image.Image, min_score: float = 0.6, nms: float = 0.3) -> list[tuple]:
        """[(score, box, landmarks 5x2)] in the image's own pixels, best first."""
        scale = min(1.0, DETECTOR_SIZE / max(image.size))
        small = image if scale == 1.0 else image.resize(
            (round(image.width * scale), round(image.height * scale)), Image.BILINEAR)
        rgb = np.asarray(small, np.float32)
        blob = np.zeros((1, 3, DETECTOR_SIZE, DETECTOR_SIZE), np.float32)
        blob[0, :, :rgb.shape[0], :rgb.shape[1]] = rgb[:, :, ::-1].transpose(2, 0, 1)  # BGR, 0-255
        out = dict(zip(self._outputs, self._detector.run(None, {"input": blob})))
        found = []
        for stride in (8, 16, 32):
            cols = DETECTOR_SIZE // stride
            score = np.sqrt(np.clip(out[f"cls_{stride}"][0, :, 0], 0, 1)
                            * np.clip(out[f"obj_{stride}"][0, :, 0], 0, 1))
            for i in np.nonzero(score >= min_score)[0]:
                row, col = divmod(int(i), cols)
                b = out[f"bbox_{stride}"][0, i]
                cx, cy = (col + b[0]) * stride, (row + b[1]) * stride
                w, h = np.exp(b[2]) * stride, np.exp(b[3]) * stride
                marks = (out[f"kps_{stride}"][0, i].reshape(5, 2) + [col, row]) * stride
                found.append((float(score[i]),
                              tuple(float(v) / scale for v in (cx - w / 2, cy - h / 2, w, h)),
                              marks / scale))
        found.sort(key=lambda f: -f[0])
        kept: list[tuple] = []
        for face in found:
            if all(iou(face[1], k[1]) < nms for k in kept):
                kept.append(face)
        return kept

    def embed(self, crop: Image.Image) -> np.ndarray:
        blob = np.asarray(crop, np.float32).transpose(2, 0, 1)[None]   # RGB, 0-255
        vector = self._recognizer.run(None, {"data": blob})[0][0].astype(np.float32)
        return vector / np.linalg.norm(vector)


def iou(a: tuple, b: tuple) -> float:
    iw = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


def similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares rotation + uniform scale + shift taking src onto dst (Umeyama), 2x3.
    All 5 points, not the eyes alone, which slip on small faces (Frigate's note)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s, d = src - mu_s, dst - mu_d
    u, sig, vt = np.linalg.svd(d.T @ s / len(src))
    sign = np.eye(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[1, 1] = -1
    rotation = u @ sign @ vt
    scale = np.trace(np.diag(sig) @ sign) / s.var(0).sum()
    m = np.zeros((2, 3))
    m[:, :2] = scale * rotation
    m[:, 2] = mu_d - scale * rotation @ mu_s
    return m


def align(image: Image.Image, landmarks: np.ndarray) -> Image.Image:
    m = similarity_transform(np.asarray(landmarks, np.float64), TEMPLATE)
    inverse = np.linalg.inv(np.vstack([m, [0, 0, 1]]))[:2]    # Pillow maps output -> input
    return image.transform((112, 112), Image.AFFINE, inverse.flatten().tolist(), Image.BICUBIC)


def laplacian_variance(gray: np.ndarray) -> float:
    lap = gray[1:-1, :-2] + gray[1:-1, 2:] + gray[:-2, 1:-1] + gray[2:, 1:-1] - 4 * gray[1:-1, 1:-1]
    return float(lap.var())
