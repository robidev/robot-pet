"""
The robot's map as geometry (PLAN.md 4.2, C1): what is floor, how far a
point is from the nearest obstacle, and how one map lines up with another.

- Valetudo's map is floor and wall pixels (`pixelSize` cm each), run-length
  coded; entities (robot, charger, path) are in cm. This robot's Valetudo
  has no room segments. A wall pixel is anything the lidar hit: walls and
  furniture look the same.
- The firmware never refuses a go_to: a goal on furniture stops it in front,
  one inside sends it round to the far side (2026-09-26). `march_back` pulls
  a goal back towards the robot until it's on open floor.
- A new map can come in a different frame (rotated ~74 deg once; the frame
  is wherever the robot was when the map began). `align` finds the rotation
  and shift that put one map's walls on another's: for each trial angle, an
  FFT cross-correlation of the two wall rasters gives the best shift, first
  coarse (10 cm pixels, every degree), then fine around the best.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Grid:
    """One map's floor and walls, cropped to what's mapped."""
    pixel_size: float                 # cm per pixel
    x0: int                           # map pixel of column 0
    y0: int                           # map pixel of row 0
    floor: np.ndarray                 # bool [rows, cols]
    wall: np.ndarray                  # bool [rows, cols]
    walls_cm: np.ndarray              # float [n, 2], wall pixel centres in map cm

    @classmethod
    def from_valetudo(cls, map_json: dict) -> "Grid":
        pixel_size = float(map_json.get("pixelSize") or 5)
        layers: dict[str, list[np.ndarray]] = {"floor": [], "wall": []}
        for layer in map_json.get("layers", []):
            kind = "wall" if layer.get("type") == "wall" else "floor"     # a segment is floor too
            layers[kind].append(_decode(layer))
        floor = np.concatenate(layers["floor"]) if layers["floor"] else np.zeros((0, 2), int)
        wall = np.concatenate(layers["wall"]) if layers["wall"] else np.zeros((0, 2), int)
        return cls.from_pixels(floor, wall, pixel_size)

    @classmethod
    def from_pixels(cls, floor_px: np.ndarray, wall_px: np.ndarray, pixel_size: float) -> "Grid":
        both = np.concatenate([floor_px, wall_px]) if len(floor_px) + len(wall_px) else np.zeros((1, 2), int)
        x0, y0 = both.min(axis=0)
        cols, rows = both.max(axis=0) - (x0, y0) + 1
        floor = np.zeros((rows, cols), bool)
        wall = np.zeros((rows, cols), bool)
        floor[floor_px[:, 1] - y0, floor_px[:, 0] - x0] = True
        wall[wall_px[:, 1] - y0, wall_px[:, 0] - x0] = True
        return cls(pixel_size, int(x0), int(y0), floor & ~wall, wall,
                   (wall_px + 0.5) * pixel_size)

    # --- points -----------------------------------------------------------------

    def _cell(self, x: float, y: float) -> Optional[tuple[int, int]]:
        col = math.floor(x / self.pixel_size) - self.x0
        row = math.floor(y / self.pixel_size) - self.y0
        if 0 <= row < self.floor.shape[0] and 0 <= col < self.floor.shape[1]:
            return row, col
        return None

    def is_floor(self, x: float, y: float) -> bool:
        cell = self._cell(x, y)
        return cell is not None and bool(self.floor[cell])

    def clearance(self, x: float, y: float) -> float:
        """cm from (x, y) to the nearest wall pixel's centre (inf with no walls)."""
        if not len(self.walls_cm):
            return math.inf
        return float(np.min(np.hypot(self.walls_cm[:, 0] - x, self.walls_cm[:, 1] - y)))

    def is_free(self, x: float, y: float, clearance_cm: float) -> bool:
        """Mapped floor, at least clearance_cm from anything the lidar has seen."""
        return self.is_floor(x, y) and self.clearance(x, y) >= clearance_cm

    def march_back(self, goal: tuple[float, float], toward: tuple[float, float],
                   clearance_cm: float, step_cm: float = 5.0) -> Optional[tuple[float, float]]:
        """
        The first free point from `goal` back along the line to `toward`
        (usually the robot), or None if there's none: a goal in or behind
        furniture becomes the open floor in front of it.
        """
        (gx, gy), (tx, ty) = goal, toward
        length = math.hypot(tx - gx, ty - gy)
        steps = max(1, math.ceil(length / step_cm))
        for i in range(steps + 1):
            f = i / steps
            x, y = gx + (tx - gx) * f, gy + (ty - gy) * f
            if self.is_free(x, y, clearance_cm):
                return x, y
        return None

    def wall_hits(self, other: "Grid", alignment: "Alignment") -> float:
        """
        The share of `other`'s walls that `alignment` puts within a pixel of
        one of these: a cheap check that an alignment still holds.
        """
        if not len(other.walls_cm) or not len(self.walls_cm):
            return 0.0
        a = math.radians(alignment.angle_deg)
        rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        moved = other.walls_cm @ rot.T + (alignment.tx, alignment.ty)
        cols = np.floor(moved[:, 0] / self.pixel_size).astype(int) - self.x0
        rows = np.floor(moved[:, 1] / self.pixel_size).astype(int) - self.y0
        near = self._near_wall()
        inside = (rows >= 0) & (rows < near.shape[0]) & (cols >= 0) & (cols < near.shape[1])
        return float(near[rows[inside], cols[inside]].sum() / len(moved))

    def _near_wall(self) -> np.ndarray:
        """The walls grown by a pixel each way."""
        near = self.wall.copy()
        near[1:, :] |= self.wall[:-1, :]
        near[:-1, :] |= self.wall[1:, :]
        grown = near.copy()
        grown[:, 1:] |= near[:, :-1]
        grown[:, :-1] |= near[:, 1:]
        return grown


# --- alignment ------------------------------------------------------------------


@dataclass(frozen=True)
class Alignment:
    """Takes a point from one map's frame into another's: rotate, then shift."""
    angle_deg: float
    tx: float
    ty: float
    score: float = 1.0                # share of walls that land on walls (within a pixel)

    def apply(self, x: float, y: float) -> tuple[float, float]:
        a = math.radians(self.angle_deg)
        return (math.cos(a) * x - math.sin(a) * y + self.tx,
                math.sin(a) * x + math.cos(a) * y + self.ty)

    def inverse(self) -> "Alignment":
        a = math.radians(-self.angle_deg)
        tx = -(math.cos(a) * self.tx - math.sin(a) * self.ty)
        ty = -(math.sin(a) * self.tx + math.cos(a) * self.ty)
        return Alignment(-self.angle_deg, tx, ty, self.score)


def align(ref: Grid, new: Grid, coarse_step_deg: float = 1.0) -> Alignment:
    """
    The rotation and shift that take `new`'s frame into `ref`'s, with the
    share of `new`'s walls that then land within a pixel of one of `ref`'s.
    Maps of the same room made in different frames scored 0.7-0.85 on
    2026-09-26; judge a low score as "not the same place" at the caller.
    A map with too little in it can match wrongly: one straight wall fits
    anywhere along any straight wall.
    """
    if not len(ref.walls_cm) or not len(new.walls_cm):
        return Alignment(0.0, 0.0, 0.0, 0.0)
    coarse = _search(ref.walls_cm, new.walls_cm, 2 * ref.pixel_size,
                     np.arange(-180.0, 180.0, coarse_step_deg), tolerant=True)
    # Fine: exact pixel hits only. With a pixel's tolerance a whole plateau of
    # shifts scores the same (the same map came out 16 cm and 0.25 deg off).
    fine = _search(ref.walls_cm, new.walls_cm, ref.pixel_size,
                   coarse.angle_deg + np.arange(-1.5, 1.51, 0.25), tolerant=False)
    return Alignment(fine.angle_deg, fine.tx, fine.ty, ref.wall_hits(new, fine))


def _search(ref_cm: np.ndarray, new_cm: np.ndarray, res: float, angles, tolerant: bool) -> Alignment:
    # The reference as one raster; tolerant: dilated by a pixel so near misses count.
    ref_px = np.floor(ref_cm / res).astype(int)
    origin = ref_px.min(axis=0)
    ref_px -= origin
    ref_ext = ref_px.max(axis=0) + 1
    centre = new_cm.mean(axis=0)
    half = int(math.ceil(np.max(np.hypot(*(new_cm - centre).T)) / res)) + 2
    size = _fft_size(max(ref_ext) + 2 * half + 4)
    ref_img = np.zeros((size, size), np.float32)
    reach = (-1, 0, 1) if tolerant else (0,)
    for dx in reach:
        for dy in reach:
            ref_img[np.clip(ref_px[:, 1] + dy, 0, size - 1), np.clip(ref_px[:, 0] + dx, 0, size - 1)] = 1
    ref_f = np.fft.rfft2(ref_img)

    best = (-1.0, 0.0, 0, 0)
    for angle in angles:
        a = math.radians(angle)
        rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        q = (new_cm - centre) @ rot.T
        k = np.floor(q / res).astype(int) + half              # in [0, 2 * half]
        img = np.zeros((size, size), np.float32)
        img[k[:, 1], k[:, 0]] = 1
        corr = np.fft.irfft2(ref_f * np.conj(np.fft.rfft2(img)), s=(size, size))
        sy, sx = np.unravel_index(int(np.argmax(corr)), corr.shape)
        hits = corr[sy, sx] / img.sum()
        if hits > best[0]:
            best = (float(hits), float(angle), int(sx), int(sy))
    hits, angle, sx, sy = best
    # A shift past the reference's far edge is a negative one, wrapped round.
    sx = sx - size if sx > ref_ext[0] + 1 else sx
    sy = sy - size if sy > ref_ext[1] + 1 else sy
    # new point p lands at pixel k + s of the reference raster:
    # p_ref = R (p - centre) + (s + half + origin) * res.
    a = math.radians(angle)
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    shift = (np.array([sx, sy]) + half + origin) * res - rot @ centre
    return Alignment(angle, float(shift[0]), float(shift[1]), hits)


def _fft_size(n: int) -> int:
    """The next size up made of 2s, 3s and 5s: those FFT fast."""
    while True:
        m = n
        for p in (2, 3, 5):
            while m % p == 0:
                m //= p
        if m == 1:
            return n
        n += 1


def _decode(layer: dict) -> np.ndarray:
    """Valetudo's RLE [x_start, y, count, ...] (or plain [x, y, ...]) -> [n, 2] pixels."""
    rle = layer.get("compressedPixels")
    if rle:
        runs = np.asarray(rle, int).reshape(-1, 3)
        counts = runs[:, 2]
        xs = np.repeat(runs[:, 0], counts) + (np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts))
        ys = np.repeat(runs[:, 1], counts)
        return np.stack([xs, ys], axis=1)
    return np.asarray(layer.get("pixels") or [], int).reshape(-1, 2)
