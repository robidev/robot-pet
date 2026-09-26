import json
import math
from pathlib import Path

import numpy as np
import pytest

from petd.spatial.mapgeo import Alignment, Grid, align

FIXTURES = Path(__file__).parent / "fixtures"


def load(name):
    return json.loads((FIXTURES / name).read_text())


@pytest.fixture(scope="module")
def room():
    """The whole room, mapped from the dock on 2026-09-26 (map 3578)."""
    return Grid.from_valetudo(load("valetudo_map_2026-09-26.json"))


def test_decodes_every_pixel_valetudo_counted():
    raw = load("valetudo_map_2026-09-26.json")
    grid = Grid.from_valetudo(raw)
    counts = {layer["type"]: layer["dimensions"]["pixelCount"] for layer in raw["layers"]}
    assert len(grid.walls_cm) == grid.wall.sum() == counts["wall"]
    assert (grid.floor | grid.wall).sum() >= counts["floor"]


def test_floor_walls_and_clearance(room):
    # In front of the dock: open floor, the nearest obstacle ~75 cm away.
    assert room.is_floor(2562, 2601) and room.is_free(2562, 2601, 20)
    assert 60 < room.clearance(2562, 2601) < 90
    # A box ~1.2 m from the dock, pixels 534-541 x 510-518: its edge is a
    # wall, and inside it nothing is mapped.
    assert not room.is_free(2672, 2572, 20)
    assert not room.is_floor(2690, 2570)
    assert not room.is_floor(9000, 9000)


def test_march_back_pulls_a_goal_in_furniture_out_in_front_of_it(room):
    # 2026-09-26 23:52: go_to (2690, 2570), inside the box, sent the robot
    # round to its far side. Marched back towards the robot on the dock,
    # the goal lands on the near side instead, in the open.
    x, y = room.march_back((2690, 2570), (2557, 2550), clearance_cm=20)
    assert x < 2670 and room.is_free(x, y, 20)
    assert math.hypot(x - 2690, y - 2570) < 60
    # A goal already free stays where it is.
    assert room.march_back((2562, 2601), (2557, 2550), 20) == (2562, 2601)
    # Nothing free between two points off the map.
    assert room.march_back((9000, 9000), (9100, 9000), 20) is None


def test_an_alignment_and_its_inverse_undo_each_other():
    a = Alignment(-74.0, 3000.0, 1500.0)
    x, y = a.inverse().apply(*a.apply(2558.0, 2531.0))
    assert x == pytest.approx(2558.0) and y == pytest.approx(2531.0)


def moved(grid, transform):
    """The same room, redrawn in a frame `transform` away."""
    def redraw(mask):
        cm = (np.argwhere(mask)[:, ::-1] + (grid.x0, grid.y0) + 0.5) * grid.pixel_size
        out = np.array([transform.apply(x, y) for x, y in cm])
        return np.unique(np.floor(out / grid.pixel_size).astype(int), axis=0)
    return Grid.from_pixels(redraw(grid.floor), redraw(grid.wall), grid.pixel_size)


@pytest.mark.parametrize("transform", [Alignment(37.0, 120.0, -85.0), Alignment(-74.0, 3000.0, 1500.0)])
def test_align_recovers_a_known_rotation_and_shift(room, transform):
    found = align(moved(room, transform), room)
    assert found.score > 0.95
    for point in [(2558, 2531), (2700, 2600), (2400, 2700)]:
        assert math.dist(found.apply(*point), transform.apply(*point)) < 6


def test_align_the_map_that_came_rotated(room):
    # Earlier on 2026-09-26 the robot had a partial map (3576) ~74 deg off.
    # Aligned onto the full map, its dock lands on the full map's dock.
    partial = Grid.from_valetudo(load("valetudo_map_2026-09-26_partial.json"))
    found = align(room, partial)
    assert 72 < found.angle_deg < 80 and found.score > 0.7
    assert math.dist(found.apply(2530, 2551), (2558, 2531)) < 15
