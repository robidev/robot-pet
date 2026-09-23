"""
Face recognition's pieces (PLAN.md 4.7, E6): alignment, centres, one attempt,
the vote; and, where the models and E6a's captures are present (neither is in
git), the whole engine on real snapshots.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from petd.config import PROJECT_ROOT, Config
from petd.vision.faces import TEMPLATE, similarity_transform
from petd.vision.matching import Guess, Vote, class_centre, classify, normalized

RNG = np.random.default_rng(3)


def test_alignment_undoes_rotation_scale_and_shift():
    angle, scale, shift = np.radians(20), 2.5, np.array([200.0, 150.0])
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    landmarks = TEMPLATE @ (scale * rot).T + shift      # a face somewhere in a snapshot
    m = similarity_transform(landmarks, TEMPLATE)
    back = landmarks @ m[:, :2].T + m[:, 2]
    assert np.allclose(back, TEMPLATE, atol=1e-6)


def test_a_centre_shrugs_off_a_wrong_sample():
    person = normalized(RNG.normal(size=128))
    samples = [normalized(person + 0.4 * normalized(RNG.normal(size=128))) for _ in range(8)]
    wrong = normalized(RNG.normal(size=128))            # somebody else, filed by mistake
    centre = class_centre(samples + [wrong])
    assert centre @ person > 0.95 and abs(centre @ wrong) < 0.2


def test_one_attempt_needs_the_threshold_and_the_margin():
    a, b = normalized(RNG.normal(size=128)), normalized(RNG.normal(size=128))
    centres = {1: a, 2: b}
    assert classify(a, centres, 0.35, 0.15).person_id == 1
    assert classify(normalized(RNG.normal(size=128)), centres, 0.35, 0.15).person_id is None
    halfway = normalized(a + b)                           # ~0.7 to both: not sure whose
    guess = classify(halfway, centres, 0.35, 0.15)
    assert guess.person_id is None and guess.similarity > 0.6
    assert classify(a, {}, 0.35, 0.15) == Guess(None, 0.0, None, -1.0)


def vote(*attempts) -> Vote:
    v = Vote()
    for person, similarity, height in attempts:
        v.add(Guess(person, similarity, person, 0.0), height * height)
    return v


def test_the_vote_needs_agreement_confidence_and_no_tie():
    args = (0.35, 0.45, 2)                                # unknown_sim, accept_sim, min_agree
    assert vote((1, 0.6, 100)).decide(*args) is None                      # one look isn't enough
    assert vote((1, 0.6, 100), (1, 0.5, 60)).decide(*args)[0] == 1
    assert vote((1, 0.6, 100), (1, 0.5, 60), (2, 0.6, 100), (2, 0.5, 60)).decide(*args) is None
    assert vote((1, 0.40, 100), (1, 0.41, 100)).decide(*args) is None     # agreed, but weakly
    assert vote((None, 0.2, 100), (None, 0.3, 100)).decide(*args) is None


def test_a_close_look_outweighs_distant_ones():
    close, far = (1, 0.80, 190), (1, 0.37, 48)
    person, mean = vote(close, far, far, far).decide(0.35, 0.45, 2)
    assert person == 1 and mean > 0.7


E6A = PROJECT_ROOT / "runtime" / "e6a" / "data"


@pytest.mark.skipif(not E6A.exists(), reason="E6a's captures are local only (runtime/e6a)")
def test_real_snapshots_from_e6a_are_told_apart():
    cfg = Config().recognition
    models = PROJECT_ROOT / cfg.models_dir
    if not (models / cfg.recognizer_model).exists():
        pytest.skip("models not fetched: scripts/fetch_face_models.py")
    from petd.vision.faces import FaceEngine
    engine = FaceEngine(models, cfg.detector_model, cfg.recognizer_model)
    shots: dict[tuple, list] = {}
    for meta in sorted(E6A.glob("*/*/*.json")):
        faces = engine.analyse(meta.with_suffix(".jpg").read_bytes())
        faces = [f for f in faces if f.height >= cfg.min_face_px and f.score >= cfg.min_detection_score]
        assert len(faces) == 1, meta
        shots.setdefault((meta.parent.parent.name, meta.parent.name.split("-")[0]), []).append(faces[0])
    people = sorted({name for name, _ in shots})
    # Enrolled close up only, the hardest case E6a measured.
    centres = {i: class_centre([f.embedding for f in shots[(name, "0.6m")]]) for i, name in enumerate(people)}
    for (name, distance), faces in shots.items():
        v = Vote()
        for f in faces:
            guess = classify(f.embedding, centres, cfg.unknown_sim, cfg.margin)
            assert guess.person_id in (None, people.index(name)), f"{name} at {distance} named wrong"
            v.add(guess, f.area)
        decided = v.decide(cfg.unknown_sim, cfg.accept_sim, cfg.min_agree)
        assert decided and decided[0] == people.index(name), f"{name} at {distance}: {decided}"
