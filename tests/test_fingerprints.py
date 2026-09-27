import numpy as np
import pytest
from PIL import Image

from petd.config import RecognitionConfig
from petd.memory.db import MemoryDB
from petd.memory.fingerprints import Refused, assign_attempt, forget_fingerprints
from petd.vision.kept import FaceKeeper
from petd.vision.matching import normalized, to_blob

RNG = np.random.default_rng(11)


def someone():
    return normalized(RNG.normal(size=128))


def view(person, noise=0.5):
    wobble = RNG.normal(size=128)
    return normalized(person + wobble * noise / np.linalg.norm(wobble))


class StubEngine:
    """embed() returns whatever the test says the crop looks like."""
    def __init__(self):
        self.next = None

    def embed(self, crop):
        return self.next


@pytest.fixture
def setup(tmp_path):
    db = MemoryDB(":memory:")
    keeper = FaceKeeper(tmp_path, keep_attempts=50, name_of=str)
    robin, noah = someone(), someone()
    people = {}
    for name, face in (("Robin", robin), ("Noah", noah)):
        person = db.add_person(name)
        people[name] = person
        for _ in range(5):
            embedding_id = db.add_face_embedding(person.id, to_blob(view(face)), source="enroll")
            Image.new("RGB", (112, 112)).save(keeper.fingerprints / f"{embedding_id}.jpg")
    return db, keeper, people, {"Robin": robin, "Noah": noah}


def attempt(keeper, name="20260927-101722.870_as-Robin_best-Robin-0.61.jpg"):
    path = keeper.attempts / name
    Image.new("RGB", (112, 112)).save(path)
    return keeper.find_attempt(name[:15])


def test_forget_removes_fingerprints_and_crops_but_never_the_last(setup):
    db, keeper, people, _ = setup
    done = forget_fingerprints(db, [1, 2], keeper)
    assert len(done) == 2 and db.face_count(people["Robin"].id) == 3
    assert not (keeper.fingerprints / "1.jpg").exists() and (keeper.fingerprints / "3.jpg").exists()
    with pytest.raises(Refused):
        forget_fingerprints(db, [3, 4, 5], keeper)
    with pytest.raises(Refused):
        forget_fingerprints(db, [999], keeper)
    forget_fingerprints(db, [3, 4, 5], keeper, force=True)
    assert db.face_count(people["Robin"].id) == 0


def test_a_misread_goes_to_the_right_person_with_a_pointer_to_the_wrong_copy(setup):
    db, keeper, people, faces = setup
    cfg = RecognitionConfig()
    engine = StubEngine()
    # Noah, misread as Robin, and grown into Robin's set from that very look.
    look = view(faces["Noah"])
    wrong = db.add_face_embedding(people["Robin"].id, to_blob(look), source="grown")
    engine.next = look
    misread = attempt(keeper)

    with pytest.raises(Refused):            # it isn't like Robin: refused for Robin
        assign_attempt(db, engine, cfg, misread, people["Robin"], keeper)
    done = assign_attempt(db, engine, cfg, misread, people["Noah"], keeper)
    assert done.similarity > cfg.unknown_sim
    assert done.elsewhere == [("Robin", wrong)]
    assert db.face_embedding(done.embedding_id)["source"] == "assigned"
    assert (keeper.fingerprints / f"{done.embedding_id}.jpg").exists()

    with pytest.raises(Refused):            # the same look again: nothing to add
        assign_attempt(db, engine, cfg, misread, people["Noah"], keeper)


def test_a_full_set_trades_a_grown_look_never_an_assigned_one(setup):
    db, keeper, people, faces = setup
    cfg = RecognitionConfig(max_per_person=7)
    engine = StubEngine()
    grown = [db.add_face_embedding(people["Noah"].id, to_blob(view(faces["Noah"], 0.3)), source="grown")
             for _ in range(2)]
    engine.next = view(faces["Noah"], 0.8)
    done = assign_attempt(db, engine, cfg, attempt(keeper), people["Noah"], keeper)
    assert done.replaced in grown
    assert db.face_count(people["Noah"].id) == 7
