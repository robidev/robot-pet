import logging

import yaml

from petd.config import Config
from petd.io.face import parse_status, reboot_detected, wifi_drop_message
from petd.log import prune_runs, start_run_log


def config(tmp_path, **log) -> Config:
    cfg = Config()
    cfg.log.dir = str(tmp_path / "logs")
    for key, value in log.items():
        setattr(cfg.log, key, value)
    return cfg


def test_a_run_gets_its_own_folder_with_the_log_and_its_config(tmp_path):
    cfg = config(tmp_path)
    cfg.speaker.volume = 0.35
    run_dir, handler = start_run_log(cfg, ("fake",), ["petd", "--fake"], now=1_000_000_000)
    record = logging.LogRecord("petd.test", logging.DEBUG, __file__, 1, "heard %r", ("hello",), None)
    handler.handle(record)
    handler.close()

    assert run_dir.name.endswith("-fake")
    assert "DEBUG" in (run_dir / "petd.log").read_text()    # the file keeps DEBUG
    meta = yaml.safe_load((run_dir / "run.yaml").read_text())
    assert meta["argv"] == ["petd", "--fake"]
    assert meta["config"]["speaker"]["volume"] == 0.35
    assert (tmp_path / "logs" / "latest").resolve() == run_dir.resolve()


def test_two_starts_in_the_same_second_get_separate_folders(tmp_path):
    cfg = config(tmp_path)
    first, h1 = start_run_log(cfg, now=1_000_000_000)
    second, h2 = start_run_log(cfg, now=1_000_000_000)
    h1.close(); h2.close()
    assert first != second and first.exists() and second.exists()


def test_only_the_newest_runs_are_kept_and_nothing_else_is_touched(tmp_path):
    cfg = config(tmp_path, keep_runs=3)
    runs = []
    for i in range(5):
        run_dir, handler = start_run_log(cfg, now=1_000_000_000 + i)
        handler.close()
        runs.append(run_dir)
    base = tmp_path / "logs"
    (base / "notes").mkdir()
    prune_runs(base, 3)
    assert [p.exists() for p in runs] == [False, False, True, True, True]
    assert (base / "notes").exists()


def test_a_face_reboot_is_uptime_going_backwards():
    assert reboot_detected(500, 20)
    assert not reboot_detected(500, 502)
    assert not reboot_detected(None, 20)      # first poll
    assert not reboot_detected(500, None)     # unreachable, or old firmware


def test_the_face_wifi_section_is_read_and_explained():
    state = parse_status({"wifi": {"rssi": -71, "disconnects": 2, "last_disconnect_reason": 15}})
    assert (state.wifi_rssi, state.wifi_disconnects, state.wifi_last_reason) == (-71, 2, 15)
    assert "key handshake timed out" in wifi_drop_message(state)
    assert parse_status({}).wifi_disconnects is None          # older firmware


def test_older_runs_are_pruned_to_a_size_budget_but_never_the_current_one(tmp_path):
    cfg = config(tmp_path, keep_runs=10)
    runs = []
    for i in range(4):
        run_dir, handler = start_run_log(cfg, now=1_000_000_000 + i)
        handler.close()
        (run_dir / "petd.log").write_bytes(b"x" * 400_000)      # ~0.4 MB each
        runs.append(run_dir)
    base = tmp_path / "logs"
    doomed = prune_runs(base, 10, max_old_mb=0.9, current=runs[-1])
    assert doomed == runs[:1]                   # 3 older runs, 1.2 MB: the oldest goes
    assert [p.exists() for p in runs] == [False, True, True, True]
    prune_runs(base, 10, max_old_mb=0.0, current=runs[-1])
    assert [p.exists() for p in runs] == [False, False, False, True]


def test_the_event_trace_rotates(tmp_path):
    import asyncio
    from petd.bus import EventBus
    from petd.events import SpeechStarted
    from petd.trace import EventTrace

    async def run():
        bus = EventBus()
        bus.bind_loop(asyncio.get_running_loop())
        trace = EventTrace(bus, tmp_path / "events.jsonl", max_bytes=2000)
        trace.start()
        for i in range(100):
            bus.publish(SpeechStarted(t_utc=float(i)))
        await asyncio.sleep(0.05)
        await trace.close()
    asyncio.run(run())
    assert (tmp_path / "events.jsonl.1").exists()
    assert (tmp_path / "events.jsonl").stat().st_size < 2000
    assert (tmp_path / "events.jsonl.1").stat().st_size < 2000 + 200
