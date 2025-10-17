import json
import logging
import shutil
from pathlib import Path

import pytest

from goggles._core.run import _RunContextManager


@pytest.fixture
def tmp_run_dir(tmp_path):
    """Provide a temporary run directory and ensure full cleanup."""
    base = tmp_path / "runs"
    base.mkdir()
    yield base
    shutil.rmtree(base, ignore_errors=True)
    assert not base.exists(), f"Temporary run dir {base} was not cleaned."


def _flush_handlers():
    root = logging.getLogger()
    for h in root.handlers:
        try:
            h.flush()
        except Exception:
            pass


def test_jsonl_created_and_receives_records(tmp_run_dir):
    msg = "hello-from-jsonl"
    ctx = _RunContextManager(
        name="jsonl-on",
        log_dir=str(tmp_run_dir),
        user_metadata={"project": "demo_project"},
        enable_wandb=False,
        enable_jsonl=True,
        enable_file=False,  # isolate jsonl behavior
    )
    with ctx as runctx:
        root = logging.getLogger()
        prev = root.level
        try:
            root.setLevel(logging.INFO)
            run_path = Path(runctx.log_dir)
            jsonl_path = run_path / "events.jsonl"

            logging.getLogger().info(msg)
            _flush_handlers()

            assert (
                jsonl_path.exists()
            ), "events.jsonl should be created when enable_jsonl=True"
            lines = [
                ln
                for ln in jsonl_path.read_text(encoding="utf-8").splitlines()
                if ln.strip()
            ]
            assert lines, "events.jsonl should contain at least one line"
            payload = json.loads(lines[-1])
            assert payload["message"] == msg
            assert payload["levelname"] == "INFO"
            assert "name" in payload and "created" in payload and "pathname" in payload
        finally:
            root.setLevel(prev)


def test_jsonl_not_created_when_disabled(tmp_run_dir):
    ctx = _RunContextManager(
        name="jsonl-off",
        log_dir=str(tmp_run_dir),
        user_metadata={"project": "demo_project"},
        enable_wandb=False,
        enable_jsonl=False,
    )
    with ctx as runctx:
        run_path = Path(runctx.log_dir)
        logging.getLogger().info("should-not-appear-in-jsonl")
        _flush_handlers()
        assert not (
            run_path / "events.jsonl"
        ).exists(), "events.jsonl must not exist if enable_jsonl=False"


def test_metadata_contains_jsonl_path_when_enabled(tmp_run_dir):
    ctx = _RunContextManager(
        name="meta-jsonl-on",
        log_dir=str(tmp_run_dir),
        user_metadata={"project": "demo_project"},
        enable_wandb=False,
        enable_jsonl=True,
    )
    with ctx as runctx:
        meta_path = Path(runctx.log_dir) / "metadata.json"
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        assert (
            "logs" in data and "jsonl" in data["logs"]
        ), "metadata should include logs.jsonl path"
        jpath = Path(data["logs"]["jsonl"])
        assert jpath.name == "events.jsonl"
        assert jpath.exists()


def test_jsonl_handler_detaches_on_exit(tmp_run_dir):
    ctx = _RunContextManager(
        name="jsonl-detach",
        log_dir=str(tmp_run_dir),
        user_metadata={"project": "demo_project"},
        enable_wandb=False,
        enable_jsonl=True,
    )
    with ctx as runctx:
        jsonl_path = Path(runctx.log_dir) / "events.jsonl"
        logging.getLogger().info("before-exit")
        _flush_handlers()
        size_before = jsonl_path.stat().st_size

    # After exit, the JSONL handler should be removed; file should not grow.
    logging.getLogger().info("after-exit")
    _flush_handlers()
    size_after = jsonl_path.stat().st_size
    assert (
        size_after == size_before
    ), "events.jsonl size should not grow after handler detach"


def test_configure_enable_jsonl_then_run_override_disable(tmp_run_dir):
    # Default on, per-run override off
    from goggles._core import config as cfg

    cfg.configure(enable_jsonl=True)

    ctx = _RunContextManager(
        name="jsonl-override-off",
        log_dir=str(tmp_run_dir),
        user_metadata={},
        enable_wandb=False,
        enable_jsonl=False,  # override should win
    )
    with ctx as runctx:
        assert not (Path(runctx.log_dir) / "events.jsonl").exists()


def test_configure_enable_jsonl_then_run_none_uses_default(tmp_run_dir):
    # Default on, run override None → uses default
    from goggles._core import config as cfg

    cfg.configure(enable_jsonl=True)

    ctx = _RunContextManager(
        name="jsonl-default-on",
        log_dir=str(tmp_run_dir),
        user_metadata={},
        enable_wandb=False,
        enable_jsonl=None,  # no opinion
    )
    with ctx as runctx:
        assert (Path(runctx.log_dir) / "events.jsonl").exists()


def test_jsonl_lines_are_valid_json_and_contain_expected_fields(tmp_run_dir):
    ctx = _RunContextManager(
        name="jsonl-structure",
        log_dir=str(tmp_run_dir),
        user_metadata={},
        enable_wandb=False,
        enable_jsonl=True,
    )
    with ctx as runctx:
        root = logging.getLogger()
        prev = root.level
        try:
            root.setLevel(logging.INFO)
            logging.getLogger().info("alpha")
            logging.getLogger().warning("beta")
            _flush_handlers()

            jsonl_path = Path(runctx.log_dir) / "events.jsonl"
            lines = [
                ln
                for ln in jsonl_path.read_text(encoding="utf-8").splitlines()
                if ln.strip()
            ]
            assert len(lines) >= 2

            objs = [json.loads(ln) for ln in lines[-2:]]
            levels = {o["levelname"] for o in objs}
            messages = {o["message"] for o in objs}
            assert {"alpha", "beta"} <= messages
            assert "WARNING" in levels and "INFO" in levels
            # sanity: required keys present
            for o in objs:
                for k in (
                    "message",
                    "name",
                    "level",
                    "levelname",
                    "created",
                    "pathname",
                    "lineno",
                ):
                    assert k in o
        finally:
            root.setLevel(prev)


def test_no_duplicate_jsonl_handlers_across_runs(tmp_run_dir):
    def count_jsonl_handlers():
        import logging
        from logging import Handler

        # Crude heuristic: our JSONL handler is a subclass of Handler with a .emit writing json.
        # Since we can't import the internal class here, detect by file extension presence after attach.
        return sum(
            1
            for h in logging.getLogger().handlers
            if h.__class__.__name__.lower().endswith("jsonlhandler")
        )

    before = count_jsonl_handlers()
    for i in range(2):
        with _RunContextManager(
            name=f"jsonl-{i}",
            log_dir=str(tmp_run_dir),
            user_metadata={},
            enable_wandb=False,
            enable_jsonl=True,
        ):
            pass
    after = count_jsonl_handlers()
    assert after == before


def test_jsonl_sink_detaches_on_exception(tmp_run_dir):
    ctx = _RunContextManager(
        name="jsonl-exc",
        log_dir=str(tmp_run_dir),
        user_metadata={},
        enable_wandb=False,
        enable_jsonl=True,
    )
    jsonl_path = None
    try:
        with ctx as runctx:
            jsonl_path = Path(runctx.log_dir) / "events.jsonl"
            assert jsonl_path.exists()
            logging.getLogger().info("before-raise")
            _flush_handlers()
            raise RuntimeError("boom")
    except RuntimeError:
        pass

    # After exit, handler is detached
    size_before = jsonl_path.stat().st_size if jsonl_path else 0
    logging.getLogger().info("after-exit-should-not-append")
    _flush_handlers()
    assert jsonl_path is not None and jsonl_path.stat().st_size == size_before


def test_jsonl_and_text_can_coexist(tmp_run_dir):
    msg1, msg2 = "to-text", "to-jsonl"
    ctx = _RunContextManager(
        name="both-sinks",
        log_dir=str(tmp_run_dir),
        user_metadata={},
        enable_wandb=False,
        enable_file=True,
        enable_jsonl=True,
    )
    with ctx as runctx:
        root = logging.getLogger()
        prev = root.level
        try:
            root.setLevel(logging.INFO)
            run_path = Path(runctx.log_dir)
            text_path = run_path / "events.log"
            jsonl_path = run_path / "events.jsonl"

            logging.getLogger().info(msg1)
            logging.getLogger().info(msg2)
            _flush_handlers()

            # Both files exist
            assert text_path.exists() and jsonl_path.exists()
            # Text contains plain message
            assert msg1 in text_path.read_text(encoding="utf-8")
            # JSONL contains structured message
            jlines = [
                ln
                for ln in jsonl_path.read_text(encoding="utf-8").splitlines()
                if ln.strip()
            ]
            assert any(json.loads(ln)["message"] == msg2 for ln in jlines)
        finally:
            root.setLevel(prev)
