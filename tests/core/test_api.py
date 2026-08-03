"""Tests for goggles' outer API."""

import importlib
import logging
import threading
from pathlib import Path
from typing import Any, ClassVar, cast

import pytest

import goggles as gg


class _ClassLevelLoggerHolder:
    """Holder whose logger is captured at class-body time.

    The ``gg.get_logger(...)`` call happens once when this module is
    imported by pytest, before any test body runs. That mirrors the
    realistic pattern of ``logger = gg.get_logger(...)`` at class scope
    in user code, with ``gg.attach(...)`` invoked later from ``main()``.

    Attributes:
        logger: Class-level Goggles text logger captured at module load.
    """

    logger = gg.get_logger("class-level-logger-test", scope="global")


# ---------------------------------------------------------------------
# Logger creation
# ---------------------------------------------------------------------


def test_get_logger_returns_expected_protocols():
    """Verify that get_logger returns proper protocol instances."""
    plain = gg.get_logger("plain")
    with_metrics = gg.get_logger("metrics", with_metrics=True)

    assert isinstance(plain, gg.TextLogger), (
        "get_logger('plain') should return a TextLogger"
    )
    assert not isinstance(plain, gg.GogglesLogger), (
        "get_logger('plain') should not return a GogglesLogger"
    )
    assert isinstance(with_metrics, gg.GogglesLogger), (
        "get_logger(with_metrics=True) should return a GogglesLogger"
    )


def test_logger_bind_creates_new_instance():
    """Ensure that bind() returns a derived logger."""
    log = gg.get_logger("base")
    bound = log.bind(scope="run", extra_field="value")
    assert isinstance(bound, gg.TextLogger), (
        "bound logger should be a TextLogger"
    )
    assert bound is not log, "bound logger should be a new instance"


# ---------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------


def test_timeit_measures_execution_time() -> None:
    """Check that @timeit executes and returns the wrapped result."""
    called = {}

    @gg.timeit()
    def fn(x):
        called["x"] = x
        return x + 1

    assert fn(1) == 2, "timeit decorated function should return correct result"
    assert called["x"] == 1, (
        "timeit decorated function should be called with correct args"
    )


def test_timeit_nested_does_not_conflict():
    """Verify nested @timeit decorators behave correctly."""

    @gg.timeit()
    @gg.timeit()
    def inner(x):
        return x * 2

    assert inner(3) == 6, (
        "Nested timeit decorated function should return correct result"
    )


def test_trace_on_error_logs_and_rethrows():
    """trace_on_error should log and re-raise the original error."""

    @gg.trace_on_error()
    def boom(x):
        return 1 / x

    with pytest.raises(ZeroDivisionError):
        boom(0)


def test_trace_on_error_with_kwargs():
    """Ensure kwargs are included in traced error info."""

    @gg.trace_on_error()
    def boom_kw(x=0):
        return 1 / x

    with pytest.raises(ZeroDivisionError):
        boom_kw(x=0)


# ---------------------------------------------------------------------
# Handlers and EventBus
# ---------------------------------------------------------------------


class DummyHandler:
    name = "dummy"
    capabilities: ClassVar[frozenset[gg.Kind]] = frozenset({"log", "metric"})
    handled: ClassVar[list[object]] = []

    def can_handle(self, kind):
        return True

    def handle(self, event):
        self.handled.append(event)

    def open(self):
        self.handled.clear()

    def close(self):
        self.handled.append("closed")

    def to_dict(self):
        return {"cls": "DummyHandler", "data": {}}

    @classmethod
    def from_dict(cls, serialized):
        return cls()


def test_attach_and_emit() -> None:
    """Attach dummy handler and ensure events are received."""
    gg.register_handler(DummyHandler)
    handler = DummyHandler()
    gg.attach(handler, scopes=["global"])

    log = gg.get_logger("demo")
    log.info("event test")
    gg.finish()

    assert "closed" in DummyHandler.handled, (
        "Handler should be closed on finish"
    )
    assert any(
        hasattr(e, "payload") for e in DummyHandler.handled if e != "closed"
    ), "Events should have a payload attribute"


def _scoped_handlers() -> dict:
    """Snapshot of (scope -> set of handler names) on the host bus."""
    transport = cast(Any, gg.get_bus())
    return dict(transport._bus.scopes)


def _bus_handlers() -> dict:
    """Snapshot of (handler name -> handler instance) on the host bus."""
    transport = cast(Any, gg.get_bus())
    return dict(transport._bus.handlers)


def test_configure_is_noop_by_default() -> None:
    """configure() with no args attaches nothing — purely a shortcut."""
    gg.finish()  # clean slate
    try:
        before = _scoped_handlers()
        gg.configure()
        assert _scoped_handlers() == before, (
            "configure() with default args must not change the bus state"
        )
    finally:
        gg.finish()


def test_configure_enable_console_attaches_console_handler() -> None:
    """configure(enable_console=True) wires a ConsoleHandler on global."""
    gg.finish()
    try:
        gg.configure(enable_console=True, console_level=logging.WARNING)
        assert "global" in _scoped_handlers(), (
            "configure(enable_console=True) should attach to 'global' by "
            "default"
        )
        consoles = [
            h
            for h in _bus_handlers().values()
            if isinstance(h, gg.ConsoleHandler)
        ]
        assert len(consoles) == 1, (
            f"Expected exactly one ConsoleHandler, found {len(consoles)}"
        )
        assert consoles[0].level == logging.WARNING, (
            "console_level should propagate to the handler"
        )
    finally:
        gg.finish()


def test_configure_respects_scopes_argument() -> None:
    """A custom scopes list routes the auto-attached handler accordingly."""
    gg.finish()
    try:
        gg.configure(enable_console=True, scopes=["train", "eval"])
        scopes = _scoped_handlers()
        for s in ("train", "eval"):
            assert s in scopes, (
                f"configure(scopes=[..., '{s}']) must attach under scope '{s}'"
            )
    finally:
        gg.finish()


def test_configure_replaces_existing_console_handler() -> None:
    """A second configure() call swaps the console handler instance."""
    gg.finish()
    try:
        gg.configure(enable_console=True, console_level=logging.INFO)
        first = next(
            h
            for h in _bus_handlers().values()
            if isinstance(h, gg.ConsoleHandler)
        )
        gg.configure(enable_console=True, console_level=logging.WARNING)
        consoles = [
            h
            for h in _bus_handlers().values()
            if isinstance(h, gg.ConsoleHandler)
        ]
        assert len(consoles) == 1, (
            f"Reconfigure must keep exactly one console handler; "
            f"saw {len(consoles)}"
        )
        assert consoles[0] is not first, (
            "Reconfigure must install a new ConsoleHandler instance, "
            "not silently keep the first one"
        )
        assert consoles[0].level == logging.WARNING, (
            "Reconfigure must apply the new console_level (saw "
            f"{consoles[0].level})"
        )
    finally:
        gg.finish()


def test_configure_is_exported_from_package_root() -> None:
    """configure() belongs to the documented public surface."""
    assert "configure" in gg.__all__, (
        "configure must be listed in goggles.__all__ to be public"
    )
    assert callable(getattr(gg, "configure", None)), (
        "configure must be importable from the goggles package root"
    )


def test_configure_uses_custom_handler_name() -> None:
    """configure(name=...) registers the console under that name."""
    gg.finish()
    try:
        gg.configure(enable_console=True, name="app.console")
        handlers = _bus_handlers()
        assert "app.console" in handlers, (
            "configure(name='app.console') must register the console "
            f"handler under that name; saw {sorted(handlers)}"
        )
        assert gg.ConsoleHandler.name not in handlers, (
            "A custom name must replace the class default, not be "
            f"registered alongside '{gg.ConsoleHandler.name}'"
        )
    finally:
        gg.finish()


def test_configure_replaces_console_handler_with_custom_name() -> None:
    """A second configure(name=...) call swaps that named handler."""
    gg.finish()
    try:
        gg.configure(
            enable_console=True,
            name="app.console",
            console_level=logging.INFO,
        )
        first = _bus_handlers()["app.console"]
        gg.configure(
            enable_console=True,
            name="app.console",
            console_level=logging.WARNING,
        )
        consoles = [
            h
            for h in _bus_handlers().values()
            if isinstance(h, gg.ConsoleHandler)
        ]
        assert len(consoles) == 1, (
            f"Reconfigure under a custom name must keep exactly one "
            f"console handler; saw {len(consoles)}"
        )
        assert consoles[0] is not first, (
            "Reconfigure must install a new ConsoleHandler instance "
            "under the custom name"
        )
        assert consoles[0].level == logging.WARNING, (
            f"The last configure() call must win (saw {consoles[0].level})"
        )
    finally:
        gg.finish()


def test_class_level_logger_does_not_hang_on_first_info() -> None:
    """Logger captured at class-body scope must not hang on first .info().

    Class-body resolution happens before any ``gg.attach()``; the first
    ``.info()`` after attach must complete promptly. The emit runs on a
    worker thread so a hang surfaces as a timed-out join rather than
    wedging the test runner.
    """
    gg.register_handler(DummyHandler)
    DummyHandler.handled.clear()
    handler = DummyHandler()
    gg.attach(handler, scopes=["global"])

    done = threading.Event()

    def _emit() -> None:
        _ClassLevelLoggerHolder().logger.info("class-var event")
        done.set()

    t = threading.Thread(target=_emit, daemon=True)
    t.start()
    assert done.wait(timeout=5.0), (
        "class-level logger.info() did not return within 5 s"
    )
    gg.finish()
    assert any(
        getattr(e, "payload", None) == "class-var event"
        for e in DummyHandler.handled
        if e != "closed"
    ), (
        "class-level logger never delivered its event — likely a stale "
        "transport reference captured at class-body time"
    )


def test_eventbus_emit_routing_and_detach():
    """Check EventBus routes correctly and detaches handlers."""
    bus = gg.EventBus()
    handler = DummyHandler()
    bus.attach([handler.to_dict()], scopes=["scope"])
    assert "scope" in bus.scopes, "'scope' should be in bus scopes after attach"
    assert handler.name in bus.handlers, (
        "Handler should be in bus handlers after attach"
    )

    # Emit dummy event
    event_dict = {
        "kind": "log",
        "scope": "scope",
        "msg": "msg",
        "payload": "msg",  # Required by Event constructor/processing
        "filepath": "test.py",
        "lineno": 1,
        "data": {},
        "timestamp": 0.0,
    }
    bus.emit(event_dict)

    bus.detach("dummy", "scope")
    assert "scope" not in bus.scopes, "'scope' should be removed after detach"
    assert "dummy" not in bus.handlers, (
        "'dummy' handler should be removed after detach"
    )


def test_detach_raises_for_invalid_scope():
    bus = gg.EventBus()
    handler = DummyHandler()
    bus.attach([handler.to_dict()], scopes=["valid"])
    with pytest.raises(ValueError):
        bus.detach("dummy", "invalid")


def test_eventbus_emit_ignores_unknown_scope():
    bus = gg.EventBus()
    bus.emit(
        {
            "kind": "log",
            "scope": "no_handlers",
            "msg": "none",
            "payload": "none",  # Required
            "filepath": "test.py",
            "lineno": 1,
            "data": {},
            "timestamp": 0.0,
        }
    )  # Should not raise


# ---------------------------------------------------------------------
# attach() handler-name conflicts
# ---------------------------------------------------------------------


class ConflictHandler:
    """Handler whose serialized config the test controls key by key."""

    name: str = "conflict"
    capabilities: ClassVar[frozenset[gg.Kind]] = frozenset({"log"})

    def __init__(self, name="conflict", level=logging.INFO, path="logs"):
        self.name = name
        self.level = level
        self.path = path

    def can_handle(self, kind):
        return True

    def handle(self, event):
        pass

    def open(self):
        pass

    def close(self):
        pass

    def to_dict(self):
        return {
            "cls": type(self).__name__,
            "data": {
                "name": self.name,
                "level": self.level,
                "path": self.path,
            },
        }

    @classmethod
    def from_dict(cls, serialized):
        return cls(**serialized)


class OtherConflictHandler(ConflictHandler):
    """A second handler class serializing under the same config keys."""


def _stderr_lines(capsys) -> list[str]:
    """Non-empty stderr lines captured so far."""
    return [line for line in capsys.readouterr().err.splitlines() if line]


def test_attach_identical_config_twice_is_silent(capsys) -> None:
    """Re-attaching the very same config is the idempotent case."""
    gg.register_handler(ConflictHandler)
    bus = gg.EventBus()
    handler = ConflictHandler(name="dup", level=logging.INFO)

    bus.attach([handler.to_dict()], scopes=["a"])
    bus.attach([handler.to_dict()], scopes=["a"])

    assert _stderr_lines(capsys) == [], (
        "Re-attaching an identical handler config must stay silent"
    )
    assert list(bus.handlers) == ["dup"], (
        "An identical re-attach must not register a second handler"
    )


def test_attach_conflicting_level_warns_and_keeps_first(capsys) -> None:
    """A differing config under a known name warns and changes nothing."""
    gg.register_handler(ConflictHandler)
    bus = gg.EventBus()
    first = ConflictHandler(name="dup", level=logging.INFO)
    second = ConflictHandler(name="dup", level=logging.DEBUG)

    bus.attach([first.to_dict()], scopes=["a"])
    bus.attach([second.to_dict()], scopes=["b"])

    warnings = _stderr_lines(capsys)
    assert len(warnings) == 1, (
        f"Expected exactly one conflict warning, got {warnings}"
    )
    assert "dup" in warnings[0], "The warning must name the handler"
    assert "level" in warnings[0], (
        "The warning must list the config key that differs"
    )
    assert "path" not in warnings[0], (
        "The warning must not list config keys that are identical"
    )
    assert "first" in warnings[0], (
        "The warning must say the first registration is kept"
    )
    assert cast(Any, bus.handlers["dup"]).level == logging.INFO, (
        "The first registration must stay in effect"
    )
    assert bus.scopes["b"] == {"dup"}, (
        "The second attach's scopes must still be merged"
    )


def test_attach_different_names_same_scope_does_not_warn(capsys) -> None:
    """Distinct names sharing a scope are not a conflict."""
    gg.register_handler(ConflictHandler)
    bus = gg.EventBus()
    first = ConflictHandler(name="one", level=logging.INFO)
    second = ConflictHandler(name="two", level=logging.DEBUG)

    bus.attach([first.to_dict(), second.to_dict()], scopes=["a"])

    assert _stderr_lines(capsys) == [], "Different handler names must not warn"
    assert set(bus.handlers) == {"one", "two"}, (
        "Both handlers should be registered under their own names"
    )
    assert bus.scopes["a"] == {"one", "two"}, (
        "Both handlers should be routed under the shared scope"
    )


def test_attach_conflicting_class_reports_the_class(capsys) -> None:
    """A same-named handler of another class reports the class itself."""
    gg.register_handler(ConflictHandler)
    gg.register_handler(OtherConflictHandler)
    bus = gg.EventBus()
    first = ConflictHandler(name="dup")
    second = OtherConflictHandler(name="dup")

    bus.attach([first.to_dict()], scopes=["a"])
    bus.attach([second.to_dict()], scopes=["a"])

    warnings = _stderr_lines(capsys)
    assert len(warnings) == 1, (
        f"Expected exactly one conflict warning, got {warnings}"
    )
    assert "dup" in warnings[0], "The warning must name the handler"
    assert "class" in warnings[0], (
        "A differing handler class must be reported as such"
    )
    assert type(bus.handlers["dup"]) is ConflictHandler, (
        "The first registration must stay in effect"
    )


# ---------------------------------------------------------------------
# LocalStorageHandler I/O
# ---------------------------------------------------------------------


def test_localstorage_handler_writes_json(tmp_path):
    path = tmp_path / "logs"
    handler = gg.LocalStorageHandler(path=path, name="api.jsonl")
    gg.attach(handler, scopes=["global"])
    log = gg.get_logger("io")
    log.info("testing write")

    gg.finish()

    jsonls = list(path.glob("*.jsonl"))
    assert jsonls, "LocalStorageHandler should create a jsonl file"
    data = jsonls[0].read_text()
    assert "testing write" in data, "Logged message should be in the jsonl file"


# ---------------------------------------------------------------------
# Environment and registry
# ---------------------------------------------------------------------


def test_environment_overrides(monkeypatch):
    monkeypatch.setenv("GOGGLES_ASYNC", "false")

    importlib.reload(gg)
    assert gg.GOGGLES_ASYNC is False, "GOGGLES_ASYNC env override failed"


def test_goggles_socket_env_picked_up_by_transport(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """GOGGLES_SOCKET should be honored by LocalTransport's default path.

    Args:
        monkeypatch: Fixture used to set the GOGGLES_SOCKET env var.
        tmp_path: Fixture used to derive a unique socket path per test.
    """
    sock = str(tmp_path / "gg-api-test.sock")
    monkeypatch.setenv("GOGGLES_SOCKET", sock)
    from goggles._core.transport import _default_socket_path  # noqa: PLC0415

    assert _default_socket_path() == sock, (
        "GOGGLES_SOCKET env should drive the default socket path"
    )


def test_get_handler_class_error():
    with pytest.raises(KeyError):
        gg._get_handler_class("UnknownHandler")


def test_register_handler_and_lookup():
    class MyHandler(DummyHandler):
        pass

    gg.register_handler(MyHandler)
    found = gg._get_handler_class("MyHandler")
    assert found is MyHandler, "Registered handler lookup failed"


# ---------------------------------------------------------------------
# Metrics and composite logger
# ---------------------------------------------------------------------


def test_goggles_logger_scalar_and_push(tmp_path):
    handler = gg.LocalStorageHandler(path=tmp_path, name="metrics.jsonl")
    gg.attach(handler, scopes=["global"])

    tlog = gg.get_logger("train", with_metrics=True)
    tlog.scalar("loss", 0.1, step=1)
    tlog.push({"accuracy": 0.9}, step=1)
    gg.finish()

    data = next(tmp_path.glob("*.jsonl")).read_text()
    assert "loss" in data and "accuracy" in data, (
        "Logged metrics missing from file"
    )


def test_logger_levels_mapping() -> None:
    """Ensure TextLogger.log dispatches correctly by severity."""
    log = gg.get_logger("severity")

    # Should not raise for standard levels
    for lvl in [gg.DEBUG, gg.INFO, gg.WARNING, gg.ERROR, gg.CRITICAL]:
        log.log(lvl, "msg", step=0)


# ---------------------------------------------------------------------
# Finish and import safety
# ---------------------------------------------------------------------


def test_finish_multiple_times_is_safe():
    """finish() should be idempotent and safe to call repeatedly."""
    gg.finish()
    gg.finish()  # second call should not crash


def test_import_adds_nullhandler() -> None:
    """Ensure module attaches a NullHandler at import time."""
    importlib.reload(gg)
    logger = logging.getLogger(gg.__name__)
    assert any(isinstance(h, logging.NullHandler) for h in logger.handlers), (
        "Goggles logger should have a NullHandler by default"
    )
