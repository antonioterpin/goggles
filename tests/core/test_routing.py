# tests/core/test_routing.py
import types
import socket
import pytest

import goggles._core.routing as routing


@pytest.fixture(autouse=True)
def reset_singletons():
    routing.__singleton_client = None
    routing.__singleton_server = None
    yield
    routing.__singleton_client = None
    routing.__singleton_server = None


def test_i_am_host_returns_true_for_localhost(monkeypatch):
    monkeypatch.setattr(routing, "GOGGLES_HOST", "localhost")
    assert routing.__i_am_host() is True


def test_i_am_host_true_for_local_ip(monkeypatch):
    monkeypatch.setattr(routing, "GOGGLES_HOST", "192.168.0.5")
    monkeypatch.setattr(socket, "gethostname", lambda: "fakehost")
    monkeypatch.setattr(socket, "gethostbyname", lambda _: "192.168.0.5")
    monkeypatch.setattr(routing.netifaces, "interfaces", lambda: ["eth0"])
    monkeypatch.setattr(
        routing.netifaces,
        "ifaddresses",
        lambda _: {routing.netifaces.AF_INET: [{"addr": "192.168.0.5"}]},
    )
    assert routing.__i_am_host() is True


def test_i_am_host_false_when_no_match(monkeypatch):
    monkeypatch.setattr(routing, "GOGGLES_HOST", "10.0.0.1")
    monkeypatch.setattr(socket, "gethostname", lambda: "fakehost")
    monkeypatch.setattr(socket, "gethostbyname", lambda _: "192.168.0.3")
    monkeypatch.setattr(routing.netifaces, "interfaces", lambda: ["eth0"])
    monkeypatch.setattr(
        routing.netifaces,
        "ifaddresses",
        lambda _: {routing.netifaces.AF_INET: [{"addr": "192.168.0.4"}]},
    )
    assert routing.__i_am_host() is False


@pytest.mark.parametrize("code,expected", [(0, True), (111, False)])
def test_is_port_in_use(monkeypatch, code, expected):
    class DummySocket:
        def __init__(self, ret_code: int):
            self._ret = ret_code
            self.timeout = None

        def settimeout(self, t):
            self.timeout = t  # just store, no-op

        def connect_ex(self, addr):
            # Simulate connect_ex return code (0 => port in use)
            return self._ret

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False  # don’t suppress exceptions

    # Ensure each call returns a fresh, valid context manager
    monkeypatch.setattr(
        routing.socket,
        "socket",
        lambda *a, **kw: DummySocket(code),
    )

    result = routing.__is_port_in_use("localhost", 1234)
    assert result is expected, f"expected {expected}, got {result}"


def test_is_port_in_use_handles_exception(monkeypatch):
    monkeypatch.setattr(
        routing.socket,
        "socket",
        lambda *a, **kw: (_ for _ in ()).throw(Exception("fail")),
    )
    assert routing.__is_port_in_use("localhost", 9999) is False


def test_get_bus_starts_server_if_host(monkeypatch):
    # Minimal fake EventBus with expected API
    fake_eventbus = types.SimpleNamespace(
        attach=lambda *a, **kw: None,
        detach=lambda *a, **kw: None,
        emit=lambda *a, **kw: None,
        shutdown=lambda *a, **kw: None,
    )
    fake_server = types.SimpleNamespace(
        bind=lambda *a, **kw: None, start=lambda **kw: None
    )

    monkeypatch.setattr(routing, "EventBus", lambda: fake_eventbus)
    monkeypatch.setattr(
        routing,
        "portal",
        types.SimpleNamespace(
            Server=lambda *a, **kw: fake_server, Client=lambda *a, **kw: "CLIENT"
        ),
    )
    monkeypatch.setattr(routing, "__i_am_host", lambda: True)
    monkeypatch.setattr(routing, "__is_port_in_use", lambda *a, **kw: False)

    result = routing.get_bus()
    assert isinstance(result, routing.GogglesClient)
    assert result._client == "CLIENT"
    assert routing.__singleton_client == result
    assert routing.__singleton_server == fake_server


def test_get_bus_reuses_existing_client(monkeypatch):
    routing.__singleton_client = "EXISTING"
    monkeypatch.setattr(routing, "__i_am_host", lambda: False)
    result = routing.get_bus()
    assert result == "EXISTING"


def test_get_bus_fallback_on_server_creation_failure(monkeypatch):
    class FailingServer:
        def __init__(self, *a, **kw):
            raise OSError("fail")

    monkeypatch.setattr(routing, "EventBus", lambda: object())
    monkeypatch.setattr(
        routing,
        "portal",
        types.SimpleNamespace(Server=FailingServer, Client=lambda *a, **kw: "CLIENT"),
    )
    monkeypatch.setattr(routing, "__i_am_host", lambda: True)
    monkeypatch.setattr(routing, "__is_port_in_use", lambda *a, **kw: False)

    result = routing.get_bus()
    assert isinstance(result, routing.GogglesClient)
    assert result._client == "CLIENT"
