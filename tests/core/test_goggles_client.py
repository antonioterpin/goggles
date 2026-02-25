from concurrent.futures import Future
from unittest.mock import MagicMock

from goggles import Event
from goggles._core.routing import GogglesClient


def make_future(done: bool) -> Future[None]:
    future: Future[None] = Future()
    if done:
        future.set_result(None)
    return future


def test_futures_property_accesses_portal_client_futures():
    """Test that futures property exposes portal client's internal dict."""
    client = GogglesClient()
    client._client = MagicMock()

    # Simulate portal client's internal futures dict.
    future1 = make_future(done=False)
    future2 = make_future(done=True)
    client._client.futures = {1: future1, 2: future2}

    # Property should return list of values from the dict.
    result = client.futures
    assert len(result) == 2, "Expected 2 futures from the property"
    assert future1 in result, "Expected future1 to be in the result"
    assert future2 in result, "Expected future2 to be in the result"

    client._client.shutdown.return_value = make_future(done=True)
    client.shutdown(timeout=0.01)


def test_emit_returns_future_from_portal_client():
    """Test that emit delegates to portal client and returns its future."""
    client = GogglesClient()
    client._client = MagicMock()

    expected_future = make_future(done=False)
    client._client.emit.return_value = expected_future

    event = Event("log", "scope", "msg", filepath="test.py", lineno=1)
    result = client.emit(event)

    assert result is expected_future, (
        "Expected emit to return the future from portal client"
    )
    client._client.emit.assert_called_once()

    client._client.shutdown.return_value = make_future(done=True)
    client.shutdown()


def test_shutdown_waits_for_futures_with_timeout():
    """Test that shutdown waits on portal client futures with timeout."""
    client = GogglesClient()
    client._client = MagicMock()

    future1 = make_future(done=False)
    future2 = make_future(done=True)

    # Mock futures property to return list from portal dict.
    client._client.futures = {1: future1, 2: future2}

    client._client.shutdown.return_value = make_future(done=True)
    client.shutdown(timeout=0.01)

    # Check that we tried to wait on futures.
    client._client.shutdown.assert_called_once()
