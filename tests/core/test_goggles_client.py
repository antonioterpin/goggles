from concurrent.futures import Future
from unittest.mock import MagicMock

from goggles import Event
from goggles._core.routing import GogglesClient


def make_future(done: bool) -> Future[None]:
    future: Future[None] = Future()
    if done:
        future.set_result(None)
    return future


def test_pruning_threshold_defaults_to_100():
    client = GogglesClient()
    assert client._pruning_threshold == 100, (
        f"Expected pruning threshold of 100, got {client._pruning_threshold}"
    )


def test_pruning_threshold_can_be_customized():
    client = GogglesClient(pruning_threshold=50)
    assert client._pruning_threshold == 50, (
        f"Expected pruning threshold of 50, got {client._pruning_threshold}"
    )


def test_no_pruning_below_threshold():
    client = GogglesClient(pruning_threshold=10)
    client._client = MagicMock()
    client._client.emit.return_value = make_future(done=False)

    # Fill up to threshold
    for _ in range(10):
        client.futures.append(make_future(done=True))

    # Logic: if len > threshold: prune.
    # Current len=10. 10 > 10 is False.
    # So no pruning expected.

    event = Event("log", "scope", "msg", filepath="test.py", lineno=1)
    client.emit(event)

    # 10 existing + 1 new = 11
    assert len(client.futures) == 11, (
        f"Expected 11 futures, got {len(client.futures)}"
    )


def test_pruning_triggered_above_threshold():
    threshold = 5
    client = GogglesClient(pruning_threshold=threshold)
    client._client = MagicMock()
    client._client.emit.return_value = make_future(done=False)

    # Add 6 finished futures (already above threshold)
    for _ in range(6):
        client.futures.append(make_future(done=True))

    # current len=6. 6 > 5 is True. Pruning triggers.
    event = Event("log", "scope", "msg", filepath="test.py", lineno=1)
    client.emit(event)

    # All 6 were done, so they result in 0.
    # Then we append the new one. Total = 1.
    assert len(client.futures) == 1, (
        f"Expected 1 future, got {len(client.futures)}"
    )


def test_only_finished_futures_are_pruned():
    threshold = 5
    client = GogglesClient(pruning_threshold=threshold)
    client._client = MagicMock()
    client._client.emit.return_value = make_future(done=False)

    # Add 4 pending futures and 2 finished futures (Total 6 > 5)
    pending = [make_future(done=False) for _ in range(4)]
    finished = [make_future(done=True) for _ in range(2)]

    client.futures.extend(pending)
    client.futures.extend(finished)

    assert len(client.futures) == 6, (
        f"Expected 6 futures initially, got {len(client.futures)}"
    )

    # Emit triggers prune
    client.emit(Event("log", "scope", "msg", filepath="test.py", lineno=1))

    # Expect: 2 finished pruned, 4 pending kept, 1 new added = 5
    assert len(client.futures) == 5, (
        f"Expected 5 futures, got {len(client.futures)}"
    )
    # The new one is also pending
    assert all(not f.done() for f in client.futures), (
        "All futures should be pending"
    )
