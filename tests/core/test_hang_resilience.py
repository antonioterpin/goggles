"""Tests for hang resilience of Goggles server and client."""

import time
import os
import sys
import subprocess
from collections.abc import Iterator
from typing import Any
import pytest
import goggles
from goggles._core import routing
import portal

# Helper to start server
SERVER_CODE = """
import portal
import time
import os
import sys

def dummy_emit(event):
    return None # ACK

def dummy(*args):
    return None

def main():
    port = int(os.environ.get("GOGGLES_PORT", 3456))
    server = portal.Server(port, name="MockServer")
    server.bind("emit", dummy_emit)
    server.bind("attach", dummy)
    server.bind("detach", dummy)
    server.bind("shutdown", dummy)
    server.start(block=False)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        server.close()

if __name__ == "__main__":
    main()
"""


@pytest.fixture
def free_port() -> int:
    """Allocate and return a free local TCP port.

    Returns:
        int: Free TCP port for test server startup.
    """
    import socket

    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture
def server_process(free_port: int) -> Iterator[subprocess.Popen[Any]]:
    """Start and yield a mock server process bound to `free_port`.

    Args:
        free_port: Port chosen for this test run.

    Yields:
        subprocess.Popen[Any]: Running mock server process.
    """
    env = os.environ.copy()
    env["GOGGLES_PORT"] = str(free_port)
    p = subprocess.Popen([sys.executable, "-c", SERVER_CODE], env=env)
    time.sleep(1)  # Wait for start
    yield p
    p.terminate()
    p.wait()


@pytest.fixture
def goggles_client(free_port: int) -> Iterator[None]:
    """Configure a client singleton connected to the mock server.

    Args:
        free_port: Port where the server process listens.

    Yields:
        None: Control returns to the test with patched client initialization.
    """
    # We must patch the singleton client or create a fresh one
    routing.__singleton_client = None

    os.environ["GOGGLES_PORT"] = str(free_port)
    # Important: Set timeout to something short for test
    os.environ["GOGGLES_TRANSPORT_TIMEOUT"] = "2.0"

    # We also need to patch GogglesClient to use small maxinflight for fast fill
    original_init = routing.GogglesClient.__init__

    def new_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # Re-create internal client with small limits
        self._client = portal.Client(
            addr=f"{os.environ.get('GOGGLES_HOST', 'localhost')}:{free_port}",
            name=kwargs.get("name", "test-client"),
            maxinflight=10,
            max_send_queue=10,
        )

    # cache original
    cached_init = routing.GogglesClient.__init__
    routing.GogglesClient.__init__ = new_init

    yield

    # Teardown
    routing.GogglesClient.__init__ = cached_init
    if routing.__singleton_client:
        routing.__singleton_client.shutdown()
        routing.__singleton_client = None


# Skip this test because it causes deadlocks in portal.ClientSocket teardown
# when using a mock server that is killed abruptly.
@pytest.mark.skip(reason="Teardown hangs due to portal deadlock on connect")
def test_consumer_death_does_not_hang_producer(
    server_process: subprocess.Popen[Any], goggles_client: None
) -> None:
    """
    Simulates consumer death and asserts that producer throws TimeoutError
    instead of hanging forever.

    Args:
        server_process: Running mock server process fixture.
        goggles_client: Fixture that patches client internals for the test.
    """
    # 1. Setup handler
    from goggles._core.integrations.wandb import WandBHandler

    handler = WandBHandler(project="test-project")
    goggles.attach(handler, scopes=["global"])
    log = goggles.get_logger("test")

    # 2. Fill the buffer
    print("Filling buffer...")
    for i in range(20):
        log.info(f"Message {i}")
        time.sleep(0.01)

    print("Buffer filled (hopefully). Killing server...")
    server_process.kill()
    server_process.wait()

    # 3. Continue sending and expect TimeoutError / Exception
    print("Sending post-kill...")

    start = time.time()
    try:
        # This loop should eventually hit the timeout or fail fast
        for i in range(1000):
            log.info(f"Post-kill message {i}")
            # If we don't sleep, we might fill remaining buffer slots faster
            time.sleep(0.01)

            if time.time() - start > 10:
                pytest.fail("Did not raise TimeoutError within 10s (hang detected?)")
    except Exception as e:
        print(f"Successfully caught exception: {type(e).__name__}: {e}")
    else:
        pytest.fail("Loop finished without raising Exception!")

    print("Test body finished.")
