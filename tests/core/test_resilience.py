"""Tests for resilience of Goggles server and client."""

import os
import time
import threading
import random
import pytest
import psutil
from unittest import mock

import goggles
import goggles._core.routing as routing


@pytest.fixture
def free_port():
    """Get a free random port."""
    import socket

    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, free_port):
    """Ensure a clean environment and specific port for each test."""
    port_str = str(free_port)
    monkeypatch.setenv("GOGGLES_PORT", port_str)
    monkeypatch.setenv("GOGGLES_ASYNC", "1")
    monkeypatch.setenv("GOGGLES_ENABLE_EVENT_BUS", "1")

    # Patch the module-level variable since it's already imported
    monkeypatch.setattr(goggles, "GOGGLES_PORT", port_str)
    monkeypatch.setattr(routing, "GOGGLES_PORT", port_str)

    # Reset singletons to force fresh server/client creation
    routing.__singleton_client = None
    routing.__singleton_server = None

    yield

    # Cleanup
    try:
        goggles.finish()
    except Exception as e:
        print(f"Warning: Failed to cleanup goggles: {e}")
        pass
    routing.__singleton_client = None
    routing.__singleton_server = None


@pytest.fixture
def chaos_monitor():
    """Context manager style monitor for chaos testing."""

    class Monitor:
        def __init__(self):
            self.stats = {"injected": 0, "successful": 0}
            self.original_writev = os.writev
            self.errors_to_inject = [BrokenPipeError, ConnectionResetError]
            self.injection_probability = 0.05
            self.stop_event = threading.Event()
            self.worker_counts = [0] * 4

        def chaotic_writev(self, fd, buffers):
            if random.random() < self.injection_probability:
                self.stats["injected"] += 1
                err = random.choice(self.errors_to_inject)
                raise err

            res = self.original_writev(fd, buffers)
            self.stats["successful"] += 1
            return res

        def start_workers(self, log):
            def worker(i):
                while not self.stop_event.is_set():
                    try:
                        log.info(f"worker {i} event", iter=self.worker_counts[i])
                        self.worker_counts[i] += 1
                        if self.worker_counts[i] % 100 == 0:
                            time.sleep(0.01)
                    except Exception:
                        # Intentionally ignore all exceptions to keep workers running
                        pass

            workers = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
            for w in workers:
                w.daemon = True
                w.start()
            return workers

    return Monitor()


@pytest.mark.resilience
@pytest.mark.isolation_group(name="goggles_singleton")
def test_server_resilience_to_broken_pipe(chaos_monitor):
    """
    Verify that Goggles server survives multiple BrokenPipe/ConnectionReset errors
    and stays responsive without entering a livelock (busy-loop).

    Supports long-running verification via GOGGLES_RESILIENCE_DURATION env var.
    """
    # 1. Start Goggles
    log = goggles.get_logger("resilience_test", with_metrics=True)
    log.info("warmup")

    # 2. Robust Server Wait (via Singleton)
    start_wait = time.time()
    while routing.__singleton_server is None:
        if time.time() - start_wait > 5:
            pytest.fail(
                "Timeout waiting for Goggles server to start. Check network/port binding."
            )
        time.sleep(0.1)

    server = routing.__singleton_server
    assert server.running, "Server should be marked as running"

    # Check underlying thread health as well
    assert server.loop.running, "Server loop thread should be alive"

    # 3. Resource Baselines
    proc = psutil.Process()
    initial_rss = proc.memory_info().rss
    initial_fds = proc.num_fds()
    print(f"\n[Baseline] RSS: {initial_rss/1024/1024:.2f} MB | FDs: {initial_fds}")

    # 4. Start workers
    workers = chaos_monitor.start_workers(log)

    # 5. Inject chaos
    duration = float(os.getenv("GOGGLES_RESILIENCE_DURATION", "5"))
    print(f"Running resilience test for {duration} seconds...")

    start_time = time.time()

    with mock.patch("os.writev", side_effect=chaos_monitor.chaotic_writev):
        while time.time() - start_time < duration:
            # Liveness check
            if not server.loop.running:
                pytest.fail("CRITICAL: Server loop thread DIED during test!")

            # Simple CPU check could go here, but is flaky in CI.
            # We rely on 'successful' writes in stats to prove we aren't stuck.
            time.sleep(1)

            # Progress status
            current_rss = proc.memory_info().rss
            client = routing.__singleton_client
            inflight = len(client.futures) if client else 0
            print(
                f" [Debug] RSS: {current_rss/1024/1024:.1f}MB | Client Futures: {inflight} | Successful: {chaos_monitor.stats['successful']}"
            )
            if current_rss > initial_rss + 100 * 1024 * 1024:
                # 100MB growth is alarming (though python GC is lazy)
                print(
                    f"WARNING: High memory growth! RSS: {current_rss/1024/1024:.2f} MB"
                )

    chaos_monitor.stop_event.set()
    for w in workers:
        w.join(timeout=2)

    # 6. Final verification
    assert server.loop.running, "Server should still be alive after chaos"

    # Stats
    injected = chaos_monitor.stats["injected"]
    successful = chaos_monitor.stats["successful"]
    print(f"[Result] Injected Errors: {injected}")
    print(f"[Result] Successful Writes: {successful}")

    assert injected > 0, "Test failed to inject any errors (is probability too low?)"
    assert successful > 100, "Too few successful writes - server might be deadlocked"

    # Resource Checks
    # GC to get a fair reading
    import gc

    gc.collect()

    final_rss = proc.memory_info().rss
    final_fds = proc.num_fds()
    print(f"[Final] RSS: {final_rss/1024/1024:.2f} MB | FDs: {final_fds}")

    # Heuristics for leaks (only enforce on longer runs to avoid noise)
    if duration >= 10:
        # File descriptors shouldn't grow unbounded.
        # Baseline file descriptors + some constant overhead for threads/sockets.
        # If it grew by > 50, we likely have a socket leak.
        fd_growth = final_fds - initial_fds
        if fd_growth > 50:
            pytest.fail(f"Potential FD leak detected! Growth: {fd_growth}")

        # Memory is harder, but let's warn
        mem_growth_mb = (final_rss - initial_rss) / 1024 / 1024
        if mem_growth_mb > 50:
            print(f"WARNING: Significant memory growth: {mem_growth_mb:.2f} MB")
