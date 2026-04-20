"""Hydra-configured benchmark for Goggles' hot-path logging latency.

The benchmark measures per-call wall-clock latency for the logger entry
points (`scalar`, `image`, `video`, `info`, `debug`, `print`) and reports
percentiles across one or more runs.

Typical invocations
-------------------

Default run (scalar, 10k calls, max throughput)::

    uv run python examples/105_benchmark.py

Named presets (see `conf/preset/`)::

    uv run python examples/105_benchmark.py +preset=scalar_1khz
    uv run python examples/105_benchmark.py +preset=scalar_10khz
    uv run python examples/105_benchmark.py +preset=scalar_max
    uv run python examples/105_benchmark.py +preset=image_sweep
    uv run python examples/105_benchmark.py +preset=video_sweep

Ad-hoc overrides (Hydra CLI)::

    uv run python examples/105_benchmark.py \
        log_type=scalar num_logs=20000 delay=0.0 wandb.enabled=false

Image resolution sweep in a single run (fast iteration)::

    uv run python examples/105_benchmark.py \
        +preset=image_sweep image.sizes='[32,128,512]' num_logs=200

Hydra multirun (one process per sweep point)::

    uv run python examples/105_benchmark.py --multirun \
        log_type=scalar delay=0.001,0.0001,0.0

W&B online vs offline
---------------------

`logger.scalar(...)` hot-path timings are the same in both modes: the
producer thread only does `queue.put(event)`, and the W&B upload runs on
W&B's own background thread. So `WANDB_MODE=offline` is representative
of the *producer cost* a user sees in a real run.

When running this benchmark (and, more generally, when to choose offline):

* **Reproducible latency numbers.** Online mode folds in network RTT,
  wandb.ai backpressure, and auth round-trips. None of those affect our
  `logger.*` measurement, but they blur your wall-clock total time and
  make runs non-deterministic across machines/regions. For a clean
  measurement of the producer hot path, use offline.
* **Airgapped / HPC nodes.** Compute nodes without outbound Internet
  (cluster workers, slurm jobs, CI runners behind a firewall). Offline
  writes locally; you ship the run out later.
* **Flaky network.** If the link drops, online runs can stall the wandb
  sync thread; offline runs never care.
* **Untrusted environment.** You want to review data before it leaves
  the host.

Empirically, at 1 kHz and 10 kHz scalar throughput the handler drain
thread keeps up (queue depth returns to 0 the moment the producer loop
finishes), so running online does not create a backlog for scalars --
it just adds background traffic.

Offline runs are written to `./wandb/offline-run-<ts>-<id>/` and can be
uploaded later with::

    wandb sync wandb/               # all offline runs
    wandb sync wandb/offline-run-ID # a specific run
"""

from __future__ import annotations

import datetime
import os
import sys
import threading
import time
from statistics import mean, median, quantiles, stdev
from typing import Any, ClassVar, cast

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

import goggles as gg
from goggles import Event, Kind

_logger = gg.get_logger(
    "goggles.benchmark",
    scope="goggles.benchmark",
    with_metrics=True,
)


class _DeliveryCounter:
    """Handler that just counts events delivered to the host.

    Used to verify end-to-end reliability: the producer emits N events,
    then ``finish()`` blocks until everything drains; this handler's
    count must equal N. Prior to the shutdown fix, bulk video benchmarks
    were losing >90 % of events here.

    Attributes:
        name: Handler identifier used by the bus's dedup logic.
        capabilities: Event kinds this handler claims to handle.
    """

    name = "goggles.benchmark.counter"
    capabilities: ClassVar[frozenset[Kind]] = frozenset(
        {
            "log",
            "metric",
            "image",
            "video",
            "artifact",
            "histogram",
            "vector",
            "vector_field",
        }
    )

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._count_by_kind: dict[str, int] = {}

    def can_handle(self, kind: Kind) -> bool:
        del kind
        return True

    def handle(self, event: Event) -> None:
        with self._lock:
            self._count_by_kind[event.kind] = (
                self._count_by_kind.get(event.kind, 0) + 1
            )

    def open(self) -> None: ...

    def close(self) -> None: ...

    @property
    def totals(self) -> dict[str, int]:
        with self._lock:
            return dict(self._count_by_kind)

    def to_dict(self) -> dict:
        return {"cls": "_DeliveryCounter", "data": {}}

    @classmethod
    def from_dict(cls, serialized: dict) -> _DeliveryCounter:
        del serialized
        return cls()


def _run_one(
    log_type: str,
    num_logs: int,
    delay: float,
    *,
    image_size: int,
    video_size: int,
    video_frames: int,
    label: str,
    verbose: bool,
    step_offset: int = 0,
) -> list[float]:
    """Run a single benchmark pass and return per-call timings in ms.

    Args:
        log_type: "scalar", "image", "video", "info", "debug", or "print".
        num_logs: Number of logging calls.
        delay: Seconds to sleep between calls (``0`` for max throughput).
        image_size: Edge length (px) of the generated square RGB image.
        video_size: Edge length (px) of each video frame.
        video_frames: Frames per video sample.
        label: Tag used in logs + as the metric name suffix.
        verbose: If True, log periodic progress lines.
        step_offset: Base step for this run. All logger calls log at
            ``step_offset + idx`` so sweep points that share a wandb run
            (same scope) don't collide on step 0..N-1 and trigger
            "step X < current step Y" rejections.

    Returns:
        A list of per-call wall-clock times, in milliseconds.

    Raises:
        ValueError: If ``log_type`` is not one of the supported values.
    """
    times: list[float] = []
    for step_in_run in range(num_logs):
        step = step_offset + step_in_run
        if log_type == "scalar":
            value: Any = np.random.randn()
        elif log_type == "image":
            value = np.random.randint(
                0,
                256,
                (image_size, image_size, 3),
                dtype=np.uint8,
            )
        elif log_type == "video":
            value = np.random.randint(
                0,
                256,
                (video_frames, 3, video_size, video_size),
                dtype=np.uint8,
            )
        elif log_type in ("info", "debug", "print"):
            value = f"{log_type} message step {step}"
        else:
            raise ValueError(f"Unsupported log_type: {log_type}")

        start = time.perf_counter()
        if log_type == "scalar":
            _logger.scalar(
                name=f"test_metric_{label}",
                value=float(value),
                step=step,
            )
        elif log_type == "image":
            _logger.image(
                name=f"test_image_{label}",
                image=value,
                step=step,
            )
        elif log_type == "video":
            _logger.video(
                name=f"test_video_{label}",
                video=value,
                step=step,
            )
        elif log_type == "info":
            _logger.info(value)
        elif log_type == "debug":
            _logger.debug(value)
        elif log_type == "print":
            print(value)

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        times.append(elapsed_ms)

        if delay > 0:
            time.sleep(delay)

        if verbose and step_in_run % max(1, num_logs // 10) == 0:
            _logger.info(
                f"[{label}] step {step_in_run}/{num_logs} - {elapsed_ms:.6f} ms"
            )

    return times


def _report(times: list[float], label: str) -> None:
    """Log min/median/p99/max/mean/stdev for a single benchmark run.

    Args:
        times: Per-call wall-clock times in milliseconds.
        label: Tag identifying the run (log_type, resolution, etc.).
    """
    if not times:
        return

    p99 = (
        quantiles(times, n=100, method="inclusive")[-1]
        if len(times) > 1
        else times[0]
    )
    p999 = (
        quantiles(times, n=1000, method="inclusive")[-1]
        if len(times) >= 1000
        else max(times)
    )
    _logger.info(f"=== [{label}] Statistics over {len(times)} calls ===")
    _logger.info(f"  Min    : {min(times):.6f} ms")
    _logger.info(f"  Median : {median(times):.6f} ms")
    _logger.info(f"  Mean   : {mean(times):.6f} ms")
    _logger.info(f"  p99    : {p99:.6f} ms")
    _logger.info(f"  p99.9  : {p999:.6f} ms")
    _logger.info(f"  Max    : {max(times):.6f} ms")
    _logger.info(
        f"  Std    : {(stdev(times) if len(times) > 1 else 0.0):.6f} ms"
    )
    _logger.info(f"  Total  : {sum(times):.6f} ms ({sum(times) / 1000:.3f} s)")


def _build_runs(cfg: DictConfig) -> list[tuple[str, dict]]:
    """Build the (label, per-run kwargs) list from the Hydra config.

    For image/video log types a list of sizes expands into multiple runs;
    everything else is a single run whose label is just the log_type.

    Args:
        cfg: Resolved Hydra configuration.

    Returns:
        Sequence of ``(label, kwargs)`` passed to :func:`_run_one`.
    """
    if cfg.log_type == "image":
        sizes = list(cfg.image.sizes) if cfg.image.sizes else [cfg.image.size]
        return [
            (
                f"image_{size}px",
                {
                    "log_type": "image",
                    "image_size": int(size),
                    "video_size": int(cfg.video.size),
                    "video_frames": int(cfg.video.frames),
                },
            )
            for size in sizes
        ]
    if cfg.log_type == "video":
        sizes = list(cfg.video.sizes) if cfg.video.sizes else [cfg.video.size]
        return [
            (
                f"video_{size}px_{cfg.video.frames}f",
                {
                    "log_type": "video",
                    "image_size": int(cfg.image.size),
                    "video_size": int(size),
                    "video_frames": int(cfg.video.frames),
                },
            )
            for size in sizes
        ]
    return [
        (
            cfg.log_type,
            {
                "log_type": cfg.log_type,
                "image_size": int(cfg.image.size),
                "video_size": int(cfg.video.size),
                "video_frames": int(cfg.video.frames),
            },
        )
    ]


def _snapshot_wandb_runs() -> list[tuple[str, str]]:
    """Capture ``(scope, "entity/project/run_id")`` for each live wandb run.

    Call this *before* ``gg.finish()`` runs — the handler's ``close()``
    clears its internal run map and resets the wandb globals, after which
    the identifiers are no longer accessible without going through the API.

    Returns:
        A list of ``(scope, run_path)`` pairs; empty if the wandb handler
        is not attached or its runs can't be introspected.
    """
    transport_any: Any = gg.get_bus()
    handler: Any = None
    bus_any: Any = getattr(transport_any, "_bus", None)
    if bus_any is not None:
        handler = bus_any.handlers.get("wandb")
    if handler is None:
        return []
    snapshots: list[tuple[str, str]] = []
    runs: Any = getattr(handler, "_runs", {})
    for scope, run in list(runs.items()):
        try:
            entity = run.entity
            project = run.project
            run_id = run.id
        except AttributeError:
            continue
        snapshots.append((scope, f"{entity}/{project}/{run_id}"))
    return snapshots


def _verify_wandb_history(
    runs: list[tuple[str, str]],
    expected_rows: int,
    poll_timeout: float = 60.0,
    poll_interval: float = 2.0,
) -> None:
    """Poll wandb for each run's history and report delivered-row counts.

    Prints a single line per run to stderr. Skipped silently in offline
    mode (``WANDB_MODE=offline``) because ``wandb.Api`` only reads the
    backend, not the local ``wandb/offline-run-*`` tree — run
    ``wandb sync`` first if you want offline runs audited.

    Args:
        runs: ``(scope, "entity/project/run_id")`` pairs from
            :func:`_snapshot_wandb_runs`.
        expected_rows: Number of history rows each run should contain.
        poll_timeout: Max seconds to wait for the wandb backend to
            reflect every logged row. wandb's sync thread is async and
            can lag the producer by a few seconds even after
            ``run.finish()``.
        poll_interval: Seconds between polls.
    """
    if not runs:
        return
    if os.environ.get("WANDB_MODE", "").lower() == "offline":
        print(
            "wandb verification skipped (offline mode; run `wandb sync` "
            "and query manually if you need delivered-row counts).",
            file=sys.stderr,
        )
        return
    try:
        import wandb  # noqa: PLC0415
    except ImportError:
        return

    try:
        api = wandb.Api()
    except Exception as exc:
        print(f"wandb verification skipped: {exc}", file=sys.stderr)
        return

    for scope, path in runs:
        deadline = time.monotonic() + poll_timeout
        last_count = -1
        while time.monotonic() < deadline:
            try:
                run = api.run(path)
                # ``scan_history`` is unsampled and reflects the true
                # server-side row count, unlike ``run.history`` which
                # caps at 500 samples by default.
                count = sum(1 for _ in run.scan_history())
            except Exception as exc:
                print(
                    f"wandb run {path} (scope={scope}): query failed: {exc}",
                    file=sys.stderr,
                )
                break
            if count >= expected_rows:
                last_count = count
                break
            last_count = count
            time.sleep(poll_interval)
        verdict = "OK" if last_count >= expected_rows else "INCOMPLETE"
        print(
            f"wandb run {path} (scope={scope}): "
            f"{last_count}/{expected_rows} history rows [{verdict}]",
            file=sys.stderr,
        )


def _kind_for_log_type(log_type: str) -> str:
    """Map a benchmark ``log_type`` to the event ``kind`` it produces.

    Args:
        log_type: Value from the Hydra config's ``log_type`` field.

    Returns:
        The corresponding event kind used in ``Event.kind``.
    """
    if log_type in {"info", "debug", "print"}:
        return "log"
    if log_type == "scalar":
        return "metric"
    return log_type


def _summary(all_results: list[tuple[str, list[float]]]) -> None:
    """Emit a concise across-run table with min/median/p99/max.

    Args:
        all_results: ``[(label, timings_ms), ...]`` for every run.
    """
    if not all_results:
        return
    _logger.info("=== Summary ===")
    _logger.info(
        f"{'run':<22} {'min(ms)':>10} {'med(ms)':>10} "
        f"{'p99(ms)':>10} {'max(ms)':>10}"
    )
    for label, timings in all_results:
        if not timings:
            continue
        p99 = (
            quantiles(timings, n=100, method="inclusive")[-1]
            if len(timings) > 1
            else timings[0]
        )
        _logger.info(
            f"{label:<22} {min(timings):>10.3f} "
            f"{median(timings):>10.3f} {p99:>10.3f} "
            f"{max(timings):>10.3f}"
        )


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entry point: attach handlers, run benchmark(s), report.

    Args:
        cfg: Resolved config, composed from ``conf/config.yaml`` plus any
            selected presets and CLI overrides.
    """
    gg.attach(
        gg.ConsoleHandler(
            name="goggles.benchmark.console",
            level=gg.INFO,
        ),
        scopes=["goggles.benchmark"],
    )

    gg.register_handler(_DeliveryCounter)
    gg.attach(_DeliveryCounter(), scopes=["goggles.benchmark"])
    # ``gg.attach`` deserializes the handler via ``from_dict`` so the
    # instance owned by the bus is not the one we constructed here; pull
    # out the canonical copy the drain thread will actually dispatch to.
    transport_any: Any = gg.get_bus()
    counter = cast(
        _DeliveryCounter,
        transport_any._bus.handlers["goggles.benchmark.counter"],
    )

    if cfg.wandb.enabled:
        # For reproducible numbers on CI / HPC / airgapped hosts, run with
        # `WANDB_MODE=offline` and sync the run afterwards with
        # `wandb sync wandb/offline-run-<id>`. See the module docstring
        # for the full offline-vs-online discussion.
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        gg.attach(
            gg.WandBHandler(
                project=cfg.wandb.project,
                config={
                    "experiment": cfg.wandb.experiment,
                    "num_logs": f"{cfg.num_logs}",
                    "log_type": cfg.log_type,
                    "run": timestamp,
                    "delay": str(cfg.delay),
                },
            ),
            scopes=["goggles.benchmark"],
        )

    runs = _build_runs(cfg)
    _logger.info("Starting logger benchmark.")
    _logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    _logger.info(f"Runs: {[label for label, _ in runs]}")

    all_results: list[tuple[str, list[float]]] = []
    # Global monotonic step counter. Every sweep point + the replay phase
    # advances it, so wandb (which tracks one global step per run) never
    # sees a regression. Prior versions let each sweep reuse step=0..N-1,
    # causing "Tried to log to step 0 < current step 499" rejections.
    next_step = 0
    try:
        for label, opts in runs:
            _logger.info(f"--- Running [{label}] ---")
            timings = _run_one(
                num_logs=int(cfg.num_logs),
                delay=float(cfg.delay),
                label=label,
                verbose=bool(cfg.verbose),
                step_offset=next_step,
                **opts,
            )
            next_step += int(cfg.num_logs)
            all_results.append((label, timings))
            _report(timings, label)
    finally:
        _summary(all_results)

        expected_wandb_rows = 0
        if cfg.wandb.enabled:
            # Replay timings into wandb. Continues the global step counter
            # past all main-loop runs so the wandb monotonicity check
            # keeps accepting values.
            for label, timings in all_results:
                for idx, log_time in enumerate(timings):
                    _logger.scalar(
                        name=f"logger_call_time_ms_{label}",
                        value=log_time,
                        step=next_step + idx,
                    )
                next_step += len(timings)
            expected_wandb_rows = next_step

        # Capture run identifiers while the wandb handler is still live;
        # ``gg.finish()`` below will call its ``close()`` which clears
        # the handler's internal run map.
        wandb_runs = _snapshot_wandb_runs() if cfg.wandb.enabled else []

        # Wait indefinitely for drain (``timeout=0`` means no deadline
        # inside ``gg.finish``). When wandb is attached, wandb.Image for
        # large payloads can cost tens of ms per call, and a bounded
        # timeout silently drops the tail of the drain queue — that was
        # the source of the "image=2063/3000 (31.2% dropped)" reports
        # from the image_sweep preset.
        print(
            "Draining... (this waits for wandb to finish uploading; "
            "set GOGGLES_SHUTDOWN_TIMEOUT to bound it).",
            file=sys.stderr,
        )
        gg.finish(timeout=0)

        # Delivery-count check: every enqueued event must have reached a
        # handler. This catches transport-level drops (e.g. the shutdown
        # race that used to lose >90 % of video frames). Print directly
        # to stderr because the transport is already shut down and any
        # _logger.* call after finish() would be a silent no-op.
        totals = counter.totals
        payload_kind = _kind_for_log_type(cfg.log_type)
        expected = len(runs) * int(cfg.num_logs)
        actual = totals.get(payload_kind, 0)
        if actual < expected:
            dropped_pct = 100.0 * (expected - actual) / max(1, expected)
            print(
                f"DELIVERY LOSS: {payload_kind}={actual}/{expected} "
                f"({dropped_pct:.1f}% dropped). Totals: {totals}",
                file=sys.stderr,
            )
        else:
            print(
                f"All {actual} {payload_kind} events delivered end-to-end. "
                f"Totals: {totals}",
                file=sys.stderr,
            )

        # Second line of defence: even if the handler saw every event,
        # wandb's own async pipeline could have dropped some. Cross-check
        # the backend's row count against what we emitted.
        if wandb_runs:
            _verify_wandb_history(wandb_runs, expected_rows=expected_wandb_rows)


if __name__ == "__main__":
    main()
