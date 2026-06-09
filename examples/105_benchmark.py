"""Hydra-configured benchmark for Goggles' hot-path logging latency.

The benchmark measures per-call wall-clock latency for the logger entry
points (`scalar`, `image`, `video`, `info`, `debug`, `print`) and reports
percentiles across one or more runs.

Typical invocations
-------------------

Default run (scalar, 10k calls, max throughput)::

    uv run python examples/105_benchmark.py

Named presets (see `conf/preset/`)::

    uv run python examples/105_benchmark.py +preset=scalar_30hz
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

Run every preset in one multirun (W&B offline)::

    WANDB_MODE=offline uv run python examples/105_benchmark.py --multirun \
        +preset=scalar_30hz,scalar_1khz,scalar_10khz,scalar_max,scalar_long,image_sweep,video_sweep,video_long

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
import json
import multiprocessing as mp
import os
import sys
import tempfile
import time
from statistics import median, quantiles, stdev
from typing import Any, cast

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

import goggles as gg

# The delivery-counter handler lives in a sibling module so the goggles
# dedicated host (a separate subprocess by default) can import it via
# GOGGLES_HOST_IMPORTS and reconstruct it there; the host inherits PYTHONPATH.
_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
if _EXAMPLES_DIR not in sys.path:
    sys.path.insert(0, _EXAMPLES_DIR)
os.environ["GOGGLES_HOST_IMPORTS"] = "_benchmark_handlers"
os.environ["PYTHONPATH"] = (
    _EXAMPLES_DIR + os.pathsep + os.environ.get("PYTHONPATH", "")
)

from _benchmark_handlers import DeliveryCounter  # noqa: E402

# ``_logger`` is populated inside the per-preset subprocess (see
# ``_run_benchmark``). Creating it at module import would make the parent
# Hydra process bind the goggles socket, which subsequently spawned
# subprocesses would then connect to as clients -- defeating the isolation
# that running each sweep point in its own process is meant to provide.
_logger: gg.GogglesLogger = cast(Any, None)


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
) -> np.ndarray:
    """Run a single benchmark pass and return per-call timings in ms.

    All payload allocations (random scalars, image/video buffers, message
    strings) happen up front, so the timed region covers only the
    ``logger.*`` call itself, not numpy or string allocation noise. The
    timings array is also preallocated to avoid Python list-grow churn at
    high logging frequencies.

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
        A 1-D ``float64`` array of per-call wall-clock times, in
        milliseconds, length ``num_logs``.

    Raises:
        ValueError: If ``log_type`` is not one of the supported values.
    """
    times = np.empty(num_logs, dtype=np.float64)

    # Pre-generate payloads outside the timed region. For image/video we
    # reuse a single buffer across calls: the transport copies the bytes
    # on its way out, so reusing the source array costs nothing in
    # realism but removes the np.random.randint allocation from the hot
    # path (which is exactly the call we're trying to measure).
    scalar_values: np.ndarray | None = None
    image_value: np.ndarray | None = None
    video_value: np.ndarray | None = None
    text_values: list[str] | None = None
    metric_name = f"test_metric_{label}"
    image_name = f"test_image_{label}"
    video_name = f"test_video_{label}"

    if log_type == "scalar":
        scalar_values = np.random.randn(num_logs)
    elif log_type == "image":
        image_value = np.random.randint(
            0, 256, (image_size, image_size, 3), dtype=np.uint8
        )
    elif log_type == "video":
        video_value = np.random.randint(
            0,
            256,
            (video_frames, 3, video_size, video_size),
            dtype=np.uint8,
        )
    elif log_type in ("info", "debug", "print"):
        text_values = [
            f"{log_type} message step {step_offset + i}"
            for i in range(num_logs)
        ]
    else:
        raise ValueError(f"Unsupported log_type: {log_type}")

    def _emit(step_in_run: int, step: int) -> None:
        if log_type == "scalar":
            assert scalar_values is not None
            _logger.scalar(
                name=metric_name,
                value=float(scalar_values[step_in_run]),
                step=step,
            )
        elif log_type == "image":
            assert image_value is not None
            _logger.image(name=image_name, image=image_value, step=step)
        elif log_type == "video":
            assert video_value is not None
            _logger.video(name=video_name, video=video_value, step=step)
        elif log_type == "info":
            assert text_values is not None
            _logger.info(text_values[step_in_run])
        elif log_type == "debug":
            assert text_values is not None
            _logger.debug(text_values[step_in_run])
        elif log_type == "print":
            assert text_values is not None
            print(text_values[step_in_run])

    for step_in_run in range(num_logs):
        step = step_offset + step_in_run

        start = time.perf_counter()
        _emit(step_in_run, step)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        times[step_in_run] = elapsed_ms

        if delay > 0:
            time.sleep(delay)

        if verbose and step_in_run % max(1, num_logs // 10) == 0:
            _logger.info(
                f"[{label}] step {step_in_run}/{num_logs} - {elapsed_ms:.6f} ms"
            )

    return times


def _report(times: np.ndarray, label: str) -> None:
    """Log min/median/p99/max/mean/stdev for a single benchmark run.

    Args:
        times: Per-call wall-clock times in milliseconds.
        label: Tag identifying the run (log_type, resolution, etc.).
    """
    if times.size == 0:
        return

    times_list = times.tolist()
    p99 = (
        quantiles(times_list, n=100, method="inclusive")[-1]
        if times.size > 1
        else times_list[0]
    )
    p999 = (
        quantiles(times_list, n=1000, method="inclusive")[-1]
        if times.size >= 1000
        else float(times.max())
    )
    total = float(times.sum())
    _logger.info(f"=== [{label}] Statistics over {times.size} calls ===")
    _logger.info(f"  Min    : {float(times.min()):.6f} ms")
    _logger.info(f"  Median : {median(times_list):.6f} ms")
    _logger.info(f"  Mean   : {float(times.mean()):.6f} ms")
    _logger.info(f"  p99    : {p99:.6f} ms")
    _logger.info(f"  p99.9  : {p999:.6f} ms")
    _logger.info(f"  Max    : {float(times.max()):.6f} ms")
    _logger.info(
        f"  Std    : {(stdev(times_list) if times.size > 1 else 0.0):.6f} ms"
    )
    _logger.info(f"  Total  : {total:.6f} ms ({total / 1000:.3f} s)")


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


def _summary(all_results: list[tuple[str, np.ndarray]]) -> None:
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
        if timings.size == 0:
            continue
        timings_list = timings.tolist()
        p99 = (
            quantiles(timings_list, n=100, method="inclusive")[-1]
            if timings.size > 1
            else timings_list[0]
        )
        _logger.info(
            f"{label:<22} {float(timings.min()):>10.3f} "
            f"{median(timings_list):>10.3f} {p99:>10.3f} "
            f"{float(timings.max()):>10.3f}"
        )


def _read_counter_totals(path: str) -> dict[str, int]:
    """Read (and remove) the delivery counter's totals file.

    Args:
        path: File the counter wrote its per-kind totals to on ``close()``.

    Returns:
        The per-kind totals, or an empty mapping if the file is unreadable.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {}
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _run_benchmark(cfg: DictConfig) -> None:
    """Execute the benchmark end-to-end for a single resolved config.

    Designed to run inside a fresh subprocess spawned by :func:`main`.
    Each Hydra sweep point gets its own Python interpreter, so the
    goggles transport, wandb handler, and any module-level state are
    constructed from scratch and torn down cleanly at process exit.
    A shared interpreter would let one preset's shutdown transport
    leak into the next preset and silently drop its events.

    Args:
        cfg: Resolved config for this sweep point.
    """
    # Rebind the module-level placeholder so helpers (``_run_one``,
    # ``_report``, ``_summary``) that reference ``_logger`` at call time
    # pick up the live instance for this subprocess.
    global _logger  # noqa: PLW0603
    _logger = gg.get_logger(
        "goggles.benchmark",
        scope="goggles.benchmark",
        with_metrics=True,
    )

    gg.attach(
        gg.ConsoleHandler(
            name="goggles.benchmark.console",
            level=gg.INFO,
        ),
        scopes=["goggles.benchmark"],
    )

    # The counter handler runs in the host -- a separate process by default --
    # so we cannot read its state in memory. It writes its per-kind totals to
    # this file on close(); we read them back after gg.finish() drains.
    counter_out = os.path.join(
        tempfile.gettempdir(), f"gg-bench-count-{os.getpid()}.json"
    )
    gg.attach(
        DeliveryCounter(out_path=counter_out), scopes=["goggles.benchmark"]
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

    all_results: list[tuple[str, np.ndarray]] = []
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
                for idx, log_time in enumerate(timings.tolist()):
                    _logger.scalar(
                        name=f"logger_call_time_ms_{label}",
                        value=log_time,
                        step=next_step + idx,
                    )
                next_step += int(timings.size)
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

        # Delivery-count check: every enqueued event must have reached
        # a handler. Catches transport-level drops (e.g. shutdown races
        # in the video path). The counter ran in the host and wrote its
        # totals on close(); read them back now that finish() has drained.
        # Print directly to stderr because the transport is already shut
        # down and any _logger.* call after finish() is a silent no-op.
        totals = _read_counter_totals(counter_out)
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


def _subprocess_entry(cfg_dict: Any, cwd: str) -> None:
    """Subprocess target: rebuild cfg and run one benchmark sweep point.

    Args:
        cfg_dict: Plain-Python copy of the Hydra config (picklable).
        cwd: Hydra's per-run working directory; inherited so relative
            paths in the config resolve consistently with the parent.
    """
    os.chdir(cwd)
    cfg = cast(DictConfig, OmegaConf.create(cfg_dict))
    _run_benchmark(cfg)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entry point: run each sweep point in its own subprocess.

    We spawn via ``multiprocessing`` with the ``spawn`` start method so
    every preset (including every point in a Hydra ``--multirun``) gets
    a fresh Python interpreter. The previous in-process loop left the
    goggles transport shut down between sweep points, causing 100%
    delivery loss from the second preset onward. Running in a subprocess
    guarantees that ``gg.finish()`` tears everything down cleanly via
    process exit, and the next preset starts from a clean slate.

    Args:
        cfg: Resolved config, composed from ``conf/config.yaml`` plus any
            selected presets and CLI overrides.

    Raises:
        KeyboardInterrupt: Re-raised after terminating the child process.
        SystemExit: Raised with the child's non-zero exit code so Hydra
            multiruns surface failures instead of silently continuing.
    """
    cfg_dict = cast(Any, OmegaConf.to_container(cfg, resolve=True))
    ctx = mp.get_context("spawn")
    proc = ctx.Process(
        target=_subprocess_entry,
        args=(cfg_dict, os.getcwd()),
    )
    proc.start()
    try:
        proc.join()
    except KeyboardInterrupt:
        proc.terminate()
        proc.join(timeout=5.0)
        raise
    if proc.exitcode not in (0, None):
        raise SystemExit(proc.exitcode)


if __name__ == "__main__":
    main()
