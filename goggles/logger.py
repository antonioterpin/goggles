"""The goggles logger."""

import os
import traceback
import pathlib
from types import MappingProxyType
from datetime import datetime
import imageio
from multiprocessing import Process, Queue
import json
import hashlib
import time
import random
from multiprocessing import shared_memory
from contextlib import contextmanager

from .config import load_configuration
from .severity import Severity

wandb = None

# --- Load defaults, which are constant across all instances and immutable ---
_consts = {
    "level": Severity.DEBUG,
    "to_terminal": False,
    "wandb_project": None,
    "enable_signal_handler": False,
    "logdir": "~/logdir",
}
_config_path = pathlib.Path.cwd() / ".goggles-default.yaml"
for key, value in load_configuration(_config_path).items():
    if key in _consts:
        _consts[key] = value
    else:
        raise ValueError(f"Unexpected default key '{key}' in .goggles-default.yaml")

# Expand the logdir, convert the level to a Severity enum
# and make the constants immutable.
_consts["logdir"] = os.path.expanduser(_consts["logdir"])
_consts["level"] = Severity.from_str(_consts["level"])
_consts = MappingProxyType(_consts)  # make it immutable

# runtime params to be shared across all instances
_config_fingerprint = hashlib.sha256(_consts["logdir"].encode("utf-8")).hexdigest()[:16]
_SHM_NAME = f"goggles_shm_{_config_fingerprint}"
_SHM_SIZE = 1024

created = False
while not created:
    created = True
    try:
        shm = shared_memory.SharedMemory(name=_SHM_NAME, create=False)
    except FileNotFoundError:
        try:
            shm = shared_memory.SharedMemory(
                name=_SHM_NAME, create=True, size=_SHM_SIZE
            )
            # ensure the segment is accessible by all users
            try:
                shm_path = f"/dev/shm/{_SHM_NAME}"
                os.chmod(shm_path, 0o666)
            except Exception as e:
                print(f"Warning: could not set permissions on shared memory: {e}")
        except FileExistsError:
            # Another process created it in the meantime, try again
            created = False
    # wait a random time
    if not created:
        time.sleep(random.uniform(0.01, 0.1))


@contextmanager
def _shm_handle(create: bool = False):
    """Context manager for shared memory."""
    try:
        if create:
            shm = shared_memory.SharedMemory(
                name=_SHM_NAME, create=True, size=_SHM_SIZE
            )
        else:
            shm = shared_memory.SharedMemory(name=_SHM_NAME, create=False)
    except FileNotFoundError:
        if create:
            raise
        # create-on-demand if it didn't exist yet
        shm = shared_memory.SharedMemory(name=_SHM_NAME, create=True, size=_SHM_SIZE)
    try:
        yield shm
    finally:
        shm.close()


def _read_shm():
    """Read the JSON blob (if any) from shared memory."""
    try:
        with _shm_handle(create=False) as shm:
            # immediately copy the whole region to bytes
            blob = bytes(shm.buf[:_SHM_SIZE])
        # split off trailing zeroes
        raw = blob.split(b"\x00", 1)[0]
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError:
        return {}
    except Exception as e:
        print(f"Failed to read shared memory: {e}")
        return {}


def _write_shm(data: dict):
    """Write our JSON blob into shared memory, zero-padding the rest."""
    try:
        raw = json.dumps(data).encode("utf-8")
        if len(raw) > _SHM_SIZE:
            raise RuntimeError("Shared-mem JSON too big")
        with _shm_handle(create=False) as shm:
            shm.buf[: len(raw)] = raw
            shm.buf[len(raw) :] = b"\x00" * (_SHM_SIZE - len(raw))
    except Exception as e:
        print(f"Failed to write to shared memory: {e}")


def _get_log_file_path() -> str:
    """Get the path to the log file."""
    _shared = _read_shm()
    if "name" not in _shared:
        return None
    return os.path.join(_consts["logdir"], _shared["name"], ".log")


def _log(severity: Severity, message: str):
    """Log a message with the specified severity level."""
    # Ensure the message is a string
    if not isinstance(message, str):
        message = str(message)

    if severity.value < _consts["level"].value:
        return  # Skip logging if below configured level

    # Get caller information
    this_file = os.path.abspath(__file__)
    # get a list of FrameSummary objects
    stack = traceback.extract_stack()
    # walk from the end (most recent call) backwards
    for frame_summary in reversed(stack[:-1]):  # skip the last entry (this line)
        if os.path.abspath(frame_summary.filename) != this_file:
            caller_filename = frame_summary.filename
            caller_line = frame_summary.lineno
            break
    else:
        caller_filename = "<unknown>"
        caller_line = 0

    # Get current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Format the log message
    formatted_message = (
        f"[{severity.name}][{timestamp}] {caller_filename}:{caller_line} - {message}"
    )

    # Log to terminal if enabled
    if _consts["to_terminal"]:
        color = severity.to_color()
        print(f"{color}{formatted_message}{severity.reset_color()}")

    # Log to wandb if enabled
    if _wandb():
        log_path = _get_log_file_path()
        if log_path is not None:
            # TODO: add lock
            with open(log_path, "a") as log_file:
                log_file.write(formatted_message + "\n")


def info(message: str):
    """Log an informational message."""
    _log(Severity.INFO, message)


def debug(message: str):
    """Log a debug message."""
    _log(Severity.DEBUG, message)


def warning(message: str):
    """Log a warning message."""
    _log(Severity.WARNING, message)


def error(message: str):
    """Log an error message."""
    _log(Severity.ERROR, message)


def stop_wandb_run():
    """Stop the current wandb run, if any."""
    if _wandb():
        logfile = _get_log_file_path()
        if logfile and os.path.exists(logfile):
            artifact = wandb.Artifact("training-logs", type="log")
            artifact.add_file(logfile)
            try:
                wandb.run.log_artifact(artifact)
            except Exception as e:
                error(f"Warning: could not upload W&B artifact: {e}")
            else:
                os.remove(logfile)

        try:
            wandb.finish()
            info("WandB run finished successfully.")
        except Exception as e:
            error(f"Warning: wandb.finish() failed: {e}")


def new_wandb_run(name: str, config: dict = None):
    """Start a new wandb run with the given name and configuration."""
    stop_workers()
    stop_wandb_run()

    global wandb
    if wandb is None:
        try:
            import wandb
        except ImportError:
            warning("wandb is not installed, skipping W&B logging.")
            return

    run = wandb.init(project=_consts["wandb_project"], name=name, config=config)
    # Set the shared run id in shared memory
    _write_shm(
        {
            "name": name,
            "wandb_run_id": run.id,
        }
    )

    # Make sure logdir exists
    run_logdir = os.path.join(_consts["logdir"], name)
    os.makedirs(run_logdir, exist_ok=True, mode=0o777)


def _is_wandb_active() -> bool:
    """Check if a wandb run is currently active."""
    return wandb.run is not None


def _wandb() -> bool:
    """Check if we can log to wandb."""
    if _consts["wandb_project"] is None or wandb is None:
        return False

    wandb_running_in_this_process = _is_wandb_active()
    _shared = _read_shm()
    shared_run_id = _shared.get("wandb_run_id", None)

    if wandb_running_in_this_process and wandb.run.id == shared_run_id:
        # If wandb is running and the shared run id matches,
        # we can log to the existing run.
        return True

    elif not wandb_running_in_this_process:
        # If wandb is not running, but we have a shared run id,
        # start a new wandb run with that id.
        wandb.init(
            project=_consts["wandb_project"],
            id=shared_run_id,
            resume="allow",
        )
        return True

    return False


def scalar(name: str, value: float):
    """Log a scalar value."""
    if _wandb():
        wandb.log({name: value})


def image(name: str, value):
    """Log an image."""
    if _wandb():
        # Assuming value is a PIL Image or similar
        wandb.log({name: wandb.Image(value)})


def vector(name: str, values):
    """Log a vector."""
    if _wandb():
        wandb.log({name: wandb.Histogram(values)})


def video(
    name: str,
    video,
    fps: int = 10,
    bitrate: str = "150k",
    crf: int = 30,
    to_file: bool = True,
):
    """Log a video."""
    shared = _read_shm()
    out_dir = os.path.join(_consts["logdir"], shared["name"], "videos")
    os.makedirs(out_dir, exist_ok=True, mode=0o777)
    path = os.path.join(out_dir, f"{name}.webm")

    writer = imageio.get_writer(
        path,
        fps=fps,
        codec="libvpx-vp9",
        bitrate=bitrate,
        pixelformat="yuv420p",
        ffmpeg_params=[
            "-crf",
            str(crf),
            "-row-mt",
            "1",
            "-auto-alt-ref",
            "1",
            "-lag-in-frames",
            "25",
        ],
    )
    for frame in video:
        writer.append_data(frame)
    writer.close()

    info(f"Saved high-compression WebM @ {path}")

    # log to W&B
    if _wandb():
        wandb.log({name: wandb.Video(path, format="webm")})
        info("Uploaded video to WandB.")

    if not to_file:
        # delete the file after logging
        os.remove(path)


_task_queue = None
_workers = []


def init_scheduler(num_workers: int = 2):
    """Initialize the scheduler with a specified number of worker processes."""
    global _task_queue, _workers
    if _task_queue is None:
        _task_queue = Queue()
        for _ in range(num_workers):
            p = Process(target=_worker_loop, args=(_task_queue,), daemon=True)
            p.start()
            _workers.append(p)


def _worker_loop(queue: Queue):
    """Worker loop to process scheduled tasks."""
    while True:
        val = queue.get()
        if val is None:
            # Signal to stop the worker
            break
        fn, args, kwargs = val
        try:
            fn(*args, **kwargs)
        except Exception as e:
            error(
                f"Error executing scheduled function {fn.__name__} with "
                f"args {args} and kwargs {kwargs}:\n{e}"
            )


def stop_workers():
    """Stop all worker processes."""
    global _task_queue, _workers
    if _task_queue is not None:
        info("Stopping Goggles workers...")
        for _ in range(len(_workers)):
            _task_queue.put(None)
        for worker in _workers:
            worker.join()
        _task_queue = None
        _workers = []


def schedule_log(fn, *args, **kwargs):
    """Schedule a function to be executed by the worker processes."""
    global _task_queue, _workers
    init_scheduler()
    _task_queue.put((fn, args, kwargs))


def ensure_tasks_finished(polling_time: float = 0.01):
    """Ensure all scheduled tasks are finished."""
    global _task_queue, _workers
    if _task_queue is not None:
        while not _task_queue.empty():
            time.sleep(polling_time)


def cleanup():
    """Cleanup resources and finish wandb runs."""
    debug("Cleaning up Goggles logger...")
    # stop the workers if they are running
    stop_workers()
    # finally, tear down the shared-memory segment
    try:
        shared_memory.SharedMemory(name=_SHM_NAME).unlink()
    except FileNotFoundError:
        pass
