"""The goggles logger."""

import wandb
import os
import json
import imageio
import signal
import inspect
from enum import Enum
from multiprocessing import Process, Queue
from datetime import datetime
from typing import Iterable, Optional
from .config import PrettyConfig, load_configuration
from .utils import safe_chmod, FileRWLock


class Severity(Enum):
    """Severity levels for logging."""

    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40

    def to_json(self):
        """Convert severity to JSON-compatible string."""
        return self.name

    @classmethod
    def from_json(cls, name):
        """Convert JSON-compatible string to Severity enum."""
        return Severity[name.upper()]


class SeverityEncoder(json.JSONEncoder):
    """Custom JSON encoder for Severity enum."""

    def default(self, obj):
        """Encode Severity enum as its name."""
        if isinstance(obj, Severity):
            return obj.name
        return super().default(obj)


# Attempt to load project-level defaults from .goggles-default.yaml in the project root
_project_yaml_path = os.path.join(os.getcwd(), ".goggles-default.yaml")
_loaded_project_defaults = {
    "level": Severity.DEBUG,
    "to_file": False,
    "to_terminal": False,
    "wandb_project": None,
    "wandb_run_id": None,
    "name": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    "logdir": os.path.expanduser("~/logdir"),
    "config": {},
    "enable_signal_handler": False,
}

if os.path.exists(_project_yaml_path):
    try:
        custom_defaults = load_configuration(_project_yaml_path)
        t = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if "level" in custom_defaults:
            custom_defaults["level"] = Severity.from_json(custom_defaults["level"])
        if "name" in custom_defaults:
            custom_defaults["name"] = custom_defaults["name"].replace("{timestamp}", t)
        if "logdir" in custom_defaults:
            custom_defaults["logdir"] = custom_defaults["logdir"].replace(
                "{timestamp}", t
            )
            custom_defaults["logdir"] = os.path.expanduser(custom_defaults["logdir"])
        _loaded_project_defaults.update(custom_defaults)
    except Exception:
        # If parsing fails, ignore and continue with built-in defaults
        pass

# Build filenames that combine both
_config_path = os.path.join(_loaded_project_defaults["logdir"], "goggles_logger.json")
# Create a lock on that unique lock-filename
_rwlock = FileRWLock(
    os.path.join(_loaded_project_defaults["logdir"], "goggles_logger.json.lock"),
)

# Ensure the file paths exist
os.makedirs(_loaded_project_defaults["logdir"], exist_ok=True, mode=0o777)


class Goggles:
    """Goggles logger for structured logging."""

    # ANSI color codes for terminal
    _COLOR_MAP = {
        Severity.DEBUG: "[34m",  # blue
        Severity.INFO: "",  # white "[32m",  # green
        Severity.WARNING: "[33m",  # yellow
        Severity.ERROR: "[31m",  # red
    }
    _COLOR_RESET = "[0m"

    # Scheduler internals
    _task_queue = None
    _workers = []

    # Wandb utils
    _run_id = None

    @classmethod
    def init_scheduler(cls, num_workers: int = 2):
        """Initialize the scheduler with a specified number of worker processes."""
        if cls._task_queue is None:
            cls._task_queue = Queue()
            for _ in range(num_workers):
                p = Process(
                    target=cls._worker_loop, args=(cls._task_queue,), daemon=True
                )
                p.start()
                cls._workers.append(p)

    @staticmethod
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
            except Exception:
                Goggles.error(
                    f"Error executing scheduled function {fn.__name__} with "
                    f"args {args} and kwargs {kwargs}"
                )

    @classmethod
    def stop_workers(cls):
        """Stop all worker processes."""
        if cls._task_queue is not None:
            Goggles.info("Stopping Goggles workers...")
            for _ in range(len(cls._workers)):
                cls._task_queue.put(None)
            for worker in cls._workers:
                worker.join()
            cls._task_queue = None
            cls._workers = []

    @classmethod
    def schedule_log(cls, fn, *args, **kwargs):
        """Schedule a function to be executed by the worker processes."""
        cls.init_scheduler()
        cls._task_queue.put((fn, args, kwargs))

    @staticmethod
    def timeit(severity=Severity.INFO, name=None, to_wandb=False):
        """Decorator to measure the execution time of a function."""

        def decorator(func):
            import time
            import os

            def wrapper(*args, **kwargs):
                start = time.perf_counter()
                result = func(*args, **kwargs)
                duration = time.perf_counter() - start
                fname = (
                    name
                    or f"{os.path.basename(func.__code__.co_filename)}:{func.__name__}"
                )
                Goggles._log(severity, f"{fname} took {duration:.6f}s")
                if to_wandb:
                    Goggles.scalar(f"timings/{fname}", duration)
                return result

            return wrapper

        return decorator

    @staticmethod
    def trace_on_error():
        """Decorator to trace errors and log function parameters."""

        def decorator(func):
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # collect parameters
                    data = {"args": args, "kwargs": kwargs}
                    # if method, collect self attributes
                    if args and hasattr(args[0], "__dict__"):
                        data["self"] = args[0].__dict__
                    Goggles.error(f"Exception in {func.__name__}: {e}; state: {data}")
                    raise

            return wrapper

        return decorator

    @classmethod
    def get_config(cls):
        """Load the configuration from the JSON file."""
        with _rwlock.read_lock():
            try:
                config = load_configuration(_config_path)
            except Exception:
                # Configuration is invalid or missing, use defaults
                with open(_config_path, "w") as f:
                    json.dump(_loaded_project_defaults, f, cls=SeverityEncoder)
                # Ensure the file is writable by any process
                safe_chmod(_config_path, 0o666)
                config = _loaded_project_defaults.copy()
            logdir = os.path.join(config["logdir"], config["name"])
            config["file_path"] = os.path.join(logdir, "log.txt")
            if not os.path.exists(config["file_path"]):
                os.makedirs(logdir, exist_ok=True, mode=0o777)
                with open(config["file_path"], "w") as f:
                    f.write("")
                safe_chmod(config["file_path"], 0o666)
            return PrettyConfig(config)

    @classmethod
    def cleanup(cls):
        """Clean up the logger by removing the lock file and resetting the run ID."""
        cls.debug("Cleaning up Goggles logger...")
        cls.stop_workers()

        with _rwlock.write_lock():
            cfg = cls.get_config()
            if cfg.get("wandb_project"):
                try:
                    artifact = wandb.Artifact("training-logs", type="log")
                    artifact.add_file(cfg["file_path"])
                    wandb.run.log_artifact(artifact)
                except Exception:
                    pass

            cls._run_id = None
            cls._task_queue = None

    @classmethod
    def set_config(
        cls,
        name: str = None,
        wandb_project: Optional[str] = None,
        to_file: Optional[bool] = None,
        to_terminal: Optional[bool] = True,
        level: Optional[Severity] = None,
        wandb_run_id: Optional[None] = None,
        logdir: Optional[str] = None,
        config: Optional[dict] = {},
    ):
        """Set the configuration for Goggles logger.

        Args:
            name (str): Name of the log file. If None, uses default name with timestamp.
            wandb_project (str): Name of the Weights & Biases project.
            to_file (bool): Whether to log to a file.
            to_terminal (bool): Whether to log to the terminal.
            level (Severity): Logging level.
            wandb_run_id (str): Optional W&B run ID to resume.
            logdir (str): Directory to store logs.
            config (dict): Additional configuration parameters to log.
        """
        name = (
            name.replace("{timestamp}", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
            if name
            else _loaded_project_defaults["name"]
        )
        data = _loaded_project_defaults
        data["wandb_project"] = wandb_project
        data["wandb_run_id"] = wandb_run_id
        data["config"] = config or {}
        if name is not None:
            data["name"] = name
        if to_file is not None:
            data["to_file"] = to_file
        if to_terminal is not None:
            data["to_terminal"] = to_terminal
        if level is not None:
            if isinstance(level, Severity):
                data["level"] = level
            else:
                data["level"] = Severity.from_json(level)
        if logdir is not None:
            data["logdir"] = logdir.replace(
                "{timestamp}", datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            )
            data["logdir"] = os.path.expanduser(data["logdir"])

        if wandb_run_id is None or (
            cls._run_id is not None and wandb_run_id != cls._run_id
        ):
            wandb.finish()
            cls._run_id = None

        with _rwlock.write_lock():
            # write JSON (all values now primitives)
            with open(_config_path, "w") as f:
                json.dump(data, f, cls=SeverityEncoder)

    @classmethod
    def _init_wandb(cls):
        """Initialize Weights & Biases (W&B) logging."""
        cfg = cls.get_config()
        init_args = {}
        init_args["project"] = cfg["wandb_project"]
        init_args["name"] = cfg["name"]
        init_args["config"] = cfg.get("config", {})
        run_id = cfg.get("wandb_run_id")
        if run_id:
            if run_id == cls._run_id:
                return
            init_args["id"] = run_id
            init_args["resume"] = "allow"
        if not cls._run_id:
            run = wandb.init(**init_args)
            cls._run_id = run.id
            cfg["wandb_run_id"] = run.id
            cls.set_config(
                wandb_run_id=run.id,
                wandb_project=cfg["wandb_project"],
                name=cfg["name"],
                to_file=cfg["to_file"],
                to_terminal=cfg["to_terminal"],
                level=Severity[cfg["level"]],
                logdir=cfg["logdir"],
                config=cfg.get("config", {}),
            )

    @classmethod
    def _log(cls, severity: Severity, message: str):
        """Log a message with the specified severity level.

        Args:
            severity (Severity): The severity level of the log message.
            message (str): The log message to be recorded.
        """
        cfg = cls.get_config()
        lvl = cfg["level"]
        lvl = lvl if isinstance(lvl, Severity) else Severity[lvl]
        if severity.value < lvl.value:
            return
        frame = inspect.stack()[2]
        filename = os.path.basename(frame.filename)
        line_no = frame.lineno
        # timestamp in ISO 8601 format
        timestamp = datetime.now().isoformat(sep=" ", timespec="milliseconds")
        line = f"[{severity.name}][{filename}:{line_no}][{timestamp}] {message}"
        if cfg.get("to_terminal"):
            color = cls._COLOR_MAP.get(severity, "")
            print(f"{color}{line}{cls._COLOR_RESET}")
        if cfg.get("to_file"):
            with _rwlock.write_lock():
                with open(cfg["file_path"], "a") as f:
                    f.write(line + "\n")
                # Ensure the file is writable by any process
                safe_chmod(cfg["file_path"], 0o666)

    @classmethod
    def debug(cls, message: str):
        """Log a debug message.

        Args:
            message (str): The debug message to be logged.
        """
        cls._log(Severity.DEBUG, message)

    @classmethod
    def info(cls, message: str):
        """Log an informational message.

        Args:
            message (str): The informational message to be logged.
        """
        cls._log(Severity.INFO, message)

    @classmethod
    def warning(cls, message: str):
        """Log a warning message.

        Args:
            message (str): The warning message to be logged.
        """
        cls._log(Severity.WARNING, message)

    @classmethod
    def error(cls, message: str):
        """Log an error message.

        Args:
            message (str): The error message to be logged.
        """
        cls._log(Severity.ERROR, message)

    @classmethod
    def scalar(cls, name: str, value: int | float):
        """Log a scalar value with the specified name.

        Args:
            name (str): The name of the scalar value.
            value (int | float): The scalar value to be logged.
        """
        cfg = cls.get_config()
        if cfg.get("wandb_project"):
            cls._init_wandb()
            wandb.log({name: value})

    @classmethod
    def vector(cls, name: str, values: Iterable):
        """Log a vector of values with the specified name.

        Args:
            name (str): The name of the vector.
            values (Iterable): The vector of values to be logged.
        """
        cfg = cls.get_config()
        if cfg.get("wandb_project"):
            cls._init_wandb()
            wandb.log({name: wandb.Histogram(values)})

    @classmethod
    def image(cls, name: str, image):
        """Log an image with the specified name.

        Args:
            name (str): The name of the image.
            image: The image to be logged.
        """
        cfg = cls.get_config()
        if cfg.get("wandb_project"):
            cls._init_wandb()
            wandb.log({name: wandb.Image(image)})

    @classmethod
    def video(
        cls,
        name: str,
        video,
        fps: int = 10,
        bitrate: str = "150k",
        crf: int = 30,
        to_file: bool = True,
    ):
        """Log a WebM/VP9 video for maximal compression."""
        cfg = cls.get_config()
        out_dir = os.path.join(cfg["logdir"], cfg["name"], "videos")
        os.makedirs(out_dir, exist_ok=True, mode=0o777)
        path = os.path.join(out_dir, f"{name}.webm")

        writer = imageio.get_writer(
            path,
            fps=fps,
            codec="libvpx-vp9",
            bitrate=bitrate,
            ffmpeg_params=[
                "-pix_fmt",
                "yuv420p",
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

        cls.info(f"Saved high-compression WebM → {path}")

        # log to W&B
        if cfg.get("wandb_project"):
            cls._init_wandb()
            wandb.log({name: wandb.Video(path, fps=fps, format="webm")})
            cls.info("Uploaded video to WandB.")

        if not to_file:
            # delete the file after logging
            os.remove(path)


# Capture the pre-existing handler before we overwrite it:
_prev_sigint_handler = signal.getsignal(signal.SIGINT)


# Install signal handlers if enabled
def _handle_sigint(signum, frame):
    Goggles.cleanup()
    # put back whatever was there before
    signal.signal(signal.SIGINT, _prev_sigint_handler)

    if _prev_sigint_handler in (signal.SIG_DFL, signal.SIG_IGN):
        # If the previous handler was the default, raise to stop execution.
        # SIG_DFL -> default behavior (KeyboardInterrupt)
        # SIG_IGN -> ignored; fall back to KeyboardInterrupt so user can still kill
        raise KeyboardInterrupt
    else:
        # Otherwise, delegate to it.
        _prev_sigint_handler(signum, frame)


# only install the handler if the flag is enabled
if _loaded_project_defaults.get("enable_signal_handler"):
    signal.signal(signal.SIGINT, _handle_sigint)
