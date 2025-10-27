"""Decorators for logging and timing function execution."""

from .severity import Severity

from . import get_logger

logger = get_logger("goggles.decorators")


def timeit(severity=Severity.INFO, name=None, to_wandb=False):
    """Measure the execution time of a function via decorators."""

    def decorator(func):
        import time
        import os

        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            duration = time.perf_counter() - start
            filename = os.path.basename(func.__code__.co_filename)
            fname = name or f"{filename}:{func.__name__}"
            logger.log(severity, f"{fname} took {duration:.6f}s")
            if to_wandb:
                logger.scalar(f"timings/{fname}", duration)
            return result

        return wrapper

    return decorator


def trace_on_error():
    """Trace errors and log function parameters via decorators."""

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
                logger.error(f"Exception in {func.__name__}: {e}; state: {data}")
                raise

        return wrapper

    return decorator
