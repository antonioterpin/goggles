"""Decorators for logging and timing function execution."""

from .severity import Severity

from . import get_logger

logger = get_logger("goggles.decorators", with_metrics=True)


def timeit(severity=Severity.INFO, name=None):
    """Measure the execution time of a function via decorators.

    Args:
        severity (Severity): Log severity level for timing message.
        name (str): Optional name for the timing entry.
            If None, uses filename:function_name.

    Example:
    >>> @timeit(severity=Severity.DEBUG, name="my_function_timing")
    ... def my_function():
    ...     # function logic here
    ...     pass
    >>> my_function()
    DEBUG: my_function_timing took 0.123456s

    """

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
            logger.metrics.scalar(f"timings/{fname}", duration)
            return result

        return wrapper

    return decorator


def trace_on_error():
    """Trace errors and log function parameters via decorators.

    Example:
    >>> @trace_on_error()
    ... def my_function(x, y):
    ...     return x / y  # may raise ZeroDivisionError
    >>> my_function(10, 0)
    ERROR: Exception in my_function: division by zero, state:
    {'args': (10, 0), 'kwargs': {}}

    """

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
