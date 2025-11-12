"""Decorators for logging and timing function execution."""

from __future__ import annotations

import functools
import logging
from typing import ParamSpec, TypeVar
from collections.abc import Callable

# TypeVar for preserving function signature
P = ParamSpec("P")
T = TypeVar("T")


def timeit(
    severity: int = logging.INFO, name: str | None = None, scope: str = "global"
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator to time function execution and log the duration.

    Args:
        severity: Log severity level for timing message.
        name: Optional name for the timing entry.
            If None, uses filename:function_name.
        scope: Scope of the logged event (e.g., "global" or "run").

    Returns:
        Decorated function with same signature as input.

    Example:
    >>> @timeit(severity=logging.DEBUG, name="my_function_timing")
    ... def my_function():
    ...     # function logic here
    ...     pass
    >>> my_function()
    DEBUG: my_function_timing took 0.123456s

    """
    import time
    import os
    from goggles import get_logger, GogglesLogger

    logger: GogglesLogger = get_logger(
        "goggles.decorators.timeit", with_metrics=True, scope=scope
    )

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            start = time.perf_counter()
            result = func(*args, **kwargs)
            duration = time.perf_counter() - start
            filename = os.path.basename(func.__code__.co_filename)
            fname = name or f"{filename}:{func.__name__}"
            logger.log(severity, f"{fname} took {duration:.6f}s")
            logger.scalar(f"timings/{fname}", duration)
            return result

        return wrapper

    return decorator


def trace_on_error(scope: str = "global") -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator to log function arguments and state on exception.

    Args:
        scope: Scope of the logged event (e.g., "global" or "run").

    Example:
    >>> @trace_on_error()
    ... def my_function(x, y):
    ...     return x / y  # may raise ZeroDivisionError
    >>> my_function(10, 0)
    ERROR: Exception in my_function: division by zero, state:
    {'args': (10, 0), 'kwargs': {}}

    """
    from goggles import get_logger

    logger = get_logger(
        "goggles.decorators.trace_on_error",
        scope=scope,
    )

    def decorator(func: Callable[P, T]) -> Callable[P, T]:

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
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
