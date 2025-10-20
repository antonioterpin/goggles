"""A library module that logs, without assuming the app configures goggles."""

from goggles import get_logger

_LIB_LOG = get_logger(__name__, component="lib")


def do_something(x: int) -> int:
    _LIB_LOG.info("lib-goggles-start", x=x)
    y = x * 2
    _LIB_LOG.info("lib-goggles-done", y=y)
    return y
