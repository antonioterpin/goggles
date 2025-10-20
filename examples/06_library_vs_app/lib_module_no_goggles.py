"""A library module that logs, without assuming the app configures goggles."""

from logging import getLogger

_LIB_LOG = getLogger(__name__)


def do_something(x: int) -> int:
    _LIB_LOG.info("lib-no-goggles-start: x=%d", x)
    y = x * 2
    _LIB_LOG.info("lib-no-goggles-done: y=%d", y)
    return y
