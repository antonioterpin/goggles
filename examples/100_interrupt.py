"""Example of using Goggles with auto-cleanup on ctrl-c."""

import signal
import time
from types import FrameType

import goggles as gg
from goggles._core.integrations import ConsoleHandler

# Instantiate a TextLogger (No metrics)
logger = gg.get_logger(name="examples.interrupt")

gg.attach(
    ConsoleHandler(name="examples.interrupt.info", level=gg.INFO), ["global"]
)

_prev_sigint_handler = signal.getsignal(signal.SIGINT)


# simulate an existing custom SIGINT handler
def custom_handler(signum: int, frame: FrameType | None) -> None:
    """Simulate a custom SIGINT handler that wraps the previous handler.

    Args:
        signum: Received signal number.
        frame: Current stack frame at signal time.

    Returns:
        None.

    """
    print("Custom handler called for SIGINT (Ctrl-C).")
    print("Now calling what was the previous handler...")
    if callable(_prev_sigint_handler):
        _prev_sigint_handler(signum, frame)


# install our wrapper
signal.signal(signal.SIGINT, custom_handler)

print("Started. Press Ctrl-C")

# main work loop
try:
    counter = 0
    while True:
        logger.info(f"Working... {counter}")
        counter += 1
        time.sleep(1)
except KeyboardInterrupt:
    print("KeyboardInterrupt caught in main; exiting.")
finally:
    gg.finish()
