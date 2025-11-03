"""Example of using Goggles with auto-cleanup on ctrl-c."""

import signal
import time
import goggles as gg

from goggles._core.integrations import ConsoleHandler

# Instantiate a TextLogger (No metrics)
logger = gg.get_logger(name="examples.interrupt")

gg.attach(ConsoleHandler(name="examples.interrupt.info", level=gg.INFO), ["global"])

_prev_sigint_handler = signal.getsignal(signal.SIGINT)


# simulate an existing custom SIGINT handler
def custom_handler(signum, frame):
    """Simulate a custom handler for SIGINT (Ctrl-C) wrapping previous handler."""
    print("Custom handler called for SIGINT (Ctrl-C).")
    print("Now calling what was the previous handler...")
    _prev_sigint_handler(signum, frame)  # call the previous handler if it exists


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
