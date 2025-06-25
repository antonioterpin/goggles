"""Example of using Goggles with auto-cleanup on ctrl-c."""

import signal
import time
import goggles

_prev_sigint_handler = signal.getsignal(signal.SIGINT)


# simulate an existing custom SIGINT handler
def custom_handler(signum, frame):
    """Custom handler for SIGINT (Ctrl-C) that wraps the previous handler."""
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
        goggles.info(f"Working... {counter}")
        counter += 1
        time.sleep(1)
except KeyboardInterrupt:
    print("KeyboardInterrupt caught in main; exiting.")
