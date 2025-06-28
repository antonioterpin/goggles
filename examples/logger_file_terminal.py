"""Example of logging to both file and terminal using Goggles."""

import goggles
import os

print("\n-- Test 1: Logging to file and terminal, level=DEBUG --")

goggles.new_wandb_run(
    name="test",
    config={"param1": 42, "param2": "value"},
)

goggles.debug("debug msg")
goggles.info("info msg")
goggles.warning("warning appears")

filepath = os.path.expanduser("~/logdir/test/.log")

print(f"Accumulated log file {filepath}:")
with open(filepath, "r") as f:
    print(f.read())
