"""Example of logging to both file and terminal using Goggles."""

import goggles
import os
from time import sleep

print("\n-- Logging to file and terminal, level=DEBUG --")

# NO WANDB RUNNING
goggles.debug("debug msg")
goggles.info("info msg")
goggles.warning("warning appears")

print("\n-- Starting a new WandB run... WandB will be imported on demand --")
sleep(5)

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
