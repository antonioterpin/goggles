"""Example: a custom W&B x-axis (e.g. physical time) instead of the step index.

Any numeric keyword passed to a metric call is logged in the same step, so it
can be selected as the chart x-axis in W&B (instead of the default ``_step``).
Nested dicts are flattened to dotted scalar keys
(``custom_step={"time": t}`` -> ``custom_step.time``), which are selectable too.

Run offline (no account needed):
    WANDB_MODE=offline python examples/09_custom_step_axis.py
"""

import math

import goggles as gg
from goggles import WandBHandler

logger: gg.GogglesLogger = gg.get_logger(
    "examples.custom_step", with_metrics=True
)
gg.attach(
    WandBHandler(project="goggles_custom_step", run_name="custom_step"),
    scopes=["global"],
)

dt = 1.0 / 30.0  # a 30 Hz process
for i in range(300):
    t = i * dt
    # `sim_time` is a flat axis; `custom_step={"time": t}` flattens to
    # `custom_step.time` -- both selectable as the chart x-axis.
    logger.scalar(
        "signal",
        math.sin(math.pi * t),
        step=i,
        sim_time=t,
        custom_step={"time": t},
    )

gg.finish()
print("In W&B, set the x-axis to `sim_time` or `custom_step.time`.")
