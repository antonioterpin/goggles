from pathlib import Path

import numpy as np

import goggles as gg

# `logger.push({...})` used to batch every value as a metric. With image
# promotion, image-shaped numpy arrays in the mapping are split out into
# individual image events while scalars stay in a single metric event.
# Image-shaped means: a 2-D array, OR a 3-D array with a trailing
# channel axis in {1, 3, 4}.

LOG_DIR = Path("examples/logs/107_push")
logger = gg.get_logger("examples.push", with_metrics=True)
gg.attach(
    gg.LocalStorageHandler(path=LOG_DIR, name="examples.push"),
)
gg.attach(
    gg.ConsoleHandler(name="examples.push.console", level=gg.INFO),
)

print(f"=== writing logs to {LOG_DIR} ===\n")

rng = np.random.default_rng(0)
gray = rng.integers(0, 255, (64, 64), dtype=np.uint8)  # (H, W) -> image
rgb = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)  # (H, W, 3) -> image
rgba = rng.integers(0, 255, (64, 64, 4), dtype=np.uint8)  # (H, W, 4) -> image

# Mixed batch: 2 scalars + 3 image-shaped arrays. The two scalars go
# out as a single metric event; each image becomes its own image event
# with the dict key as its name.
logger.push(
    {
        "loss": 0.123,
        "accuracy": 0.91,
        "samples/gray": gray,
        "samples/rgb": rgb,
        "samples/rgba": rgba,
    },
    step=0,
)

# Vector-shaped arrays (1-D, scalar) stay on the metric path. Mix
# them with images in a single push without thinking about routing.
logger.push(
    {
        "logits": np.array([0.1, 0.6, 0.3]),
        "samples/rgb_2": rgb,
    },
    step=1,
)

# When the mapping is all images, no empty metric event is emitted.
# Inspect log.jsonl after the run -- step=2 only has image entries.
logger.push(
    {"samples/rgb_3": rgb, "samples/gray_2": gray},
    step=2,
)

gg.finish()

print()
print("Inspect:")
print(f"- {LOG_DIR}/log.jsonl  -- one event per line")
print(f"- {LOG_DIR}/images/    -- individual PNGs per promoted key")
