import time

import wandb
import goggles as gg
from goggles import WandBHandler
import numpy as np

# In this example, we set up a logger that outputs to Weights & Biases (W&B).
logger: gg.GogglesLogger = gg.get_logger("examples.basic", with_metrics=True)
handler = WandBHandler(project="goggles_example")
gg.attach(handler, scopes=["global"])


logger.info(
    "Logging to Weights & Biases!"
)  # This will be ignored because there's no log handler attached yet
for i in range(100):
    logger.scalar("accuracy", i, step=i)

# Generate and log an image
image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
logger.image(image, name="Random image")

# Generate and log a video
video = np.random.randint(
    0, 255, (30, 3, 64, 64), dtype=np.uint8
)  # 30 frames of 64x64 RGB
logger.video(video, name="Random Video", fps=10)

# Load and log artifact
artifact = np.random.rand(100, 100, 3)
logger.artifact(artifact, name="Random Artifact")

# Add extra fields to any metric logged to be used as x-axis in W&B
for i in range(101, 151):
    logger.scalar(
        "loss",
        150 - i,
        step=i,
    )
    if i % 10 == 0:
        logger.image(
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8),
            name="Random image with custom step",
            step=i,  # Global step
            custom_step={
                "custom_step": i // 10 - 10
            },  # Extra field to be used as x-axis
        )

# Log a static histogram (that does not change over time)
data = np.random.randn(1000)
logger.histogram(data, name="Random Values Histogram", static=True)

# Or a dynamic histogram (that changes over time)
for i in range(10):
    data = np.random.randn(1000) + i  # Shift mean over time
    logger.histogram(data, name="Dynamic Random Values Histogram", step=151 + i)

time.sleep(10)
# When using asynchronous logging (like wandb), make sure to finish
gg.finish()
