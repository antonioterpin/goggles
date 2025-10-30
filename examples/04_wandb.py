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
