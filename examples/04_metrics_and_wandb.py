import goggles as gg
from goggles import WandBHandler
import numpy as np

# In this basic example, we set up a logger that outputs to the console.
logger: gg.GogglesLogger = gg.get_logger("examples.basic")
gg.attach(WandBHandler(project="goggles_example"), scopes=["global"])


logger.info("Logging to Weights & Biases!")
logger.scalar("accuracy", 0.95)

# Generate and log an image
image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
logger.image("Goggles Logo", image)  # TODO: replace with actual logo

# Generate and log a video
video = np.random.randint(
    0, 255, (30, 64, 64, 3), dtype=np.uint8
)  # 30 frames of 64x64 RGB
logger.video("Sample Video", video)

# # Load and log artifact
# artifact = np.random.rand(100, 100, 3
#                           )
# logger.artifact("Sample Artifact", artifact)
