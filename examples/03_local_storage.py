import goggles as gg
from pathlib import Path
import numpy as np
import logging

# In this example, we set up a logger that stores events
# in a structured directory:
# - examples/logs/log.jsonl: Main JSONL log file with all events
# - examples/logs/images/: Directory for image files
# - examples/logs/videos/: Directory for video files
# - examples/logs/artifacts/: Directory for other artifact files

# Get a logger with metrics support
logger = gg.get_logger("examples.jsonl", with_metrics=True)

# Set up JSONL handler with a base directory
gg.attach(
    gg.LocalStorageHandler(
        path=Path("examples/logs"),
        name="examples.jsonl",
    )
)
gg.attach(
    gg.ConsoleHandler(name="examples.jsonl.console", level=logging.INFO),
)

print("=== Goggles Local Storage Handler Example ===")
print("Creating structured logs in: examples/logs/")
print("- examples/logs/log.jsonl (main log file)")
print("- examples/logs/images/ (image files)")
print("- examples/logs/videos/ (video files)")
print("- examples/logs/artifacts/ (other files)")
print()

# Scalar metrics
logger.scalar("accuracy", 0.95, step=1)
logger.scalar("loss", 0.123, step=1)
logger.scalar("learning_rate", 0.001, step=1)

# Batch metrics
logger.push(
    {
        "train/accuracy": 0.92,
        "train/loss": 0.156,
        "val/accuracy": 0.88,
        "val/loss": 0.201,
    },
    step=2,
)

# Image logging
logger.info("Logging sample image...")
dummy_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
logger.image(dummy_image, format="png", step=1)
logger.image(dummy_image, name="custom_name", format="jpg", step=1)

# Video logging
logger.info("Logging sample video...")
T, H, W = 10, 128, 128
dummy_video_gif = np.random.rand(T, H, W, 3).astype(np.float32)
dummy_video_mp4 = np.random.rand(T, H, W, 3).astype(np.float32)
logger.video(dummy_video_gif, format="gif", fps=30, step=1)
logger.video(dummy_video_mp4, name="custom_name", format="mp4", fps=24, step=1)

# # Artifact logging (with text data as example)
# print("Logging sample artifact...")
# artifact_data = "This is sample artifact content\nLine 2\nLine 3"
# logger.artifact("text_artifact", artifact_data, format="txt", step=1)

# # If the format is unsupported, the event will be skipped with a warning
# logger.artifact(
#     "unsupported_artifact", artifact_data, format="unknown_format", step=1)

print()
print("✓ All events logged successfully!")
print()
print("Check the created structure:")
print("- examples/logs/log.jsonl (contains all events with media file paths)")
print("- examples/logs/images/ (contains saved image files)")
print("- examples/logs/videos/ (contains saved video files)")
print("- Media files are referenced by relative paths in the JSONL log")
