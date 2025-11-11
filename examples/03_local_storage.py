import goggles as gg
from pathlib import Path
import numpy as np

# In this example, we set up a logger that stores events
# in a structured directory:
# - examples/logs/log.jsonl: Main JSONL log file with all events
# - examples/logs/images/: Directory for image files
# - examples/logs/videos/: Directory for video files
# - examples/logs/artifacts/: Directory for other artifact files
# - examples/logs/vector_fields/: Directory for vector field files
# - examples/logs/histograms/: Directory for histogram files

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
    gg.ConsoleHandler(name="examples.jsonl.console", level=gg.INFO),
)

print("=== Goggles Local Storage Handler Example ===")
print("Creating structured logs in: examples/logs/")
print("- examples/logs/log.jsonl (main log file)")
print("- examples/logs/images/ (image files)")
print("- examples/logs/videos/ (video files)")
print("- examples/logs/artifacts/ (other files)")
print()

# Scalar metrics
logger.scalar("accuracy", 0.95)
logger.scalar("loss", 0.123)
logger.scalar("learning_rate", 0.001)

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
logger.image(dummy_image, format="png")
logger.image(dummy_image, name="custom_name", format="jpg")

# Video logging
logger.info("Logging sample video...")
T, H, W = 10, 128, 128
dummy_video_gif = np.random.rand(T, H, W, 3).astype(np.float32)
dummy_video_mp4 = np.random.rand(T, H, W, 3).astype(np.float32)
logger.video(dummy_video_gif, format="gif", fps=30)
logger.video(dummy_video_mp4, name="custom_name", format="mp4", fps=24)

# Artifact logging (with text data as example)
logger.info("Logging sample artifact...")
artifact_txt = "This is sample artifact content\nLine 2\nLine 3"
logger.artifact(artifact_txt, name="text_artifact", format="txt")
artifact_json = {"key": "value"}
logger.artifact(artifact_json, name="json_artifact", format="json")
artifact_yaml = {
    "key": "value",
    "list": [1, 2, 3],
    "nested": {"a": 10, "b": [20, 30]},
}
logger.artifact(artifact_yaml, name="yaml_artifact", format="yaml")
artifact_csv = "col1,col2\n1,2\n3,4"
logger.artifact(artifact_csv, name="csv_artifact", format="csv")
artifact_unknown = "This is some unknown format data"
logger.artifact(artifact_unknown, name="unknown_artifact", format="myformat")

# Vector field logging
logger.info("Logging sample vector field...")


def make_lamb_oseen_vortices(
    H: int,
    W: int,
    vortices: list[tuple[float, float, float, float]],
    domain: tuple[float, float, float, float] = (-1.0, 1.0, -1.0, 1.0),
    uniform: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """
    Build a 2D incompressible flow from Lamb–Oseen vortices.

    vortices: list of (x0, y0, Gamma, sigma), in domain coords.
      - (x0, y0): vortex center (same units as 'domain')
      - Gamma: circulation (sign sets rotation direction)
      - sigma: core size (Gaussian radius)
    domain: (xmin, xmax, ymin, ymax) spanning the grid
    uniform: (U, V) optional background flow
    Returns: (H, W, 2) with (u, v).
    """
    xmin, xmax, ymin, ymax = domain
    y = np.linspace(ymin, ymax, H, dtype=np.float32)
    x = np.linspace(xmin, xmax, W, dtype=np.float32)
    X, Y = np.meshgrid(x, y)

    u = np.full((H, W), uniform[0], dtype=np.float32)
    v = np.full((H, W), uniform[1], dtype=np.float32)

    eps = 1e-6
    for x0, y0, Gamma, sigma in vortices:
        dx = X - x0
        dy = Y - y0
        r2 = dx * dx + dy * dy
        r = np.sqrt(r2) + eps
        # Lamb–Oseen tangential speed:
        # v_theta(r) = (Gamma / (2π r)) * (1 - exp(-r^2 / (2 sigma^2)))
        v_theta = (
            (Gamma / (2.0 * np.pi)) * (1.0 - np.exp(-r2 / (2.0 * sigma * sigma))) / r
        )
        # Tangential direction (-dy/r, dx/r)
        u += -dy * (v_theta / r)
        v += dx * (v_theta / r)

    return np.stack([u, v], axis=-1)


VF_H, VF_W = 128, 128
vortices = [
    (-0.4, 0.0, +5.0, 0.15),  # CCW
    (+0.4, 0.0, -5.0, 0.15),  # CW
]
dummy_vector_field = make_lamb_oseen_vortices(VF_H, VF_W, vortices, uniform=(0.2, 0.0))
logger.vector_field(
    dummy_vector_field,
    name="lamb_oseen_dipole",
    store_visualization=False,  # optional, False by default
)
# We can also log with visualization saving enabled, but will take longer atm.
logger.vector_field(
    dummy_vector_field,
    name="lamb_oseen_dipole_magnitude",
    store_visualization=True,
    mode="magnitude",  # optional
)
# Default is a magnitude plot, but we can also visualize the vorticity
# and we can add a colorbar to the visualization
logger.vector_field(
    dummy_vector_field,
    name="lamb_oseen_dipole_vorticity",
    store_visualization=True,
    mode="vorticity",
    add_colorbar=True,
)

# Histogram logging
logger.info("Logging sample histogram...")
histogram_data = np.random.randn(1000)
logger.histogram(histogram_data, name="sample_histogram")

print()
print("✓ All events logged successfully!")
print()
print("Check the created structure:")
print("- examples/logs/log.jsonl (contains all events with media file paths)")
print("- examples/logs/images/ (contains saved image files)")
print("- examples/logs/videos/ (contains saved video files)")
print("- examples/logs/artifacts/ (contains saved artifact files)")
print("- examples/logs/vector_fields/ (contains saved vector field files)")
print("- examples/logs/histograms/ (contains saved histogram files)")
print("- Media files are referenced by relative paths in the JSONL log")

gg.finish()
