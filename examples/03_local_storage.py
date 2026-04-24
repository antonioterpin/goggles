from pathlib import Path

import numpy as np

import goggles as gg

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
# Step is required to ensure proper ordering across different handlers
logger.scalar("accuracy", 0.95, step=0)
logger.scalar("loss", 0.123, step=0)
logger.scalar("learning_rate", 0.001, step=0)

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

# `push` also accepts image-shaped arrays (2-D, or 3-D with trailing
# channel axis in {1, 3, 4}) and promotes them to individual image
# events, so scalars + figures can land in a single call.
logger.push(
    {
        "train/loss": 0.120,
        "val/loss": 0.190,
        "eval/confusion": np.random.randint(
            0, 255, (32, 32, 3), dtype=np.uint8
        ),
        "eval/attention": np.random.randint(0, 255, (32, 32), dtype=np.uint8),
    },
    step=2,
)

# Image logging
logger.info("Logging sample image...")
dummy_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
logger.image(dummy_image, format="png", step=3)
logger.image(dummy_image, name="custom_name", format="jpg", step=3)

# RGBA image logging (with alpha channel)
logger.info("Logging sample RGBA image...")
dummy_rgba_image = np.random.randint(0, 255, (64, 64, 4), dtype=np.uint8)
# Make some areas transparent by setting alpha to different values
dummy_rgba_image[16:32, 16:32, 3] = 128  # Semi-transparent center
dummy_rgba_image[48:64, 48:64, 3] = 0  # Fully transparent corner
logger.image(dummy_rgba_image, name="rgba_image", format="png", step=4)

# Video logging
logger.info("Logging sample video...")
T, H, W = 10, 128, 128
dummy_video_gif = np.random.rand(T, H, W, 3).astype(np.float32)
dummy_video_mp4 = np.random.rand(T, H, W, 3).astype(np.float32)
logger.video(dummy_video_gif, format="gif", fps=30, step=5)
logger.video(dummy_video_mp4, name="custom_name", format="mp4", fps=24, step=5)

# RGBA video logging (with alpha channel)
logger.info("Logging sample RGBA video...")
dummy_rgba_video = np.random.rand(T, H, W, 4).astype(np.float32)
# Add some interesting alpha patterns
for t in range(T):
    # Create a moving transparent circle
    y, x = np.ogrid[:H, :W]
    center_y, center_x = (
        H // 2 + 20 * np.sin(2 * np.pi * t / T),
        W // 2 + 20 * np.cos(2 * np.pi * t / T),
    )
    mask = (x - center_x) ** 2 + (y - center_y) ** 2 < 400
    dummy_rgba_video[t, :, :, 3] = 0.8  # Base alpha
    dummy_rgba_video[t, mask, 3] = 0.2  # Transparent circle
logger.video(dummy_rgba_video, name="rgba_video", format="mp4", fps=15, step=6)

# Artifact logging (with text data as example)
logger.info("Logging sample artifact...")
artifact_txt = "This is sample artifact content\nLine 2\nLine 3"
logger.artifact(artifact_txt, name="text_artifact", format="txt", step=7)
artifact_json = {"key": "value"}
logger.artifact(artifact_json, name="json_artifact", format="json", step=7)
artifact_yaml = {
    "key": "value",
    "list": [1, 2, 3],
    "nested": {"a": 10, "b": [20, 30]},
}
logger.artifact(artifact_yaml, name="yaml_artifact", format="yaml", step=7)
artifact_csv = "col1,col2\n1,2\n3,4"
logger.artifact(artifact_csv, name="csv_artifact", format="csv", step=7)
artifact_unknown = "This is some unknown format data"
logger.artifact(
    artifact_unknown, name="unknown_artifact", format="myformat", step=7
)

# Vector field logging
logger.info("Logging sample vector field...")


def make_lamb_oseen_vortices(
    H: int,
    W: int,
    vortices: list[tuple[float, float, float, float]],
    domain: tuple[float, float, float, float] = (-1.0, 1.0, -1.0, 1.0),
    uniform: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Build a 2D incompressible flow from Lamb-Oseen vortices.

    Args:
        H: Grid height in pixels.
        W: Grid width in pixels.
        vortices: List of `(x0, y0, Gamma, sigma)` tuples in domain coordinates.
        domain: Spatial domain as `(xmin, xmax, ymin, ymax)`.
        uniform: Optional uniform background flow `(U, V)`.

    Returns:
        Velocity field array of shape `(H, W, 2)` containing `(u, v)`.

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
        # Lamb-Oseen tangential speed:
        # v_theta(r) = (Gamma / (2*pi*r)) * (1 - exp(-r^2/(2*sigma^2)))
        v_theta = (
            (Gamma / (2.0 * np.pi))
            * (1.0 - np.exp(-r2 / (2.0 * sigma * sigma)))
            / r
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
dummy_vector_field = make_lamb_oseen_vortices(
    VF_H, VF_W, vortices, uniform=(0.2, 0.0)
)
logger.vector_field(
    dummy_vector_field,
    name="lamb_oseen_dipole",
    store_visualization=False,  # optional, False by default
    step=8,
)
# We can also log with visualization saving enabled, but will take longer atm.
logger.vector_field(
    dummy_vector_field,
    name="lamb_oseen_dipole_magnitude",
    store_visualization=True,
    mode="magnitude",  # optional
    step=8,
)
# Default is a magnitude plot, but we can also visualize the vorticity
# and we can add a colorbar to the visualization
logger.vector_field(
    dummy_vector_field,
    name="lamb_oseen_dipole_vorticity",
    store_visualization=True,
    mode="vorticity",
    add_colorbar=True,
    step=8,
)

# Trajectories logging
# `logger.trajectories` accepts arrays shaped (N, L, dim) with dim in {2, 3};
# set `store_visualization=True` to also save a PNG preview alongside the .npy.
logger.info("Logging sample trajectories (2D + 3D)...")
rng = np.random.default_rng(0)
N, L = 6, 40

# 2-D random walk per trajectory
trajectories_2d = np.cumsum(rng.normal(size=(N, L, 2), scale=0.1), axis=1)
logger.trajectories(
    trajectories_2d,
    name="random_walk_2d",
    store_visualization=True,
    step=9,
)

# 3-D random walk per trajectory (matplotlib picks a 3D projection)
trajectories_3d = np.cumsum(rng.normal(size=(N, L, 3), scale=0.1), axis=1)
logger.trajectories(
    trajectories_3d,
    name="random_walk_3d",
    store_visualization=True,
    step=9,
)

# Histogram logging
logger.info("Logging sample histogram...")
histogram_data = np.random.randn(1000)
logger.histogram(histogram_data, name="sample_histogram", step=9)

print()
print("✓ All events logged successfully!")
print()
print("Check the created structure:")
print("- examples/logs/log.jsonl (contains all events with media file paths)")
print("- examples/logs/images/ (contains saved image files)")
print("- examples/logs/videos/ (contains saved video files)")
print("- examples/logs/artifacts/ (contains saved artifact files)")
print("- examples/logs/vector_fields/ (contains saved vector field files)")
print("- examples/logs/trajectories/ (contains saved trajectory files)")
print("- examples/logs/histograms/ (contains saved histogram files)")
print("- Media files are referenced by relative paths in the JSONL log")

gg.finish()
