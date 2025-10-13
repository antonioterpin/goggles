"""Example of using schedule log to prepare time-intensive visuals to log."""

import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import goggles


def generate_random_flow(width=64, height=64):
    """Generate a random optical flow field.

    Args:
        width (int): Width of the flow field.
        height (int): Height of the flow

    Returns:
        np.ndarray: A 2D array of shape (height, width, 2)

    """
    u = np.random.uniform(-1, 1, size=(height, width))
    v = np.random.uniform(-1, 1, size=(height, width))
    flow = np.stack((u, v), axis=-1)
    return flow


def viz_and_log(flow, idx):
    """Visualize the optical flow and log it.

    Args:
        flow (np.ndarray): Optical flow field of shape (height, width, 2).
        idx (int): Index for logging.

    """
    u, v = flow[..., 0], flow[..., 1]
    angle = np.arctan2(v, u)  # [-pi, pi]
    magnitude = np.sqrt(u**2 + v**2)

    hue = (angle + np.pi) / (2 * np.pi)  # [0,1]
    mag_norm = np.clip(magnitude / (np.max(magnitude) + 1e-6), 0, 1)

    hsv = np.zeros(flow.shape[:2] + (3,), dtype=np.float32)
    hsv[..., 0] = hue
    hsv[..., 1] = 1.0
    hsv[..., 2] = mag_norm

    rgb = plt.cm.hsv(hsv[..., 0])[:, :, :3] * hsv[..., 2][..., None]
    viz = Image.fromarray((rgb * 255).astype(np.uint8))

    # Log the visualization
    goggles.image(f"flow_viz_{idx}", viz)


# Configure Goggles
goggles.init_scheduler(num_workers=10)

for i in range(1000):
    flow = generate_random_flow()

    queue_size = goggles._task_queue.qsize()
    goggles.scalar("queue_size", queue_size)
    # Schedule visualization and logging
    start = time.perf_counter()
    goggles.schedule_log(viz_and_log, flow, i)
    schedule_time = time.perf_counter() - start
    goggles.scalar("schedule_time", schedule_time)

goggles.cleanup()
