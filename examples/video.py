"""Example of combining Goggles schedule log for saving file & video logging."""

import time
import numpy as np
import matplotlib.pyplot as plt
from goggles.logger import Goggles
import os

os.makedirs("/tmp/flows", exist_ok=True)


def generate_random_flow(width=16, height=16):
    """Generate a random optical flow field.

    Args:
        width (int): Width of the flow field.
        height (int): Height of the flow field.

    Returns:
        np.ndarray: A 2D array of shape (height, width, 2)
    """
    u = np.random.uniform(-1, 1, size=(height, width))
    v = np.random.uniform(-1, 1, size=(height, width))
    flow = np.stack((u, v), axis=-1)
    return flow


def save_flow_to_file(flow, idx):
    """Save the flow field as a numpy array to a file.

    Args:
        flow (np.ndarray): Optical flow field of shape (height, width, 2).
        idx (int): Index for logging.
    """
    np.save(f"/tmp/flows/flow_{idx}.npy", flow)


def viz_and_log(n_steps):
    """Visualize the optical flow fields and log them as a video.

    Args:
        n_steps (int): Number of flow fields to visualize.
    """
    frames = []
    for i in range(n_steps):
        # load flow from file
        flow = np.load(f"/tmp/flows/flow_{i}.npy")

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
        frames.append((rgb * 255).astype(np.uint8))

    Goggles.video("flow_viz", np.stack(frames, axis=0))


# Configure Goggles
Goggles.set_config(wandb_project="test")
Goggles.init_scheduler(num_workers=4)
N_FLOWS = 1000

for i in range(N_FLOWS):
    flow = generate_random_flow()

    queue_size = Goggles._task_queue.qsize()
    Goggles.scalar("queue_size", queue_size)
    # Schedule asynchronous logs
    start = time.perf_counter()
    Goggles.schedule_log(save_flow_to_file, flow, i)
    schedule_time = time.perf_counter() - start

    # Log how much time it took to generate the flow and schedule the log
    Goggles.scalar("schedule_time", schedule_time)

Goggles.stop_workers()
# Now visualize and log the flows
viz_and_log(N_FLOWS)
