import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np

import goggles as gg
from goggles import WandBHandler

# In this example, we set up a logger that outputs to Weights & Biases (W&B).
# `group` and `tags` flow straight through to `wandb.init`, so every run this
# handler creates lands in the same W&B group with the same searchable tags.
# Use `wandb_init_kwargs` for less-common `wandb.init` options that Goggles
# does not expose directly. Here, W&B saves the entry-point source file and
# looks for code relative to the repository root.
logger: gg.GogglesLogger = gg.get_logger("examples.basic", with_metrics=True)
handler = WandBHandler(
    project="goggles_example",
    run_name="example_run",
    group="goggles_example_group",
    tags=["example", "smoke-test"],
    wandb_init_kwargs={
        "save_code": True,
        "settings": {"code_dir": "."},
    },
)
gg.attach(handler, scopes=["global"])


logger.info(
    "Logging to Weights & Biases!"
)  # This will be ignored because there's no log handler attached yet
for i in range(100):
    logger.scalar("accuracy", i, step=i)

# Generate and log an image
image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
logger.image(image, name="Random image", step=100)

# Generate and log an RGBA image (with alpha channel)
rgba_image = np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8)
# Create a gradient alpha effect
for i in range(100):
    rgba_image[i, :, 3] = int(255 * (i / 100))  # Vertical alpha gradient
logger.image(rgba_image, name="Random RGBA image", step=100)

# Grayscale GIF video
gray_video_gif = np.random.randint(
    0, 255, (30, 64, 64), dtype=np.uint8
)  # (F, H, W), also accepted as (F, 1, H, W)
logger.video(
    gray_video_gif,
    name="Grayscale GIF Video",
    fps=15,
    format="gif",
    step=100,
)

# Grayscale MP4 video
gray_video_mp4 = np.random.randint(
    0, 255, (30, 64, 64), dtype=np.uint8
)  # (F, H, W), also accepted as (F, 1, H, W)
logger.video(
    gray_video_mp4,
    name="Grayscale MP4 Video",
    fps=10,
    format="mp4",
    step=100,
)

# Generate and log a video
video = np.random.randint(
    0, 255, (30, 3, 64, 64), dtype=np.uint8
)  # 30 frames of 64x64 RGB
logger.video(video, name="Random Video", fps=10, step=100)

# Generate and log an RGBA video (with alpha channel).
# Layout is channels-first: (frames, channels, H, W) = (15, 4, 64, 64).
# The alpha channel is axis 1 index 3 — index that, not the trailing dim.
rgba_video = np.random.randint(0, 255, (15, 4, 64, 64), dtype=np.uint8)
# Create a fading effect over time
for t in range(15):
    alpha_value = int(255 * (1 - t / 14))  # Fade out over time
    rgba_video[t, 3, :, :] = alpha_value
logger.video(rgba_video, name="Random RGBA Video", fps=5, step=100)

# Load and log artifact: WandBHandler expects {path, name, type} where
# `path` points to a file on disk to be uploaded as a W&B artifact.
# WandB uploads the file asynchronously, so the local copy must outlive
# every later logger call (and gg.finish()) — that's why the temp dir
# is held open until the very end and the cleanup is wrapped in a
# try/finally so a crash anywhere in the example still removes it.
artifact_dir = tempfile.TemporaryDirectory(prefix="goggles-example-")
try:
    artifact_file = Path(artifact_dir.name) / "random_artifact.npy"
    np.save(artifact_file, np.random.rand(100, 100, 3))
    logger.artifact(
        {
            "path": str(artifact_file),
            "name": "random_artifact",
            "type": "misc",
        },
        step=100,
    )

    # Directory uploads + aliases: an Orbax/PyTorch checkpoint is a
    # tree of files, so pass the *directory* as `path` and W&B will
    # add it recursively. `aliases` tags the version inside the W&B
    # artifact collection (think `:best`, `:latest`) — handy when you
    # want a stable URL pointing to your best checkpoint.
    ckpt_dir = Path(artifact_dir.name) / "checkpoint_step_100"
    ckpt_dir.mkdir()
    np.save(ckpt_dir / "params.npy", np.random.rand(8, 8))
    np.save(ckpt_dir / "opt_state.npy", np.random.rand(8, 8))
    logger.artifact(
        {
            "path": str(ckpt_dir),
            "name": "model_checkpoint",
            "type": "checkpoint",
            "aliases": ["best", "step-100"],
        },
        step=100,
    )

    def lamb_oseen_velocity(
        points: np.ndarray,
        vortices: list[tuple[float, float, float, float]],
        uniform: tuple[float, float] = (0.0, 0.0),
    ) -> np.ndarray:
        """Evaluate a sum of Lamb-Oseen vortices at arbitrary 2D points.

        Args:
            points: Array of shape ``(..., 2)`` of ``(x, y)`` positions.
            vortices: List of tuples ``(x0, y0, Gamma, sigma)`` defining the
                position, circulation strength, and core size of each vortex.
            uniform: ``(u, v)`` uniform flow component to add to the field.

        Returns:
            Velocity array of shape ``(..., 2)``.
        """
        x = points[..., 0]
        y = points[..., 1]
        u = np.full_like(x, uniform[0], dtype=np.float32)
        v = np.full_like(y, uniform[1], dtype=np.float32)

        eps = 1e-6
        for x0, y0, Gamma, sigma in vortices:
            dx = x - x0
            dy = y - y0
            r2 = dx * dx + dy * dy
            r = np.sqrt(r2) + eps
            v_theta = (
                (Gamma / (2.0 * np.pi))
                * (1.0 - np.exp(-r2 / (2.0 * sigma * sigma)))
                / r
            )
            u += -dy * (v_theta / r)
            v += dx * (v_theta / r)
        return np.stack([u, v], axis=-1)

    def make_lamb_oseen_vortices(
        H: int,
        W: int,
        vortices: list[tuple[float, float, float, float]],
        domain: tuple[float, float, float, float] = (-1.0, 1.0, -1.0, 1.0),
        uniform: tuple[float, float] = (0.0, 0.0),
    ) -> np.ndarray:
        """Build a 2D incompressible flow from Lamb-Oseen vortices.

        Args:
            H: Height of the output vector field.
            W: Width of the output vector field.
            vortices: List of tuples (x0, y0, Gamma, sigma) defining the
                position, circulation strength, and core size of each vortex.
            domain: (xmin, xmax, ymin, ymax) defining the spatial extent of
                the field.
            uniform: (u, v) uniform flow component to add to the field.

        Returns:
            A (H, W, 2) array containing the (u, v) vector field.
        """
        xmin, xmax, ymin, ymax = domain
        ys = np.linspace(ymin, ymax, H, dtype=np.float32)
        xs = np.linspace(xmin, xmax, W, dtype=np.float32)
        X, Y = np.meshgrid(xs, ys)
        grid = np.stack([X, Y], axis=-1)
        return lamb_oseen_velocity(grid, vortices, uniform=uniform)

    def abc_flow_velocity(
        points: np.ndarray,
        A: float = 1.0,
        B: float = float(np.sqrt(2.0 / 3.0)),
        C: float = float(np.sqrt(1.0 / 3.0)),
    ) -> np.ndarray:
        """Evaluate the Arnold-Beltrami-Childress flow at arbitrary 3D points.

        Args:
            points: Array of shape ``(..., 3)`` of ``(x, y, z)`` positions.
            A: Coefficient on the ``sin(z)/cos(z)`` terms.
            B: Coefficient on the ``sin(x)/cos(x)`` terms.
            C: Coefficient on the ``cos(y)/sin(y)`` terms.

        Returns:
            Velocity array of shape ``(..., 3)``.
        """
        x = points[..., 0]
        y = points[..., 1]
        z = points[..., 2]
        u = A * np.sin(z) + C * np.cos(y)
        v = B * np.sin(x) + A * np.cos(z)
        w = C * np.sin(y) + B * np.cos(x)
        return np.stack([u, v, w], axis=-1).astype(np.float32)

    def integrate_particles(
        init_positions: np.ndarray,
        velocity_fn: Callable[[np.ndarray], np.ndarray],
        n_steps: int,
        dt: float,
    ) -> np.ndarray:
        """Integrate particles through a velocity field with RK4.

        Args:
            init_positions: Array of shape ``(N, dim)`` with seed positions.
            velocity_fn: Callable mapping ``(..., dim)`` positions to
                ``(..., dim)`` velocities.
            n_steps: Number of integration steps.
            dt: Step size.

        Returns:
            Trajectory array of shape ``(N, n_steps + 1, dim)``.
        """
        N, dim = init_positions.shape
        out = np.empty((N, n_steps + 1, dim), dtype=np.float32)
        out[:, 0] = init_positions
        p = init_positions.astype(np.float32)
        for s in range(n_steps):
            k1 = velocity_fn(p)
            k2 = velocity_fn(p + 0.5 * dt * k1)
            k3 = velocity_fn(p + 0.5 * dt * k2)
            k4 = velocity_fn(p + dt * k3)
            p = p + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            out[:, s + 1] = p
        return out

    # Vector field logging to W&B
    VF_H, VF_W = 128, 128
    vortices = [
        (-0.4, 0.0, +5.0, 0.15),
        (+0.4, 0.0, -5.0, 0.15),
    ]
    dummy_vector_field = make_lamb_oseen_vortices(
        VF_H, VF_W, vortices, uniform=(0.2, 0.0)
    )

    logger.vector_field(
        dummy_vector_field,
        name="lamb_oseen_dipole_magnitude",
        mode="magnitude",
        step=101,
    )
    logger.vector_field(
        dummy_vector_field,
        name="lamb_oseen_dipole_vorticity",
        mode="vorticity",
        add_colorbar=True,
        step=102,
    )

    # Trajectories logging. `logger.trajectories` accepts (N, L, dim)
    # arrays with dim in {2, 3}. Here we seed a grid of particles and
    # advect them through the same Lamb-Oseen dipole used above, so the
    # result is the swirling flow rendered as paths.
    seed_axis = np.linspace(-0.7, 0.7, 8, dtype=np.float32)
    seed_x, seed_y = np.meshgrid(seed_axis, seed_axis)
    init_positions_2d = np.stack([seed_x.ravel(), seed_y.ravel()], axis=-1)
    trajectories_2d = integrate_particles(
        init_positions_2d,
        lambda p: lamb_oseen_velocity(p, vortices, uniform=(0.2, 0.0)),
        n_steps=120,
        dt=0.02,
    )
    logger.trajectories(
        trajectories_2d,
        name="lamb_oseen_dipole_trajectories",
        step=103,
    )

    # 3D trajectories — particles advected through the chaotic ABC flow.
    # matplotlib picks a 3D projection automatically when dim == 3.
    rng = np.random.default_rng(0)
    init_positions_3d = rng.uniform(0.0, 2.0 * np.pi, size=(48, 3)).astype(
        np.float32
    )
    trajectories_3d = integrate_particles(
        init_positions_3d,
        abc_flow_velocity,
        n_steps=200,
        dt=0.05,
    )
    logger.trajectories(
        trajectories_3d,
        name="abc_flow_trajectories",
        step=104,
    )

    # Add extra fields to any metric logged to be used as x-axis in W&B.
    # Start at 104 so we stay monotonic with the trajectories above.
    for i in range(104, 154):
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
    logger.histogram(
        data, name="Random Values Histogram", static=True, step=154
    )

    # Or a dynamic histogram (that changes over time)
    for i in range(10):
        data = np.random.randn(1000) + i  # Shift mean over time
        logger.histogram(
            data, name="Dynamic Random Values Histogram", step=154 + i
        )

    # When using asynchronous logging (like wandb), make sure to finish.
    # ``gg.finish()`` waits indefinitely by default so no queued events are
    # dropped; pass ``timeout=T`` if you need a bound.
    gg.finish()
finally:
    artifact_dir.cleanup()
