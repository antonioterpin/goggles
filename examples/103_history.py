"""Demonstration of the History module.

This script shows how to:
1. Define a HistorySpec.
2. Create GPU-resident histories.
3. Update them over time with resets.
4. Move data between host and device.
5. Compare JIT vs non-JIT update performance.
"""

import os
import time
from collections.abc import Callable

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0 if available
import jax
import jax.numpy as jnp

from goggles.history import (
    create_history,
    to_host,
    update_history,
)
from goggles.history.spec import HistorySpec
from goggles.history.types import History

try:
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def main() -> None:
    # Define a simple config and parse it into a HistorySpec
    config = {
        "state": {
            "length": 4,
            "shape": (2,),
            "dtype": jnp.float32,
            "init": "zeros",
        },
        "reward": {
            "length": 4,
            "shape": (),
            "dtype": jnp.float32,
            "init": "randn",
        },
    }

    spec = HistorySpec.from_config(config)

    rng = jax.random.key(0)
    batch_size = 3

    # Create history on device (GPU if available).
    history = create_history(spec, batch_size, rng)

    print("\nCreated history:")
    for k, v in history.items():
        print(f"  {k:>8}: shape={v.shape}, dtype={v.dtype}, device={v.device}")

    # Define update function
    def simulate_updates(
        update_fn: Callable, n_steps: int = 20
    ) -> tuple[History, float]:
        hist = history
        t0 = time.perf_counter()
        for t in range(n_steps):
            rng_t = jax.random.fold_in(rng, t)
            new_data = {
                "state": jnp.ones((batch_size, 1, 2), jnp.float32) * t,
                "reward": jnp.full((batch_size, 1), float(t), jnp.float32),
            }
            reset_mask = jax.random.bernoulli(rng_t, p=0.2, shape=(batch_size,))
            hist = update_fn(hist, new_data, reset_mask)

        jax.block_until_ready(hist)
        elapsed = time.perf_counter() - t0
        return hist, elapsed

    # Non-JIT execution
    print("\nRunning sequential updates (non-JIT)...")
    _nonjit_hist, nonjit_elapsed = simulate_updates(update_history)
    print(f"Non-JIT execution time: {nonjit_elapsed:.4f} s")

    # JIT execution
    print("\nRunning JIT-compiled updates...")
    update_history_jit = jax.jit(update_history)
    jit_hist, jit_elapsed = simulate_updates(update_history_jit)
    print(f"JIT execution time: {jit_elapsed:.4f} s")

    # Move back to host and inspect
    host_history = to_host(jit_hist)
    print("\nFinal host history:")
    for k, v in host_history.items():
        print(f"  {k}: mean={v.mean():.3f}, shape={v.shape}")

    # Optional visualization
    if HAS_MPL:
        rewards = host_history["reward"]
        timesteps = jnp.arange(rewards.shape[1])
        plt.figure(figsize=(6, 3))
        for b in range(batch_size):
            plt.plot(timesteps, rewards[b], label=f"batch {b}")
        plt.xlabel("Time step")
        plt.ylabel("Reward")
        plt.title("Reward history trajectories")
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        print("\n(Matplotlib not installed - skipping plot.)")

    # Timing summary
    print("\nTiming summary:")
    print(f"  Non-JIT: {nonjit_elapsed:.4f} s")
    print(f"  JIT:     {jit_elapsed:.4f} s")
    print("=" * 80)

    # Further examples
    print(
        "\nFor a complete demonstration of how GPU-resident histories are "
        "used "
        "in real environments and estimators, check out Flow Gym:"
    )
    print("  🔗 https://github.com/antonioterpin/flowgym")


if __name__ == "__main__":
    main()
