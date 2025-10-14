"""Demonstration of the History module.

This script shows how to:
1. Define a HistorySpec.
2. Create GPU-resident histories.
3. Update them over time with resets.
4. Move data between host and device.
5. Compare JIT vs non-JIT update performance.
"""

import time
import os
from typing import Callable, Dict

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0 if available
import jax
import jax.numpy as jnp

from goggles.history import (
    create_history,
    update_history,
    to_device,
    to_host,
)
from goggles.history.spec import HistorySpec
from goggles.history.types import History

try:
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def main() -> None:  # noqa: D103
    print("=" * 80)
    print(" History Module Demonstration")
    print("=" * 80)
    print(f"Available devices: {[str(d) for d in jax.devices()]}")

    # ------------------------------------------------------------------
    # 1. Define a simple config and parse it into a HistorySpec
    # ------------------------------------------------------------------
    config = {
        "state": {"length": 4, "shape": (2,), "dtype": jnp.float32, "init": "zeros"},
        "reward": {"length": 4, "shape": (), "dtype": jnp.float32, "init": "randn"},
    }

    spec = HistorySpec.from_config(config)

    rng = jax.random.key(0)
    batch_size = 3

    # ------------------------------------------------------------------
    # 2. Create the history and move it to device
    # ------------------------------------------------------------------
    history = create_history(spec, batch_size, rng)
    history = to_device(history)

    print("\nCreated history:")
    for k, v in history.items():
        print(f"  {k:>8}: shape={v.shape}, dtype={v.dtype}, device={v.device}")

    # ------------------------------------------------------------------
    # 3. Define update function (same for JIT and non-JIT)
    # ------------------------------------------------------------------
    def simulate_updates(
        update_fn: Callable, n_steps: int = 20
    ) -> Dict[str, History | float]:
        hist = history
        t0 = time.perf_counter()
        for t in range(n_steps):
            rng_t, subkey = jax.random.split(jax.random.fold_in(rng, t))
            new_data = {
                "state": jnp.ones((batch_size, 1, 2), jnp.float32) * t,
                "reward": jnp.full((batch_size, 1), float(t), jnp.float32),
            }
            reset_mask = jax.random.bernoulli(subkey, p=0.2, shape=(batch_size,))
            hist = update_fn(hist, new_data, reset_mask)

        jax.block_until_ready(hist)
        elapsed = time.perf_counter() - t0
        return {"history": hist, "elapsed": elapsed}

    # ------------------------------------------------------------------
    # 4. Non-JIT execution
    # ------------------------------------------------------------------
    print("\nRunning sequential updates (non-JIT)...")
    out_nonjit = simulate_updates(update_history)
    print(f"Non-JIT execution time: {out_nonjit['elapsed']:.4f} s")

    # ------------------------------------------------------------------
    # 5. JIT execution
    # ------------------------------------------------------------------
    print("\nRunning JIT-compiled updates...")
    update_history_jit = jax.jit(update_history)
    out_jit = simulate_updates(update_history_jit)
    print(f"JIT execution time: {out_jit['elapsed']:.4f} s")

    # ------------------------------------------------------------------
    # 6. Move back to host and inspect
    # ------------------------------------------------------------------
    host_history = to_host(out_jit["history"])  # type: ignore
    print("\nFinal host history:")
    for k, v in host_history.items():
        print(f"  {k}: mean={v.mean():.3f}, shape={v.shape}")

    # ------------------------------------------------------------------
    # 7. Optional visualization
    # ------------------------------------------------------------------
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
        print("\n(Matplotlib not installed — skipping plot.)")

    # ------------------------------------------------------------------
    # 8. Timing summary
    # ------------------------------------------------------------------
    print("\nTiming summary:")
    print(f"  Non-JIT: {out_nonjit['elapsed']:.4f} s")
    print(f"  JIT:     {out_jit['elapsed']:.4f} s")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 9. Further examples
    # ------------------------------------------------------------------
    print(
        "\nFor a more complete demonstration of how GPU-resident histories are used "
        "in real environments and estimators, check out Flow Gym:"
    )
    print("  🔗 https://github.com/antonioterpin/flowgym")


if __name__ == "__main__":
    main()
