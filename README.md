# 😎 Goggles

A lightweight, flexible Python logging and monitoring library designed to simplify and enhance experiment tracking, performance profiling, and error tracing. Integrates with terminal, file-based logs, and W\&B (Weights & Biases). It is thought primarily for research projects in robotics.

```bash
pip install "goggles @ git+ssh://git@github.com/antonioterpin/goggles.git"
```

## Features

- 🤖 **Multi-process compatible**
  Synchronize logs from all spawned processes via shared memory.

- 🎯 **Multi-output logging**
  Log to terminal and/or file (configurable via `.goggles-default.yaml`).

- 🕒 **Performance profiling**
  `@goggles.timeit` decorator measures and logs runtime.

- 🐞 **Error tracing**
  `@goggles.trace_on_error` auto-logs full stack on exceptions.

- 📊 **Metrics tracking**
  `goggles.scalar`, `goggles.vector`, `goggles.image`, `goggles.video` → Weights & Biases.

- 🚦 **Graceful shutdown**
  Call `goggles.cleanup()` (or hook into your own `signal` handler).

- ⚙️ **Asynchronous scheduling**
  Offload heavy logging tasks via `goggles.schedule_log(...)`.

- 📁 **Pretty configuration loading**
  `goggles.load_configuration(...)` loads YAML with validation.

## Quickstart

1. **Create `.goggles-default.yaml`** in your project root:

   ```yaml
   logdir: ~/my_logs
   to_terminal: true
   level: INFO
   wandb_project: demo_project
   enable_signal_handler: false

2. **Ready to log**:

    ```python
    import goggles

    goggles.debug("Debugging details…")
    goggles.info("Experiment started")
    goggles.warning("This is a warning")
    goggles.error("An error occurred")
    ```

We cleanup all the resources automatically at exit.

## Configuration

Pretty logging of configuration files.

```python
import goggles

# Load from examples/example_config.yaml
config = goggles.load_configuration("examples/example_config.yaml")
print(config)

# Access as dict
print(f"time_per_experiment = {config['time_per_experiment']}")
```

## Decorators: `@goggles.timeit` and `@goggles.trace_on_error`

Measure execution time of methods or functions:

```python
import goggles

class Worker:
    @goggles.timeit(severity=goggles.Severity.DEBUG)
    def compute_heavy(self, n):
        return sum(range(n))

    @goggles.trace_on_error()
    def risky_division(self, x, y):
        return x / y

g = Worker()
g.compute_heavy(1_000_000)

try:
    g.risky_division(1, 0)
except ZeroDivisionError:
    pass  # Full traceback was logged
```

## File & Terminal Logging

All driven by your .goggles-default.yaml.

## W\&B Integration

Log scalars, vectors, images, and videos directly to Weights & Biases:

```python
import goggles
from PIL import Image
import numpy as np

# Start or switch a W&B run
goggles.new_wandb_run(name="exp-run", config={"lr":1e-3, "batch":32})

# Scalars & histograms
goggles.scalar("accuracy", 0.92)
goggles.vector("loss_curve", [0.5,0.4,0.3])

# Images
img = Image.fromarray((np.random.rand(64,64,3)*255).astype(np.uint8))
goggles.image("random_image", img)
```

## Graceful Shutdown

Cleanly handle interrupts (e.g., Ctrl-C) and perform cleanup:

```python
import goggles
from PIL import Image
import numpy as np

# Start or switch a W&B run
goggles.new_wandb_run(name="exp-run", config={"lr":1e-3, "batch":32})

# Scalars & histograms
goggles.scalar("accuracy", 0.92)
goggles.vector("loss_curve", [0.5,0.4,0.3])

# Images
img = Image.fromarray((np.random.rand(64,64,3)*255).astype(np.uint8))
goggles.image("random_image", img)

```

## Asynchronous Logging & Video

Offload heavy logging tasks to worker threads and log video sequences:

```python
import goggles, numpy as np, time
from PIL import Image

goggles.new_wandb_run("video_demo", {})
goggles.init_scheduler(num_workers=4)

def save_and_log_frame(frame, idx):
    path = f"/tmp/frame_{idx}.png"
    frame.save(path)
    goggles.image(f"frame_{idx}", frame)

for i in range(100):
    arr = (np.random.rand(64,64,3)*255).astype(np.uint8)
    img = Image.fromarray(arr)
    goggles.schedule_log(save_and_log_frame, img, i)
    goggles.scalar("queue_size", goggles._task_queue.qsize())

goggles.stop_workers()
```

## Full Examples

See the `examples/` folder for scripts covering:

- Config loading
- Decorators
- File vs. terminal logs
- W&B scalar/vector/image/video
- Graceful shutdown
- Async scheduling

## Contributing

PRs, issues, and feature requests are welcome! Open an issue or submit a PR on GitHub.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## ⚡ GPU-Resident History Module (JAX)

Goggles now provides a **GPU-resident temporal history subsystem** designed for reusable,
JAX-based on-device memory management.
It’s primarily used by FlowGym and other JAX pipelines that need batched temporal buffers.

### Installation

To use the GPU history module:

#### CPU only

```bash
pip install "goggles[jax]"
```

#### With CUDA (GPU)

Install JAX with GPU support **before** Goggles:

```bash
# Example for CUDA 12
pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install "goggles[jax]"
```

### Example

```python
from goggles.history import HistorySpec, create_history, update_history
import jax.numpy as jnp

spec = HistorySpec.from_config({
    "images": {"length": 4, "shape": (64, 64, 2), "dtype": jnp.float32},
    "flow":   {"length": 2, "shape": (64, 64, 2), "dtype": jnp.float32},
})

history = create_history(spec, batch_size=8)
print(history["images"].shape)  # (8, 4, 64, 64, 2)
```

### Design Notes

- Implements **pure functional**, **JIT-safe** buffer updates using `jax.numpy` and `jax.lax`.
- Each field follows the convention `(B, T, *shape)`.
- Supports **per-field history lengths** and **reset masks** for episodic updates.
- Integrated with FlowGym’s `EstimatorState` for GPU-local temporal memory.

---

📦 **Optional Dependencies**

Goggles keeps JAX optional.
Install JAX support only if you need GPU-resident histories:

```toml
[project.optional-dependencies]
jax = ["jax>=0.4.0", "jaxlib>=0.4.0"]
```

---

### 🧠 Future Directions

- NumPy backend for CPU-only fallback
- Device sharding support (GSPMD)
- Seamless integration with RL replay buffers
