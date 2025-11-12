# 😎 Goggles - Observability for Robotics Research

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![GitHub stars](https://img.shields.io/github/stars/antonioterpin/robostack?style=social)](https://github.com/antonioterpin/goggles/stargazers)
[![codecov](https://codecov.io/gh/antonioterpin/goggles/graph/badge.svg?token=J49B8TFDSM)](https://codecov.io/gh/antonioterpin/goggles)
[![Tests](https://github.com/antonioterpin/goggles/actions/workflows/test.yaml/badge.svg)](https://github.com/antonioterpin/goggles/actions/workflows/test.yaml)
[![Code Style](https://github.com/antonioterpin/goggles/actions/workflows/code-style.yaml/badge.svg)](https://github.com/antonioterpin/goggles/actions/workflows/code-style.yaml)
[![PyPI version](https://img.shields.io/pypi/v/robo-goggles.svg)](https://pypi.org/project/robo-goggles)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)


A lightweight, flexible Python observability framework designed for robotics research. Goggles provides structured logging, experiment tracking, performance profiling, and device-resident temporal memory management for JAX-based pipelines.

## ✨ Features

- 🤖 **Multi-process (and multi-machines) logging** - Synchronize logs across spawned processes reliably and efficiently (shared memory when available).
- 🎯 **Multi-output support** - Log to console, files, and remote services simultaneously.
- 📊 **Experiment tracking** - Native integration with Weights & Biases for metrics, images, and videos.
- 🕒 **Performance profiling** - `@goggles.timeit` decorator for automatic runtime measurement.
- 🐞 **Error tracing** - `@goggles.trace_on_error` auto-logs full stack traces on exceptions.
- 🧠 **Device-resident histories** - JAX-based GPU memory management for efficient, long-running experiments metrics.
- 🚦 **Graceful shutdown** - Automatic cleanup of resources and handlers.
- ⚙️ **Structured configuration** - YAML-based config loading with validation.
- 🔌 **Extensible handlers** - Plugin architecture for custom logging backends.

## 🏗️ Projects Built with Goggles

This framework has been battle-tested across multiple research projects:

[![FluidsControl](https://img.shields.io/badge/GitHub-antonioterpin%2Ffluidscontrol-2ea44f?logo=github)](https://github.com/antonioterpin/fluidscontrol)
[![FlowGym](https://img.shields.io/badge/GitHub-antonioterpin%2Fflowgym-2ea44f?logo=github)](https://github.com/antonioterpin/flowgym)
[![SynthPix](https://img.shields.io/badge/GitHub-antonioterpin%2Fsynthpix-2ea44f?logo=github)](https://github.com/antonioterpin/synthpix)
[![Πnet](https://img.shields.io/badge/GitHub-antonioterpin%2Fpinet-2ea44f?logo=github)](https://github.com/antonioterpin/pinet)
[![Glitch](https://img.shields.io/badge/GitHub-antonioterpin%2Fglitch-2ea44f?logo=github)](https://github.com/antonioterpin/glitch)

## 🚀 Quick Start

### Installation

```bash
# Basic installation
uv add robo-goggles # or pip install robo-goggles

# With Weights & Biases support
uv add "robo-goggles[wandb]"

# With JAX device-resident histories
uv add "robo-goggles[jax]"
```

For the development installation, see our [How to contribute](./CONTRIBUTING.md) page.

> [!WARNING]
> **Port selection**: Goggles requires a port for communication, and works multi-process, multi-machine, multi-user. If different projects have the same port, the behavior is undefined. You can set a unique port for each projet by setting in `.env` the variable `GOGGLES_PORT`.


### Basic usage

```python
import goggles as gg
import logging

# Set up console logging
logger = gg.get_logger("my_experiment")
gg.attach(
    gg.ConsoleHandler(name="console", level=logging.INFO),
)

# Basic logging
logger.info("Experiment started")
logger.warning("This is a warning")
logger.error("An error occurred")

# Goggles works by default in async mode,
# to ensure all the jobs are finished use
gg.finish()
```

See also [Example 1](./examples/01_basic_run.py), which you can run after cloning the repo with
```bash
uv run examples/01_basic_run.py
```

### Experiment tracking with W&B

```python
import goggles as gg
import numpy as np

# Enable metrics logging
logger = gg.get_logger("experiment", with_metrics=True)
gg.attach(
    gg.WandBHandler(project="my_project", name="run_1"),
)

# Log metrics, images, and videos
for step in range(100):
    logger.scalar("loss", np.random.random(), step=step)
    logger.scalar("accuracy", 0.8 + 0.2 * np.random.random(), step=step)

# Log images and videos
image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
logger.image(image, name="sample_image", step=100)

video = np.random.randint(0, 255, (30, 3, 64, 64), dtype=np.uint8)
logger.video(video, name="sample_video", fps=10, step=100)

gg.finish()
```

### Performance profiling and error tracking

```python
import goggles as gg
import logging

class Trainer:
    @gg.timeit(severity=logging.INFO)
    def train_step(self, batch):
        # Your training logic here
        return {"loss": 0.1}

    @gg.trace_on_error()
    def risky_operation(self, data):
        # This will log full traceback on any exception
        return data / 0  # Will trigger trace logging

trainer = Trainer()
trainer.train_step({"x": [1, 2, 3]})  # Logs execution time

try:
    trainer.risky_operation(10)
except ZeroDivisionError:
    pass  # Full traceback was automatically logged
```

### Configuration Management

Load and validate YAML configurations:

```python
import goggles

# Load configuration with automatic validation
config = goggles.load_configuration("config.yaml")
print(config) # Pretty print
print(config["learning_rate"])  # Access as dict

# Pretty-print configuration
goggles.save_configuration(config, "output.yaml")
```

### Supported Platforms 💻

| Platform | Basic | W&B | JAX/GPU | Development |
|----------|-------|-----|---------|-------------|
| Linux    | ✅    | ✅   | ✅      | ✅          |
| macOS    | ✅    | ✅   | ✅      | ✅          |
| Windows  | ✅    | ✅   | ❌      | ✅          |

*GPU support requires CUDA-compatible hardware and drivers*

## 🔥 Examples

Explore the `examples/` directory for comprehensive usage patterns:

```bash
# Basic logging setup
uv run examples/01_basic_run.py

# Advanced: Multi-scope logging
uv run examples/02_multi_scope.py

# File-based logging (local storage)
uv run examples/03_local_storage.py

# Weights & Biases integration
uv run examples/04_wandb.py

# Advanced: Weights & Biases multi-run setup
uv run examples/05_wandb_multiple_runs.py

# Advanced: Custom handler
uv run exacmples/06_custom_handler.py

# Graceful shutdown utils
uv run examples/100_interrupt.py

# Pretty and convenient utils for configuration laoding
uv run examples/101_config.py

# Advanced: Performance decorators
uv run examples/102_decorators.py

# Advanced: JAX device-resident histories
uv run examples/103_history.py
```

## 🧠 For Goggles power user

This section includes some cool functionalities of `goggles`. Enjoy!

### Multi-scope logging
Goggles allow easily to set up different handlers for different scopes. That is, one can have an handler attached to multiple scopes, and a scope having multiple handlers. Each logger is associated to a single scope (by default: `global`), and logging with that logger will invoke all the loggers associated with the scope.

#### Why?
Within the same run, we may have logs that belong to different scopes. An example is training in Reinforcement Learning, where in a single training run there are multiple episodes. A complete example for this is provided in the [multiple runs in WandB](#multiple-runs-in-wandb) section.

#### Usage

```python
# In this example, we set up a handlers associated
# to different scopes.
handler1 = gg.ConsoleHandler(name="examples.basic.console.1", level=logging.INFO)
gg.attach(handler1, scopes=["global", "scope1"])

handler2 = gg.ConsoleHandler(name="examples.basic.console.2", level=logging.INFO)
gg.attach(handler2, scopes=["global", "scope2"])

# We need to get separate loggers for each scope
logger_scope1 = gg.get_logger("examples.basic.scope1", scope="scope1")
logger_scope2 = gg.get_logger("examples.basic.scope2")
logger_scope2.bind(scope="scope2")  # You can also bind the scope after creation
logger_global = gg.get_logger("examples.basic.global", scope="global")

# Now we can log messages to different scopes, so that only the interested
# handlers will process them.
logger_scope1.info(f"This will be logged only by {handler1.name}")
logger_scope2.info(f"This will be logged only by {handler2.name}")
logger_global.info("This will be logged by both handlers.")

# The same result can be achieved using namespaces,
# which are indicated by dot notation.
logger_namespace = gg.get_logger("examples.basic.namespace", scope="namespace")
logger_namespace.info("This will be logged by both handlers.")

gg.finish()
```

See also [examples/02_multi_scope.py](./examples/02_multi_scope.py) for a running example.

### Multiple runs in WandB
An example of the benefit of scopes is given by the WandBHandler, which instantiate a different WandB run for each scope and groups them together:

```python
import goggles as gg
from goggles import WandBHandler

# In this example, we set up multiple runs in Weights & Biases (W&B).
# All runs created by the handler will be grouped under
# the same project and group.
logger: gg.GogglesLogger = gg.get_logger("examples.basic", with_metrics=True)
handler = WandBHandler(
    project="goggles_example", reinit="create_new", group="multiple_runs"
)

# In particular, we set up multiple runs in an RL training loop, with each
# episode being a separate W&B run and a global run tracking all episodes.
num_episodes = 3
episode_length = 10
scopes = [f"episode_{episode}" for episode in range(num_episodes + 1)]
scopes.append("global")
gg.attach(handler, scopes=scopes)


def my_episode(index: int):
    episode_logger = gg.get_logger(scope=f"episode_{index}", with_metrics=True)
    for step in range(episode_length):
        # Supports scopes transparently
        # and has its own step counter
        episode_logger.scalar("env/reward", index * episode_length + step, step=step)


for i in range(num_episodes):
    my_episode(i)
    logger.scalar("total_reward", i, step=i)

gg.finish()
```

### Fully asynchronous logging
As in the WandB example, all the handlers work in the background. By default, the logging calls are not blocking, but can be made blocking by setting the environment variable `GOGGLES_ASYNC` to `0` or `false`. When you use the async mode, remember to call `gg.finish()` at the end from your host machine!
>[!WARNING]
> This functionality still needs thorough tesing, as well as a better documentation. Help is appreciated! 🤗

### Multi-machine logging
Goggles provides options to synchronize logging across machines, since there is always only a single server active. The relevant environment variables here are `GOGGLES_HOST` and `GOGGLES_PORT`.
>[!WARNING]
> This functionality still needs thorough tesing, as well as a better documentation. Help is appreciated! 🤗

### Adding a custom handler
> [!NOTE]
> Ideally, you should open a PR: We would love to integrate your work!

Adding a custom handler is straightforward:

```python
import goggles as gg
import logging


class CustomConsoleHandler(gg.ConsoleHandler):
    """A custom console handler that adds a prefix to each log message."""

    def handle(self, event: gg.Event) -> None:
        dict = event.to_dict()

        dict["payload"] = f"[CUSTOM PREFIX] {dict['payload']}"

        event = gg.Event.from_dict(dict)
        super().handle(event)


# Register the custom handler so it can be serialized/deserialized
gg.register_handler(CustomConsoleHandler)

# In this basic example, we set up a logger that outputs to the console.
logger = gg.get_logger("examples.custom_handler")


gg.attach(
    CustomConsoleHandler(name="examples.custom.console", level=logging.INFO),
    scopes=["global"],
)
# Because the logging level is set to INFO, the debug message will not be shown.
logger.info("Hello, world!")
logger.debug("you won't see this at INFO")

gg.finish()
```

See also [examples/05_custom_handler.py](./examples/06_custom_handler.py) for a complete example.

### Device-resident histories
For long-running GPU experiments that need efficient temporal memory management:

#### Why?

During development of fluid control experiments and reinforcement learning pipelines, we needed to:
- Track detailed metrics during GPU-accelerated training
- Avoid expensive device-to-host transfers
- Maintain temporal state across episodes
- Support JIT compilation for maximum performance

#### Features

- **Pure functional** and **JIT-safe** buffer updates
- **Per-field history lengths** with episodic reset support
- **Batch-first convention**: `(B, T, *shape)` for all tensors
- **Zero host-device synchronization** during updates
- **Integrated with FlowGym's** `EstimatorState` for temporal RL memory

#### Usage

```python
from goggles.history import HistorySpec, create_history, update_history
import jax.numpy as jnp

# Define what to track over time
spec = HistorySpec.from_config({
    "states": {"length": 100, "shape": (64, 64, 2), "dtype": jnp.float32},
    "actions": {"length": 50, "shape": (8,), "dtype": jnp.float32},
    "rewards": {"length": 100, "shape": (), "dtype": jnp.float32},
})

# Create GPU-resident history buffers
history = create_history(spec, batch_size=32)
print(history["states"].shape)  # (32, 100, 64, 64, 2)

# Update buffers during training (JIT-compiled)
new_state = jnp.ones((32, 64, 64, 2))
history = update_history(history, {"states": new_state})
```

See also [examples/103_history.py](./examples/103_history.py) for a running example.


## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for detailed information on:

• Development workflow and environment setup
• Code style requirements and automated checks
• Testing standards and coverage expectations
• PR preparation and commit message conventions

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
