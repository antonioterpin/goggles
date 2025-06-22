# 😎 Goggles

A lightweight, flexible Python logging and monitoring library designed to simplify and enhance experiment tracking, performance profiling, and error tracing. Integrates with terminal, file-based logs, and W\&B (Weights & Biases). It is thought primarily for research projects in robotics.

```bash
pip install "goggles @ git+ssh://git@github.com/antonioterpin/goggles.git"
```

## Features

* 🤖 **Multi-process compatible**: Thought for robots, Goggles synchronize the logging of all processes in your project.
* 🎯 **Multi-output logging**: Log to terminal, file, or both.
* 🕒 **Performance profiling**: `@Goggles.timeit` decorator to profile runtimes.
* 🐞 **Error tracing**: `@Goggles.trace_on_error` to capture stack traces on failure.
* 📊 **Metrics tracking**: Log scalars, vectors, images, and videos to W\&B.
* 🚦 **Graceful Sshutdown**: Handle interrupts with cleanup logic.
* ⚙️ **Asynchronous scheduling**: Offload heavy logging tasks to worker threads.
* 📁 **Pretty configuration logging**: Load settings from YAML and print them nicely.

## Quickstart

```python
from goggles import Goggles, Severity

# Configure logging to both terminal and file at DEBUG level
Goggles.set_config(
    name="my-experiment",
    to_terminal=True,
    to_file=True,
    level=Severity.DEBUG
)

# Log messages
Goggles.debug("Debugging details...")
Goggles.info("Experiment started")
Goggles.warning("This is a warning")
Goggles.error("An error occurred")
```

You can modify the config (e.g., to switch runs):
```python
from goggles import Goggles, Severity

Goggles.set_config(
    name="new-run-name",
    to_file=False,
    to_terminal=True,
    level=Severity.WARNING,
    wandb_project="my_wandb_project"
)
```

## Configuration

Pretty logging of configuration files.

```python
from goggles import load_configuration

# Load from examples/example_config.yaml
config = load_configuration("examples/example_config.yaml")
print(config)

# Access as dict
print(f"time_per_experiment = {config['time_per_experiment']}")
```

## Decorators

### `@Goggles.timeit`

Measure execution time of methods or functions:

```python
from goggles import Goggles, Severity

class MyWorker:
    @Goggles.timeit(severity=Severity.DEBUG)
    def compute_heavy(self, n):
        return sum(range(n))

Goggles.set_config(
    to_terminal=True, level=Severity.DEBUG)
MyWorker().compute_heavy(1000000)
```

### `@Goggles.trace_on_error`

Automatically log a full stack trace when an exception is raised:

```python
from goggles import Goggles

class Processor:
    @Goggles.trace_on_error()
    def divide(self, x, y):
        return x / y

Goggles.set_config(to_terminal=True)
try:
    Processor().divide(10, 0)
except ZeroDivisionError:
    pass  # trace will have been printed
```

## File & Terminal Logging

```python
from goggles import Goggles, Severity

# Default config
print(Goggles.get_config())

# File + terminal, DEBUG
Goggles.set_config(name="run1", to_file=True, to_terminal=True, level=Severity.DEBUG)
Goggles.info("Hello file + terminal")

# Terminal only, WARNING
Goggles.set_config(to_file=False, to_terminal=True, level=Severity.WARNING)
Goggles.info("This won't show up")
Goggles.warning("Visible in terminal only")
```

## W\&B Integration

Log scalars, vectors, images, and videos directly to Weights & Biases:

```python
from goggles import Goggles, Severity
from PIL import Image
import numpy as np

Goggles.set_config(wandb_project="demo_project")

# Scalars and vectors
Goggles.scalar("accuracy", 0.92)
Goggles.vector("loss_curve", [0.5, 0.4, 0.35, 0.3])

# Images
img = Image.fromarray((np.random.rand(64,64,3)*255).astype(np.uint8))
Goggles.image("random_image", img)
```

Switch W\&B runs or log configuration:

```python
Goggles.set_config(name="new-run", wandb_project="demo_project")
Goggles.image("after_switch", img)

# Log custom config dict
Goggles.set_config(
    name="config-log",
    wandb_project="demo_project",
    config={"lr": 1e-3, "batch_size": 32}
)
```

## Graceful Shutdown

Cleanly handle interrupts (e.g., Ctrl-C) and perform cleanup:

```python
import time
from goggles import Goggles, Severity, GracefulShutdown

Goggles.set_config(to_terminal=True, level=Severity.DEBUG)
Goggles.info("Press Ctrl-C to stop loop.")

with GracefulShutdown("Shutting down...") as gs:
    while not gs.stop:
        Goggles.debug("Running...")
        time.sleep(1)

Goggles.info("Exited cleanly.")
```

## Asynchronous Logging & Video

Offload heavy logging tasks to worker threads and log video sequences:

```python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from goggles import Goggles

# Initialize
Goggles.set_config(wandb_project="video_demo")
Goggles.init_scheduler(num_workers=4)

# Generate and schedule logging of flow fields
for i in range(100):
    flow = np.random.uniform(-1,1,(16,16,2))
    # Schedule async save + video log
    Goggles.schedule_log(lambda f,i: np.save(f"/tmp/f{i}.npy", f), flow, i)
    Goggles.scalar("queue_size", Goggles._task_queue.qsize())

# Stop workers and log video
Goggles.stop_workers()
# ... then stack frames and call Goggles.video()
```

## Full working examples
We provide full working examples in the `examples` folder.

## Contributing

Contributions, issues, and feature requests are welcome! Please open a GitHub issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
