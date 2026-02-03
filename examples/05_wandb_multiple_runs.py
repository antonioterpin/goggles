from pathlib import Path
import numpy as np

import goggles as gg
from goggles import WandBHandler

# In this example, we set up multiple runs in Weights & Biases (W&B).
# All runs created by the handler will be grouped under
# the same project and group.

# To group together e.g. different groups, you can use the config parameter
# and group in wandb first by "experiment", then by "group".

logger: gg.GogglesLogger = gg.get_logger(
    "examples.wandb", scope="training", with_metrics=True
)

# In particular, we set up multiple runs in an RL training loop, with each
# episode being a separate W&B run and a global run tracking all episodes.
num_training_runs = 3
num_episodes = 5
episode_length = 10

dummy_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)


def my_episode(run_index: int, episode_index: int):
    episode_logger = gg.get_logger(
        scope=f"training.{run_index}.{episode_index}", with_metrics=True
    )
    for step in range(episode_length):
        # Supports scopes transparently
        # and has its own step counter
        episode_logger.scalar(
            "env/reward", run_index * episode_index * episode_length + step, step=step
        )
        # Example of image logging, which will also be stored
        # using a namespace
        episode_logger.image(dummy_image, name="observations/image_stepped", step=step)


def my_training_run(run_index: int):
    run_logger = gg.get_logger(scope=f"training.{run_index}", with_metrics=True)
    for episode_index in range(num_episodes):
        my_episode(run_index, episode_index)
        run_logger.scalar(
            "run/total_reward",
            run_index * episode_index * episode_length + episode_index,
            step=episode_index,
        )


for i in range(num_training_runs):
    # For each training run, we create a handler
    handler = WandBHandler(
        project="goggles_wandb",
        reinit="create_new",
        group=f"training.{i}",
        config={"experiment": "my-experiment"},
    )
    # We also attach a local storage handler to keep a local copy of the logs
    local_storage_handler = gg.LocalStorageHandler(
        path=Path(f"examples/logs/training/{i}"),
        name=f"training.{i}.local",
    )
    gg.attach(handler, scopes=[f"training.{i}"])
    gg.attach(local_storage_handler, scopes=[f"training.{i}"])
    my_training_run(i)
    # Sync
    gg.finish()
