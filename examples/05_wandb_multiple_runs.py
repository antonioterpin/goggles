import goggles as gg
from goggles import WandBHandler
import numpy as np

# In this example, we set up multiple runs in Weights & Biases (W&B).
logger: gg.GogglesLogger = gg.get_logger("examples.basic", with_metrics=True)
handler = WandBHandler(project="goggles_example", reinit="create_new")

# In particular, we set up multiple runs in an RL training loop, with each
# episode being a separate W&B run and a global run tracking all episodes.
num_episodes = 3
episode_length = 10
scopes = [f"scope_{episode}" for episode in range(num_episodes)]
scopes.append("global")
gg.attach(handler, scopes=scopes)

for i in range(num_episodes):
    episode_logger = gg.get_logger(scope=f"scope_{i}", with_metrics=True)
    for j in range(episode_length):
        episode_logger.scalar("step_reward", i * episode_length + j, step=j)
    logger.scalar("reward", i, step=i)

# When using asynchronous logging (like wandb), make sure to finish
gg.finish()
