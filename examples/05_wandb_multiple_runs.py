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

# When using asynchronous logging (like wandb), make sure to finish
gg.finish()
