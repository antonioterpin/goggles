"""Example of loading a configuration file using Goggles."""

from goggles import load_configuration

# Example usage
config = load_configuration("examples/example_config.yaml")
print(config)
# we can also access the config like a normal dict
print(
    "We can also access the config like a normal dict:",
    f"{config['time_per_experiment']=}",
)
