"""Example of using Goggles with Weights & Biases (W&B) for logging various data types."""

from goggles import Goggles
from PIL import Image
import numpy as np

print("-- Test 1: Logging scalar, vector, image --")
Goggles.set_config(wandb_project="test_proj")
# Scalar
Goggles.scalar("test_scalar", 1.23)
# Vector
Goggles.vector("test_vector", [1, 2, 3, 4, 5])
# Image: random image
img = Image.fromarray((np.random.rand(64, 64, 3) * 255).astype(np.uint8))
Goggles.image("test_image", img)

print("-- Test 2: W&B run switch --")
Goggles.set_config(name="test-run-switch", wandb_project="test_proj")
Goggles.scalar("test_scalar_after", 2.34)
Goggles.image("test_image", img)

print("-- Test 3: W&B log config --")
Goggles.set_config(
    name="test-config-log",
    wandb_project="test_proj",
    config={"param1": 42, "param2": "test"},
)
Goggles.scalar("test_scalar_after", 2.34)
