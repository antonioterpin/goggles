"""Example of using Goggles with Weights & Biases (W&B) for logging various data types."""

from PIL import Image
import numpy as np
import goggles

print("-- Test 1: Logging scalar, vector, image --")
goggles.new_wandb_run(name="test-run", config={"param1": 41, "param2": "test2"})
# Scalar
goggles.scalar("test_scalar", 1.23)
# Vector
goggles.vector("test_vector", [1, 2, 3, 4, 5])
# Image: random image
img = Image.fromarray((np.random.rand(64, 64, 3) * 255).astype(np.uint8))
goggles.image("test_image", img)

print("-- Test 2: W&B run switch --")
goggles.new_wandb_run(name="test-run-switch", config={"param1": 42, "param2": "test"})
goggles.scalar("test_scalar_after", 2.34)
goggles.image("test_image", img)
