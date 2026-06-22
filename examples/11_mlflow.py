from pathlib import Path

import numpy as np

import goggles as gg

# This example forwards Goggles events to MLflow (https://mlflow.org).
#
# Install the optional extra first:
#   uv add "robo-goggles[mlflow]"
#
# Two ways to view the data via a URL:
#
# 1. LOCAL store (this example). We log to a local sqlite database and a
#    local artifact directory, then browse it with the MLflow UI:
#
#        mlflow ui \
#            --backend-store-uri sqlite:///examples/logs/mlflow/mlflow.db \
#            --port 5000
#        # open http://localhost:5000
#
#    Running training on a remote box? Start the UI there with
#    `--host 0.0.0.0` and reach it over an SSH tunnel
#    (`ssh -L 5000:localhost:5000 <remote>`), then open
#    http://localhost:5000.
#
# 2. REMOTE tracking server. Point the handler at a running server and it
#    streams there over HTTP; anyone can watch live at the same URL:
#
#        handler = gg.MLflowHandler(
#            experiment="goggles-example",
#            tracking_uri="http://<server>:5000",
#        )
#
# NOTE: as of MLflow 3 the bare `./mlruns` file store is in maintenance
# mode and rejected by default, so we use a sqlite backend here.

store_dir = Path("examples/logs/mlflow")
store_dir.mkdir(parents=True, exist_ok=True)
tracking_uri = f"sqlite:///{store_dir / 'mlflow.db'}"
artifact_location = (store_dir / "artifacts").resolve().as_uri()

logger = gg.get_logger("examples.mlflow", with_metrics=True, scope="training")

gg.attach(
    gg.MLflowHandler(
        experiment="goggles-example",
        tracking_uri=tracking_uri,
        artifact_location=artifact_location,
        run_name="example_run",
        params={"lr": 3e-4, "optimizer": "adam"},
        tags={"project": "goggles", "stage": "demo"},
    ),
    scopes=["training"],
)
gg.attach(
    gg.ConsoleHandler(name="examples.mlflow.console", level=gg.INFO),
    scopes=["training"],
)

print("=== Goggles MLflow Handler Example ===")
print(f"Tracking store: {tracking_uri}")
print(f"Artifacts:      {artifact_location}")
print(
    "View with: mlflow ui --backend-store-uri "
    f"sqlite:///{store_dir / 'mlflow.db'} --port 5000"
)
print()

logger.info("MLflow example started")

# Scalar metrics -> MLflow metrics (plottable, with a step slider).
rng = np.random.default_rng(0)
for step in range(50):
    loss = float(np.exp(-step / 10) + 0.05 * rng.random())
    logger.scalar("loss", loss, step=step)
    logger.scalar("accuracy", float(0.5 + 0.01 * step), step=step)

# Images -> MLflow's stepped image viewer.
image = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
logger.image(image, name="camera", step=50)

# Videos -> encoded to a GIF artifact (MLflow has no native video type).
video = rng.integers(0, 255, (15, 48, 48, 3), dtype=np.uint8)
logger.video(video, name="rollout", fps=10, format="gif", step=51)

# Trajectories -> rendered to an image via goggles.media.
trajectories = np.cumsum(rng.normal(size=(8, 40, 2), scale=0.1), axis=1)
logger.trajectories(trajectories, name="paths", step=52)

# Vector fields -> rendered to an image via goggles.media.
grid_y, grid_x = np.mgrid[0:16, 0:16]
field = np.stack([-(grid_y - 8.0), grid_x - 8.0], axis=-1).astype(np.float32)
logger.vector_field(field, name="swirl", mode="magnitude", step=53)

# Histograms -> rendered to a Matplotlib figure artifact.
logger.histogram(rng.standard_normal(1000), name="weights", step=54)

# Artifacts -> uploaded to the run's artifact store.
artifact_file = store_dir / "config.txt"
artifact_file.write_text("example artifact\n")
logger.artifact({"path": str(artifact_file), "name": "configs"}, step=55)

print("✓ All events logged. Start the MLflow UI to inspect them.")

gg.finish()
