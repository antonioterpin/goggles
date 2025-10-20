import argparse
from pathlib import Path
import goggles as gg


def main(args):
    gg.configure(enable_console=False, enable_jsonl=True)

    with gg.run("metrics", log_dir=args.log_dir) as ctx:
        # Scalars are always JSONL-friendly and mirrored to W&B if enabled.
        gg.scalar("train/loss", 0.123, step=1)
        gg.scalar("train/acc", 0.91, step=1, split="train")

        # Images/videos: we record metadata in JSONL; binaries go to backends (e.g., W&B).
        try:
            import numpy as np

            img = (np.linspace(0, 255, 32 * 32 * 3).reshape(32, 32, 3)).astype("uint8")
            vid = np.zeros((8, 16, 16, 3), dtype="uint8")
            gg.image("samples/img", img, step=1, split="val")
            gg.video("rollout", vid, step=1, fps=4)
        except Exception:
            gg.get_logger("examples.metrics").warning(
                "numpy not available; skipping image/video"
            )

        print(f"jsonl at: {Path(ctx.log_dir) / 'events.jsonl'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--log-dir", default="runs")
    main(p.parse_args())
