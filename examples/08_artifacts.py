import argparse
from pathlib import Path
import goggles as gg


# TODO: check if it works with W&B once we have full support for scalars/images/videos.
def main(args):
    gg.configure(enable_console=False, enable_file=True, enable_artifacts=True)

    with gg.run(
        "artifacts",
        log_dir=args.log_dir,
        enable_artifacts=True,  # can also be set via configure()
        artifact_name="goggles-artifacts",
        artifact_type="goggles-run",
    ) as ctx:
        # Produce some small file to piggyback alongside metadata/logs.
        f = Path(ctx.log_dir) / "notes.txt"
        f.write_text("hello artifacts\n", encoding="utf-8")
        gg.get_logger("examples.artifacts").info("wrote-notes", path=str(f))
        print("run dir:", ctx.log_dir)
        print("if W&B enabled, artifacts will be uploaded on context exit")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--log-dir", default="runs")
    main(p.parse_args())
