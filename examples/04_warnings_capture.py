import argparse, warnings
from pathlib import Path
import goggles as gg


def main(args):
    gg.configure(enable_console=False, enable_file=True, capture_warnings=True)

    with gg.run("warnings", log_dir=args.log_dir) as ctx:
        warnings.simplefilter("always", category=UserWarning)
        warnings.warn("hello-warning", UserWarning)

        gg.get_logger("examples.warnings").info("after-warning")
        print(f"check {Path(ctx.log_dir) / 'events.log'} for 'hello-warning'")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--log-dir", default="runs")
    main(p.parse_args())
