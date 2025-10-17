import argparse
import goggles as gg


def main(args):
    # Optional process-wide defaults; can be overridden by run(...)
    gg.configure(log_level="INFO", enable_console=True, enable_file=True)

    with gg.run("basic", log_dir=args.log_dir) as ctx:
        log = gg.get_logger("examples.basic", seed=0)
        log.info("hello-from-basic")
        log.debug("you won't see this at INFO")
        print(f"run dir: {ctx.log_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--log-dir", default="runs", help="base directory for runs")
    main(p.parse_args())
