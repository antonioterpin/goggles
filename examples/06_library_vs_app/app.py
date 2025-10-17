import argparse
import goggles as gg
from .lib_module import do_something


# NOTE: run this as a module: python -m examples.06_library_vs_app.app
def main(args):
    # App configures sinks; library just calls get_logger().
    gg.configure(enable_console=True, enable_file=True, enable_jsonl=True)

    with gg.run("app-uses-lib", log_dir=args.log_dir) as ctx:
        log = gg.get_logger("examples.app", role="driver")
        log.info("calling-lib", val=3)
        out = do_something(3)
        log.info("lib-result", out=out)
        print("run dir:", ctx.log_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--log-dir", default="runs")
    main(p.parse_args())
