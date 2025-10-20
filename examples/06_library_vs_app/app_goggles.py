import argparse
import goggles as gg
from .lib_module_goggles import do_something as do_something_with_goggles
from .lib_module_no_goggles import do_something as do_something_no_goggles


# NOTE: run this as a module: python -m examples.06_library_vs_app.app_goggles
def main(args):
    # App configures sinks; library just calls get_logger().
    gg.configure(enable_console=True, enable_file=True, enable_jsonl=True)

    with gg.run("app-uses-lib", log_dir=args.log_dir) as ctx:
        log = gg.get_logger("examples.app", role="driver")
        log.info("calling-lib", val=3)
        out_no_goggles = do_something_with_goggles(3)
        out_no_goggles = do_something_no_goggles(3)
        log.info("lib-result-goggles", out=out_no_goggles)
        log.info("lib-result-no-goggles", out=out_no_goggles)
        print("run dir:", ctx.log_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--log-dir", default="runs")
    main(p.parse_args())
