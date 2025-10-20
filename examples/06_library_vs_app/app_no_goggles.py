import argparse

from logging import getLogger
from .lib_module_goggles import do_something as do_something_with_goggles
from .lib_module_no_goggles import do_something as do_something_no_goggles

logger = getLogger("examples.app")


# NOTE: run this as a module: python -m examples.06_library_vs_app.app_no_goggles
def main(args):
    logger.info("calling-libs, val=3")
    out_no_goggles = do_something_no_goggles(3)
    out_with_goggles = do_something_with_goggles(3)
    logger.info("lib-result-with-goggles, out=%d", out_with_goggles)
    logger.info("lib-result-no-goggles, out=%d", out_no_goggles)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--log-dir", default="runs")
    main(p.parse_args())
