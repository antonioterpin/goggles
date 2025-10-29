import argparse
import goggles as gg
import logging

logger = gg.get_logger("examples.basic", seed=42)
gg.attach(
    gg.ConsoleHandler(name="examples.basic.console", level=logging.INFO),
    scopes=["global"],
)


def main(args):
    logger.info("Hello, world!")
    logger.debug("you won't see this at INFO")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--log-dir", default="runs", help="base directory for runs")
    main(p.parse_args())
