"""Class-level loggers.

Capturing the logger as a class attribute is a common Python pattern:

    class Worker:
        logger = gg.get_logger(__name__)

        def run(self):
            self.logger.info("...")

The ``get_logger(...)`` call is evaluated once at class-body time -- before
``main()`` runs and before any ``gg.attach(...)``. Goggles resolves its
transport on every emit, so the same logger instance routes through whatever
handlers the application installs later. No re-bind is needed.
"""

import goggles as gg


class Worker:
    """A class that holds its logger as a class attribute.

    Attributes:
        logger: Class-level Goggles logger captured at class-body time.

    """

    logger = gg.get_logger("examples.class_level.worker")

    def __init__(self, name: str) -> None:
        """Store the worker's name for use in log messages.

        Args:
            name: Identifier included in each emitted message.

        """
        self.name = name

    def run(self, n: int) -> int:
        """Do a small amount of work and log the result.

        Args:
            n: Number of iterations to simulate.

        Returns:
            The product ``n * 2``.

        """
        self.logger.info(f"{self.name}: starting (n={n})")
        result = n * 2
        self.logger.info(f"{self.name}: done -> {result}")
        return result


# Two instances share the same class-level logger. Both routes through
# the same handler set installed below.
w1 = Worker("alpha")
w2 = Worker("beta")

# Application setup happens *after* Worker has been imported and its
# class-level logger has already been built.
gg.attach(
    gg.ConsoleHandler(name="examples.class_level.console", level=gg.INFO),
    scopes=["global"],
)

w1.run(3)
w2.run(5)

gg.finish()
