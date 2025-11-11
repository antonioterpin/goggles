"""Example: library vs application logging with goggles

This example shows a recommended pattern:
    - libraries declare a stable scope and obtain a logger with that scope
    - applications perform the runtime "setup", attaching handlers (console, local
	storage, optional WandB) to the appropriate scopes

Run this script to see the effect of attaching handlers from the app side.
"""

from pathlib import Path
import time
import goggles as gg


# ------------------ Library code (what a library would ship) ------------------
LIBRARY_SCOPE = "library"

# Library authors should only declare a scope and obtain a logger. They should
# not attach handlers or configure global logging policies.
library_logger = gg.get_logger(__name__, scope=LIBRARY_SCOPE)


def library_work():
    """A small function that logs at different levels from inside the library.

    The app decides which handlers (and therefore which levels and outputs)
    are active for the library's scope.
    """
    library_logger.info("library: doing useful work")
    library_logger.debug("library: debug details that may be filtered by handlers")
    library_logger.warning("library: finished with a warning")


# ------------------ Application code (what an app would do) ------------------
def setup_logging(project_root: Path | str = "examples/logs"):
    """Set up handlers for the application.

    - Console handler is attached globally so general info is shown on stdout.
    - LocalStorageHandler is attached to the library scope so the library's
        events are saved to disk.
    - If available, a WandB handler is attached under the library scope.
    """
    project_root = Path(project_root)
    # Console for general messages (global)
    gg.attach(gg.ConsoleHandler(name="app.console", level=gg.INFO), scopes=["global"])

    # Local storage for library events (stored under examples/logs/library)
    gg.attach(
        gg.LocalStorageHandler(path=project_root / "library", name="app.local"),
        scopes=[LIBRARY_SCOPE],
    )

    # Optional: WandB. Only attach it when the integration is importable.
    try:
        from goggles import WandBHandler

        gg.attach(
            WandBHandler(project="goggles-demo", run_name="app_run", reinit="create_new"),
            scopes=[LIBRARY_SCOPE],
        )
    except Exception:
        # wandb or the integration might not be installed in the user's env.
        # We silently skip it so the example remains runnable.
        pass


if __name__ == "__main__":
    print("=== Example: library vs app logging ===")
    print("Step 1: call library function BEFORE app.setup_logging() -- nothing should be logged")
    library_work()

    print()
    print("Step 2: application sets up handlers and calls the library again")
    setup_logging()
    library_work()

    print()
    print("Check the directory 'examples/logs/library' to find saved events (if any).")

    # Clean shutdown for handlers that need it
    gg.finish()
