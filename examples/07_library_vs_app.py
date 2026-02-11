"""Example: library vs application logging with goggles.

This example shows a recommended pattern:
- libraries declare a stable scope and obtain a logger with that scope
- applications perform runtime setup by attaching handlers to scopes

Run this script to see the effect of attaching handlers from the app side.
"""

from pathlib import Path

import goggles as gg

try:
    from goggles import WandBHandler
except Exception:  # pragma: no cover - optional dependency in example
    WandBHandler = None

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
    library_logger.debug(
        "library: debug details that may be filtered by handlers"
    )
    library_logger.warning("library: finished with a warning")


# ------------------ Application code (what an app would do) ------------------
def setup_logging(project_root: Path | str = "examples/logs") -> None:
    """Set up handlers for the application.

    Args:
        project_root: Base directory where local logs are written.

    - Console handler is attached globally so general info is shown on stdout.
    - LocalStorageHandler is attached to the library scope so the library's
        events are saved to disk.
    - If available, a WandB handler is attached under the library scope.

    Returns:
        None.

    """
    project_root = Path(project_root)
    # Console for general messages (global)
    gg.attach(
        gg.ConsoleHandler(name="app.console", level=gg.INFO), scopes=["global"]
    )

    # Local storage for library events (stored under examples/logs/library)
    gg.attach(
        gg.LocalStorageHandler(path=project_root / "library", name="app.local"),
        scopes=[LIBRARY_SCOPE],
    )

    # Optional: WandB. Attach only when the integration is importable.
    if WandBHandler is not None:
        gg.attach(
            WandBHandler(
                project="goggles-demo", run_name="app_run", reinit="create_new"
            ),
            scopes=[LIBRARY_SCOPE],
        )


if __name__ == "__main__":
    print("=== Example: library vs app logging ===")
    print(
        "Step 1: call library function BEFORE app.setup_logging(); "
        "nothing should be logged"
    )
    library_work()

    print()
    print("Step 2: application sets up handlers and calls the library again")
    setup_logging()
    library_work()

    print()
    print("Check 'examples/logs/library' to find saved events (if any).")

    # Clean shutdown for handlers that need it
    gg.finish()
