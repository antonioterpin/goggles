import logging

import goggles as gg

# A console handler that lets DEBUG and above through. Per-logger level
# filtering happens *before* the handler, so the handler stays permissive
# and each logger decides what it emits.
gg.attach(
    gg.ConsoleHandler(name="examples.levels.console", level=gg.DEBUG),
    scopes=["global"],
)

# Default level is logging.NOTSET -- emit everything.
log_a = gg.get_logger("examples.levels.a")
log_a.debug("A debug -- appears (NOTSET)")
log_a.info("A info  -- appears (NOTSET)")

# Pin a logger at INFO at construction time. Useful for noisy modules
# where you want to drop DEBUG noise without touching the handler.
log_b = gg.get_logger("examples.levels.b", level=logging.INFO)
log_b.debug("B debug -- dropped at the logger before the handler sees it")
log_b.info("B info  -- appears")

# set_level(...) raises (or lowers) the bar after construction. Sibling
# loggers obtained via separate get_logger(...) calls are unaffected.
log_c = gg.get_logger("examples.levels.c")
log_c.debug("C debug -- appears (still NOTSET here)")
log_c.set_level(logging.WARNING)
log_c.info("C info  -- dropped after set_level(WARNING)")
log_c.warning("C warn  -- appears")

# Independent of C's level change.
log_d = gg.get_logger("examples.levels.d")
log_d.debug("D debug -- appears (sibling of C, unaffected)")

gg.finish()
