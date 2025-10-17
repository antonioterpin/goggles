# A library module that logs, without assuming the app configures goggles.
import goggles as gg

_LIB_LOG = gg.get_logger("my_lib", component="lib")  # safe before run()


def do_something(x: int) -> int:
    _LIB_LOG.info("lib-start", x=x)
    y = x * 2
    _LIB_LOG.info("lib-done", y=y)
    return y
