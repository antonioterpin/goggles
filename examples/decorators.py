"""Example of using Goggles decorators for measuring performance and tracing errors."""

from goggles import Goggles, Severity


class TestClass:
    """A class to demonstrate the use of Goggles decorators."""

    def __init__(self, x):
        """Initialize the class with a value."""
        self.x = x

    @Goggles.timeit(severity=Severity.DEBUG)
    def compute(self, n):
        """Compute the sum of integers from 0 to n-1."""
        total = 0
        for i in range(n):
            total += i
        return total

    @Goggles.trace_on_error()
    def fail_method(self, y):
        """A method that intentionally raises a ZeroDivisionError."""
        return self.x / y


# configure logger for terminal only
Goggles.set_config(to_file=False, to_terminal=True, level=Severity.DEBUG)

# Test timeit
tc = TestClass(0)
tc.compute(100000)

# Test trace_on_error
try:
    tc = TestClass(10)
    tc.fail_method(0)
except ZeroDivisionError:
    pass
