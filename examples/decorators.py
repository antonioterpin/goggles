"""Example of using Goggles decorators for measuring performance and tracing errors."""

import goggles


class TestClass:
    """A class to demonstrate the use of Goggles decorators."""

    def __init__(self, x):
        """Initialize the class with a value."""
        self.x = x

    @goggles.timeit(severity=goggles.Severity.DEBUG)
    def compute(self, n):
        """Compute the sum of integers from 0 to n-1."""
        total = 0
        for i in range(n):
            total += i
        return total

    @goggles.trace_on_error()
    def fail_method(self, y):
        """Intentionally raise a ZeroDivisionError."""
        return self.x / y


# Test timeit
tc = TestClass(0)
tc.compute(100000)

# Test trace_on_error
try:
    tc = TestClass(10)
    tc.fail_method(0)
except ZeroDivisionError:
    pass
