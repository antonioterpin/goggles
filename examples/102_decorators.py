"""Example of using Goggles decorators.

NOTE: In certain configurations, mypy may consider timeit and trace_on_error as
untyped. You can safely ignore these warnings with # type: ignore[misc].
"""

import goggles as gg

gg.attach(
    gg.ConsoleHandler(
        name="examples.basic.console", level=gg.DEBUG
    ),  # Changed to DEBUG to see timeit output
    scopes=["global"],
)


class TestClass:
    """A class to demonstrate the use of Goggles decorators."""

    def __init__(self, x: float) -> None:
        """Initialize the class with a value.

        Args:
            x: Numerator used by `fail_method`.

        """
        self.x = x

    @gg.timeit(severity=gg.DEBUG, name="compute_sum")
    def compute(self, n: int) -> int:
        """Compute the sum of integers from 0 to n-1.

        Args:
            n: Upper bound (exclusive).

        Returns:
            Sum of integers from `0` to `n - 1`.

        """
        total = 0
        for i in range(n):
            total += i
        return total

    @gg.timeit(
        severity=gg.INFO, name="fast_computation"
    )  # This will also show timing
    def fast_compute(self, n: int) -> int:
        """Compute sum using mathematical formula (faster).

        Args:
            n: Upper bound (exclusive).

        Returns:
            Sum of integers from `0` to `n - 1`.

        """
        return n * (n - 1) // 2

    @gg.trace_on_error()
    def fail_method(self, y: float) -> float:
        """Divide the stored value by `y`.

        Args:
            y: Divisor value.

        Returns:
            Quotient of `self.x / y`.

        """
        return self.x / y


# Test timeit decorator with different methods
print("=== Testing @timeit decorator ===")
tc = TestClass(0)

# Test slow computation (loop-based)
print("Testing slow computation...")
result1 = tc.compute(100000)
print(f"Slow result: {result1}")

# Test fast computation (formula-based)
print("Testing fast computation...")
result2 = tc.fast_compute(100000)
print(f"Fast result: {result2}")

print(f"Results match: {result1 == result2}")

# Test trace_on_error decorator
print("\n=== Testing @trace_on_error decorator ===")
try:
    tc = TestClass(10)
    tc.fail_method(0)
except ZeroDivisionError:
    pass
finally:
    gg.finish()
