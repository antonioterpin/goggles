"""Utility functions for validation."""

from typing import Union


def round_up_to_multiple(number: float, multiple: int) -> int:
    """Round up a number to the nearest multiple.

    Args:
        number (float): The number to round up.
        multiple (int): The multiple to round up to.

    Returns:
        int: The rounded up number.

    """
    if number % multiple == 0:
        return number
    return (number + multiple - 1) // multiple * multiple


def is_int(val: Union[int, float]) -> bool:
    """Check if a value is an integer.

    Args:
        val (Union[int, float]): The value to check.

    Returns:
        bool: True if the value is an integer, False otherwise.ge

    """
    if isinstance(val, int):
        return True
    if isinstance(val, float):
        if abs(val - int(val)) < 1e-6:
            return True
    return False
