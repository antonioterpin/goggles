"""Test suite for drag filter implementations."""

import numpy as np
import pytest

from goggles.filters import (
    ScaleFilter,
    AverageFilter,
    ConcatFilter,
    ExpAverageFilter,
    FilterConfig,
    MedianFilter,
    MinMaxFilter,
    QuantizationFilter,
    create_concat_filter,
)


# Test MinMaxFilter
@pytest.mark.parametrize(
    "scalar_input, expected_scalar",
    [
        (5.0, 0.75),
        (10.0, 1.0),
        (15.0, 1.0),
        (-5.0, 0.25),
        (-10.0, 0.0),
        (-15.0, 0.0),
        (0.0, 0.5),
    ],
)
@pytest.mark.parametrize("shape", [(1,), (3,), (2, 3)])
def test_minmaxfilter_step(
    scalar_input: float, expected_scalar: float, shape: tuple[int, ...]
) -> None:
    """Test MinMaxFilter with batched array inputs.

    Args:
        scalar_input: Scalar used to fill the input array.
        expected_scalar: Expected normalized scalar value.
        shape: Shape of the generated test arrays.
    """
    f = MinMaxFilter(min_val=-10.0, max_val=10.0)

    input_array = np.full(shape, scalar_input)
    expected_array = np.full(shape, expected_scalar)

    # Test step
    output = f.step(input_array)
    assert np.allclose(
        output, expected_array
    ), f"step failed: input={input_array}, expected={expected_array}, got={output}"

    # Test __call__
    output_call = f(input_array)
    assert np.allclose(
        output_call, expected_array
    ), f"call failed: input={input_array}, expected={expected_array}, got={output_call}"


@pytest.mark.parametrize(
    "scalar_input, expected_scalar",
    [
        (5.0, 0.75),
        (10.0, 1.0),
        (15.0, 1.0),
        (-5.0, 0.25),
        (-10.0, 0.0),
        (-15.0, 0.0),
        (0.0, 0.5),
    ],
)
@pytest.mark.parametrize("shape", [(1,), (3,), (2, 3)])
def test_minmaxfilter_reset(
    scalar_input: float, expected_scalar: float, shape: tuple[int, ...]
) -> None:
    """Test MinMaxFilter reset does not affect stateless behavior for array inputs.

    Args:
        scalar_input: Scalar used to fill the input array.
        expected_scalar: Expected normalized scalar value.
        shape: Shape of the generated test arrays.
    """
    f = MinMaxFilter(min_val=-10.0, max_val=10.0)

    input_array = np.full(shape, scalar_input)
    expected_array = np.full(shape, expected_scalar)

    # Step before reset
    output1 = f.step(input_array)
    assert np.allclose(output1, expected_array), "Initial step failed"

    # Reset (should do nothing)
    f.reset()

    # Step after reset
    output2 = f.step(input_array)
    assert np.allclose(output2, expected_array), "Reset affected the filter"


def test_minmaxfilter_name() -> None:
    """Test the name method of MinMaxFilter."""
    f = MinMaxFilter(min_val=-10.0, max_val=10.0)
    name = f.name
    assert "10.0" in name, f"Expected '10.0' in name, got {name}"
    assert "MinMaxFilter" in name, f"Expected 'MinMaxFilter' in name, got {name}"


@pytest.mark.parametrize(
    "max_abs_value",
    [0.0, -1.0],
)
def test_minmaxfilter_invalid_init(max_abs_value: float) -> None:
    """Test that bad init raises a ValueError for invalid max_abs_value.

    Args:
        max_abs_value: The maximum absolute value for the MinMaxFilter.
    """
    with pytest.raises(ValueError):
        MinMaxFilter(min_val=-max_abs_value, max_val=max_abs_value)


# Test AverageFilter
@pytest.mark.parametrize(
    "inputs",
    [
        np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]),  # shape (5, 1)
        np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]),  # shape (3, 2)
        np.random.rand(10, 3),  # shape (10, 3)
    ],
)
def test_averagefilter_step(inputs: np.ndarray) -> None:
    """Test the step method of AverageFilter for computing moving averages over arrays.

    Args:
        inputs: A batch of input vectors (2D array).
    """
    window_size = inputs.shape[0]
    f = AverageFilter(window_size=window_size)
    outputs = [f.step(x) for x in inputs]
    expected = np.mean(inputs, axis=0)

    np.testing.assert_allclose(
        outputs[-1],
        expected,
        rtol=1e-6,
        err_msg=(f"step: input {inputs}, expected {expected}, got {outputs[-1]}"),
    )

    f.reset()
    outputs = [f(x) for x in inputs]
    np.testing.assert_allclose(
        outputs[-1],
        expected,
        rtol=1e-6,
        err_msg=(f"call: input {inputs}, expected {expected}, got {outputs[-1]}"),
    )


def test_averagefilter_name() -> None:
    """Test the name method of AverageFilter."""
    f = AverageFilter(window_size=3)
    name = f.name
    assert "3" in name, f"Expected '3' in name, got {name}"
    assert "AverageFilter" in name, f"Expected 'AverageFilter' in name, got {name}"


@pytest.mark.parametrize(
    "window_size",
    [0, -1, 2.5, -3.5],
)
def test_averagefilter_invalid_init(window_size: int) -> None:
    """Test that bad init raises a ValueError for invalid window sizes.

    Args:
        window_size: The window size for the AverageFilter.
    """
    with pytest.raises(ValueError):
        AverageFilter(window_size=window_size)


# Test ExpAverageFilter
@pytest.mark.parametrize(
    "alpha",
    [-0.1, 1.1],
)
def test_expaveragefilter_invalid_init(alpha: float) -> None:
    """Test that bad init raises a ValueError for invalid alpha values.

    Args:
        alpha: The smoothing factor for the ExpAverageFilter.
    """
    with pytest.raises(ValueError):
        ExpAverageFilter(alpha=alpha)


@pytest.mark.parametrize("alpha", [0.0, 0.5, 0.75, 0.99, 1.0])
@pytest.mark.parametrize(
    "inputs",
    [
        np.array([[1.0], [2.0], [3.0]]),
        np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]]),
        np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]]),
    ],
)
def test_expaveragefilter_step(alpha: float, inputs: np.ndarray) -> None:
    """Test the step method of ExpAverageFilter with batched input vectors.

    Args:
        alpha: Smoothing factor for exponential averaging.
        inputs: Sequence of batched vectors passed through the filter.
    """
    f = ExpAverageFilter(alpha=alpha)
    outputs = [f.step(x) for x in inputs]

    # Manual EMA computation for vector inputs
    def ema(values: np.ndarray) -> np.ndarray:
        n = len(values)
        if n == 0:
            return np.zeros_like(values[0])
        exponents = np.arange(n - 1, -1, -1)
        weights = alpha * (1 - alpha) ** exponents
        weights[0] = (1 - alpha) ** (n - 1)
        weighted = weights[:, None] * values
        return np.sum(weighted, axis=0)

    expected = [ema(inputs[: i + 1]) for i in range(len(inputs))]
    for out, exp in zip(outputs, expected, strict=False):
        np.testing.assert_allclose(out, exp, rtol=1e-3, err_msg="Step failed")

    # Test __call__ path
    f.reset()
    outputs = [f(x) for x in inputs]
    for out, exp in zip(outputs, expected, strict=False):
        np.testing.assert_allclose(out, exp, rtol=1e-3, err_msg="Call failed")


@pytest.mark.parametrize(
    "alpha, inputs, expected",
    [
        (
            1.0,
            np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]),
            np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]),
        ),
        (
            0.0,
            np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]),
            np.array([[1.0, 10.0], [1.0, 10.0], [1.0, 10.0]]),
        ),
    ],
)
def test_expaveragefilter_step_edge_cases(
    alpha: float, inputs: np.ndarray, expected: np.ndarray
) -> None:
    """Test ExpAverageFilter with edge cases for alpha (0 and 1), using batched input.

    Args:
        alpha: Smoothing factor for exponential averaging.
        inputs: Sequence of batched vectors passed through the filter.
        expected: Expected output vectors for each input step.
    """
    f = ExpAverageFilter(alpha=alpha)
    outputs = [f.step(x) for x in inputs]
    for out, exp in zip(outputs, expected, strict=False):
        np.testing.assert_allclose(out, exp, rtol=1e-6, err_msg="Edge case failed")


def test_expaveragefilter_reset() -> None:
    """Test the reset method of ExpAverageFilter with batched inputs.

    After reset, the next input should re-initialize the filter state.
    """
    f = ExpAverageFilter(alpha=0.5)
    x1 = np.array([1.0, 10.0])
    x2 = np.array([2.0, 20.0])
    x3 = np.array([3.0, 30.0])

    f.step(x1)
    f.step(x2)
    f.reset()
    out = f.step(x3)

    np.testing.assert_allclose(
        out, x3, rtol=1e-6, err_msg="Reset did not reinitialize state"
    )


def test_expaveragefilter_name() -> None:
    """Test the name method of ExpAverageFilter."""
    a = 0.5
    f = ExpAverageFilter(alpha=a)
    name = f.name
    assert str(a) in name, f"Expected '{a}' in name, got {name}"
    assert (
        "ExpAverageFilter" in name
    ), f"Expected 'ExpAverageFilter' in name, got {name}"


@pytest.mark.parametrize(
    "inputs_float",
    [
        [[1.0, 3.0], [2.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
        [[5.0, 4.0], [3.0, 2.0], [1.0, 0.0]],
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]],
    ],
)
def test_medianfilter_step(inputs_float: list[list[float]]) -> None:
    """Test the step method of MedianFilter with batched inputs.

    Args:
        inputs_float: A list of batched input vectors.
    """
    inputs = [np.array(x) for x in inputs_float]
    f = MedianFilter(window_size=len(inputs))

    outputs = [f.step(x) for x in inputs]
    stacked = np.stack(inputs)
    expected = np.median(stacked, axis=0)

    np.testing.assert_allclose(
        outputs[-1], expected, rtol=1e-6, err_msg="Median filter step failed"
    )

    f.reset()
    outputs = [f(x) for x in inputs]
    np.testing.assert_allclose(
        outputs[-1], expected, rtol=1e-6, err_msg="Median filter call failed"
    )


def test_medianfilter_reset() -> None:
    """Test the reset method of MedianFilter with batched inputs."""
    f = MedianFilter(window_size=3)

    f.step(np.array([1.0, 3.0]))
    f.step(np.array([4.0, 2.0]))

    f.reset()

    out = f.step(np.array([2.0, 5.0]))
    expected = np.array(
        [2.0, 5.0]
    )  # After reset, first input should be returned directly

    np.testing.assert_allclose(
        out, expected, rtol=1e-6, err_msg="Median filter reset failed"
    )


def test_medianfilter_name() -> None:
    """Test the name method of MedianFilter."""
    window_size = 3
    f = MedianFilter(window_size=3)
    name = f.name
    assert str(window_size) in name, f"Expected '{window_size}' in name, got {name}"
    assert "MedianFilter" in name, f"Expected 'MedianFilter' in name, got {name}"


@pytest.mark.parametrize(
    "window_size",
    [0, -1, 2.5, -3.5],
)
def test_medianfilter_invalid_init(window_size: int) -> None:
    """Test that bad init raises a ValueError for invalid window sizes.

    Args:
        window_size: The window size for the MedianFilter.
    """
    with pytest.raises(ValueError):
        MedianFilter(window_size=window_size)


# Test QuantizationFilter
def test_quantizationfilter_step() -> None:
    """Test the step method of QuantizationFilter for clamping and quantization on batched input."""

    f = QuantizationFilter(min_value=-0.150, max_value=0.150, step_size=0.00015)

    inputs = np.array(
        [
            -0.160,  # Clamped to -0.150
            0.160,  # Clamped to 0.150
            0.00005,  # Quantized to 0.0
            0.0001,  # Quantized to 0.00015
            0.0,  # Should stay 0.0
            0.00015,  # Should stay 0.00015
        ]
    )
    expected = np.array(
        [
            -0.150,
            0.150,
            0.0,
            0.00015,
            0.0,
            0.00015,
        ]
    )

    output = f.step(inputs)
    np.testing.assert_allclose(
        output,
        expected,
        rtol=1e-6,
        atol=1e-6,
        err_msg="QuantizationFilter vectorized step failed",
    )

    # Also test __call__
    output_call = f(inputs)
    np.testing.assert_allclose(
        output_call,
        expected,
        rtol=1e-6,
        atol=1e-6,
        err_msg="QuantizationFilter vectorized __call__ failed",
    )


def test_quantizationfilter_reset() -> None:
    """Test the reset method of QuantizationFilter with batched input.

    Since QuantizationFilter is stateless, reset should have no effect.
    """
    f = QuantizationFilter(min_value=-0.150, max_value=0.150, step_size=0.00015)

    input_array = np.array([0.0001, -0.0001, 0.0])
    expected = np.array([0.00015, -0.00015, 0.0])

    output1 = f.step(input_array)
    np.testing.assert_allclose(
        output1, expected, rtol=1e-6, atol=1e-6, err_msg="Initial step failed"
    )

    f.reset()  # Should do nothing

    output2 = f.step(input_array)
    np.testing.assert_allclose(
        output2, expected, rtol=1e-6, atol=1e-6, err_msg="Reset affected the filter"
    )


def test_quantizationfilter_name() -> None:
    """Test the name method of QuantizationFilter."""
    min = -0.150
    max = 0.150
    step = 0.00015
    f = QuantizationFilter(min_value=min, max_value=max, step_size=step)
    name = f.name
    assert str(min) in name, f"Expected '{min}' in name, got {name}"
    assert str(max) in name, f"Expected '{max}' in name, got {name}"
    assert str(step) in name, f"Expected '{step}' in name, got {name}"
    assert (
        "QuantizationFilter" in name
    ), f"Expected 'QuantizationFilter' in name, got {name}"


# Test ConcatFilter
def test_concatfilter_step() -> None:
    """Test the step method of ConcatFilter with vectorized (batched) input."""
    f1 = MinMaxFilter(min_val=-10.0, max_val=10.0)
    f2 = AverageFilter(window_size=2)
    concat_f = ConcatFilter(filters=[f1, f2])

    inputs = np.array(
        [
            [15.0, -15.0],  # → MinMax: [1.0, 0.0]
            [5.0, -5.0],  # → MinMax: [0.75, 0.25]
            [20.0, -20.0],  # → MinMax: [1.0, 0.0]
        ]
    )

    # Step-by-step filtering
    outputs = []
    for x in inputs:
        outputs.append(concat_f.step(x))

    outputs = np.stack(outputs)

    # Manually compute expected values
    # First input → [1.0, 0.0]
    # Second input → [0.75, 0.25], avg over [1.0, 0.75] = 0.875, [0.0, 0.25] = 0.125
    # Third input → [1.0, 0.0], avg over [0.75, 1.0] = 0.875, [0.25, 0.0] = 0.125

    expected = np.array(
        [
            [1.0, 0.0],
            [0.875, 0.125],
            [0.875, 0.125],
        ]
    )

    np.testing.assert_allclose(outputs, expected, rtol=1e-6)

    # Now test __call__
    concat_f.reset()
    output_call_1 = concat_f(inputs[0])
    np.testing.assert_allclose(output_call_1, [1.0, 0.0], rtol=1e-6)
    output_call_2 = concat_f(inputs[1])
    np.testing.assert_allclose(output_call_2, [0.875, 0.125], rtol=1e-6)


def test_concatfilter_empty() -> None:
    """Test ConcatFilter with an empty filter list and batched input."""
    concat_f = ConcatFilter(filters=[])

    inputs = np.array(
        [
            [5.0, -5.0],
            [10.0, -10.0],
        ]
    )

    # Step should return the same inputs unchanged
    assert np.allclose(
        concat_f.step(inputs[0]), inputs[0]
    ), "Failed step on first input"
    concat_f.reset()  # Should do nothing
    assert np.allclose(concat_f.step(inputs[1]), inputs[1]), "Failed step after reset"
    assert np.allclose(concat_f(inputs[0]), inputs[0]), "Failed call"


def test_concatfilter_reset() -> None:
    """Test the reset method of ConcatFilter with batched input."""
    f1 = AverageFilter(window_size=3)
    f2 = ExpAverageFilter(alpha=0.5)
    concat_f = ConcatFilter(filters=[f1, f2])

    batch1 = np.array([1.0, 2.0])
    batch2 = np.array([2.0, 4.0])
    batch3 = np.array([3.0, 6.0])

    # Warm-up steps
    concat_f.step(batch1)
    concat_f.step(batch2)

    # Reset
    concat_f.reset()

    # After reset, the next input should pass through cleanly
    output = concat_f.step(batch3)
    assert np.allclose(
        output, batch3
    ), f"Concat filter reset failed: got {output}, expected {batch3}"


def test_concatfilter_name() -> None:
    """Test the name method of ConcatFilter."""
    f1 = MinMaxFilter(min_val=-10.0, max_val=10)
    f2 = AverageFilter(window_size=3)
    concat_f = ConcatFilter(filters=[f1, f2])
    name = concat_f.name
    assert f1.name in name, f"Expected '{f1.name}' in name, got {name}"
    assert f2.name in name, f"Expected '{f2.name}' in name, got {name}"
    assert "ConcatFilter" in name, f"Expected 'ConcatFilter' in name, got {name}"


# Test create_concat_filter
def test_create_concat_filter_valid() -> None:
    """Test create_concat_filter with valid filter configurations."""
    config = [
        FilterConfig(
            type="MinMaxFilter",
            parameters={"min_val": -10.0, "max_val": 10.0},
        ),
        FilterConfig(
            type="AverageFilter",
            parameters={"window_size": 3},
        ),
    ]
    concat_f = create_concat_filter(config)
    assert isinstance(concat_f, ConcatFilter), "Expected ConcatFilter instance"
    assert len(concat_f.filters) == 2, "Expected 2 filters in ConcatFilter"
    assert isinstance(
        concat_f.filters[0], MinMaxFilter
    ), "Expected MinMaxFilter as first filter"
    assert concat_f.filters[0].max_val == 10.0, "Incorrect max_val in MinMaxFilter"
    assert isinstance(
        concat_f.filters[1], AverageFilter
    ), "Expected AverageFilter as second filter"
    assert (
        concat_f.filters[1].window_size == 3
    ), "Incorrect window_size in AverageFilter"


def test_create_concat_filter_invalid_type() -> None:
    """Test create_concat_filter with an invalid filter type."""
    config = [
        FilterConfig(
            type="InvalidFilter",
            parameters={},
        )
    ]
    with pytest.raises(ValueError):
        create_concat_filter(config)


def test_create_concat_filter_invalid_params() -> None:
    """Test create_concat_filter with invalid filter parameters."""
    config = [
        FilterConfig(
            type="MinMaxFilter",
            parameters={"min_val": 10.0, "max_val": -10.0},  # Invalid params
        )
    ]
    with pytest.raises(ValueError):
        create_concat_filter(config)


# Test ScaleFilter
@pytest.mark.parametrize(
    "scale, scalar_input, expected_scalar",
    [
        (2.0, 1.0, 2.0),
        (0.5, -2.0, -1.0),
        (-1.0, 3.0, -3.0),
        (0.0, 5.0, 0.0),
    ],
)
@pytest.mark.parametrize("shape", [(1,), (2,), (2, 3)])
def test_scalefilter_step(
    scale: float, scalar_input: float, expected_scalar: float, shape: tuple[int]
) -> None:
    """Test ScaleFilter step method with batched input arrays.

    Args:
        scale: The scaling factor.
        scalar_input: The scalar to fill the input array.
        expected_scalar: The expected output after scaling.
        shape: The shape of the input array.
    """
    f = ScaleFilter(scale=scale)

    input_array = np.full(shape, scalar_input)
    expected_array = np.full(shape, expected_scalar)

    output = f.step(input_array)
    np.testing.assert_allclose(
        output,
        expected_array,
        rtol=1e-6,
        err_msg=f"ScaleFilter step failed: input={input_array}, expected={expected_array}, got={output}",
    )

    # Test __call__
    output_call = f(input_array)
    np.testing.assert_allclose(
        output_call,
        expected_array,
        rtol=1e-6,
        err_msg=f"ScaleFilter call failed: input={input_array}, expected={expected_array}, got={output_call}",
    )


@pytest.mark.parametrize(
    "scale",
    ["a", None, object()],
)
def test_scalefilter_invalid_init(scale: object) -> None:
    """Test ScaleFilter initialization with invalid scale types.

    Args:
        scale: The invalid scale value to test.
    """
    with pytest.raises(ValueError):
        ScaleFilter(scale=scale)


def test_scalefilter_reset() -> None:
    """Test ScaleFilter reset does not affect stateless behavior."""
    f = ScaleFilter(scale=2.0)

    input_array = np.array([1.0, 2.0])
    expected = np.array([2.0, 4.0])

    output1 = f.step(input_array)
    f.reset()
    output2 = f.step(input_array)

    np.testing.assert_allclose(output1, expected, rtol=1e-6)
    np.testing.assert_allclose(output2, expected, rtol=1e-6)


def test_scalefilter_name() -> None:
    """Test the name method of ScaleFilter includes class name and scale value."""
    scale = 1.5
    f = ScaleFilter(scale=scale)
    name = f.name
    assert "ScaleFilter" in name, f"Expected 'ScaleFilter' in name, got {name}"
    assert str(scale) in name, f"Expected '{scale}' in name, got {name}"


def test_scalefilter_concatfilter() -> None:
    """Test ScaleFilter inside a ConcatFilter with batched input."""
    f1 = ScaleFilter(scale=2.0)
    f2 = ScaleFilter(scale=0.5)
    concat_f = ConcatFilter(filters=[f1, f2])

    input_array = np.array([[1.0, 2.0], [3.0, 4.0]])
    expected = input_array  # Because 2.0 * 0.5 = 1.0

    output = np.stack([concat_f.step(x) for x in input_array])
    np.testing.assert_allclose(output, expected, rtol=1e-6)
