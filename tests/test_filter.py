"""Test suite for drag filter implementations."""

from typing import Any, cast

import jax.numpy as jnp
import numpy as np
import pytest

from goggles.filters import (
    AverageFilter,
    ConcatFilter,
    ExpAverageFilter,
    FilterConfig,
    MedianFilter,
    MinMaxFilter,
    QuantizationFilter,
    RangeRejectFilter,
    ScaleFilter,
    StdRejectFilter,
    create_concat_filter,
)

ARRAY_BACKENDS = [np, jnp]


@pytest.fixture(params=ARRAY_BACKENDS)
def xp(request: pytest.FixtureRequest) -> Any:
    """Array backend (np or jnp).

    Args:
        request: The pytest request object that provides the parameter value.

    Returns:
        The array backend module (np or jnp) to be used in tests.
    """
    return request.param


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
    xp: Any,
    scalar_input: float,
    expected_scalar: float,
    shape: tuple[int, ...],
) -> None:
    """Test MinMaxFilter with batched array inputs.

    Args:
        xp: The array backend (np or jnp) provided by the pytest fixture.
        scalar_input: Scalar used to fill the input array.
        expected_scalar: Expected normalized scalar value.
        shape: Shape of the generated test arrays.
    """
    f = MinMaxFilter(min_val=-10.0, max_val=10.0)

    input_array = xp.full(shape, scalar_input)
    expected_array = xp.full(shape, expected_scalar)

    # Test step
    output = f.step(input_array)
    assert xp.allclose(output, expected_array), (
        "step failed: "
        f"input={input_array}, expected={expected_array}, got={output}"
    )

    # Test __call__
    output_call = f(input_array)
    assert xp.allclose(output_call, expected_array), (
        "call failed: "
        f"input={input_array}, expected={expected_array}, got={output_call}"
    )


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
    xp: Any,
    scalar_input: float,
    expected_scalar: float,
    shape: tuple[int, ...],
) -> None:
    """Test MinMaxFilter reset for stateless behavior with array inputs.

    Args:
        xp: The array backend (np or jnp) provided by the pytest fixture.
        scalar_input: Scalar used to fill the input array.
        expected_scalar: Expected normalized scalar value.
        shape: Shape of the generated test arrays.
    """
    f = MinMaxFilter(min_val=-10.0, max_val=10.0)

    input_array = xp.full(shape, scalar_input)
    expected_array = xp.full(shape, expected_scalar)

    # Step before reset
    output1 = f.step(input_array)
    assert xp.allclose(output1, expected_array), "Initial step failed"

    # Reset (should do nothing)
    f.reset()

    # Step after reset
    output2 = f.step(input_array)
    assert xp.allclose(output2, expected_array), "Reset affected the filter"


def test_minmaxfilter_name() -> None:
    """Test the name method of MinMaxFilter."""
    f = MinMaxFilter(min_val=-10.0, max_val=10.0)
    name = f.name
    assert "10.0" in name, f"Expected '10.0' in name, got {name}"
    assert "MinMaxFilter" in name, (
        f"Expected 'MinMaxFilter' in name, got {name}"
    )


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
def test_averagefilter_step(xp: Any, inputs: np.ndarray) -> None:
    """Test `AverageFilter.step` moving averages over array inputs.

    Args:
        xp: The array backend (np or jnp) provided by the pytest fixture.
        inputs: A batch of input vectors (2D array).
    """
    inputs_array = xp.array(inputs, dtype=xp.float32)
    window_size = inputs_array.shape[0]
    f = AverageFilter(window_size=window_size)
    outputs = [f.step(x) for x in inputs_array]
    expected = xp.mean(inputs_array, axis=0)

    np.testing.assert_allclose(
        np.asarray(outputs[-1]),
        np.asarray(expected),
        rtol=1e-6,
        err_msg=(
            f"step: input {inputs_array}, "
            f"expected {expected}, got {outputs[-1]}"
        ),
    )

    f.reset()
    outputs = [f(x) for x in inputs_array]
    np.testing.assert_allclose(
        np.asarray(outputs[-1]),
        np.asarray(expected),
        rtol=1e-6,
        err_msg=(
            f"call: input {inputs_array}, "
            f"expected {expected}, got {outputs[-1]}"
        ),
    )


def test_averagefilter_name() -> None:
    """Test the name method of AverageFilter."""
    f = AverageFilter(window_size=3)
    name = f.name
    assert "3" in name, f"Expected '3' in name, got {name}"
    assert "AverageFilter" in name, (
        f"Expected 'AverageFilter' in name, got {name}"
    )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: AverageFilter(window_size=3),
        lambda: MedianFilter(window_size=3),
        lambda: StdRejectFilter(
            std_factor=2.0,
            window_size=3,
            fallback_filter=[
                {"type": "AverageFilter", "parameters": {"window_size": 3}}
            ],
        ),
    ],
    ids=["AverageFilter", "MedianFilter", "StdRejectFilter"],
)
def test_windowbuffer_filter_init_state(factory: Any) -> None:
    f = factory()
    assert f.buffer is None
    assert f.n_seen == 0
    assert f.index == 0


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
def test_expaveragefilter_step(
    xp: Any, alpha: float, inputs: np.ndarray
) -> None:
    """Test the step method of ExpAverageFilter with batched input vectors.

    Args:
        xp: The array backend (np or jnp) provided by the pytest fixture.
        alpha: Smoothing factor for exponential averaging.
        inputs: Sequence of batched vectors passed through the filter.
    """
    inputs_array = xp.array(inputs, dtype=xp.float32)
    f = ExpAverageFilter(alpha=alpha)
    outputs = [f.step(x) for x in inputs_array]

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

    expected = [ema(inputs_array[: i + 1]) for i in range(len(inputs_array))]
    for out, exp in zip(outputs, expected, strict=False):
        np.testing.assert_allclose(
            np.asarray(out), exp, rtol=1e-3, err_msg="Step failed"
        )

    # Test __call__ path
    f.reset()
    outputs = [f(x) for x in inputs_array]
    for out, exp in zip(outputs, expected, strict=False):
        np.testing.assert_allclose(
            np.asarray(out), exp, rtol=1e-3, err_msg="Call failed"
        )


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
    xp: Any, alpha: float, inputs: np.ndarray, expected: np.ndarray
) -> None:
    """Test `ExpAverageFilter` edge cases for alpha values 0 and 1.

    Args:
        xp: The array backend (np or jnp) provided by the pytest fixture.
        alpha: Smoothing factor for exponential averaging.
        inputs: Sequence of batched vectors passed through the filter.
        expected: Expected output vectors for each input step.
    """
    inputs_array = xp.array(inputs, dtype=xp.float32)
    f = ExpAverageFilter(alpha=alpha)
    outputs = [f.step(x) for x in inputs_array]
    for out, exp in zip(outputs, expected, strict=False):
        np.testing.assert_allclose(
            np.asarray(out), exp, rtol=1e-6, err_msg="Edge case failed"
        )


def test_expaveragefilter_reset(xp: Any) -> None:
    """Test the reset method of ExpAverageFilter with batched inputs.

    Args:
        xp: The array backend (np or jnp) provided by the pytest fixture.

    After reset, the next input should re-initialize the filter state.
    """
    f = ExpAverageFilter(alpha=0.5)
    x1 = xp.array([1.0, 10.0])
    x2 = xp.array([2.0, 20.0])
    x3 = xp.array([3.0, 30.0])

    f.step(x1)
    f.step(x2)
    f.reset()
    out = f.step(x3)

    np.testing.assert_allclose(
        np.asarray(out),
        x3,
        rtol=1e-6,
        err_msg="Reset did not reinitialize state",
    )


def test_expaveragefilter_name() -> None:
    """Test the name method of ExpAverageFilter."""
    a = 0.5
    f = ExpAverageFilter(alpha=a)
    name = f.name
    assert str(a) in name, f"Expected '{a}' in name, got {name}"
    assert "ExpAverageFilter" in name, (
        f"Expected 'ExpAverageFilter' in name, got {name}"
    )


@pytest.mark.parametrize(
    "inputs_float",
    [
        [[1.0, 3.0], [2.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
        [[5.0, 4.0], [3.0, 2.0], [1.0, 0.0]],
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]],
    ],
)
def test_medianfilter_step(xp: Any, inputs_float: list[list[float]]) -> None:
    """Test the step method of MedianFilter with batched inputs.

    Args:
        xp: The array backend (np or jnp) provided by the pytest fixture.
        inputs_float: A list of batched input vectors.
    """
    inputs = [xp.array(x) for x in inputs_float]
    f = MedianFilter(window_size=len(inputs))

    outputs = [f.step(x) for x in inputs]
    stacked = xp.stack(inputs)
    expected = xp.median(stacked, axis=0)

    np.testing.assert_allclose(
        np.asarray(outputs[-1]),
        np.asarray(expected),
        rtol=1e-6,
        err_msg="Median filter step failed",
    )

    f.reset()
    outputs = [f(x) for x in inputs]
    np.testing.assert_allclose(
        np.asarray(outputs[-1]),
        np.asarray(expected),
        rtol=1e-6,
        err_msg="Median filter call failed",
    )


def test_medianfilter_reset(xp: Any) -> None:
    """Test the reset method of MedianFilter with batched inputs.

    Args:
        xp: The array backend (np or jnp) provided by the pytest fixture.
    """
    f = MedianFilter(window_size=3)

    f.step(xp.array([1.0, 3.0]))
    f.step(xp.array([4.0, 2.0]))

    f.reset()

    out = f.step(xp.array([2.0, 5.0]))
    expected = xp.array(
        [2.0, 5.0]
    )  # After reset, first input should be returned directly

    np.testing.assert_allclose(
        np.asarray(out),
        np.asarray(expected),
        rtol=1e-6,
        err_msg="Median filter reset failed",
    )


def test_medianfilter_name() -> None:
    """Test the name method of MedianFilter."""
    window_size = 3
    f = MedianFilter(window_size=3)
    name = f.name
    assert str(window_size) in name, (
        f"Expected '{window_size}' in name, got {name}"
    )
    assert "MedianFilter" in name, (
        f"Expected 'MedianFilter' in name, got {name}"
    )


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
def test_quantizationfilter_step(xp: Any) -> None:
    """Test QuantizationFilter clamping and quantization on batched input.

    Args:
        xp: The array backend (np or jnp) provided by the pytest fixture.
    """

    f = QuantizationFilter(min_value=-0.150, max_value=0.150, step_size=0.00015)

    inputs = xp.array(
        [
            -0.160,  # Clamped to -0.150
            0.160,  # Clamped to 0.150
            0.00005,  # Quantized to 0.0
            0.0001,  # Quantized to 0.00015
            0.0,  # Should stay 0.0
            0.00015,  # Should stay 0.00015
        ]
    )
    expected = xp.array(
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
        np.asarray(output),
        np.asarray(expected),
        rtol=0.1,
        atol=1e-5,
        err_msg="QuantizationFilter vectorized step failed",
    )

    # Also test __call__
    output_call = f(inputs)
    np.testing.assert_allclose(
        np.asarray(output_call),
        np.asarray(expected),
        rtol=0.1,
        atol=1e-5,
        err_msg="QuantizationFilter vectorized __call__ failed",
    )


def test_quantizationfilter_reset(xp: Any) -> None:
    """Test the reset method of QuantizationFilter with batched input.

    Args:
        xp: The array backend (np or jnp) provided by the pytest fixture.

    Since QuantizationFilter is stateless, reset should have no effect.
    """
    f = QuantizationFilter(min_value=-0.150, max_value=0.150, step_size=0.00015)

    input_array = xp.array([0.0001, -0.0001, 0.0])
    expected = xp.array([0.00015, -0.00015, 0.0])

    output1 = f.step(input_array)
    np.testing.assert_allclose(
        np.asarray(output1),
        np.asarray(expected),
        rtol=0.1,
        atol=1e-5,
        err_msg="Initial step failed",
    )

    f.reset()  # Should do nothing

    output2 = f.step(input_array)
    np.testing.assert_allclose(
        np.asarray(output2),
        np.asarray(expected),
        rtol=0.1,
        atol=1e-5,
        err_msg="Reset affected the filter",
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
    assert "QuantizationFilter" in name, (
        f"Expected 'QuantizationFilter' in name, got {name}"
    )


# Test ConcatFilter
def test_concatfilter_step(xp: Any) -> None:
    """Test the step method of ConcatFilter with vectorized (batched) input.

    Args:
        xp: The array backend (np or jnp) provided by the pytest fixture.
    """
    f1 = MinMaxFilter(min_val=-10.0, max_val=10.0)
    f2 = AverageFilter(window_size=2)
    concat_f = ConcatFilter(filters=[f1, f2])

    inputs = xp.array(
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

    outputs = xp.stack(outputs)

    # Manually compute expected values
    # First input → [1.0, 0.0]
    # Second input -> [0.75, 0.25], avg over [1.0, 0.75] and [0.0, 0.25].
    # Third input -> [1.0, 0.0], avg over [0.75, 1.0] and [0.25, 0.0].

    expected = xp.array(
        [
            [1.0, 0.0],
            [0.875, 0.125],
            [0.875, 0.125],
        ]
    )

    np.testing.assert_allclose(
        np.asarray(outputs), np.asarray(expected), rtol=1e-6
    )

    # Now test __call__
    concat_f.reset()
    output_call_1 = concat_f(inputs[0])
    np.testing.assert_allclose(np.asarray(output_call_1), [1.0, 0.0], rtol=1e-6)
    output_call_2 = concat_f(inputs[1])
    np.testing.assert_allclose(
        np.asarray(output_call_2), [0.875, 0.125], rtol=1e-6
    )


def test_concatfilter_empty(xp: Any) -> None:
    """Test ConcatFilter with an empty filter list and batched input.

    Args:
        xp: The array backend (np or jnp) provided by the pytest fixture.
    """
    concat_f = ConcatFilter(filters=[])

    inputs = xp.array(
        [
            [5.0, -5.0],
            [10.0, -10.0],
        ]
    )

    # Step should return the same inputs unchanged
    assert xp.allclose(concat_f.step(inputs[0]), inputs[0]), (
        "Failed step on first input"
    )
    concat_f.reset()  # Should do nothing
    assert xp.allclose(concat_f.step(inputs[1]), inputs[1]), (
        "Failed step after reset"
    )
    assert xp.allclose(concat_f(inputs[0]), inputs[0]), "Failed call"


def test_concatfilter_reset(xp: Any) -> None:
    """Test the reset method of ConcatFilter with batched input.

    Args:
        xp: The array backend (np or jnp) provided by the pytest fixture.
    """
    f1 = AverageFilter(window_size=3)
    f2 = ExpAverageFilter(alpha=0.5)
    concat_f = ConcatFilter(filters=[f1, f2])

    batch1 = xp.array([1.0, 2.0])
    batch2 = xp.array([2.0, 4.0])
    batch3 = xp.array([3.0, 6.0])

    # Warm-up steps
    concat_f.step(batch1)
    concat_f.step(batch2)

    # Reset
    concat_f.reset()

    # After reset, the next input should pass through cleanly
    output = concat_f.step(batch3)
    assert xp.allclose(output, batch3), (
        f"Concat filter reset failed: got {output}, expected {batch3}"
    )


def test_concatfilter_name() -> None:
    """Test the name method of ConcatFilter."""
    f1 = MinMaxFilter(min_val=-10.0, max_val=10)
    f2 = AverageFilter(window_size=3)
    concat_f = ConcatFilter(filters=[f1, f2])
    name = concat_f.name
    assert f1.name in name, f"Expected '{f1.name}' in name, got {name}"
    assert f2.name in name, f"Expected '{f2.name}' in name, got {name}"
    assert "ConcatFilter" in name, (
        f"Expected 'ConcatFilter' in name, got {name}"
    )


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
    assert isinstance(concat_f.filters[0], MinMaxFilter), (
        "Expected MinMaxFilter as first filter"
    )
    assert concat_f.filters[0].max_val == 10.0, (
        "Incorrect max_val in MinMaxFilter"
    )
    assert isinstance(concat_f.filters[1], AverageFilter), (
        "Expected AverageFilter as second filter"
    )
    assert concat_f.filters[1].window_size == 3, (
        "Incorrect window_size in AverageFilter"
    )


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
    xp: Any,
    scale: float,
    scalar_input: float,
    expected_scalar: float,
    shape: tuple[int],
) -> None:
    """Test ScaleFilter step method with batched input arrays.

    Args:
        xp: The array backend (np or jnp) provided by the pytest fixture.
        scale: The scaling factor.
        scalar_input: The scalar to fill the input array.
        expected_scalar: The expected output after scaling.
        shape: The shape of the input array.
    """
    f = ScaleFilter(scale=scale)

    input_array = xp.full(shape, scalar_input)
    expected_array = xp.full(shape, expected_scalar)

    output = f.step(input_array)
    np.testing.assert_allclose(
        np.asarray(output),
        np.asarray(expected_array),
        rtol=1e-6,
        err_msg=(
            "ScaleFilter step failed: "
            f"input={input_array}, expected={expected_array}, got={output}"
        ),
    )

    # Test __call__
    output_call = f(input_array)
    np.testing.assert_allclose(
        np.asarray(output_call),
        np.asarray(expected_array),
        rtol=1e-6,
        err_msg=(
            "ScaleFilter call failed: "
            f"input={input_array}, expected={expected_array}, got={output_call}"
        ),
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
    with pytest.raises(TypeError):
        ScaleFilter(scale=cast(Any, scale))


def test_scalefilter_reset(xp: Any) -> None:
    """Test ScaleFilter reset does not affect stateless behavior.

    Args:
        xp: The array backend (np or jnp) provided by the pytest fixture.
    """
    f = ScaleFilter(scale=2.0)

    input_array = xp.array([1.0, 2.0])
    expected = xp.array([2.0, 4.0])

    output1 = f.step(input_array)
    f.reset()
    output2 = f.step(input_array)

    np.testing.assert_allclose(
        np.asarray(output1), np.asarray(expected), rtol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(output2), np.asarray(expected), rtol=1e-6
    )


def test_scalefilter_name() -> None:
    """Test ScaleFilter name includes class name and scale value."""
    scale = 1.5
    f = ScaleFilter(scale=scale)
    name = f.name
    assert "ScaleFilter" in name, f"Expected 'ScaleFilter' in name, got {name}"
    assert str(scale) in name, f"Expected '{scale}' in name, got {name}"


def test_scalefilter_concatfilter(xp: Any) -> None:
    """Test ScaleFilter inside a ConcatFilter with batched input.

    Args:
        xp: The array backend (np or jnp) provided by the pytest fixture.
    """
    f1 = ScaleFilter(scale=2.0)
    f2 = ScaleFilter(scale=0.5)
    concat_f = ConcatFilter(filters=[f1, f2])

    input_array = xp.array([[1.0, 2.0], [3.0, 4.0]])
    expected = input_array  # Because 2.0 * 0.5 = 1.0

    output = xp.stack([concat_f.step(x) for x in input_array])
    np.testing.assert_allclose(
        np.asarray(output), np.asarray(expected), rtol=1e-6
    )


def test_rangerejectfilter_replace_only_invalid(xp: Any) -> None:
    """Only out-of-range values should be replaced.

    Args:
        xp: The array backend (np or jnp) provided by the pytest fixture.
    """
    f = RangeRejectFilter(
        min_value=-1.0,
        max_value=1.0,
        fallback_filter=[{"type": "ScaleFilter", "parameters": {"scale": 0.0}}],
    )

    data = xp.array([-2.0, -0.5, 0.2, 3.0])
    out = f.step(data)

    # invalid → replaced by fallback (=0), valid unchanged
    expected = xp.array([0.0, -0.5, 0.2, 0.0])
    np.testing.assert_allclose(np.asarray(out), np.asarray(expected), rtol=1e-6)


def test_rangerejectfilter_fallback_chain_applied(xp: Any) -> None:
    """Test that the fallback chain is applied for out-of-range values.

    Args:
        xp: The array backend (np or jnp) provided by the pytest fixture.
    """
    f = RangeRejectFilter(
        min_value=0.0,
        max_value=1.0,
        fallback_filter=[
            {"type": "ScaleFilter", "parameters": {"scale": 2.0}},
            {"type": "ScaleFilter", "parameters": {"scale": 0.5}},
        ],
    )

    data = xp.array([-1.0, 0.5, 2.0])
    out = f.step(data)

    # midpoint = 0.5 → fallback sees [0.5,0.5,0.5]
    expected = xp.array([0.5, 0.5, 0.5])
    np.testing.assert_allclose(np.asarray(out), np.asarray(expected), rtol=1e-6)


def test_rangerejectfilter_partial_vector_replacement(xp: Any) -> None:
    """Test that only out-of-range values are replaced in a vector input.

    Args:
        xp: The array backend (np or jnp) provided by the pytest fixture.
    """
    f = RangeRejectFilter(
        min_value=-1.0,
        max_value=1.0,
        fallback_filter=[
            {"type": "ScaleFilter", "parameters": {"scale": 10.0}}
        ],
    )

    data = xp.array([-2.0, 0.1, 2.0])
    out = f.step(data)

    expected = xp.array([0.0, 0.1, 0.0])
    np.testing.assert_allclose(np.asarray(out), np.asarray(expected), rtol=1e-6)


def test_rangerejectfilter_fallback_state_updates(xp: Any) -> None:
    """Test that fallback filter state updates with out-of-range inputs.

    Args:
        xp: The array backend (np or jnp) provided by the pytest fixture.
    """
    f = RangeRejectFilter(
        min_value=-1.0,
        max_value=1.0,
        fallback_filter=[
            {"type": "AverageFilter", "parameters": {"window_size": 2}}
        ],
    )

    x1 = xp.array([0.5])
    x2 = xp.array([0.6])
    x3 = xp.array([5.0])  # invalid

    f.step(x1)
    f.step(x2)

    out = f.step(x3)

    # fallback sees [0.6, 0.6]
    expected = xp.array([0.6])
    np.testing.assert_allclose(np.asarray(out), np.asarray(expected), rtol=1e-6)


def test_rangerejectfilter_reset(xp: Any) -> None:
    """Test RangeRejectFilter reset with fallback filter that has state.

    Args:
        xp: The array backend (np or jnp) provided by the pytest fixture.
    """
    f = RangeRejectFilter(
        min_value=-1.0,
        max_value=1.0,
        fallback_filter=[
            {"type": "ExpAverageFilter", "parameters": {"alpha": 0.5}}
        ],
    )

    f.step(xp.array([1.0]))
    f.step(xp.array([2.0]))

    f.reset()

    out = f.step(xp.array([3.0]))  # invalid

    expected = xp.array([0.0])  # midpoint path
    np.testing.assert_allclose(np.asarray(out), np.asarray(expected), rtol=1e-6)


def test_rangerejectfilter_accepts_filterconfig_objects(xp: Any) -> None:
    """Fallback should accept FilterConfig instances directly.

    Args:
        xp: The array backend (np or jnp) provided by the pytest fixture.
    """
    cfg = FilterConfig(
        type="ScaleFilter",
        parameters={"scale": 0.0},
    )

    f = RangeRejectFilter(
        min_value=-1.0,
        max_value=1.0,
        fallback_filter=[cfg],
    )

    out = f.step(xp.array([5.0]))
    np.testing.assert_allclose(np.asarray(out), np.asarray([0.0]), rtol=1e-6)


def test_rangerejectfilter_invalid_init() -> None:
    """Invalid range must raise."""
    with pytest.raises(ValueError):
        RangeRejectFilter(
            min_value=1.0,
            max_value=-1.0,
            fallback_filter=[],
        )


def test_stdrejectfilter_warmup_returns_input(xp: Any) -> None:
    """Before window is full, StdRejectFilter should pass values through.

    Args:
        xp: The array backend (np or jnp) provided by the pytest fixture.
    """
    f = StdRejectFilter(
        std_factor=1.0,
        window_size=3,
        fallback_filter=[{"type": "ScaleFilter", "parameters": {"scale": 0.0}}],
    )

    x1 = xp.array([1.0, 2.0])
    x2 = xp.array([100.0, -50.0])

    np.testing.assert_allclose(
        np.asarray(f.step(x1)), np.asarray(x1), rtol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(f.step(x2)), np.asarray(x2), rtol=1e-6
    )


def test_stdrejectfilter_per_element_replacement_after_warmup(xp: Any) -> None:
    """After warmup, rejection is computed independently per element.

    Args:
        xp: The array backend (np or jnp) provided by the pytest fixture.
    """
    f = StdRejectFilter(
        std_factor=2.0,
        window_size=3,
        fallback_filter=[{"type": "ScaleFilter", "parameters": {"scale": 0.0}}],
    )

    # Build history with non-zero std on first element and ~0 std on second.
    f.step(xp.array([0.0, 1.0]))
    f.step(xp.array([1.0, 1.0]))
    f.step(xp.array([2.0, 1.0]))

    out = f.step(xp.array([10.0, 1.0]))

    # First element is outlier -> replaced by fallback (0.0).
    # Second element is inlier and should stay unchanged.
    expected = xp.array([0.0, 1.0])
    np.testing.assert_allclose(np.asarray(out), np.asarray(expected), rtol=1e-6)


def test_stdrejectfilter_reset_restarts_warmup(xp: Any) -> None:
    """After reset, filter should behave like a fresh instance.

    Args:
        xp: The array backend (np or jnp) provided by the pytest fixture.
    """
    f = StdRejectFilter(
        std_factor=1.0,
        window_size=2,
        fallback_filter=[
            {"type": "ExpAverageFilter", "parameters": {"alpha": 0.5}}
        ],
    )

    f.step(xp.array([0.0]))
    f.step(xp.array([1.0]))
    f.step(xp.array([100.0]))

    f.reset()

    x = xp.array([42.0])
    out = f.step(x)
    np.testing.assert_allclose(np.asarray(out), np.asarray(x), rtol=1e-6)


@pytest.mark.parametrize(
    "std_factor, window_size",
    [
        (0.0, 3),
        (-1.0, 3),
        (1.0, 0),
        (1.0, -2),
    ],
)
def test_stdrejectfilter_invalid_init(
    std_factor: float, window_size: int
) -> None:
    """Invalid StdRejectFilter constructor args must raise.

    Args:
        std_factor: The standard deviation factor for outlier rejection.
        window_size:
            The size of the window for computing mean and std deviation.
    """
    with pytest.raises(ValueError):
        StdRejectFilter(
            std_factor=std_factor,
            window_size=window_size,
            fallback_filter=[],
        )


def test_stdrejectfilter_inliers_pass_after_window(xp: Any) -> None:
    """After warmup, inliers should pass unchanged.

    Args:
        xp: The array backend (np or jnp) provided by the pytest fixture.
    """
    f = StdRejectFilter(
        std_factor=2.0,
        window_size=3,
        fallback_filter=[{"type": "ScaleFilter", "parameters": {"scale": 0.0}}],
    )

    # Warmup
    f.step(xp.array([0.0]))
    f.step(xp.array([1.0]))
    f.step(xp.array([2.0]))

    # Mean=1, std≈0.816 -> 1.5 is valid
    x = xp.array([1.5])
    out = f.step(x)

    np.testing.assert_allclose(np.asarray(out), np.asarray(x), rtol=1e-6)


def test_stdrejectfilter_stateful_replacement_sequence(xp: Any) -> None:
    """Multiple outliers should produce consistent stateful outputs.

    Args:
        xp: The array backend (np or jnp) provided by the pytest fixture.
    """
    f = StdRejectFilter(
        std_factor=1.0,
        window_size=3,
        fallback_filter=[{"type": "ScaleFilter", "parameters": {"scale": 0.0}}],
    )

    # Warmup
    f.step(xp.array([0.0]))
    f.step(xp.array([1.0]))
    f.step(xp.array([2.0]))

    # Outlier -> replaced by 0
    out1 = f.step(xp.array([100.0]))
    np.testing.assert_allclose(np.asarray(out1), np.asarray([0.0]), rtol=1e-6)

    # Another outlier -> still stable
    out2 = f.step(xp.array([200.0]))
    np.testing.assert_allclose(np.asarray(out2), np.asarray([0.0]), rtol=1e-6)


def test_stdrejectfilter_mixed_elements_over_time(xp: Any) -> None:
    """Different elements should evolve independently across time.

    Args:
        xp: The array backend (np or jnp) provided by the pytest fixture.
    """
    f = StdRejectFilter(
        std_factor=1.5,
        window_size=3,
        fallback_filter=[{"type": "ScaleFilter", "parameters": {"scale": 0.0}}],
    )

    # Warmup
    f.step(xp.array([0.0, 0.0]))
    f.step(xp.array([1.0, 0.0]))
    f.step(xp.array([2.0, 0.0]))

    # First element outlier, second still valid
    out = f.step(xp.array([50.0, 0.0]))
    expected = xp.array([0.0, 0.0])
    np.testing.assert_allclose(np.asarray(out), np.asarray(expected), rtol=1e-6)

    # Now both valid again
    out2 = f.step(xp.array([1.0, 0.0]))
    np.testing.assert_allclose(
        np.asarray(out2), np.asarray([1.0, 0.0]), rtol=1e-6
    )
