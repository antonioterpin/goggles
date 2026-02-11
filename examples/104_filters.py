"""Demonstration of the Filters module.

This script shows how to:
1. Use different filter types (ScaleFilter, MinMaxFilter, AverageFilter)
2. Chain filters together using ConcatFilter
3. Work with both numpy arrays and JAX arrays
4. Reset filter states
5. Create filters from configuration dictionaries
"""

import numpy as np

try:
    import jax
    import jax.numpy as jnp

    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jnp = None
    jax = None

import goggles as gg
from goggles.filters import (
    AverageFilter,
    ConcatFilter,
    ExpAverageFilter,
    FilterConfig,
    MedianFilter,
    MinMaxFilter,
    QuantizationFilter,
    ScaleFilter,
    create_concat_filter,
)

# Set up logging
gg.attach(
    gg.ConsoleHandler(name="examples.filters.console", level=gg.INFO),
    scopes=["global"],
)
logger = gg.get_logger("examples.filters")


def demo_basic_filters():
    """Demonstrate basic filter usage with different types."""
    logger.info("=== Basic Filter Demonstration ===")

    # Generate some noisy data
    np.random.seed(42)
    data = np.array([1.0, 2.5, 1.8, 3.2, 2.1, 4.0, 3.5, 2.8, 3.9, 4.2])
    logger.info(f"Original data: {data}")

    # 1. ScaleFilter - multiply by constant
    scale_filter = ScaleFilter(scale=2.0)
    scaled_data = [scale_filter.step(x) for x in data]
    logger.info(f"Scaled by 2.0: {scaled_data}")

    # 2. MinMaxFilter - normalize to [0, 1]
    minmax_filter = MinMaxFilter(min_val=0.0, max_val=5.0)
    normalized_data = [minmax_filter.step(x) for x in data]
    logger.info(f"MinMax normalized [0,5] -> [0,1]: {normalized_data}")

    # 3. AverageFilter - moving average with window
    avg_filter = AverageFilter(window_size=3)
    averaged_data = []
    for x in data:
        averaged_data.append(avg_filter.step(x))
    logger.info(f"Moving average (window=3): {averaged_data}")

    # 4. ExpAverageFilter - exponential moving average
    exp_avg_filter = ExpAverageFilter(alpha=0.3)
    exp_averaged_data = []
    for x in data:
        exp_averaged_data.append(exp_avg_filter.step(x))
    logger.info(f"Exponential average (alpha=0.3): {exp_averaged_data}")


def demo_stateful_filters():
    """Demonstrate filters with state and reset functionality."""
    logger.info("\n=== Stateful Filters and Reset ===")

    data = np.array([1.0, 5.0, 2.0, 8.0, 3.0])

    # Create a median filter
    median_filter = MedianFilter(window_size=3)

    logger.info("Processing data with MedianFilter:")
    results_before_reset = []
    for i, x in enumerate(data):
        result = median_filter.step(x)
        results_before_reset.append(result)
        logger.info(f"  Step {i + 1}: input={x}, median={result}")

    # Reset and process again
    logger.info("Resetting filter and processing same data:")
    median_filter.reset()
    results_after_reset = []
    for i, x in enumerate(data):
        result = median_filter.step(x)
        results_after_reset.append(result)
        logger.info(f"  Step {i + 1}: input={x}, median={result}")

    # Show that results are identical after reset
    is_identical = np.allclose(results_before_reset, results_after_reset)
    logger.info(f"Results identical after reset: {is_identical}")


def demo_jax_compatibility() -> None:
    """Demonstrate filter compatibility with JAX arrays.

    Returns:
        None.

    """
    logger.info("\n=== JAX Array Compatibility ===")

    if not HAS_JAX:
        logger.warning("JAX not available - skipping JAX compatibility demo")
        logger.info("Install JAX with: pip install jax jaxlib")
        return
    assert jnp is not None

    # Create JAX arrays
    jax_data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    logger.info(f"JAX input data: {jax_data}")
    logger.info(f"Data type: {type(jax_data)}")

    # Apply filters to JAX arrays
    scale_filter = ScaleFilter(scale=0.5)
    avg_filter = AverageFilter(window_size=2)

    scaled_jax = scale_filter.step(jax_data)
    logger.info(f"Scaled JAX array: {scaled_jax}")
    logger.info(f"Output type: {type(scaled_jax)}")

    # Process each element through averaging filter
    jax_averaged = []
    for x in jax_data:
        result = avg_filter.step(x)
        jax_averaged.append(result)

    logger.info(f"JAX moving average: {jax_averaged}")


def demo_quantization_filter():
    """Demonstrate the quantization filter."""
    logger.info("\n=== Quantization Filter ===")

    # Generate continuous values
    continuous_data = np.array([0.0123, -0.0456, 0.0789, -0.1234, 0.1567])
    logger.info(f"Continuous data: {continuous_data}")

    # Create quantization filter with custom parameters
    quant_filter = QuantizationFilter(
        min_value=-0.2, max_value=0.2, step_size=0.01
    )

    quantized_data = []
    for x in continuous_data:
        quantized = quant_filter.step(x)
        quantized_data.append(quantized)
        logger.info(f"  {x:.4f} -> {float(quantized):.4f}")

    logger.info(f"Quantized data: {quantized_data}")


def demo_concat_filter():
    """Demonstrate chaining filters together."""
    logger.info("\n=== Chaining Filters with ConcatFilter ===")

    # Create a sequence of filters
    filters = [
        ScaleFilter(scale=10.0),  # Scale up by 10
        MinMaxFilter(
            min_val=0.0, max_val=100.0
        ),  # Normalize assuming max is 100
        AverageFilter(window_size=2),  # Smooth with moving average
        QuantizationFilter(
            min_value=0.0, max_value=1.0, step_size=0.1
        ),  # Quantize
    ]

    concat_filter = ConcatFilter(filters)
    logger.info(f"Filter chain: {concat_filter.name}")

    # Process data through the chain
    raw_data = np.array([0.1, 0.3, 0.2, 0.8, 0.6, 0.4, 0.9, 0.5])
    logger.info(f"Raw data: {raw_data}")

    processed_data = []
    for i, x in enumerate(raw_data):
        result = concat_filter.step(x)
        processed_data.append(result)
        logger.info(f"  Step {i + 1}: {x:.1f} -> {float(result):.1f}")

    logger.info(f"Final processed data: {processed_data}")


def demo_config_based_filters():
    """Demonstrate creating filters from configuration."""
    logger.info("\n=== Configuration-based Filter Creation ===")

    # Define filter configurations
    filter_configs = [
        FilterConfig(type="ScaleFilter", parameters={"scale": 5.0}),
        FilterConfig(
            type="MinMaxFilter", parameters={"min_val": 0.0, "max_val": 10.0}
        ),
        FilterConfig(type="ExpAverageFilter", parameters={"alpha": 0.2}),
    ]

    # Create concat filter from configs
    config_filter = create_concat_filter(filter_configs)
    logger.info(f"Config-based filter: {config_filter.name}")

    # Test with sample data
    test_data = np.array([1.0, 1.5, 2.0, 1.8, 2.2])
    logger.info(f"Test data: {test_data}")

    results = []
    for x in test_data:
        result = config_filter.step(x)
        results.append(float(result))

    logger.info(f"Processed results: {results}")


def demo_multidimensional_data():
    """Demonstrate filters working with multidimensional arrays."""
    logger.info("\n=== Multidimensional Data Processing ===")

    # Create 2D data (e.g., image patches, feature vectors)
    np.random.seed(123)
    data_2d = np.random.randn(5, 3, 3)  # 5 samples of 3x3 data
    logger.info(f"2D data shape: {data_2d.shape}")
    logger.info(f"First sample:\n{data_2d[0]}")

    # Apply filters to multidimensional data
    scale_filter = ScaleFilter(scale=0.1)
    avg_filter = AverageFilter(window_size=3)

    # Process each sample
    scaled_results = []
    averaged_results = []

    for i, sample in enumerate(data_2d):
        scaled = scale_filter.step(sample)
        averaged = avg_filter.step(sample)

        scaled_results.append(scaled)
        averaged_results.append(averaged)

        if i == 0:  # Show details for first sample
            logger.info(f"Sample {i + 1} scaled:\n{scaled}")
            logger.info(f"Sample {i + 1} averaged:\n{averaged}")

    logger.info(f"Processed {len(scaled_results)} multidimensional samples")


def main():
    """Run all filter demonstrations."""
    logger.info("Starting Goggles Filters demonstration...")

    demo_basic_filters()
    demo_stateful_filters()
    demo_jax_compatibility()
    demo_quantization_filter()
    demo_concat_filter()
    demo_config_based_filters()
    demo_multidimensional_data()

    logger.info("\n=== Filter Demonstration Complete ===")
    logger.info("Key takeaways:")
    logger.info("- Filters can process both numpy and JAX arrays")
    logger.info("- Stateful filters maintain history and can be reset")
    logger.info("- Filters can be chained together with ConcatFilter")
    logger.info("- Filters work with multidimensional data")
    logger.info("- Configuration-based filter creation enables flexibility")


if __name__ == "__main__":
    main()
    gg.finish()
