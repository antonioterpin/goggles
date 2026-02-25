"""Script to benchmark various logger functions' performance.

This script measures how long it takes to call different logging functions
(scalar, image, info, debug, video, print) to WandB.

Example:
    GOGGLES_PORT=8374 uv run python \
        -m tests.benchmark.benchmark_logger \
        --log-type scalar --num-logs 10000

    GOGGLES_PORT=8374 uv run python \
        -m tests.benchmark.benchmark_logger \
        --log-type image --image-size 128 --num-logs 10000

    GOGGLES_PORT=8374 uv run python \
        -m tests.benchmark.benchmark_logger \
        --log-type video --video-size 64 --num-logs 1500

    GOGGLES_PORT=8374 uv run python \
        -m tests.benchmark.benchmark_logger \
        --log-type info --num-logs 10000

    GOGGLES_PORT=8374 uv run python \
        -m tests.benchmark.benchmark_logger \
        --log-type debug --num-logs 10000

    GOGGLES_PORT=8374 uv run python \
        -m tests.benchmark.benchmark_logger \
        --log-type print --num-logs 10000

    GOGGLES_PORT=8374 uv run python \
        -m tests.benchmark.benchmark_logger \
        --log-type scalar --num-logs 10000 --delay 0.001
"""

import argparse
import datetime
import time

import numpy as np

import goggles as gg

logger = gg.get_logger(
    "goggles.benchmark",
    scope="goggles.benchmark",
    with_metrics=True,
)

gg.attach(
    gg.ConsoleHandler(
        name="goggles.benchmark.console",
        level=gg.INFO,
    ),
    scopes=["goggles.benchmark"],
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark different logger functions"
    )
    parser.add_argument(
        "--num-logs",
        type=int,
        default=10000,
        help="Number of logging calls to benchmark (default: 10000)",
    )
    parser.add_argument(
        "--log-type",
        type=str,
        choices=["scalar", "image", "info", "debug", "video", "print"],
        default="scalar",
        help=(
            "Type of logging to benchmark: scalar, image, info, debug, "
            "video, print (default: scalar)"
        ),
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=64,
        help=(
            "Size of test images (NxN) for image logging benchmark "
            "(default: 64)"
        ),
    )
    parser.add_argument(
        "--video-size",
        type=int,
        default=64,
        help="Size of test video frames (NxN) "
        "for video logging benchmark (default: 64)",
    )
    parser.add_argument(
        "--video-frames",
        type=int,
        default=10,
        help=(
            "Number of frames per video for video logging benchmark "
            "(default: 10)"
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help=(
            "Optional delay in seconds between consecutive logging calls "
            "(default: 0.0)"
        ),
    )
    args = parser.parse_args()

    # Setup WandB logging
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    gg.attach(
        gg.WandBHandler(
            project="goggles",
            config={
                "experiment": "benchmark_goggles",
                "num_logs": f"{args.num_logs}",
                "log_type": args.log_type,
                "run": timestamp,
            },
        ),
        scopes=["goggles.benchmark"],
    )

    logger.info("Starting logger benchmark.")
    logger.info(f"Log type: {args.log_type}")
    logger.info(f"Number of logging calls: {args.num_logs}")
    logger.info(f"Verbose mode: {args.verbose}")
    logger.info(f"Delay between calls: {args.delay:.6f} s")
    if args.log_type == "image":
        logger.info(f"Image size: {args.image_size}x{args.image_size}")
    elif args.log_type == "video":
        logger.info(f"Video frame size: {args.video_size}x{args.video_size}")
        logger.info(f"Frames per video: {args.video_frames}")

    logger_times = []  # Store all logging times

    try:
        for step in range(args.num_logs):
            # Generate test data based on log type
            if args.log_type == "scalar":
                value = np.random.randn()
            elif args.log_type == "image":
                # Generate random RGB image
                value = np.random.randint(
                    0,
                    256,
                    (args.image_size, args.image_size, 3),
                    dtype=np.uint8,
                )
            elif args.log_type == "video":
                # Generate random RGB video (frames, height, width, channels)
                value = np.random.randint(
                    0,
                    256,
                    (args.video_frames, 3, args.video_size, args.video_size),
                    dtype=np.uint8,
                )
            elif args.log_type == "info":
                value = f"Info message step {step}"
            elif args.log_type == "debug":
                value = f"Debug message step {step}"
            elif args.log_type == "print":
                value = f"Print message step {step}"

            # Time the logging call
            start_log_time = time.time()

            if args.log_type == "scalar":
                assert isinstance(value, float)
                logger.scalar(
                    name="test_metric",
                    value=value,
                    step=step,
                )
            elif args.log_type == "image":
                assert isinstance(value, np.ndarray)
                logger.image(
                    name="test_image",
                    image=value,
                    step=step,
                )
            elif args.log_type == "video":
                assert isinstance(value, np.ndarray)
                logger.video(
                    name="test_video",
                    video=value,
                    step=step,
                )
            elif args.log_type == "info":
                assert isinstance(value, str)
                logger.info(value)
            elif args.log_type == "debug":
                assert isinstance(value, str)
                logger.debug(value)
            elif args.log_type == "print":
                print(value)

            log_time = (time.time() - start_log_time) * 1000  # Convert to ms
            logger_times.append(log_time)

            if args.delay > 0:
                time.sleep(args.delay)

            if args.verbose and step % 1000 == 0:
                logger.info(
                    f"Step {step}/{args.num_logs} - Log time: {log_time:.6f} ms"
                )

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise
    finally:
        # Compute statistics
        if logger_times:
            from statistics import mean, median, stdev

            logger.info("\n\n")
            logger.info("=== Logger Benchmark Statistics ===")
            logger.info(f"Logging function: {args.log_type}")
            logger.info(f"Total logging calls: {len(logger_times)}")

            min_time = min(logger_times)
            max_time = max(logger_times)
            mean_time = mean(logger_times)
            median_time = median(logger_times)
            stdev_time = stdev(logger_times) if len(logger_times) > 1 else 0.0
            total_time = sum(logger_times)

            logger.info(f"Min time: {min_time:.6f} ms")
            logger.info(f"Max time: {max_time:.6f} ms")
            logger.info(f"Mean time: {mean_time:.6f} ms")
            logger.info(f"Median time: {median_time:.6f} ms")
            logger.info(f"Std dev: {stdev_time:.6f} ms")
            logger.info(
                f"Total time: {total_time:.6f} ms ({total_time / 1000:.6f} s)"
            )

            # Log all individual times to WandB
            logger.info("\n\n")
            logger.info("=== Logging all individual times ===")
            for idx, log_time in enumerate(logger_times):
                logger.scalar(
                    name="logger_call_time_ms",
                    value=log_time,
                    step=idx + args.num_logs,
                    custom_step={
                        "idx": idx
                    },  # Extra field to be used as x-axis
                )

        gg.finish()  # Close WandB run and clean up loggers
