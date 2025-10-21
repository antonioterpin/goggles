"""Worker thread for materializing device arrays into encoded host byte payloads.

This module defines the `MaterializerWorker`, which continuously consumes metric
events from an input `MetricsQueue`, materializes any JAX device arrays into host
NumPy arrays, performs downscaling and quantization, and re-encodes the data as
PNG, JPEG, or MP4 for downstream consumption.

Typical use:
    - A producer thread enqueues `MetricEvent`s with JAX arrays as payloads.
    - The materializer converts these arrays into byte-encoded formats.
    - A consumer thread (e.g., exporter) dequeues the processed events.

Notes:
    This class is primarily intended for internal metrics pipelines that operate
    asynchronously. It is thread-safe but not multi-process safe.

"""

import io
import logging
import os
import tempfile
import threading
from dataclasses import replace
from typing import Any, Callable, Optional

import numpy as np
from PIL import Image

from .queue import MetricsQueue
from .event import MetricEvent

try:
    import jax
except Exception:  # pragma: no cover
    jax = None

try:
    import imageio.v2 as imageio
except Exception:  # pragma: no cover
    imageio = None


class MaterializerWorker(threading.Thread):
    """Thread worker that materializes arrays and emits encoded host bytes.

    This thread continuously polls an input `MetricsQueue` for `MetricEvent`
    instances, converts their payloads to NumPy arrays, optionally downsamples
    and quantizes them, encodes them as images or videos, and finally enqueues
    the resulting encoded bytes to an output `MetricsQueue`.

    Attributes:
        _in_q (MetricsQueue): Input queue providing metric events.
        _out_q (MetricsQueue): Output queue for materialized events.
        _stop (threading.Event): Shared stop signal for cooperative shutdown.
        _image_fmt (str): Target image format ("png", "jpeg").
        _video_fmt (str): Target video format ("mp4").
        _downscale (int): Spatial downscaling factor.
        _quantize_policy (Optional[callable]): Optional user-defined quantizer.
        _fps (int): Frames per second for encoded videos.
        _sleep (float): Idle sleep time between polling cycles.
        _log (logging.Logger): Logger used for debug and error messages.

    """

    def __init__(
        self,
        in_queue: MetricsQueue,
        out_queue: MetricsQueue,
        stop_event: threading.Event,
        *,
        image_format: str = "png",
        video_format: str = "mp4",
        downscale_factor: int = 1,
        quantize_policy: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        fps: int = 12,
        logger: Optional[logging.Logger] = None,
        idle_sleep: float = 0.001,
    ) -> None:
        """Initialize the worker and validate configuration.

        Args:
            in_queue (MetricsQueue): Queue providing raw metric events.
            out_queue (MetricsQueue): Queue to receive materialized events.
            stop_event (threading.Event): Stop flag for cooperative shutdown.
            image_format (str): Image encoding format ("png", "jpeg", or "jpg").
            video_format (str): Video encoding format ("mp4" only supported).
            downscale_factor (int): Factor for spatial downscaling (>= 1).
            quantize_policy (Optional[callable]): Custom float→uint8 quantizer.
            fps (int): Target frame rate for encoded videos.
            logger (Optional[logging.Logger]): Optional custom logger.
            idle_sleep (float): Polling delay when input queue is empty.

        Raises:
            ValueError: If invalid formats or parameters are provided.

        """
        super().__init__(daemon=True)
        if image_format.lower() not in {"png", "jpeg", "jpg"}:
            raise ValueError("image_format must be one of {'png','jpeg','jpg'}.")
        if video_format.lower() not in {"mp4"}:
            raise ValueError("video_format must be 'mp4'.")
        if downscale_factor < 1:
            raise ValueError("downscale_factor must be >= 1.")
        if fps < 1:
            raise ValueError("fps must be >= 1.")

        self._in_q = in_queue
        self._out_q = out_queue
        self._stop = stop_event
        self._image_fmt = (
            "jpeg" if image_format.lower() == "jpg" else image_format.lower()
        )
        self._video_fmt = video_format.lower()
        self._downscale = downscale_factor
        self._quantize_policy = quantize_policy
        self._fps = fps
        self._sleep = idle_sleep
        self._log = logger or logging.getLogger(__name__)

    def run(self) -> None:
        """Continuously process events until stopped."""
        while not self._stop.is_set():
            event = self._in_q.dequeue()
            if event is None:
                # Avoid busy-waiting when queue is empty
                if self._sleep > 0:
                    try:
                        import time

                        time.sleep(self._sleep)
                    except Exception:
                        pass
                continue

            try:
                out = self.process_event(event)
            except Exception as err:  # pragma: no cover
                self._log.exception("Materializer failed: %s", err)
                out = self._attach_error(event, f"{type(err).__name__}: {err}")

            ok = self._out_q.enqueue(out)
            if not ok:
                # Queue overflow handled by MetricsQueue policy
                self._log.debug(
                    "Materializer output queue full; event dropped by policy"
                )

    def process_event(self, event: MetricEvent) -> MetricEvent:
        """Materialize a single `MetricEvent`.

        Performs:
            1. Device-to-host transfer.
            2. Optional downscaling.
            3. Quantization to uint8.
            4. Encoding to image or video bytes.

        Args:
            event (MetricEvent): Input event containing an array payload.

        Returns:
            MetricEvent: New event with encoded `payload`, `encoding` metadata,
                and `materialized=True`.

        Raises:
            TypeError: If event payload is missing or unsupported.
            ValueError: If payload has unsupported dimensions.

        """
        payload = (
            getattr(event, "payload", None)
            if hasattr(event, "payload")
            else (event.get("payload") if isinstance(event, dict) else None)
        )
        if payload is None:
            raise TypeError("Event has no 'payload'.")

        arr = self._to_numpy(payload)

        if self._downscale > 1:
            s = self._downscale
            # Downscale by simple striding for efficiency
            if arr.ndim in (2, 3):
                arr = arr[::s, ::s, ...]
            elif arr.ndim == 4:
                arr = arr[:, ::s, ::s, ...]
            else:
                raise ValueError("Payload must be 2D/3D (image) or 4D (video)")

        arr = self._quantize(arr)

        if arr.ndim in (2, 3):
            data, meta = self._encode_image(arr, self._image_fmt)
        elif arr.ndim == 4:
            data, meta = self._encode_video(arr, self._video_fmt, self._fps)
        else:
            raise ValueError("Unsupported payload rank")

        # Preserve event type (dataclass or dict)
        if hasattr(event, "__dict__"):
            try:
                return replace(event, payload=data, encoding=meta, materialized=True)
            except Exception:
                setattr(event, "payload", data)
                setattr(event, "encoding", meta)
                setattr(event, "materialized", True)
                return event
        if isinstance(event, dict):
            e = dict(event)
            e["payload"] = data
            e["encoding"] = meta
            e["materialized"] = True
            # Construct and return a MetricEvent so the return type is consistent.
            return MetricEvent(**e)
        return event

    def _to_numpy(self, x: Any) -> np.ndarray:
        """Convert a supported input to a NumPy array."""
        if jax is not None and hasattr(jax, "Array") and isinstance(x, jax.Array):  # type: ignore[attr-defined]
            return np.array(x)
        if hasattr(x, "__array__"):
            return np.asarray(x)
        raise TypeError(f"Unsupported payload type: {type(x)!r}")

    def _quantize(self, x: np.ndarray) -> np.ndarray:
        """Quantize array to uint8 using a linear mapping.

        Args:
            x (np.ndarray): Input array, float or uint8.

        Returns:
            np.ndarray: Quantized array in [0, 255], dtype uint8.

        Raises:
            ValueError: If custom quantizer does not return uint8.

        """
        if x.dtype == np.uint8:
            return x
        if self._quantize_policy is not None:
            y = self._quantize_policy(x)
            if y.dtype != np.uint8:
                raise ValueError("quantize_policy must return uint8.")
            return y
        x = np.asarray(x, dtype=np.float32)
        finite = np.isfinite(x)
        if not finite.any():
            return np.zeros_like(x, dtype=np.uint8)
        xmin, xmax = np.nanmin(x[finite]), np.nanmax(x[finite])
        # Choose scaling based on known range
        if xmin >= 0.0 and xmax <= 1.0:
            y = np.clip(x * 255.0, 0.0, 255.0)
        else:
            rng = max(1e-12, float(xmax - xmin))
            y = (x - xmin) * (255.0 / rng)
        return y.astype(np.uint8)

    def _encode_image(self, img: np.ndarray, fmt: str) -> tuple[bytes, dict]:
        """Encode a 2D or 3D array as PNG or JPEG bytes."""
        if img.ndim == 2:
            pil = Image.fromarray(img, mode="L")
        elif img.ndim == 3 and img.shape[-1] in (1, 3, 4):
            mode = {1: "L", 3: "RGB", 4: "RGBA"}[img.shape[-1]]
            pil = Image.fromarray(
                img.squeeze(-1) if img.shape[-1] == 1 else img, mode=mode
            )
        else:
            raise ValueError("Image must be HxW, HxWx1, HxWx3, or HxWx4.")
        buf = io.BytesIO()
        save_kwargs = {}
        if fmt == "jpeg":
            save_kwargs.update(quality=90, optimize=True)
        pil.save(buf, format="JPEG" if fmt == "jpeg" else "PNG", **save_kwargs)
        return buf.getvalue(), {
            "content_type": "image/jpeg" if fmt == "jpeg" else "image/png",
            "shape": tuple(img.shape),
        }

    def _encode_video(self, vid: np.ndarray, fmt: str, fps: int) -> tuple[bytes, dict]:
        """Encode a 4D array as an MP4 video.

        Args:
            vid (np.ndarray): Array of shape (T, H, W, C).
            fmt (str): Must be 'mp4'.
            fps (int): Frames per second.

        Returns:
            tuple[bytes, dict]: Encoded bytes and metadata.

        Raises:
            RuntimeError: If `imageio` is unavailable.
            ValueError: If invalid channel count or format.

        """
        if fmt != "mp4":
            raise ValueError("Only MP4 video is supported.")
        if imageio is None:
            raise RuntimeError("imageio is required for MP4 encoding.")
        if vid.ndim == 3:
            vid = vid[..., None]
        if vid.shape[-1] == 1:
            vid = np.repeat(vid, 3, axis=-1)
        if vid.shape[-1] != 3:
            raise ValueError("Video frames must have 1 or 3 channels.")
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            path = tmp.name
        try:
            writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8)
            with writer:
                for frame in vid:
                    writer.append_data(frame.astype(np.uint8))
            with open(path, "rb") as f:
                data = f.read()
        finally:
            try:
                os.remove(path)
            except Exception:
                pass
        return data, {
            "content_type": "video/mp4",
            "shape": tuple(vid.shape),
            "fps": fps,
        }

    def _attach_error(self, event: Any, msg: str) -> Any:
        """Attach an error message to an event."""
        if hasattr(event, "__dict__"):
            setattr(event, "materializer_error", msg)
            return event
        if isinstance(event, dict):
            e = dict(event)
            e["materializer_error"] = msg
            return e
        return event
