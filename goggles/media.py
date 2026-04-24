"""Media utilities for saving images and videos from numpy arrays."""

import io
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from ruamel.yaml import YAML


class _FrameWriter(Protocol):
    """Subset of imageio writer API used by this module."""

    def append_data(self, image: np.ndarray) -> None:
        """Append a single frame to the output stream.

        Args:
            image: Frame to append.
        """


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    """Convert a NumPy array to uint8 format for image/video saving.

    Args:
        arr: Input array of arbitrary dtype and range.

    Returns:
        uint8 array with values in [0, 255], suitable for saving
            as an image or video.
    """
    if arr.dtype == np.uint8:
        return arr
    a = arr.astype(np.float32)
    vmin = float(np.min(a))
    vmax = float(np.max(a))
    if np.isclose(vmax, vmin):
        return np.zeros_like(a, dtype=np.uint8)
    if vmin >= 0.0 and vmax <= 1.0:
        a = a * 255.0
    else:
        a = (a - vmin) / (vmax - vmin) * 255.0
    return np.clip(a, 0, 255).astype(np.uint8)


def _normalize_frames(frames: np.ndarray) -> tuple[np.ndarray, str]:
    """Normalize input frames to uint8 and determine mode.

    Args:
        frames: Input clip as a NumPy array of shape
            (T, H, W) or (T, H, W, C) where C is 1 or 3.

    Returns:
        Tuple of (uint8 array, mode string).

    Raises:
        ValueError: If input shape or channel count is invalid.
    """
    arr = np.asarray(frames)
    if arr.ndim < 3:
        raise ValueError("Expected shape (T, *image_shape[, C]).")
    # channels
    if arr.ndim == 3:  # (T,H,W)
        C = 1
    else:  # (T,H,W,C)
        C = arr.shape[-1]
        if C not in (1, 3, 4):
            raise ValueError(
                f"Last dimension must be 1 or 3 channels, got {C}."
            )
    if arr.ndim >= 4 and C == 1:
        arr = arr[..., 0]  # -> (T,H,W)
    arr_u8 = _to_uint8(arr)
    mode = "L" if (arr_u8.ndim == 3 or C == 1) else "RGB"
    return arr_u8, mode


def _ensure_even_hw(u8: np.ndarray) -> np.ndarray:
    """Ensure height and width are even by padding if necessary.

    Args:
        u8: Input uint8 array of shape (T, H, W) or (T, H, W, C).

    Returns:
        Padded uint8 array with even height and width.
    """
    if u8.ndim == 3:  # (T,H,W)
        T, H, W = u8.shape
        pad_h, pad_w = H % 2, W % 2
        if pad_h or pad_w:
            out = np.zeros((T, H + pad_h, W + pad_w), dtype=np.uint8)
            out[:, :H, :W] = u8
            return out
        return u8
    else:  # (T,H,W,3)
        T, H, W, C = u8.shape
        pad_h, pad_w = H % 2, W % 2
        if pad_h or pad_w:
            out = np.zeros((T, H + pad_h, W + pad_w, C), dtype=np.uint8)
            out[:, :H, :W, :] = u8
            return out
        return u8


def save_numpy_gif(
    frames: np.ndarray, out_path: str, fps: int = 10, loop: int = 0
) -> None:
    """Save a NumPy clip to GIF using imageio.

    Args:
        frames: Input clip as a NumPy array of shape
            (T, H, W) or (T, H, W, C) where C is 1 or 3.
        out_path: Output file path.
        fps: Frames per second.
        loop: Number of times the GIF should loop (0 = infinite).
    """
    arr_u8, _ = _normalize_frames(frames)
    imgs = [arr_u8[i] for i in range(arr_u8.shape[0])]
    imageio.mimwrite(  # pyright: ignore[reportCallIssue]
        out_path,
        imgs,
        format="GIF",  # pyright: ignore[reportArgumentType]
        duration=1.0 / float(fps),
        loop=loop,
        palettesize=256,
        subrectangles=True,
    )


def save_numpy_mp4(
    frames: np.ndarray,
    out_path: Path,
    fps: int = 30,
    codec: str = "libx264",
    pix_fmt: str = "yuv420p",
    bitrate: str | None = None,
    crf: int | None = 18,
    convert_gray_to_rgb: bool = True,
    preset: str | None = "medium",
) -> None:
    """Save a NumPy clip to MP4 using imageio-ffmpeg.

    Args:
        frames: Input clip as a NumPy array of shape
            (T, H, W) or (T, H, W, C) where C is 1 or 3.
        out_path: Output file path.
        fps: Frames per second.
        codec: Video codec to use.
        pix_fmt: Pixel format for ffmpeg.
        bitrate: Bitrate string for ffmpeg (e.g. "4M").
        crf: Constant Rate Factor for quality control.
        convert_gray_to_rgb: Whether to convert grayscale to RGB.
        preset: ffmpeg preset for speed/quality tradeoff.

    """
    arr_u8, mode = _normalize_frames(frames)

    # Prefer RGB for broad player compatibility
    if mode == "L" and convert_gray_to_rgb:
        arr_u8 = np.stack([arr_u8] * 3, axis=-1)  # (T,H,W) -> (T,H,W,3)

    # Ensure even dims for yuv420p
    arr_u8 = _ensure_even_hw(arr_u8)

    # Build writer kwargs
    writer_kwargs = {
        "fps": fps,
        "codec": codec,
        "macro_block_size": None,
        "format": "FFMPEG",
    }
    if bitrate is not None:
        writer_kwargs["bitrate"] = bitrate

    ffmpeg_params = []
    if crf is not None:
        ffmpeg_params += ["-crf", str(crf)]
    if preset is not None:
        ffmpeg_params += ["-preset", preset]
    if pix_fmt is not None:
        ffmpeg_params += ["-pix_fmt", pix_fmt]
    if ffmpeg_params:
        writer_kwargs["ffmpeg_params"] = ffmpeg_params

    with imageio.get_writer(out_path, **writer_kwargs) as writer_raw:
        writer = cast(_FrameWriter, writer_raw)
        for i in range(arr_u8.shape[0]):
            writer.append_data(arr_u8[i])


def save_numpy_image(image: np.ndarray, out_path: str, format: str) -> None:
    """Save a NumPy image to file using imageio.

    Args:
        image: Input image as a NumPy array of shape
            (H, W) or (H, W, C) where C is 1 or 3.
        out_path: Output file path.
        format: Image format (e.g., 'png', 'jpg', 'jpeg').

    """
    arr_u8, _ = _normalize_frames(image[np.newaxis, ...])
    imageio.imwrite(out_path, arr_u8[0], format=format)  # pyright: ignore[reportCallIssue, reportArgumentType]


def save_numpy_vector_field_visualization(
    vector_field: np.ndarray,
    dir: Path,
    name: str,
    mode: Literal["vorticity", "magnitude"] = "magnitude",
    arrow_stride: int = 8,
    dpi: int = 300,
    add_colorbar: bool = True,
) -> None:
    """Save a 2D vector field visualization as a PNG image.

    Args:
        vector_field: Input vector field of shape (H, W, 2).
        dir: Output directory path.
        name: Base name for the output PNG file (without extension).
        mode: Visualization mode.
        arrow_stride: Stride for downsampling arrows (every Nth point).
        dpi: Resolution of the output image.
        add_colorbar: Whether to include a colorbar.
    """
    # Store original backend to restore later
    original_backend = matplotlib.get_backend()
    matplotlib.use("Agg")

    try:
        fig, bbox_setting, pad_setting = _build_vector_field_figure(
            vector_field=vector_field,
            mode=mode,
            arrow_stride=arrow_stride,
            dpi=dpi,
            add_colorbar=add_colorbar,
        )

        # Save
        dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            str(dir / f"{name}.png"),
            dpi=dpi,
            bbox_inches=bbox_setting,
            pad_inches=pad_setting,
            facecolor="white",
            edgecolor="none",
        )
        plt.close(fig)

    finally:
        # Restore original matplotlib backend
        matplotlib.use(original_backend)


def create_numpy_vector_field_visualization(
    vector_field: np.ndarray,
    mode: Literal["vorticity", "magnitude"] = "magnitude",
    arrow_stride: int = 8,
    dpi: int = 300,
    add_colorbar: bool = True,
) -> np.ndarray:
    """Create a vector field visualization as an RGB NumPy image.

    Args:
        vector_field: Input vector field of shape (H, W, 2).
        mode: Visualization mode.
        arrow_stride: Stride for downsampling arrows (every Nth point).
        dpi: Resolution used by matplotlib while rendering.
        add_colorbar: Whether to include a colorbar.

    Returns:
        Rendered image as a uint8 array with shape (H, W, 3).
    """
    # Store original backend to restore later
    original_backend = matplotlib.get_backend()
    matplotlib.use("Agg")

    try:
        fig, bbox_setting, pad_setting = _build_vector_field_figure(
            vector_field=vector_field,
            mode=mode,
            arrow_stride=arrow_stride,
            dpi=dpi,
            add_colorbar=add_colorbar,
        )

        buf = io.BytesIO()
        fig.savefig(
            buf,
            format="png",
            dpi=dpi,
            bbox_inches=bbox_setting,  # "tight"
            pad_inches=pad_setting,  # your 0.02
            facecolor="white",
            edgecolor="none",
        )
        plt.close(fig)
        buf.seek(0)
        rgba = imageio.imread(buf)  # (H,W,4) uint8 RGBA

        image = np.ascontiguousarray(rgba[..., :3])
        return image

    finally:
        # Restore original matplotlib backend
        matplotlib.use(original_backend)


def _build_vector_field_figure(
    vector_field: np.ndarray,
    mode: Literal["vorticity", "magnitude"],
    arrow_stride: int,
    dpi: int,
    add_colorbar: bool,
) -> tuple[Any, str, float]:
    """Build a matplotlib figure visualizing a 2D vector field.

    Args:
        vector_field: Input vector field of shape (H, W, 2).
        mode: Visualization mode.
        arrow_stride: Stride for downsampling arrows (every Nth point).
        dpi: Resolution used by matplotlib while rendering.
        add_colorbar: Whether to include a colorbar.

    Returns:
        Tuple of (figure, bbox_setting, pad_setting) where:
        - figure: The matplotlib figure object.
        - bbox_setting: Value for `bbox_inches` to use when saving.
        - pad_setting: Value for `pad_inches` to use when saving.

    Raises:
        ValueError: If `mode` is invalid or input shape is incorrect.
    """
    H = vector_field.shape[0]
    W = vector_field.shape[1]

    # Create figure that matches pixel aspect; keep margins zero by default
    fig, ax = plt.subplots(figsize=(W / 50, H / 50), dpi=dpi)
    ax.set_aspect("equal")

    # Compute scalar field and arrow color based on the selected mode
    if mode == "magnitude":
        scalar_field = np.linalg.norm(vector_field, axis=-1)
        cmap = plt.get_cmap("viridis")
        arrow_color = "white"
    elif mode == "vorticity":
        # Compute vorticity (curl) of the vector field: dVx/dy - dVy/dx
        dy = np.gradient(vector_field[..., 0], axis=0)
        dx = np.gradient(vector_field[..., 1], axis=1)
        scalar_field = dx - dy
        cmap = plt.get_cmap("RdBu_r")
        arrow_color = "black"
    else:
        raise ValueError("mode must be 'magnitude' or 'vorticity'")

    # Display scalar field as background
    im = ax.imshow(
        scalar_field,
        cmap=cmap,
        origin="lower",
        extent=(0, W, 0, H),
        interpolation="bilinear",
    )

    # Arrow grid
    y_coords, x_coords = np.mgrid[0:H:arrow_stride, 0:W:arrow_stride]
    u_sampled = vector_field[::arrow_stride, ::arrow_stride, 0]
    v_sampled = vector_field[::arrow_stride, ::arrow_stride, 1]

    # Plot arrows
    ax.quiver(
        x_coords,
        y_coords,
        u_sampled,
        v_sampled,
        color=arrow_color,
        alpha=0.9,
        scale_units="xy",
        scale=1,
        width=0.002,
        headwidth=4,
        headlength=5,
        headaxislength=4.5,
        linewidth=0.5,
        edgecolor="none",
    )

    # Remove axes and ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    # (1) Colorbar exactly same height as the axes
    if add_colorbar:
        divider = make_axes_locatable(ax)
        # size can be tweaked; "3%" is a nice thin bar, pad is the gap
        cax = divider.append_axes("right", size="3%", pad=0.02)
        cb = fig.colorbar(im, cax=cax)
        cb.ax.tick_params(length=2)

        # Leave a tiny margin so ticks/labels aren't clipped
        fig.subplots_adjust(left=0.0, right=0.98, bottom=0.0, top=1.0)
        bbox_setting = "tight"
        pad_setting = 0.02
    else:
        # (2) No colorbar: strip all outer boundaries/margins
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        bbox_setting = "tight"
        pad_setting = 0.0

    return fig, bbox_setting, pad_setting


def _build_trajectories_figure(
    trajectories: np.ndarray,
    dpi: int,
) -> Any:
    """Build a matplotlib figure visualizing a batch of trajectories.

    Args:
        trajectories: Array of shape ``(N, L, dim)`` with ``dim`` in
            ``{2, 3}``.
        dpi: Resolution used by matplotlib while rendering.

    Returns:
        The matplotlib figure.

    Raises:
        ValueError: If the input shape or dimension is invalid.
    """
    if trajectories.ndim != 3:
        raise ValueError(
            "Trajectories must have shape (N, L, dim); "
            f"got {trajectories.shape}."
        )
    N, _, dim = trajectories.shape
    if dim not in (2, 3):
        raise ValueError(f"Trajectories dim must be 2 or 3; got {dim}.")

    subplot_kw: dict[str, Any] = {"projection": "3d"} if dim == 3 else {}
    fig, ax = plt.subplots(dpi=dpi, subplot_kw=subplot_kw)
    for n in range(N):
        ax.plot(
            *(trajectories[n, :, i] for i in range(dim)),
            linestyle="-",
            marker=".",
            markersize=2,
            linewidth=0.8,
        )
    if dim == 2:
        ax.set_aspect("equal")
    return fig


def save_numpy_trajectories_visualization(
    trajectories: np.ndarray,
    dir: Path,
    name: str,
    dpi: int = 200,
) -> None:
    """Save a trajectories visualization as a PNG image.

    Args:
        trajectories: Array of shape ``(N, L, dim)``.
        dir: Output directory.
        name: Base name for the output PNG file (no extension).
        dpi: Rendering resolution.
    """
    original_backend = matplotlib.get_backend()
    matplotlib.use("Agg")
    try:
        fig = _build_trajectories_figure(trajectories, dpi=dpi)
        dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            str(dir / f"{name}.png"),
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.05,
            facecolor="white",
            edgecolor="none",
        )
        plt.close(fig)
    finally:
        matplotlib.use(original_backend)


def create_numpy_trajectories_visualization(
    trajectories: np.ndarray,
    dpi: int = 200,
) -> np.ndarray:
    """Render a trajectories visualization to a uint8 RGB image.

    Args:
        trajectories: Array of shape ``(N, L, dim)``.
        dpi: Rendering resolution.

    Returns:
        Rendered image as a ``uint8`` array with shape ``(H, W, 3)``.
    """
    original_backend = matplotlib.get_backend()
    matplotlib.use("Agg")
    try:
        fig = _build_trajectories_figure(trajectories, dpi=dpi)
        buf = io.BytesIO()
        fig.savefig(
            buf,
            format="png",
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.05,
            facecolor="white",
            edgecolor="none",
        )
        plt.close(fig)
        buf.seek(0)
        rgba = imageio.imread(buf)
        return np.ascontiguousarray(rgba[..., :3])
    finally:
        matplotlib.use(original_backend)


def _to_python(obj: Any) -> Any:
    """Recursively convert NumPy types to their native Python equivalents.

    Args:
        obj: Arbitrary Python value, possibly nested, possibly containing
            NumPy scalars or arrays.

    Returns:
        A value composed solely of ``dict``/``list``/builtin scalars,
        safe to hand to a plain YAML dumper. Sequence inputs such as
        tuples are normalized to lists.
    """
    if isinstance(obj, np.ndarray):
        return _to_python(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {_to_python(k): _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_python(x) for x in obj]
    return obj


def yaml_dump(obj: Any) -> str:
    """Dump an object to a YAML string, handling NumPy types.

    Args:
        obj: The object to dump. NumPy arrays/scalars are normalized to
            native Python types before serialization.

    Returns:
        YAML string representation of the object.
    """
    yaml = YAML(typ="safe", pure=True)
    buf = io.StringIO()
    yaml.dump(_to_python(obj), buf)
    return buf.getvalue()
