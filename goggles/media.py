"""Media utilities for saving images and videos from numpy arrays."""

import yaml
from typing import Literal
import numpy as np
import imageio
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    """Convert a NumPy array to uint8 format for image/video saving.

    Args:
        arr: Input array of arbitrary dtype and range.
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
            raise ValueError(f"Last dimension must be 1 or 3 channels, got {C}.")
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
    imageio.mimsave(
        out_path,
        imgs,
        format="GIF",
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

    with imageio.get_writer(out_path, **writer_kwargs) as writer:
        if arr_u8.ndim == 3:
            for i in range(arr_u8.shape[0]):
                writer.append_data(arr_u8[i])
        else:
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
    imageio.imwrite(out_path, arr_u8[0], format=format)


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
        H, W, _ = vector_field.shape

        # Create figure that matches pixel aspect; keep margins zero by default
        fig, ax = plt.subplots(figsize=(W / 50, H / 50), dpi=dpi)
        ax.set_aspect("equal")

        # Compute scalar field and arrow color based on the selected mode
        if mode == "magnitude":
            scalar_field = np.linalg.norm(vector_field, axis=-1)
            cmap = plt.cm.viridis
            arrow_color = "white"
        elif mode == "vorticity":
            # Compute vorticity (curl) of the vector field: dVx/dy - dVy/dx
            dy = np.gradient(vector_field[..., 0], axis=0)
            dx = np.gradient(vector_field[..., 1], axis=1)
            scalar_field = dx - dy
            cmap = plt.cm.RdBu_r
            arrow_color = "black"
        else:
            raise ValueError("mode must be 'magnitude' or 'vorticity'")

        # Display scalar field as background
        im = ax.imshow(
            scalar_field,
            cmap=cmap,
            origin="lower",
            extent=[0, W, 0, H],
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


class NumpyDumper(yaml.SafeDumper):
    """YAML Dumper that handles NumPy data types."""

    pass


def _represent_numpy_scalar(dumper, value):
    # Order matters: bool first (bool is a subclass of integer in Python)
    if isinstance(value, np.bool_):
        return dumper.represent_bool(bool(value))
    if isinstance(value, np.integer):
        return dumper.represent_int(int(value))
    if isinstance(value, np.floating):
        return dumper.represent_float(float(value))
    # Fallback: just stringify
    return dumper.represent_str(str(value))


def _represent_ndarray(dumper, data: np.ndarray):
    if data.ndim == 0:
        # 0-D array -> scalar
        return _represent_numpy_scalar(dumper, data.item())
    # Higher-D -> nested lists
    return dumper.represent_list(data.tolist())


def _represent_numpy_generic(dumper, data: np.generic):
    # Handles np.int64, np.float32, np.bool_, etc.
    return _represent_numpy_scalar(dumper, data)


# Register representers
NumpyDumper.add_representer(np.ndarray, _represent_ndarray)
NumpyDumper.add_multi_representer(np.generic, _represent_numpy_generic)


def yaml_dump(obj, **kwargs) -> str:
    """Dump an object to a YAML string, handling NumPy types.

    Args:
        obj: The object to dump.
        **kwargs: Additional keyword arguments to pass to `yaml.dump`.

    Returns:
        YAML string representation of the object.
    """
    return yaml.dump(obj, Dumper=NumpyDumper, **kwargs)
