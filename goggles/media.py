"""Media utilities for saving images and videos from numpy arrays."""

import numpy as np
import imageio


def _to_uint8(arr: np.ndarray) -> np.ndarray:
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


def _normalize_frames(frames: np.ndarray):
    arr = np.asarray(frames)
    if arr.ndim < 3:
        raise ValueError("Expected shape (T, *image_shape[, C]).")
    # channels
    if arr.ndim == 3:  # (T,H,W)
        C = 1
    else:  # (T,H,W,C)
        C = arr.shape[-1]
        if C not in (1, 3):
            raise ValueError(f"Last dimension must be 1 or 3 channels, got {C}.")
    if arr.ndim >= 4 and C == 1:
        arr = arr[..., 0]  # -> (T,H,W)
    arr_u8 = _to_uint8(arr)
    mode = "L" if (arr_u8.ndim == 3 or C == 1) else "RGB"
    return arr_u8, mode


def _ensure_even_hw(u8: np.ndarray) -> np.ndarray:
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
        frames (np.ndarray): Input clip as a NumPy array of shape
            (T, H, W) or (T, H, W, C) where C is 1 or 3.
        out_path (str): Output file path.
        fps (int): Frames per second.
        loop (int): Number of times the GIF should loop (0 = infinite).

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
    out_path: str,
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
        frames (np.ndarray): Input clip as a NumPy array of shape
            (T, H, W) or (T, H, W, C) where C is 1 or 3.
        out_path (str): Output file path.
        fps (int): Frames per second.
        codec (str): Video codec to use.
        pix_fmt (str): Pixel format for ffmpeg.
        bitrate (str | None): Bitrate string for ffmpeg (e.g. "4M").
        crf (int | None): Constant Rate Factor for quality control.
        convert_gray_to_rgb (bool): Whether to convert grayscale to RGB.
        preset (str | None): ffmpeg preset for speed/quality tradeoff.

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
        image (np.ndarray): Input image as a NumPy array of shape
            (H, W) or (H, W, C) where C is 1 or 3.
        out_path (str): Output file path.
        format (str): Image format (e.g., 'png', 'jpg', 'jpeg').

    """
    arr_u8, _ = _normalize_frames(image[np.newaxis, ...])
    imageio.imwrite(out_path, arr_u8[0], format=format)
