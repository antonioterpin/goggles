"""JSONL integration for Goggles logging framework."""

import json
import threading
from pathlib import Path
from typing import Any, ClassVar
from uuid import uuid4
from typing_extensions import Self
import logging
import numpy as np

from goggles.types import Event, Kind
from goggles.media import (
    save_numpy_gif,
    save_numpy_image,
    save_numpy_mp4,
    save_numpy_vector_field_visualization,
    yaml_dump,
)


class LocalStorageHandler:
    """Write events to a structured directory locally.

    This handler creates a directory structure:
    - {base_path}/log.jsonl: Main JSONL log file with all events
    - {base_path}/images/: Directory for image files
    - {base_path}/videos/: Directory for video files
    - {base_path}/artifacts/: Directory for other artifact files

    For media events (image, video, artifact), the binary data is saved to
    the appropriate subdirectory and the relative path is logged in the
    JSONL file instead of the raw data.

    Thread-safe and line-buffered, ensuring atomic writes per event.

    Attributes:
        name (str): Stable handler identifier.
        capabilities (set[str]): Supported event kinds.

    """

    name: str = "jsonl"
    capabilities: ClassVar[frozenset[Kind]] = frozenset(
        {"log", "metric", "image", "video", "artifact", "vector_field", "histogram"}
    )

    def __init__(self, path: Path, name: str = "jsonl") -> None:
        """Initialize the handler with a base directory.

        Args:
            path: Base directory for logs and media files. Will be created if it doesn't exist.
            name: Handler identifier (for logging diagnostics).

        """
        self.name = name
        self._base_path = Path(path)

    def open(self) -> None:
        """Create directory structure and open the JSONL file for appending."""
        self._lock = threading.Lock()

        # Create directory structure
        self._log_file = self._base_path / "log.jsonl"
        self._images_dir = self._base_path / "images"
        self._videos_dir = self._base_path / "videos"
        self._artifacts_dir = self._base_path / "artifacts"
        self._vector_fields_dir = self._base_path / "vector_fields"
        self._histograms_dir = self._base_path / "histograms"
        self._base_path.mkdir(parents=True, exist_ok=True)
        self._images_dir.mkdir(exist_ok=True)
        self._videos_dir.mkdir(exist_ok=True)
        self._artifacts_dir.mkdir(exist_ok=True)
        self._vector_fields_dir.mkdir(exist_ok=True)
        self._histograms_dir.mkdir(exist_ok=True)

        # Open log file
        self._fp = open(self._log_file, "a", encoding="utf-8", buffering=1)

        # Open logger for diagnostics
        self._logger = logging.getLogger(self.name)

    def close(self) -> None:
        """Flush and close the JSONL file."""
        if self._fp and not self._fp.closed:
            with self._lock:
                self._fp.flush()
                self._fp.close()

    def can_handle(self, kind: Kind) -> bool:
        """Return True if this handler supports the given event kind.

        Args:
            kind: Kind of event ("log", "metric", "image", "artifact").

        Returns:
            True if the kind is supported, False otherwise.

        """
        return kind in self.capabilities

    def handle(self, event: Event) -> None:
        """Write a single event to the JSONL file.

        Args:
            event: The event to serialize.

        """
        event_dict = event.to_dict()

        # Handle media events by saving files and updating payload
        kind = event_dict["kind"]
        if kind == "image":
            event_dict = self._save_image_to_file(event_dict)
        elif kind == "video":
            event_dict = self._save_video_to_file(event_dict)
        elif kind == "artifact":
            event_dict = self._save_artifact_to_file(event_dict)
        elif kind == "vector_field":
            event_dict = self._save_vector_field_to_file(event_dict)
        elif kind == "histogram":
            event_dict = self._save_histogram_to_file(event_dict)

        if event_dict is None:
            self._logger.warning(
                "Skipping event logging due to unsupported media format."
            )
            return

        try:
            with self._lock:
                json.dump(
                    event_dict,
                    self._fp,
                    ensure_ascii=False,
                    default=self._json_serializer,
                )
                self._fp.write("\n")
                self._fp.flush()
        except Exception:
            logging.getLogger(self.name).exception("Failed to write JSONL event")

    def to_dict(self) -> dict:
        """Serialize handler configuration to dictionary."""
        return {
            "cls": self.__class__.__name__,
            "data": {
                "path": str(self._base_path),
                "name": self.name,
            },
        }

    @classmethod
    def from_dict(cls, serialized: dict) -> Self:
        """Reconstruct a handler from its serialized representation.

        Args:
            serialized: Serialized handler dictionary.

        Returns:
            Reconstructed LocalStorageHandler instance.
        """
        data = serialized.get("data", serialized)
        return cls(
            path=Path(data["path"]),
            name=data["name"],
        )

    def _json_serializer(self, obj: Any) -> str:
        """Serialize object to JSON-compatible format.

        Args:
            obj: Object to serialize.

        Returns:
            JSON-serializable representation of the object.
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return str(obj.item())

        # For other non-serializable objects, convert to string
        return str(obj)

    def _save_image_to_file(self, event: dict) -> dict | None:
        """Save image data to file and update event with file path.

        Args:
            event: Event dictionary.

        Returns:
            Updated event with file path instead of raw data.
            None if the image could not be saved.

        """
        image_format = "png"
        if "extra.format" in event:
            image_format = event["extra.format"]

        image_path = self._make_media_name(event, self._images_dir, image_format)
        try:
            save_numpy_image(
                event["payload"],
                str(image_path),
                format=image_format,
            )
        except Exception:
            return None

        event["payload"] = str(image_path.relative_to(self._base_path))
        return event

    def _save_video_to_file(self, event: dict) -> dict | None:
        """Save video data to file and update event with file path.

        Args:
            event: Event dictionary.

        Returns:
            Updated event with file path instead of raw data.
            None if the video could not be saved.

        """
        video_format = "mp4"
        if "extra.format" in event:
            video_format = event["extra.format"]
        if video_format not in {"mp4", "gif"}:
            return None

        video_path = self._make_media_name(event, self._videos_dir, video_format)

        fps = 1.0
        if "extra.fps" in event:
            fps = float(event["extra.fps"])

        if video_format == "gif":
            video_data: np.ndarray = event["payload"]
            loop = 0
            if "extra.loop" in event:
                loop = event["extra.loop"]
            save_numpy_gif(video_data, str(video_path), fps=int(fps), loop=loop)
            event["payload"] = str(video_path.relative_to(self._base_path))
        elif video_format == "mp4":
            video_data: np.ndarray = event["payload"]
            video_codec = "libx264"
            pix_fmt = "yuv420p"
            bitrate = None
            crf = 18
            convert_gray_to_rgb = True
            preset = "medium"
            if "extra.codec" in event:
                video_codec = event["extra.codec"]
            if "extra.pix_fmt" in event:
                pix_fmt = event["extra.pix_fmt"]
            if "extra.bitrate" in event:
                bitrate = event["extra.bitrate"]
            if "extra.crf" in event:
                crf = event["extra.crf"]
            if "extra.convert_gray_to_rgb" in event:
                convert_gray_to_rgb = event["extra.convert_gray_to_rgb"]
            if "extra.preset" in event:
                preset = event["extra.preset"]

            save_numpy_mp4(
                video_data,
                video_path,
                fps=int(fps),
                codec=video_codec,
                pix_fmt=pix_fmt,
                bitrate=bitrate,
                crf=crf,
                convert_gray_to_rgb=convert_gray_to_rgb,
                preset=preset,
            )
            event["payload"] = str(video_path.relative_to(self._base_path))
        return event

    def _save_artifact_to_file(self, event: dict) -> dict | None:
        """Save artifact data to file and update event with file path.

        Args:
            event: Event dictionary.

        Returns:
            Updated event with file path instead of raw data.
            If the artifact format is unknown, returns None.

        """
        artifact_format = "txt"
        if "extra.format" in event:
            artifact_format = event["extra.format"]

        artifact_path = self._make_media_name(
            event, self._artifacts_dir, artifact_format
        )

        if artifact_format not in {"txt", "csv", "json", "yaml"}:
            return None

        if artifact_format == "json":
            event["payload"] = json.dumps(event["payload"], indent=2)

        if artifact_format == "yaml":
            event["payload"] = yaml_dump(event["payload"])

        with open(artifact_path, "w") as f:
            f.write(event["payload"])

        event["payload"] = str(artifact_path.relative_to(self._base_path))
        return event

    def _save_vector_field_to_file(self, event: dict) -> dict | None:
        """Save vector field data to file and update event with file path.

        Args:
            event: Event dictionary.

        Returns:
            dict | None: Updated event with file path instead of raw data.

        """
        vector_field_path = self._make_media_name(event, self._vector_fields_dir, "npy")

        if "extra.store_visualization" in event:
            add_colorbar = False
            if "extra.add_colorbar" in event:
                add_colorbar = event["extra.add_colorbar"]

            mode = "magnitude"
            if "extra.mode" in event:
                mode = event["extra.mode"]

            if mode not in {"vorticity", "magnitude"}:
                self._logger.warning(
                    f"Unknown vector field visualization mode '{mode}'."
                    " Supported modes are: 'vorticity', 'magnitude'."
                    " The vector field visualization will not be saved."
                )
            else:
                save_numpy_vector_field_visualization(
                    event["payload"],
                    dir=vector_field_path.parent / Path("visualizations"),
                    name=vector_field_path.stem,
                    mode=mode,
                    add_colorbar=add_colorbar,
                )

        np.save(vector_field_path, event["payload"])

        event["payload"] = str(vector_field_path.relative_to(self._base_path))
        return event

    def _save_histogram_to_file(self, event: dict) -> dict | None:
        """Save histogram data to file and update event with file path.

        Args:
            event: Event dictionary.

        Returns:
            dict: Updated event with file path instead of raw data.

        """
        histogram_path = self._make_media_name(event, self._histograms_dir, "npy")
        np.save(histogram_path, event["payload"])

        event["payload"] = str(histogram_path.relative_to(self._base_path))
        return event

    def _make_media_name(self, event: dict, media_dir: Path, ext: str) -> Path:
        """Get the name of the media from the event extra, or generate a UUID.

        Args:
            event: Event dictionary.
            media_dir: Directory to save the media file.
            ext: File extension for the media file.

        Returns:
            Path: Path to the media file.
        """
        media_name = str(uuid4())
        if "extra.name" in event:
            media_name = event["extra.name"]
        if "step" in event and event["step"] is not None:
            media_name += f"_{event['step']}"
        path = media_dir / Path(f"{media_name}.{ext}")
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
