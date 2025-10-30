"""JSONL integration for Goggles logging framework."""

import json
import threading
from pathlib import Path
from typing import Any, FrozenSet, Optional
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
    capabilities: FrozenSet[str] = frozenset(
        {"log", "metric", "image", "video", "artifact", "vector_field", "histogram"}
    )

    def __init__(self, path: Path, name: str = "jsonl") -> None:
        """Initialize the handler with a base directory.

        Args:
            path (Path): Base directory for logs and media files. Will be created if it doesn't exist.
            name (str): Handler identifier (for logging diagnostics).

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
            kind (Kind): Kind of event ("log", "metric", "image", "artifact").

        Returns:
            bool: True if the kind is supported, False otherwise.

        """
        return kind in self.capabilities

    def handle(self, event: Event) -> None:
        """Write a single event to the JSONL file.

        Args:
            event (Event): The event to serialize.

        """
        event = event.to_dict()

        # Handle media events by saving files and updating payload
        kind = event["kind"]
        if kind == "image":
            event = self._save_image_to_file(event)
        elif kind == "video":
            event = self._save_video_to_file(event)
        elif kind == "artifact":
            event = self._save_artifact_to_file(event)
        elif kind == "vector_field":
            event = self._save_vector_field_to_file(event)
        elif kind == "histogram":
            event = self._save_histogram_to_file(event)

        if event is None:
            self._logger.warning(
                "Skipping event logging due to unsupported media format."
            )
            return

        try:
            with self._lock:
                json.dump(
                    event, self._fp, ensure_ascii=False, default=self._json_serializer
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
        """Reconstruct a handler from its serialized representation."""
        data = serialized.get("data", serialized)
        return cls(
            path=Path(data["path"]),
            name=data["name"],
        )

    def _json_serializer(self, obj: Any) -> str:
        """Serialize object to JSON-compatible format.

        Args:
            obj: Object to serialize.

        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()

        # For other non-serializable objects, convert to string
        return str(obj)

    def _save_image_to_file(self, event: dict) -> dict:
        """Save image data to file and update event with file path.

        Args:
            event (dict): Event dictionary.

        Returns:
            dict: Updated event with file path instead of raw data.

        """
        image_format = "png"
        if event["extra"] and "format" in event["extra"]:
            image_format = event["extra"]["format"]

        image_name = str(uuid4())
        if event["extra"] and "name" in event["extra"]:
            image_name = event["extra"]["name"]

        image_path = self._images_dir / Path(f"{image_name}.{image_format}")
        save_numpy_image(
            event["payload"],
            image_path,
            format=image_format,
        )

        event["payload"] = str(image_path.relative_to(self._base_path))
        return event

    def _save_video_to_file(self, event: dict) -> dict:
        """Save video data to file and update event with file path.

        Args:
            event (dict): Event dictionary.

        Returns:
            dict: Updated event with file path instead of raw data.

        """
        video_format = "mp4"
        if "format" in event["extra"]:
            video_format = event["extra"]["format"]
        if video_format not in {"mp4", "gif"}:
            self._logger.warning(
                f"Unknown video format '{video_format}'."
                " Supported formats are: 'mp4', 'gif'."
                " The video will not be logged."
            )
            return None

        video_name = str(uuid4())
        if event["extra"] and "name" in event["extra"]:
            video_name = event["extra"]["name"]

        fps = 1.0
        if event["extra"] and "fps" in event["extra"]:
            fps = float(event["extra"]["fps"])

        if video_format == "gif":
            video_data: np.ndarray = event["payload"]
            loop = 0
            if event["extra"] and "loop" in event["extra"]:
                loop = event["extra"]["loop"]
            gif_path = self._videos_dir / Path(f"{video_name}.gif")
            save_numpy_gif(video_data, gif_path, fps=fps, loop=loop)
            event["payload"] = str(gif_path.relative_to(self._base_path))
        elif video_format == "mp4":
            video_data: np.ndarray = event["payload"]
            video_codec = "libx264"
            pix_fmt = "yuv420p"
            bitrate = None
            crf = 18
            convert_gray_to_rgb = True
            preset = "medium"
            if event["extra"]:
                if "codec" in event["extra"]:
                    video_codec = event["extra"]["codec"]
                if "pix_fmt" in event["extra"]:
                    pix_fmt = event["extra"]["pix_fmt"]
                if "bitrate" in event["extra"]:
                    bitrate = event["extra"]["bitrate"]
                if "crf" in event["extra"]:
                    crf = event["extra"]["crf"]
                if "convert_gray_to_rgb" in event["extra"]:
                    convert_gray_to_rgb = event["extra"]["convert_gray_to_rgb"]
                if "preset" in event["extra"]:
                    preset = event["extra"]["preset"]

            mp4_path = self._videos_dir / Path(f"{video_name}.mp4")
            save_numpy_mp4(
                video_data,
                mp4_path,
                fps=fps,
                codec=video_codec,
                pix_fmt=pix_fmt,
                bitrate=bitrate,
                crf=crf,
                convert_gray_to_rgb=convert_gray_to_rgb,
                preset=preset,
            )
            event["payload"] = str(mp4_path.relative_to(self._base_path))
        return event

    def _save_artifact_to_file(self, event: dict) -> Optional[dict]:
        """Save artifact data to file and update event with file path.

        Args:
            event (dict): Event dictionary.

        Returns:
            Optional[dict]: Updated event with file path instead of raw data.
                If the artifact format is unknown, returns None.

        """
        artifact_format = "txt"
        if event["extra"] and "format" in event["extra"]:
            artifact_format = event["extra"]["format"]

        if artifact_format not in {"txt", "csv", "json", "yaml"}:
            self._logger.warning(
                f"Unknown artifact format '{artifact_format}'."
                " Supported formats are: 'txt', 'csv', 'json', 'yaml'."
                " The artifact will not be logged."
            )
            return None

        if artifact_format == "json":
            import json

            event["payload"] = json.dumps(event["payload"], indent=2)

        if artifact_format == "yaml":
            import yaml

            event["payload"] = yaml.dump(event["payload"])

        artifact_name = str(uuid4())
        if event["extra"] and "name" in event["extra"]:
            artifact_name = event["extra"]["name"]

        artifact_path = self._artifacts_dir / Path(f"{artifact_name}.{artifact_format}")

        with open(artifact_path, "w") as f:
            f.write(event["payload"])

        event["payload"] = str(artifact_path.relative_to(self._base_path))
        return event

    def _save_vector_field_to_file(self, event: dict) -> Optional[dict]:
        """Save vector field data to file and update event with file path.

        Args:
            event (dict): Event dictionary.

        Returns:
            dict: Updated event with file path instead of raw data.

        """
        vector_field_name = str(uuid4())
        if event["extra"] and "name" in event["extra"]:
            vector_field_name = event["extra"]["name"]

        if event["extra"] and "store_visualization" in event["extra"]:
            add_colorbar = False
            if event["extra"] and "add_colorbar" in event["extra"]:
                add_colorbar = event["extra"]["add_colorbar"]

            mode = "magnitude"
            if event["extra"] and "mode" in event["extra"]:
                mode = event["extra"]["mode"]

            if mode not in {"vorticity", "magnitude"}:
                self._logger.warning(
                    f"Unknown vector field visualization mode '{mode}'."
                    " Supported modes are: 'vorticity', 'magnitude'."
                    " The vector field visualization will not be saved."
                )
            else:
                save_numpy_vector_field_visualization(
                    event["payload"],
                    dir=self._vector_fields_dir,
                    name=f"{vector_field_name}_visualization",
                    mode=mode,
                    add_colorbar=add_colorbar,
                )

        vector_field_path = self._vector_fields_dir / Path(f"{vector_field_name}.npy")
        np.save(vector_field_path, event["payload"])

        event["payload"] = str(vector_field_path.relative_to(self._base_path))
        return event

    def _save_histogram_to_file(self, event: dict) -> Optional[dict]:
        """Save histogram data to file and update event with file path.

        Args:
            event (dict): Event dictionary.

        Returns:
            dict: Updated event with file path instead of raw data.

        """
        histogram_name = str(uuid4())
        if event["extra"] and "name" in event["extra"]:
            histogram_name = event["extra"]["name"]

        histogram_path = self._histograms_dir / Path(f"{histogram_name}.npy")
        np.save(histogram_path, event["payload"])

        event["payload"] = str(histogram_path.relative_to(self._base_path))
        return event
