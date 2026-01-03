import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from goggles._core.integrations.storage import LocalStorageHandler


@pytest.fixture
def tmp_handler(tmp_path):
    handler = LocalStorageHandler(tmp_path)
    handler.open()
    yield handler
    handler.close()


def make_event(kind, payload=None, extra=None):
    event = MagicMock()
    event_data = {
        "kind": kind,
        "payload": payload,
        "extra": {},  # Keep top-level empty for typing compatibility if needed
    }
    if extra:
        for k, v in extra.items():
            event_data[f"extra.{k}"] = v

    event.to_dict.return_value = event_data
    return event


# -------------------------------------------------------------------------
# Initialization and structure
# -------------------------------------------------------------------------


def test_open_creates_directories(tmp_path):
    handler = LocalStorageHandler(tmp_path)
    handler.open()
    for sub in [
        "images",
        "videos",
        "artifacts",
        "vector_fields",
        "histograms",
    ]:
        assert (tmp_path / sub).exists()
    assert (tmp_path / "log.jsonl").exists()


def test_to_dict_and_from_dict_roundtrip(tmp_path):
    handler = LocalStorageHandler(tmp_path, name="test_jsonl")
    data = handler.to_dict()
    rebuilt = LocalStorageHandler.from_dict(data)
    assert rebuilt.name == handler.name
    assert rebuilt._base_path == handler._base_path


# -------------------------------------------------------------------------
# Media serialization helpers
# -------------------------------------------------------------------------


def test_json_serializer_numpy_types(tmp_handler):
    arr = np.array([1, 2, 3])
    assert tmp_handler._json_serializer(arr) == [1, 2, 3]
    assert isinstance(tmp_handler._json_serializer(np.int64(5)), int)
    assert isinstance(tmp_handler._json_serializer(np.float32(1.0)), float)
    assert "object" in tmp_handler._json_serializer(object())


@patch("goggles._core.integrations.storage.save_numpy_image")
def test_save_image_to_file_calls_helper(mock_save, tmp_handler):
    event = {
        "payload": np.zeros((4, 4, 3)),
        "extra.format": "png",
        "extra.name": "img_name",
    }
    updated = tmp_handler._save_image_to_file(event)
    assert "images/img_name.png" in updated["payload"]
    mock_save.assert_called_once()


@patch("goggles._core.integrations.storage.save_numpy_mp4")
def test_save_video_to_file_mp4(mock_save, tmp_handler):
    event = {
        "payload": np.zeros((2, 2, 2, 3)),
        "payload": np.zeros((2, 2, 2, 3)),
        "extra.format": "mp4",
        "extra.name": "vid",
        "extra.fps": 5.0,
    }
    updated = tmp_handler._save_video_to_file(event)
    assert "videos/vid.mp4" in updated["payload"]
    mock_save.assert_called_once()


@patch("goggles._core.integrations.storage.save_numpy_gif")
def test_save_video_to_file_gif(mock_save, tmp_handler):
    event = {
        "payload": np.zeros((2, 2, 2, 3)),
        "payload": np.zeros((2, 2, 2, 3)),
        "extra.format": "gif",
        "extra.name": "anim",
        "extra.fps": 10.0,
        "extra.loop": 1,
    }
    updated = tmp_handler._save_video_to_file(event)
    assert "videos/anim.gif" in updated["payload"]
    mock_save.assert_called_once()


def test_save_video_to_file_unknown_format_warns(tmp_handler, caplog):
    event = {"payload": np.zeros((2, 2)), "extra.format": "avi"}
    res = tmp_handler._save_video_to_file(event)
    assert res is None
    assert any("Unknown video format" in m for m in caplog.messages)


def test_save_artifact_to_file_json(tmp_handler):
    payload = {"key": "value"}
    event = {"payload": payload, "extra.format": "json", "extra.name": "art"}
    updated = tmp_handler._save_artifact_to_file(event)
    path = tmp_handler._base_path / updated["payload"]
    assert path.exists()
    with open(path) as f:
        data = json.load(f)
    assert data == payload


def test_save_artifact_to_file_yaml(tmp_handler):
    payload = {"key": "value"}
    event = {"payload": payload, "extra.format": "yaml", "extra.name": "art2"}
    updated = tmp_handler._save_artifact_to_file(event)
    path = tmp_handler._base_path / updated["payload"]
    assert path.exists()
    text = path.read_text()
    assert "key" in text


def test_save_artifact_to_file_unknown_format_warns(tmp_handler, caplog):
    event = {"payload": "data", "extra.format": "bin"}
    res = tmp_handler._save_artifact_to_file(event)
    assert res is None
    assert any("Unknown artifact format" in m for m in caplog.messages)


@patch("goggles._core.integrations.storage.save_numpy_vector_field_visualization")
def test_save_vector_field_to_file(mock_save, tmp_handler):
    event = {
        "payload": np.zeros((2, 2, 2)),
        "payload": np.zeros((2, 2, 2)),
        "extra.store_visualization": True,
        "extra.mode": "magnitude",
        "extra.name": "vf",
    }
    updated = tmp_handler._save_vector_field_to_file(event)
    path = tmp_handler._base_path / updated["payload"]
    assert path.exists()
    mock_save.assert_called_once()


def test_save_vector_field_to_file_with_unknown_mode_warns(tmp_handler, caplog):
    event = {
        "payload": np.zeros((2, 2, 2)),
        "payload": np.zeros((2, 2, 2)),
        "extra.store_visualization": True,
        "extra.mode": "unknown",
    }
    tmp_handler._save_vector_field_to_file(event)
    assert any("Unknown vector field visualization mode" in m for m in caplog.messages)


def test_save_histogram_to_file(tmp_handler):
    event = {"payload": np.arange(10), "extra.name": "hist"}
    updated = tmp_handler._save_histogram_to_file(event)
    path = tmp_handler._base_path / updated["payload"]
    assert np.load(path).shape == (10,)


# -------------------------------------------------------------------------
# Handle and can_handle
# -------------------------------------------------------------------------


def test_can_handle_recognized_and_unrecognized(tmp_handler):
    assert tmp_handler.can_handle("image")
    assert not tmp_handler.can_handle("nonsense")


def test_handle_writes_jsonl(tmp_handler):
    event = make_event("log", {"msg": "hello"})
    tmp_handler.handle(event)
    log_path = tmp_handler._base_path / "log.jsonl"
    content = log_path.read_text()
    assert '"msg": "hello"' in content


@patch.object(LocalStorageHandler, "_save_image_to_file", autospec=True)
def test_handle_image_event_uses_helper(mock_save, tmp_handler):
    e = make_event("image", np.zeros((2, 2, 3)))
    tmp_handler.handle(e)
    mock_save.assert_called_once()


def test_handle_invalid_media_warns(tmp_handler, caplog):
    with patch.object(LocalStorageHandler, "_save_video_to_file", return_value=None):
        event = make_event("video", np.zeros((2, 2, 2, 3)))
        tmp_handler.handle(event)
        assert any("Skipping event logging" in m for m in caplog.messages)
