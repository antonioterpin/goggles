import json
from pathlib import Path
import re
import pytest

import goggles
from goggles._core.run import _RunContextManager, _ACTIVE_RUN


def test_run_creates_directory_and_metadata(tmp_path: Path):
    """A new run should create its directory and write metadata.json."""
    cm = _RunContextManager(
        name="test", log_dir=str(tmp_path), user_metadata={"seed": 123}
    )

    with cm as ctx:
        run_dir = Path(ctx.log_dir)
        metadata_path = run_dir / "metadata.json"

        # Directory and metadata file should exist
        assert run_dir.exists() and run_dir.is_dir()
        assert metadata_path.is_file()

        # Metadata content should include required fields
        data = json.loads(metadata_path.read_text())
        for key in ["run_id", "run_name", "created_at", "pid", "host", "python"]:
            assert key in data

        # User metadata must be preserved
        assert data["metadata"] == {"seed": 123}

    # On exit, finished_at should be added
    data = json.loads(metadata_path.read_text())
    assert "finished_at" in data


def test_nested_run_raises(tmp_path: Path):
    """Starting a run inside another run must raise RuntimeError."""
    cm_outer = _RunContextManager(name="outer", log_dir=str(tmp_path), user_metadata={})
    cm_inner = _RunContextManager(name="inner", log_dir=str(tmp_path), user_metadata={})

    with cm_outer:
        with pytest.raises(RuntimeError):
            with cm_inner:
                pass


def test_active_run_context_is_cleared(tmp_path: Path):
    """Ensure _ACTIVE_RUN is set on enter and cleared on exit."""
    cm = _RunContextManager(name="check", log_dir=str(tmp_path), user_metadata={})

    assert _ACTIVE_RUN.get() is None
    with cm as ctx:
        assert _ACTIVE_RUN.get() == ctx
    assert _ACTIVE_RUN.get() is None


def test_public_run_creates_metadata(tmp_path: Path):
    """The public API should mirror _RunContextManager behavior."""
    with goggles.run(name="public", log_dir=str(tmp_path), seed=7) as ctx:
        run_dir = Path(ctx.log_dir)
        metadata_path = run_dir / "metadata.json"

        assert run_dir.exists()
        assert metadata_path.exists()

        data = json.loads(metadata_path.read_text())
        assert data["metadata"]["seed"] == 7
        assert data["run_name"] == "public"

    data = json.loads(metadata_path.read_text())
    assert "finished_at" in data


def test_public_nested_run_raises(tmp_path: Path):
    """Nested run(...) contexts via the public API must raise."""
    with goggles.run(name="outer", log_dir=str(tmp_path)):
        with pytest.raises(RuntimeError):
            with goggles.run(name="inner", log_dir=str(tmp_path)):
                pass


RUN_DIR_PATTERN = re.compile(r"^(?P<name>[\w\-]+)-\d{8}_\d{6}-[0-9a-f]{8}$")
UUID4_PATTERN = re.compile(r"^[0-9a-f]{32}$")
ISO_TS_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+(\+00:00|Z)?$")


@pytest.mark.parametrize("cls_or_api", ["core", "public"])
def test_run_directory_naming_and_metadata_consistency(tmp_path: Path, cls_or_api: str):
    """Ensure both core and public APIs produce consistent directory and metadata layout."""
    if cls_or_api == "core":
        cm = _RunContextManager(
            name="demo", log_dir=str(tmp_path), user_metadata={"seed": 1}
        )
        context_manager = cm
    else:
        context_manager = goggles.run(name="demo", log_dir=str(tmp_path), seed=1)

    with context_manager as ctx:
        run_dir = Path(ctx.log_dir)
        metadata_path = run_dir / "metadata.json"

        # 1. Directory exists and follows the naming convention
        assert run_dir.exists(), "Run directory should exist"
        assert RUN_DIR_PATTERN.match(
            run_dir.name
        ), f"Unexpected dir name: {run_dir.name}"

        # 2. metadata.json exists and includes expected keys
        assert metadata_path.exists(), "metadata.json should exist"
        data = json.loads(metadata_path.read_text())

        required_fields = [
            "run_id",
            "run_name",
            "created_at",
            "pid",
            "host",
            "python",
            "metadata",
        ]
        for key in required_fields:
            assert key in data, f"Missing {key} in metadata.json"

        # 3. Field values have correct formats
        assert UUID4_PATTERN.match(
            data["run_id"]
        ), f"run_id not UUID4: {data['run_id']}"
        assert ISO_TS_PATTERN.match(
            data["created_at"]
        ), f"Invalid ISO timestamp: {data['created_at']}"
        assert isinstance(data["metadata"], dict)
        assert data["metadata"].get("seed") == 1

    # 4. After exiting the context, metadata.json should contain finished_at
    data = json.loads(metadata_path.read_text())
    assert "finished_at" in data, "finished_at should be appended on exit"
    assert ISO_TS_PATTERN.match(
        data["finished_at"]
    ), "finished_at should be valid ISO timestamp"


def test_run_directories_are_unique(tmp_path: Path):
    """Two runs should always produce distinct directories and run_ids."""
    dirs, ids = [], []
    for _ in range(2):
        with goggles.run(name="uniq", log_dir=str(tmp_path)) as ctx:
            dirs.append(Path(ctx.log_dir))
            ids.append(ctx.run_id)
    # Different UUIDs and directories
    assert ids[0] != ids[1]
    assert dirs[0] != dirs[1]
