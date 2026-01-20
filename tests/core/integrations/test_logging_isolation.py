from goggles._core.integrations.console import ConsoleHandler
from goggles._core.integrations.storage import LocalStorageHandler
from goggles._core.integrations.wandb import WandBHandler


def test_console_handler_isolation():
    """Verify that ConsoleHandler logger has propagation disabled."""
    handler = ConsoleHandler(name="test.console")
    handler.open()
    try:
        assert handler._logger.propagate is False
    finally:
        handler.close()


def test_storage_handler_isolation(tmp_path):
    """Verify that LocalStorageHandler logger has propagation disabled."""
    handler = LocalStorageHandler(path=tmp_path, name="test.jsonl")
    handler.open()
    try:
        assert handler._logger.propagate is False
    finally:
        handler.close()


def test_wandb_handler_isolation():
    """Verify that WandBHandler logger has propagation disabled."""
    # WandBHandler sets propagate = False in __init__
    handler = WandBHandler()
    assert handler._logger.propagate is False
