from importlib import import_module
from typing import Any


def import_from_path(path: str) -> Any:
    """
    Import an object from a fully-qualified Python path.

    Example:
        "commstudy.communication.identity.IdentityComm"
    """
    module_path, object_name = path.rsplit(".", 1)

    module = import_module(module_path)

    try:
        return getattr(module, object_name)
    except AttributeError as exc:
        raise ImportError(
            f"'{object_name}' does not exist in module '{module_path}'"
        ) from exc