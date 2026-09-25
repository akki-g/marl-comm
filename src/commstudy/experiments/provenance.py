"""Content fingerprints for executable research sources."""

import hashlib
from pathlib import Path


def source_fingerprint(repo_root: Path | None = None) -> str:
    """Hash sorted package paths and bytes, excluding docs, caches, and runs."""
    source = (
        Path(__file__).resolve().parents[1]
        if repo_root is None
        else (Path(repo_root) / "src" / "commstudy")
    )
    digest = hashlib.sha256()
    for path in sorted(source.rglob("*.py")):
        digest.update(path.relative_to(source).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
