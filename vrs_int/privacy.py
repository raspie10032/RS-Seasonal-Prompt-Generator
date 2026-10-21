"""Keep generated local data out of ordinary Git staging.

This is not a network sandbox or an override for files already tracked by Git.
Never modify the user's original image directories or their permissions.
"""
import os
from pathlib import Path


IGNORE = b"\n# VRS local data: never include in a source-code repository.\n*\n"


def private_directory(directory):
    """Protect an output directory before writing metadata, preserving old rules."""
    directory = Path(directory)
    if (directory / ".git").exists():
        raise ValueError("Choose a dedicated VRS data directory, not a Git repository root")
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = directory / ".gitignore"
    if path.is_symlink():
        raise ValueError("VRS data .gitignore must be a regular local file")
    if path.exists() and path.read_bytes().endswith(IGNORE):
        return
    # Append instead of replacing user rules. O_NOFOLLOW also closes the symlink
    # race on platforms supporting it; Windows still rejects existing links.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(fd, "ab") as stream:
        stream.write(IGNORE)
        stream.flush()
        os.fsync(stream.fileno())
