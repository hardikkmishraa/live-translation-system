import os
import uuid
from pathlib import Path
from typing import Set
from fastapi import HTTPException, status

ALLOWED_EXTENSIONS: Set[str] = {
    ".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm", ".mp4"
}

def generate_safe_filename(prefix: str, extension: str) -> str:
    """Generates an unguessable UUID-based safe internal filename."""
    clean_ext = extension.lower().strip()
    if not clean_ext.startswith("."):
        clean_ext = f".{clean_ext}"
    if clean_ext not in ALLOWED_EXTENSIONS:
        clean_ext = ".wav"
    safe_id = uuid.uuid4().hex
    return f"{prefix}_{safe_id}{clean_ext}"

def validate_safe_path(target_path: Path, base_dir: Path) -> Path:
    """
    Prevents path traversal attacks by validating that the resolved path
    strictly resides within the intended base directory.
    """
    resolved_base = base_dir.resolve()
    resolved_target = target_path.resolve()

    try:
        # Check if resolved_target is a child of resolved_base
        resolved_target.relative_to(resolved_base)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: Invalid file path or path traversal attempt detected."
        )
    return resolved_target

def safe_unlink(path: Path) -> bool:
    """Safely removes a file, ignoring missing file errors."""
    try:
        if path.exists() and path.is_file():
            path.unlink(missing_ok=True)
            return True
    except Exception:
        pass
    return False
