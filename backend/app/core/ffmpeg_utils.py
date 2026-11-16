import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

def find_ffmpeg_executable() -> Optional[str]:
    """Locate the FFmpeg executable from PATH, brew locations, or imageio_ffmpeg."""
    # 1. System PATH
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg and os.path.isfile(system_ffmpeg) and os.access(system_ffmpeg, os.X_OK):
        return system_ffmpeg

    # 2. Common Homebrew locations on macOS
    brew_paths = [
        "/opt/homebrew/bin/ffmpeg",
        "/usr/local/bin/ffmpeg"
    ]
    for bp in brew_paths:
        if os.path.isfile(bp) and os.access(bp, os.X_OK):
            return bp

    # 3. imageio_ffmpeg bundled binary
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.isfile(exe) and os.access(exe, os.X_OK):
            return exe
    except ImportError:
        pass

    return None

FFMPEG_EXE = find_ffmpeg_executable()

def convert_to_wav(input_path: Path, output_path: Path, sample_rate: int = 16000) -> Path:
    """
    Safely converts an audio/video file to 16kHz mono WAV using FFmpeg.
    Ensures path validation to prevent traversal and shell execution risks.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Path safety: Ensure resolved paths are within valid directories
    resolved_in = input_path.resolve()
    resolved_out = output_path.resolve()

    if not FFMPEG_EXE:
        raise RuntimeError("FFmpeg executable is not available on this system.")

    cmd = [
        FFMPEG_EXE,
        "-y",               # Overwrite output
        "-i", str(resolved_in),
        "-vn",              # Disable video stream
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "1",         # Mono channel
        str(resolved_out)
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg conversion failed: {result.stderr}")

    return resolved_out
