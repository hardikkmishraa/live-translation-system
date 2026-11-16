import logging
import sys
from typing import Any, Dict

def setup_logger(name: str = "voiceaffect") -> logging.Logger:
    """
    Configures a structured, privacy-preserving logger.
    Ensures that raw binary audio, base64 payloads, and user transcript contents
    are never leaked into server log files.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger

logger = setup_logger()

def sanitize_log_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Strips sensitive audio buffers and raw transcript text from log outputs."""
    sanitized = {}
    sensitive_keys = {"audio_bytes", "audio_base64", "source_transcript", "translated_text", "text"}
    for k, v in data.items():
        if k in sensitive_keys:
            if isinstance(v, (str, bytes)):
                sanitized[k] = f"<{len(v)} bytes/chars omitted for privacy>"
            else:
                sanitized[k] = "<omitted>"
        elif isinstance(v, dict):
            sanitized[k] = sanitize_log_data(v)
        else:
            sanitized[k] = v
    return sanitized
