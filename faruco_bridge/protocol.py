"""Protocol constants and small helpers for the faruco_worker pipe protocol.

Protocol v1 is deliberately simple: UTF-8 command headers are newline-terminated,
image payloads immediately follow as packed uint8 bytes, and each response is one
JSON object terminated by a newline. Standard output is reserved for protocol data;
worker diagnostics go to standard error.
"""

from __future__ import annotations

import json
from typing import Any, BinaryIO

PROTOCOL_VERSION = 1
SUPPORTED_FRACTAL_TYPES = (
    "FRACTAL_2L_6",
    "FRACTAL_3L_6",
    "FRACTAL_4L_6",
    "FRACTAL_5L_6",
)


class ProtocolError(RuntimeError):
    """Raised when the worker violates the pipe protocol."""


def read_json_line(stream: BinaryIO) -> dict[str, Any]:
    """Read and decode one worker response line."""
    line = stream.readline()
    if not line:
        raise ProtocolError("FArUco worker closed stdout unexpectedly")
    try:
        payload = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        excerpt = repr(line[:200])
        raise ProtocolError(f"Invalid worker response {excerpt}") from exc
    if not isinstance(payload, dict):
        raise ProtocolError("Worker response must be a JSON object")
    return payload


def encode_command(command: str) -> bytes:
    """Encode a newline-delimited ASCII command header."""
    if "\n" in command or "\r" in command:
        raise ValueError("Protocol commands cannot contain newlines")
    return command.encode("ascii") + b"\n"
