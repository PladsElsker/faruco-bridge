"""Python interface for ArUco2 Fractal ArUco markers."""

from .client import (
    Detection,
    DetectionResult,
    Detector,
    FarucoError,
    WorkerError,
    WorkerNotFoundError,
    generate_marker,
)
from .protocol import PROTOCOL_VERSION, SUPPORTED_FRACTAL_TYPES, ProtocolError

__all__ = [
    "Detection",
    "DetectionResult",
    "Detector",
    "FarucoError",
    "ProtocolError",
    "PROTOCOL_VERSION",
    "SUPPORTED_FRACTAL_TYPES",
    "WorkerError",
    "WorkerNotFoundError",
    "generate_marker",
]
