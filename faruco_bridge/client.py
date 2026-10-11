"""High-level Python client for the persistent ArUco2 FArUco worker."""

from __future__ import annotations

import os
import shutil
import subprocess
from importlib import metadata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .protocol import (
    PROTOCOL_VERSION,
    SUPPORTED_FRACTAL_TYPES,
    ProtocolError,
    encode_command,
    read_json_line,
)


class FarucoError(RuntimeError):
    """Base exception for FArUco bridge errors."""


class WorkerNotFoundError(FarucoError):
    """Raised when the C++ worker executable cannot be located."""


class WorkerError(FarucoError):
    """Raised when the worker reports an error or exits unexpectedly."""


@dataclass(frozen=True)
class Detection:
    """One detected Fractal ArUco marker.

    ``corners`` are the four outer corners in image pixels, ordered as returned by
    ArUco2. ``object_points`` and ``image_points`` are the full matched correspondences
    when requested; object points are scaled into ``marker_size_m`` units.
    """

    id: int
    type: str
    corners: np.ndarray
    object_points: Optional[np.ndarray] = None
    image_points: Optional[np.ndarray] = None
    rvec: Optional[np.ndarray] = None
    tvec: Optional[np.ndarray] = None
    reprojection_error_px: Optional[float] = None

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "Detection":
        pose = payload.get("pose")
        object_points = payload.get("object_points")
        image_points = payload.get("image_points")
        corners_array = np.asarray(payload.get("corners", []), dtype=np.float32)
        corners_array = corners_array.reshape((-1, 2)) if corners_array.size else np.empty((0, 2), dtype=np.float32)
        object_array = None if object_points is None else np.asarray(object_points, dtype=np.float32)
        if object_array is not None:
            object_array = object_array.reshape((-1, 3)) if object_array.size else np.empty((0, 3), dtype=np.float32)
        image_array = None if image_points is None else np.asarray(image_points, dtype=np.float32)
        if image_array is not None:
            image_array = image_array.reshape((-1, 2)) if image_array.size else np.empty((0, 2), dtype=np.float32)
        return cls(
            id=int(payload["id"]),
            type=str(payload["type"]),
            corners=corners_array,
            object_points=object_array,
            image_points=image_array,
            rvec=(None if not pose else np.asarray(pose["rvec"], dtype=np.float64).reshape((3, 1))),
            tvec=(None if not pose else np.asarray(pose["tvec"], dtype=np.float64).reshape((3, 1))),
            reprojection_error_px=(None if not pose else float(pose["reprojection_error_px"])),
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return {
            "id": self.id,
            "type": self.type,
            "corners": self.corners.tolist(),
            "object_points": None if self.object_points is None else self.object_points.tolist(),
            "image_points": None if self.image_points is None else self.image_points.tolist(),
            "pose": None if self.rvec is None or self.tvec is None else {
                "rvec": self.rvec.reshape(-1).tolist(),
                "tvec": self.tvec.reshape(-1).tolist(),
                "reprojection_error_px": self.reprojection_error_px,
            },
        }


@dataclass(frozen=True)
class DetectionResult:
    """Result from one frame submitted to :class:`Detector`."""

    request_id: int
    elapsed_ms: float
    markers: tuple[Detection, ...]

    @property
    def marker_count(self) -> int:
        return len(self.markers)

    def as_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "elapsed_ms": self.elapsed_ms,
            "marker_count": self.marker_count,
            "markers": [marker.as_dict() for marker in self.markers],
        }

    def annotate(self, image: np.ndarray, draw_correspondences: bool = True) -> np.ndarray:
        """Return a copy of ``image`` with detections drawn using Python OpenCV.

        OpenCV is imported lazily, so it is not required just to use the protocol
        client. The input image is not mutated.
        """
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - depends on optional local OpenCV
            raise FarucoError("DetectionResult.annotate() requires Python OpenCV (cv2)") from exc

        output = np.ascontiguousarray(image).copy()
        if output.ndim == 3 and output.shape[2] == 1:
            output = output[:, :, 0]
        if output.ndim == 2:
            output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
        for marker in self.markers:
            corners = np.rint(marker.corners).astype(np.int32).reshape((-1, 1, 2))
            if len(corners) >= 4:
                cv2.polylines(output, [corners[:4]], True, (0, 255, 0), 2, cv2.LINE_AA)
                x, y = np.rint(marker.corners[0]).astype(int)
                cv2.circle(output, (int(x), int(y)), 4, (0, 0, 255), -1)
                center = np.mean(marker.corners[:4], axis=0).astype(int)
                cv2.putText(output, str(marker.id), tuple(center), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2, cv2.LINE_AA)
            if draw_correspondences and marker.image_points is not None:
                for point in np.rint(marker.image_points).astype(np.int32):
                    cv2.circle(output, tuple(point), 2, (255, 160, 0), -1)
        return output


def _resolve_worker(worker_path: Optional[os.PathLike[str] | str] = None) -> str:
    candidates: list[Path] = []
    if worker_path is not None:
        candidates.append(Path(worker_path).expanduser())
    env_worker = os.environ.get("FARUCO_WORKER")
    if env_worker:
        candidates.append(Path(env_worker).expanduser())

    # Normal wheel installs bundle the platform-built executable beside this
    # module. An explicit path or FARUCO_WORKER takes precedence for overrides.
    package_dir = Path(__file__).resolve().parent
    bundled_name = "faruco_worker.exe" if os.name == "nt" else "faruco_worker"
    candidates.append(package_dir / "bin" / bundled_name)

    # Keep support for source checkouts and manually compiled development builds.
    package_root = package_dir.parent
    candidates.extend([
        package_root / "build" / "faruco_worker",
        package_root / "build" / "bin" / "faruco_worker",
    ])

    # Running an example directly from a source checkout puts that checkout
    # first on sys.path. Its faruco_bridge package (and possibly a local
    # faruco_bridge.egg-info directory) shadows the installed Python package,
    # even though pip has installed the native worker in site-packages. Search
    # every matching distribution rather than metadata.distribution(...), which
    # may resolve to the checkout's local egg-info first.
    relative_worker = Path("faruco_bridge") / "bin" / bundled_name
    try:
        distributions = metadata.distributions(name="faruco-bridge")
        for distribution in distributions:
            try:
                candidates.append(Path(distribution.locate_file(relative_worker)))
            except (AttributeError, OSError, TypeError, ValueError):
                # A malformed/incomplete metadata entry should not prevent
                # another installed distribution from supplying the worker.
                continue
    except (OSError, TypeError, ValueError):
        # Worker discovery should still support source builds and PATH below
        # if distribution metadata cannot be enumerated.
        pass

    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate.resolve())
        if worker_path is not None and candidate == Path(worker_path).expanduser():
            # Explicit paths should fail loudly rather than silently falling back.
            raise WorkerNotFoundError(f"Worker executable is not executable: {candidate}")

    found = shutil.which("faruco_worker")
    if found:
        return found
    raise WorkerNotFoundError(
        "The FArUco native component is missing from this installation. "
        "Reinstall faruco-bridge; if the problem persists, the installation may be incomplete."
    )


def _flatten_numeric_array(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != shape:
        raise ValueError(f"{label} must have shape {shape}; received {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} must contain only finite numbers")
    return arr


def generate_marker(
    fractal_type: str = "FRACTAL_3L_6",
    output: os.PathLike[str] | str = "fractal_marker.png",
    bit_size: int = 20,
    worker_path: Optional[os.PathLike[str] | str] = None,
) -> Path:
    """Generate a Fractal ArUco PNG using ArUco2's native generator.

    ``fractal_type`` selects one of ArUco2's predefined nesting configurations.
    ``bit_size`` is the pixel size of a bit cell in the innermost marker, not the
    overall image size or physical marker dimension. The current upstream Fractal
    ArUco implementation provides one built-in ID, so this wrapper intentionally
    does not expose an ID parameter. The function returns the output path.
    """
    if fractal_type not in SUPPORTED_FRACTAL_TYPES:
        raise ValueError(f"fractal_type must be one of {SUPPORTED_FRACTAL_TYPES}")
    if int(bit_size) < 2:
        raise ValueError("bit_size must be >= 2")
    executable = _resolve_worker(worker_path)
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        executable, "generate", "--type", fractal_type,
        "--bit-size", str(int(bit_size)), "--output", str(output_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise WorkerError(
            f"Marker generation failed with exit code {completed.returncode}: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    if not output_path.is_file():
        raise WorkerError("Worker returned success but did not create the marker image")
    return output_path


class Detector:
    """Persistent client for ArUco2's Fractal ArUco detector.

    The native worker is started lazily on the first valid call to ``detect()``.

    Parameters
    ----------
    fractal_type:
        One of ``FRACTAL_2L_6`` through ``FRACTAL_5L_6``.
    marker_size_m:
        Physical side length of the outer marker, in metres. This scales the 3-D
        correspondences returned for pose estimation.
    camera_matrix:
        Optional 3x3 camera intrinsic matrix. If provided, the C++ worker also
        estimates ``rvec``/``tvec`` and reprojection error for each detection.
    dist_coeffs:
        Optional OpenCV distortion coefficients. Defaults to five zeros when a
        camera matrix is supplied. Valid lengths: 4, 5, 8, 12, or 14.
    worker_path:
        Optional path to ``faruco_worker``. Otherwise FARUCO_WORKER/PATH are used.
    include_points:
        Include object/image correspondences in each result. Set False to reduce
        result JSON size when only outer corners or the worker's pose are needed.
    """

    def __init__(
        self,
        fractal_type: str = "FRACTAL_3L_6",
        marker_size_m: float = 0.10,
        camera_matrix: Optional[Any] = None,
        dist_coeffs: Optional[Any] = None,
        worker_path: Optional[os.PathLike[str] | str] = None,
        include_points: bool = True,
    ) -> None:
        if fractal_type not in SUPPORTED_FRACTAL_TYPES:
            raise ValueError(f"fractal_type must be one of {SUPPORTED_FRACTAL_TYPES}")
        if not np.isfinite(marker_size_m) or marker_size_m <= 0:
            raise ValueError("marker_size_m must be a finite positive number")
        if dist_coeffs is not None and camera_matrix is None:
            raise ValueError("dist_coeffs requires camera_matrix")

        self.fractal_type = fractal_type
        self.marker_size_m = float(marker_size_m)
        self.include_points = bool(include_points)
        self._request_id = 0
        self._closed = False
        self._worker_path = worker_path
        import threading
        self._lock = threading.Lock()
        self._process: Optional[subprocess.Popen[bytes]] = None
        self._command_args = [
            "worker", "--type", self.fractal_type,
            "--marker-size", format(self.marker_size_m, ".17g"),
        ]

        if camera_matrix is not None:
            camera = _flatten_numeric_array(camera_matrix, (3, 3), "camera_matrix")
            if camera[0, 0] <= 0 or camera[1, 1] <= 0:
                raise ValueError("camera_matrix fx and fy must be positive")
            if dist_coeffs is None:
                dist = np.zeros((5,), dtype=np.float64)
            else:
                dist = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)
                if dist.size not in (4, 5, 8, 12, 14):
                    raise ValueError("dist_coeffs must contain 4, 5, 8, 12, or 14 values")
                if not np.all(np.isfinite(dist)):
                    raise ValueError("dist_coeffs must contain only finite numbers")
            self._command_args.extend([
                "--camera-matrix", ",".join(format(float(x), ".17g") for x in camera.reshape(-1)),
                "--dist-coeffs", ",".join(format(float(x), ".17g") for x in dist),
            ])

    def _ensure_started(self) -> None:
        """Start the persistent worker on the first valid detection request."""
        with self._lock:
            if self._closed:
                raise WorkerError("FArUco worker is closed")

            if self._process is not None:
                if self._process.poll() is not None:
                    raise WorkerError("FArUco worker is closed or has exited")
                return

            executable = _resolve_worker(self._worker_path)
            command = [executable, *self._command_args]
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=None,  # Do not pipe diagnostics: avoids a full stderr pipe blocking the worker.
                bufsize=-1,
            )
            self._process = process
            try:
                assert process.stdout is not None
                ready = read_json_line(process.stdout)
                if (not ready.get("ok") or ready.get("event") != "ready" or
                        ready.get("protocol_version") != PROTOCOL_VERSION):
                    raise ProtocolError(f"Unexpected worker startup response: {ready}")
            except Exception:
                self._terminate_worker()
                raise

    def detect(self, image: Any, include_points: Optional[bool] = None) -> DetectionResult:
        """Detect fractal markers in a uint8 grayscale or BGR NumPy image.

        Frames are sent as contiguous raw bytes through the worker's persistent pipe.
        The method is synchronous: it returns when the worker completes that frame.
        """
        if self._closed:
            raise WorkerError("FArUco worker is closed")
        frame = np.asarray(image)
        if frame.dtype != np.uint8:
            raise ValueError(f"image must have dtype uint8, got {frame.dtype}")
        if frame.ndim == 2:
            height, width = frame.shape
            channels = 1
        elif frame.ndim == 3 and frame.shape[2] in (1, 3):
            height, width, channels = frame.shape
            if channels == 1:
                frame = frame[:, :, 0]
        else:
            raise ValueError("image must be HxW grayscale or HxWx3 BGR uint8")
        if height <= 0 or width <= 0:
            raise ValueError("image dimensions must be positive")
        frame = np.ascontiguousarray(frame)
        payload = memoryview(frame).cast("B")
        if len(payload) > 256 * 1024 * 1024:
            raise ValueError("image exceeds the 256 MiB worker protocol limit")
        use_points = self.include_points if include_points is None else bool(include_points)
        self._ensure_started()

        with self._lock:
            process = self._process
            if self._closed or process is None or process.poll() is not None:
                raise WorkerError("FArUco worker is closed or has exited")
            self._request_id += 1
            request_id = self._request_id
            header = f"DETECT {request_id} {width} {height} {channels} {len(payload)} {1 if use_points else 0}\n".encode("ascii")
            try:
                assert process.stdin is not None and process.stdout is not None
                process.stdin.write(header)
                process.stdin.write(payload)
                process.stdin.flush()
                response = read_json_line(process.stdout)
            except (BrokenPipeError, OSError, ProtocolError) as exc:
                raise WorkerError(f"Communication with FArUco worker failed: {exc}") from exc

        if not response.get("ok"):
            raise WorkerError(response.get("error", f"Worker failed for request {request_id}"))
        if int(response.get("request_id", -1)) != request_id:
            raise ProtocolError(
                f"Mismatched response ID: expected {request_id}, got {response.get('request_id')}"
            )
        try:
            detections = tuple(Detection.from_payload(item) for item in response.get("markers", []))
            return DetectionResult(
                request_id=request_id,
                elapsed_ms=float(response.get("elapsed_ms", 0.0)),
                markers=detections,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError(f"Malformed detection payload: {exc}") from exc

    def ping(self) -> bool:
        """Check worker health without starting an uninitialized worker."""
        process = self._process
        if self._closed or process is None or process.poll() is not None:
            return False
        with self._lock:
            process = self._process
            if self._closed or process is None or process.poll() is not None:
                return False
            assert process.stdin is not None and process.stdout is not None
            process.stdin.write(encode_command("PING"))
            process.stdin.flush()
            response = read_json_line(process.stdout)
        return bool(response.get("ok") and response.get("event") == "pong")

    def close(self) -> None:
        """Shut down the worker cleanly; safe to call more than once."""
        lock = getattr(self, "_lock", None)
        if lock is None:
            self._closed = True
            self._terminate_worker()
            return
        with lock:
            if self._closed:
                return
            self._closed = True
            process = self._process
            if process is None:
                return
            if process.poll() is None:
                try:
                    assert process.stdin is not None
                    process.stdin.write(encode_command("QUIT"))
                    process.stdin.flush()
                    if process.stdout is not None:
                        # Consume the acknowledgement if the process returns one.
                        read_json_line(process.stdout)
                    process.wait(timeout=3.0)
                except (BrokenPipeError, OSError, ProtocolError, subprocess.TimeoutExpired):
                    self._terminate_worker()
            else:
                self._terminate_worker()
            for stream in (process.stdin, process.stdout):
                try:
                    if stream:
                        stream.close()
                except OSError:
                    pass

    def _terminate_worker(self) -> None:
        process = getattr(self, "_process", None)
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2.0)

    def __enter__(self) -> "Detector":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - interpreter shutdown behavior
        try:
            self.close()
        except Exception:
            pass
