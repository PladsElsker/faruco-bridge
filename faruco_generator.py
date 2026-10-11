"""Generate ArUco2 Fractal ArUco markers without using Detector directly.

The marker-generation implementation lives in ArUco2's native code. This module
provides a small Python API and CLI that invokes the bundled ``faruco_worker``
only for marker generation; it does not import ``faruco_bridge`` or its detector
client, and it does not require NumPy or OpenCV's Python bindings.

Python API::

    from faruco_generator import generate_marker

    path = generate_marker(
        fractal_type="FRACTAL_3L_6",
        output="fractal_3l_6.png",
        bit_size=20,
    )

CLI::

    python -m faruco_generator --type FRACTAL_3L_6 --bit-size 20 --output marker.png
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Sequence

SUPPORTED_FRACTAL_TYPES = (
    "FRACTAL_2L_6",
    "FRACTAL_3L_6",
    "FRACTAL_4L_6",
    "FRACTAL_5L_6",
)

_WORKER_FILENAME = "faruco_worker.exe" if os.name == "nt" else "faruco_worker"
_DISTRIBUTION_NAME = "faruco-bridge"


class FarucoGeneratorError(RuntimeError):
    """Base exception raised by the marker generator."""


class GeneratorNotFoundError(FarucoGeneratorError):
    """Raised when the bundled native generation executable cannot be found."""


def _is_executable(path: Path) -> bool:
    return path.is_file() and (os.name == "nt" or os.access(path, os.X_OK))


def _resolve_worker(worker_path: os.PathLike[str] | str | None = None) -> Path:
    """Locate the worker bundled in the wheel or a local development checkout.

    ``worker_path`` and ``FARUCO_WORKER`` are explicit overrides. Installed
    distribution metadata is searched in full because an example run from the
    source checkout can put local ``faruco_bridge.egg-info`` ahead of the wheel's
    metadata, even when the compiled worker is already installed in site-packages.
    """
    override: Path | None = None
    if worker_path is not None:
        override = Path(worker_path).expanduser()
    elif os.environ.get("FARUCO_WORKER"):
        override = Path(os.environ["FARUCO_WORKER"]).expanduser()

    if override is not None:
        if _is_executable(override):
            return override.resolve()
        raise GeneratorNotFoundError(
            f"The configured FArUco worker is missing or not executable: {override}"
        )

    candidates: list[Path] = []
    module_dir = Path(__file__).resolve().parent

    # Support either a top-level faruco_generator.py or placement inside the
    # faruco_bridge package. One path typically exists; checking both keeps this
    # script usable wherever it is added to the project.
    candidates.extend(
        (
            module_dir / "faruco_bridge" / "bin" / _WORKER_FILENAME,
            module_dir / "bin" / _WORKER_FILENAME,
            module_dir / "build" / _WORKER_FILENAME,
            module_dir / "build" / "bin" / _WORKER_FILENAME,
        )
    )

    relative_worker = Path("faruco_bridge") / "bin" / _WORKER_FILENAME
    try:
        distributions = metadata.distributions(name=_DISTRIBUTION_NAME)
        for distribution in distributions:
            try:
                candidates.append(Path(distribution.locate_file(relative_worker)))
            except (AttributeError, OSError, TypeError, ValueError):
                # Ignore incomplete metadata and continue to other installations.
                continue
    except (OSError, TypeError, ValueError):
        # Source-tree and PATH discovery should still work if metadata is broken.
        pass

    for candidate in candidates:
        if _is_executable(candidate):
            return candidate.resolve()

    path_worker = shutil.which(_WORKER_FILENAME.removesuffix(".exe"))
    if path_worker:
        return Path(path_worker).resolve()

    raise GeneratorNotFoundError(
        "The FArUco marker generator is not available in this installation. "
        "Reinstall faruco-bridge so its bundled native component is installed, "
        "or set FARUCO_WORKER to an existing worker executable."
    )


def generate_marker(
    fractal_type: str = "FRACTAL_3L_6",
    output: os.PathLike[str] | str = "fractal_marker.png",
    bit_size: int = 20,
    *,
    worker_path: os.PathLike[str] | str | None = None,
) -> Path:
    """Generate a grayscale PNG using ArUco2's predefined Fractal ArUco marker.

    Args:
        fractal_type: One of ``FRACTAL_2L_6``, ``FRACTAL_3L_6``,
            ``FRACTAL_4L_6``, or ``FRACTAL_5L_6``.
        output: Destination path for the PNG. Parent directories are created.
        bit_size: Pixel size of a bit cell in the innermost marker; must be >= 2.
        worker_path: Optional explicit path to the bundled native worker. Normally
            it is discovered automatically and this argument is not needed.

    Returns:
        The absolute path to the generated image.

    Note:
        The current ArUco2 implementation has one built-in fractal marker ID and
        does not expose custom IDs. The selected fractal type controls a predefined
        nesting layout; levels and relative geometry cannot be set independently.
    """
    if fractal_type not in SUPPORTED_FRACTAL_TYPES:
        raise ValueError(
            f"fractal_type must be one of {', '.join(SUPPORTED_FRACTAL_TYPES)}; "
            f"got {fractal_type!r}"
        )
    if isinstance(bit_size, bool):
        raise ValueError("bit_size must be an integer >= 2")
    try:
        size = int(bit_size)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("bit_size must be an integer >= 2") from exc
    if size < 2 or str(bit_size).strip() not in (str(size), f"+{size}"):
        raise ValueError("bit_size must be an integer >= 2")

    executable = _resolve_worker(worker_path)
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        str(executable),
        "generate",
        "--type",
        fractal_type,
        "--bit-size",
        str(size),
        "--output",
        str(output_path),
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise FarucoGeneratorError(f"Could not start the FArUco generator: {exc}") from exc

    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no diagnostic output"
        raise FarucoGeneratorError(
            f"Marker generation failed (exit code {completed.returncode}): {detail}"
        )
    if not output_path.is_file():
        raise FarucoGeneratorError(
            "The native generator reported success but did not create the output image."
        )
    return output_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate an ArUco2 Fractal ArUco marker PNG."
    )
    parser.add_argument(
        "--type",
        dest="fractal_type",
        choices=SUPPORTED_FRACTAL_TYPES,
        default="FRACTAL_3L_6",
        help="predefined nesting configuration (default: FRACTAL_3L_6)",
    )
    parser.add_argument(
        "--bit-size",
        type=int,
        default=20,
        help="pixel size of one bit cell in the innermost marker (default: 20; min: 2)",
    )
    parser.add_argument(
        "--output",
        default="fractal_marker.png",
        help="destination PNG path (default: fractal_marker.png)",
    )
    parser.add_argument(
        "--worker",
        dest="worker_path",
        help=argparse.SUPPRESS,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        result = generate_marker(
            fractal_type=args.fractal_type,
            output=args.output,
            bit_size=args.bit_size,
            worker_path=args.worker_path,
        )
    except (FarucoGeneratorError, ValueError) as exc:
        print(f"faruco_generator: error: {exc}", file=sys.stderr)
        return 2
    print(result)
    return 0


__all__ = [
    "FarucoGeneratorError",
    "GeneratorNotFoundError",
    "SUPPORTED_FRACTAL_TYPES",
    "generate_marker",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
