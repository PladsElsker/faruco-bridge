#!/usr/bin/env python3
"""Run live FArUco detection on a Wayland desktop capture using PipeWire.

The xdg-desktop-portal prompts you to select a screen or window to share; no
webcam is opened. The first received frame is saved as a screenshot unless
--skip-initial-screenshot is passed. The prediction is shown in a separate
live OpenCV window.

For stable physical-camera pose estimates, screen capture is not a substitute
for a calibrated camera: rvec/tvec describe a *virtual* pinhole view of the
captured pixels. Supply calibration only if it matches this source and its
resolution; otherwise an approximate virtual camera matrix is used.

When sharing a full monitor on a single-monitor desktop, use --region to crop
the source content and keep the prediction window outside that crop. The portal
API does not expose monitor placement, so --preview-position can be used to
place the prediction window explicitly. Otherwise its output may appear in the
next captured frame and cause visual feedback.

Controls
--------
q / Esc : quit
s       : save the current prediction window
m       : save the current captured source frame
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Allow this example to run directly from a bridge checkout without installing
# the Python package or setting PYTHONPATH.
BRIDGE_ROOT = Path(__file__).resolve().parents[1]
if str(BRIDGE_ROOT) not in sys.path:
    sys.path.insert(0, str(BRIDGE_ROOT))

import faruco_bridge as faruco


# Relative square side lengths in the built-in fractal layouts. Level 1 is the
# outer marker, and the remaining squares are nested inside it.
LEVEL_SIDE_FRACTIONS = {
    "FRACTAL_2L_6": (1.0, 1.0 / 3.0),
    "FRACTAL_3L_6": (1.0, 3.0 / 7.0, 1.0 / 7.0),
    "FRACTAL_4L_6": (1.0, 7.0 / 15.0, 3.0 / 15.0, 1.0 / 15.0),
    "FRACTAL_5L_6": (1.0, 5.0 / 13.0, 7.0 / 39.0, 3.0 / 39.0, 1.0 / 39.0),
}
LEVEL_COLORS = (
    (0, 255, 255),   # outer level: yellow
    (255, 80, 255),  # magenta
    (0, 200, 0),     # green
    (255, 160, 0),   # blue/orange
    (255, 255, 0),   # cyan
)


def load_calibration(path: Optional[str]) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if not path:
        return None, None
    with np.load(path) as calibration:
        if "camera_matrix" not in calibration:
            raise SystemExit("Calibration NPZ must contain a 'camera_matrix' array")
        camera_matrix = np.asarray(calibration["camera_matrix"], dtype=np.float64)
        dist_coeffs = (
            np.asarray(calibration["dist_coeffs"], dtype=np.float64).reshape(-1)
            if "dist_coeffs" in calibration else np.zeros(5, dtype=np.float64)
        )
    if camera_matrix.shape != (3, 3):
        raise SystemExit("camera_matrix in calibration NPZ must have shape (3, 3)")
    if dist_coeffs.size not in (4, 5, 8, 12, 14):
        raise SystemExit("dist_coeffs must contain 4, 5, 8, 12, or 14 values")
    return camera_matrix, dist_coeffs


def approximate_camera_matrix(width: int, height: int, horizontal_fov_deg: float) -> np.ndarray:
    """Create virtual pinhole intrinsics from capture size and assumed horizontal FOV."""
    if not 1.0 < horizontal_fov_deg < 179.0:
        raise ValueError("horizontal FOV must be between 1 and 179 degrees")
    focal_px = 0.5 * width / np.tan(np.deg2rad(horizontal_fov_deg) * 0.5)
    return np.array(
        [[focal_px, 0.0, (width - 1.0) / 2.0],
         [0.0, focal_px, (height - 1.0) / 2.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def crop_bgr(frame: np.ndarray, region: Optional[list[int]]) -> np.ndarray:
    """Optionally crop a BGRA/BGR-converted frame using source-relative X/Y/W/H."""
    if not region:
        return frame
    x, y, width, height = region
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise SystemExit("--region requires non-negative X/Y and positive WIDTH/HEIGHT")
    source_h, source_w = frame.shape[:2]
    if x + width > source_w or y + height > source_h:
        raise SystemExit(
            f"--region {x} {y} {width} {height} exceeds selected source size "
            f"{source_w}x{source_h}"
        )
    return np.ascontiguousarray(frame[y:y + height, x:x + width])


def pipewire_frame_to_bgr(frame: np.ndarray) -> np.ndarray:
    """Validate a PipeWire frame (BGRA) and return a contiguous BGR image."""
    pixels = np.asarray(frame)
    if pixels.ndim != 3 or pixels.shape[2] not in (3, 4):
        raise RuntimeError(f"Unexpected PipeWire frame shape: {pixels.shape}")
    if pixels.shape[2] == 4:
        return cv2.cvtColor(pixels, cv2.COLOR_BGRA2BGR)
    return np.ascontiguousarray(pixels)


def wait_for_pipewire_frame(stream, timeout_s: float = 15.0) -> np.ndarray:
    """Wait for the first valid image, surfacing portal/stream errors clearly."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if stream.window_invalid:
            detail = getattr(stream, "error", None)
            raise RuntimeError(detail or "The shared screen/window stream ended")
        pixels = stream.get_frame()
        if pixels is not None:
            return pipewire_frame_to_bgr(pixels)
        time.sleep(0.01)
    raise TimeoutError(
        "Timed out waiting for the first PipeWire frame. Check the KDE screen-share "
        "permission dialog and try selecting the screen/window again."
    )


def save_image(path_text: str, image: np.ndarray, description: str) -> None:
    path = Path(path_text).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        print(f"Could not save {description}: {path}", flush=True)
    else:
        print(f"{description} saved: {path}", flush=True)


def cube_vertices(side_length_m: float, depth_ratio: float) -> np.ndarray:
    half = side_length_m * 0.5
    depth = side_length_m * depth_ratio
    return np.array(
        [
            [-half, -half, 0.0], [half, -half, 0.0],
            [half, half, 0.0], [-half, half, 0.0],
            [-half, -half, depth], [half, -half, depth],
            [half, half, depth], [-half, half, depth],
        ],
        dtype=np.float32,
    )


def draw_wireframe_cube(
    image: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    side_length_m: float,
    depth_ratio: float,
    color: tuple[int, int, int],
    label: str,
) -> None:
    vertices = cube_vertices(side_length_m, depth_ratio)
    projected, _ = cv2.projectPoints(vertices, rvec, tvec, camera_matrix, dist_coeffs)
    points = np.rint(projected.reshape(-1, 2)).astype(np.int32)
    edges = (
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    )
    thickness = max(1, int(round(image.shape[1] / 900.0)))
    for start, end in edges:
        cv2.line(image, tuple(points[start]), tuple(points[end]), color, thickness, cv2.LINE_AA)
    anchor = tuple(points[4])
    cv2.putText(
        image, label, (anchor[0] + 4, anchor[1] - 4),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, max(1, thickness), cv2.LINE_AA,
    )


def draw_all_level_cubes(
    image: np.ndarray,
    result: faruco.DetectionResult,
    fractal_type: str,
    marker_size_m: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    cube_depth_ratio: float,
) -> None:
    fractions = LEVEL_SIDE_FRACTIONS[fractal_type]
    for marker in result.markers:
        if marker.rvec is None or marker.tvec is None:
            if len(marker.corners):
                center = tuple(np.rint(np.mean(marker.corners, axis=0)).astype(int))
            else:
                center = (12, 50)
            cv2.putText(
                image, f"ID {marker.id}: pose unavailable", center,
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA,
            )
            continue

        # The nested marker squares are concentric. Reuse this detected pose and
        # scale the cube footprint to each level's side length.
        for level, fraction in enumerate(fractions, start=1):
            draw_wireframe_cube(
                image, marker.rvec, marker.tvec, camera_matrix, dist_coeffs,
                marker_size_m * fraction, cube_depth_ratio,
                LEVEL_COLORS[(level - 1) % len(LEVEL_COLORS)], f"ID {marker.id} L{level}",
            )

        t = np.asarray(marker.tvec).reshape(-1)
        pose_text = f"ID {marker.id} t=[{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}] m"
        if marker.reprojection_error_px is not None:
            pose_text += f" | RMS {marker.reprojection_error_px:.1f}px"
        center = tuple(np.rint(np.mean(marker.corners, axis=0)).astype(int))
        cv2.putText(
            image, pose_text, (max(6, center[0] - 140), max(20, center[1])),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA,
        )


def choose_preview_placement(
    requested_position: Optional[list[int]],
    region: Optional[list[int]],
    frame_shape: tuple[int, ...],
    source_shape: tuple[int, ...],
) -> tuple[tuple[int, int], tuple[int, int], str]:
    """Choose output location/size using source dimensions (portal omits monitor origin)."""
    capture_w, capture_h = frame_shape[1], frame_shape[0]
    max_w, max_h = min(1100, capture_w), min(800, capture_h)
    if requested_position:
        target = (requested_position[0], requested_position[1])
        note = "using requested --preview-position"
    elif region:
        # For the usual primary-monitor origin, this puts preview to the right of the crop.
        # Wayland compositors may ignore app-requested window positions; manual dragging may
        # still be required. Users with multi-monitor/non-zero-origin layouts can override.
        x, y, width, height = region
        target = (x + width + 24, y)
        free_right = source_shape[1] - (x + width + 24)
        free_bottom = source_shape[0] - y
        if free_right >= 320:
            max_w, max_h = min(max_w, free_right), min(max_h, free_bottom)
            note = "placing prediction in the free area right of --region when the compositor permits"
        else:
            note = (
                "WARNING: there is little room to the right of --region; drag the prediction "
                "window outside the crop or set --preview-position"
            )
    else:
        target = (20, 40)
        note = (
            "WARNING: the full selected source is captured; the prediction may be included "
            "in subsequent frames. Use --region and keep the preview outside it"
        )

    scale = min(max_w / capture_w, max_h / capture_h, 1.0)
    display_size = (max(320, int(capture_w * scale)), max(240, int(capture_h * scale)))
    return target, display_size, note


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--type", default="FRACTAL_3L_6", choices=faruco.SUPPORTED_FRACTAL_TYPES)
    parser.add_argument("--marker-size", type=float, default=0.10,
                        help="Assumed outer marker side length in metres")
    parser.add_argument("--calibration", help="Optional NPZ with camera_matrix and optional dist_coeffs")
    parser.add_argument("--horizontal-fov", type=float, default=70.0,
                        help="Assumed virtual-camera horizontal FOV in degrees when uncalibrated")
    parser.add_argument("--cube-depth-ratio", type=float, default=0.75,
                        help="Cube depth relative to each nested level's side length")
    parser.add_argument("--worker", help="Path to faruco_worker; can also use FARUCO_WORKER")
    parser.add_argument("--region", type=int, nargs=4, metavar=("X", "Y", "W", "H"),
                        help="Optional crop (X Y W H) relative to the selected portal source frame")
    parser.add_argument("--preview-position", type=int, nargs=2, metavar=("X", "Y"),
                        help="Prediction-window position in desktop coordinates; useful to avoid feedback")
    parser.add_argument("--monitor-screenshot", default="monitor_screenshot.png",
                        help="File for the startup/current captured source screenshot")
    parser.add_argument("--prediction-screenshot", default="faruco_prediction.png",
                        help="File used when s is pressed")
    parser.add_argument("--skip-initial-screenshot", action="store_true",
                        help="Do not save the initial screen capture automatically")
    parser.add_argument("--capture-interval", type=float, default=1.0 / 30.0,
                        help="PipeWire capture polling interval in seconds (default: 1/30)")
    args = parser.parse_args()

    if not 0.0 < args.cube_depth_ratio <= 5.0:
        raise SystemExit("--cube-depth-ratio must be greater than 0 and no more than 5")
    if args.marker_size <= 0:
        raise SystemExit("--marker-size must be positive")
    if args.capture_interval < 0:
        raise SystemExit("--capture-interval must be non-negative")

    try:
        from pipewire_capture import CaptureStream, PortalCapture, is_available
    except ImportError as exc:
        raise SystemExit(
            "This Wayland example needs pipewire-capture. Install it in the Python "
            "environment used to run the example with: python3 -m pip install 'pipewire-capture>=0.3.0'"
        ) from exc

    if not is_available():
        raise SystemExit(
            "The Wayland ScreenCast portal is unavailable on this session. Check that "
            "xdg-desktop-portal and a KDE ScreenCast backend are installed and running."
        )

    camera_matrix, dist_coeffs = load_calibration(args.calibration)
    if dist_coeffs is None:
        dist_coeffs = np.zeros(5, dtype=np.float64)
    intrinsics_are_approximate = camera_matrix is None

    portal = PortalCapture()
    try:
        print("Select the screen or window to capture in the KDE sharing dialog…", flush=True)
        session = portal.select_window()
    except Exception as exc:
        raise SystemExit(
            f"Could not start Wayland screen capture: {exc}\n"
            "On KDE Plasma, check that xdg-desktop-portal and xdg-desktop-portal-kde are "
            "running. KWin screen capture also requires OpenGL compositing."
        ) from exc
    if session is None:
        raise SystemExit("Screen capture was cancelled in the desktop sharing dialog")

    stream = None
    try:
        stream = CaptureStream(
            session.fd, session.node_id, session.width, session.height,
            capture_interval=args.capture_interval,
        )
        stream.start()
        try:
            source_frame = wait_for_pipewire_frame(stream)
        except Exception as exc:
            raise SystemExit(f"Could not get a PipeWire screen frame: {exc}") from exc

        # --region is applied to the shared monitor/window's local coordinates.
        frame = crop_bgr(source_frame, args.region)
        if not args.skip_initial_screenshot:
            save_image(args.monitor_screenshot, frame, "Captured source screenshot")

        height, width = frame.shape[:2]
        if camera_matrix is None:
            camera_matrix = approximate_camera_matrix(width, height, args.horizontal_fov)
            print(
                f"Using approximate virtual intrinsics for {width}x{height} capture at "
                f"{args.horizontal_fov:.1f}° horizontal FOV. The pose is relative to this "
                "virtual camera; use a matching calibration NPZ for calibrated pose.",
                flush=True,
            )
        elif camera_matrix[0, 0] <= 0 or camera_matrix[1, 1] <= 0:
            raise SystemExit("Calibration camera_matrix fx and fy must be positive")

        placement, display_size, placement_note = choose_preview_placement(
            args.preview_position, args.region, frame.shape, source_frame.shape,
        )
        print(f"Prediction window: {placement_note}", flush=True)

        try:
            with faruco.Detector(
                fractal_type=args.type,
                marker_size_m=args.marker_size,
                camera_matrix=camera_matrix,
                dist_coeffs=dist_coeffs,
                worker_path=args.worker,
                include_points=True,
            ) as detector:
                cv2.namedWindow("FArUco screen prediction", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("FArUco screen prediction", *display_size)
                cv2.moveWindow("FArUco screen prediction", *placement)

                recent_fps = 0.0
                last_time = time.perf_counter()
                last_processed_at = 0.0
                while True:
                    if stream.window_invalid:
                        detail = getattr(stream, "error", None)
                        if detail:
                            print(f"PipeWire capture ended: {detail}", flush=True)
                        else:
                            print("The shared screen/window capture ended.", flush=True)
                        break

                    now = time.perf_counter()
                    # Avoid repeatedly running detection on the same latest frame faster than
                    # the capture stream's configured polling interval.
                    if args.capture_interval and now - last_processed_at < args.capture_interval:
                        key = cv2.waitKey(1) & 0xFF
                        if key in (27, ord("q")):
                            break
                        time.sleep(0.001)
                        continue

                    pixels = stream.get_frame()
                    if pixels is None:
                        key = cv2.waitKey(1) & 0xFF
                        if key in (27, ord("q")):
                            break
                        time.sleep(0.005)
                        continue

                    try:
                        source_frame = pipewire_frame_to_bgr(pixels)
                        frame = crop_bgr(source_frame, args.region)
                    except Exception as exc:
                        print(f"Screen frame processing failed: {exc}", flush=True)
                        break
                    last_processed_at = time.perf_counter()

                    result = detector.detect(frame)
                    view = result.annotate(frame, draw_correspondences=False)
                    draw_all_level_cubes(
                        view, result, args.type, args.marker_size,
                        camera_matrix, dist_coeffs, args.cube_depth_ratio,
                    )

                    now = time.perf_counter()
                    dt = now - last_time
                    if dt > 0:
                        instant_fps = 1.0 / dt
                        recent_fps = instant_fps if recent_fps == 0 else 0.9 * recent_fps + 0.1 * instant_fps
                    last_time = now
                    pose_source = "approx virtual K" if intrinsics_are_approximate else "provided K"
                    cv2.putText(
                        view,
                        f"WAYLAND/PIPEWIRE | {pose_source} | {result.marker_count} marker(s) | "
                        f"detector {result.elapsed_ms:.1f} ms | {recent_fps:.1f} FPS",
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 240, 240), 2, cv2.LINE_AA,
                    )
                    cv2.imshow("FArUco screen prediction", view)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break
                    if key == ord("s"):
                        save_image(args.prediction_screenshot, view, "Prediction screenshot")
                    elif key == ord("m"):
                        save_image(args.monitor_screenshot, frame, "Captured source screenshot")
        finally:
            cv2.destroyAllWindows()
    finally:
        if stream is not None:
            stream.stop()
        session.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
