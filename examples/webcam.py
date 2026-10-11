#!/usr/bin/env python3
"""Minimal live-camera FArUco demo.

Optional calibration NPZ should contain `camera_matrix` and `dist_coeffs` arrays.
Without calibration the example still displays outer-corner detections, but not pose.
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

import faruco_bridge as faruco


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--type", default="FRACTAL_3L_6", choices=faruco.SUPPORTED_FRACTAL_TYPES)
    parser.add_argument("--marker-size", type=float, default=0.10, help="Outer marker side length in metres")
    parser.add_argument("--calibration", help="Optional NPZ containing camera_matrix and dist_coeffs")
    args = parser.parse_args()

    camera_matrix = dist_coeffs = None
    if args.calibration:
        with np.load(args.calibration) as calibration:
            camera_matrix = calibration["camera_matrix"]
            dist_coeffs = calibration["dist_coeffs"] if "dist_coeffs" in calibration else None

    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera}")

    try:
        with faruco.Detector(
            fractal_type=args.type,
            marker_size_m=args.marker_size,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
        ) as detector:
            recent_fps = 0.0
            last_time = time.perf_counter()
            while True:
                ok, frame = capture.read()
                if not ok:
                    print("Camera frame capture failed", flush=True)
                    break

                result = detector.detect(frame)
                view = result.annotate(frame)
                now = time.perf_counter()
                dt = now - last_time
                if dt > 0:
                    instant_fps = 1.0 / dt
                    recent_fps = instant_fps if recent_fps == 0 else 0.9 * recent_fps + 0.1 * instant_fps
                last_time = now
                cv2.putText(view, f"detector {result.elapsed_ms:.1f} ms | loop {recent_fps:.1f} FPS",
                            (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 240, 240), 2, cv2.LINE_AA)
                cv2.imshow("FArUco worker", view)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
    finally:
        capture.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
