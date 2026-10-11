from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from faruco_bridge import Detector, ProtocolError, WorkerError


FAKE_WORKER = r'''#!/usr/bin/env python3
import json
import sys

print(json.dumps({"ok": True, "event": "ready", "protocol_version": 1,
                  "fractal_type": "FRACTAL_3L_6", "pose_enabled": False}), flush=True)
for line in sys.stdin.buffer:
    parts = line.decode("ascii").strip().split()
    if not parts:
        continue
    if parts[0] == "QUIT":
        print(json.dumps({"ok": True, "event": "stopped"}), flush=True)
        break
    if parts[0] == "PING":
        print(json.dumps({"ok": True, "event": "pong"}), flush=True)
        continue
    if parts[0] != "DETECT" or len(parts) != 7:
        print(json.dumps({"ok": False, "error": "bad request"}), flush=True)
        continue
    _, rid, width, height, channels, nbytes, points = parts
    raw = sys.stdin.buffer.read(int(nbytes))
    if len(raw) != int(nbytes):
        print(json.dumps({"ok": False, "request_id": int(rid), "error": "short payload"}), flush=True)
        break
    markers = [{
        "id": 42,
        "type": "FRACTAL_3L_6",
        "corners": [[1, 2], [30, 2], [30, 25], [1, 25]],
        "object_points": [[-0.05, -0.05, 0], [0.05, -0.05, 0], [0.05, 0.05, 0], [-0.05, 0.05, 0]] if points == "1" else None,
        "image_points": [[1, 2], [30, 2], [30, 25], [1, 25]] if points == "1" else None,
        "pose": None,
    }]
    print(json.dumps({"ok": True, "request_id": int(rid), "elapsed_ms": 1.25,
                      "marker_count": len(markers), "markers": markers}), flush=True)
'''


@pytest.fixture
def fake_worker(tmp_path: Path) -> Path:
    script = tmp_path / "fake_faruco_worker"
    script.write_text(FAKE_WORKER, encoding="utf-8")
    script.chmod(0o755)
    return script


def test_persistent_worker_round_trip(fake_worker: Path) -> None:
    with Detector(worker_path=fake_worker) as detector:
        assert detector.ping()
        # Non-contiguous ROI exercises contiguous-buffer conversion.
        image = np.zeros((24, 32, 3), dtype=np.uint8)[:, ::2, :]
        result = detector.detect(image)
        assert result.request_id == 1
        assert result.marker_count == 1
        assert result.elapsed_ms == pytest.approx(1.25)
        marker = result.markers[0]
        assert marker.id == 42
        assert marker.corners.shape == (4, 2)
        assert marker.object_points.shape == (4, 3)
        assert marker.image_points.shape == (4, 2)


def test_detections_without_correspondences(fake_worker: Path) -> None:
    with Detector(worker_path=fake_worker, include_points=False) as detector:
        result = detector.detect(np.zeros((10, 11), dtype=np.uint8))
        assert result.markers[0].object_points is None
        assert result.markers[0].image_points is None


def test_rejects_wrong_image_shape_before_writing(fake_worker: Path) -> None:
    with Detector(worker_path=fake_worker) as detector:
        with pytest.raises(ValueError, match="image must be"):
            detector.detect(np.zeros((10, 10, 4), dtype=np.uint8))


def test_camera_matrix_validation(fake_worker: Path) -> None:
    with pytest.raises(ValueError, match="camera_matrix must have shape"):
        Detector(worker_path=fake_worker, camera_matrix=np.eye(4))


def test_context_manager_closes_worker(fake_worker: Path) -> None:
    detector = Detector(worker_path=fake_worker)
    process = detector._process
    detector.close()
    assert process.poll() == 0
    detector.close()  # idempotent


def test_generator_uses_only_upstream_supported_controls(
    fake_worker: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from subprocess import CompletedProcess
    import faruco_bridge.client as client_module
    from faruco_bridge import generate_marker

    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> CompletedProcess[str]:
        calls.append(command)
        output_path = Path(command[command.index("--output") + 1])
        output_path.write_bytes(b"fake png")
        return CompletedProcess(command, 0, stdout="generated", stderr="")

    monkeypatch.setattr(client_module.subprocess, "run", fake_run)
    output = tmp_path / "fractal.png"
    returned = generate_marker(
        fractal_type="FRACTAL_4L_6",
        output=output,
        bit_size=24,
        worker_path=fake_worker,
    )

    assert returned == output.resolve()
    assert calls and calls[0][1:5] == [
        "generate", "--type", "FRACTAL_4L_6", "--bit-size"
    ]
    assert "24" in calls[0]
    assert "--output" in calls[0]
    assert "--id" not in calls[0]
