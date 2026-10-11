# FArUco Bridge

A small, vendorable wrapper around **ArUco2's Fractal ArUco (FArUco) implementation**. It builds one C++ executable and includes a Python client that communicates with a long-lived worker over stdin/stdout pipes.

This repository is intended to be cloned separately or included in another project as a Git submodule. The Python code is imported directly from this checkout. **There is no Python package installation step**, and the worker does not need to be installed into a system directory. All fractal generation and detection use ArUco2's C++ implementation; this project does not replace or modify upstream ArUco2.

## Quick start

The normal workflow is:

1. Clone this bridge (or add it as a Git submodule).
2. Install native build dependencies and fetch ArUco2's source.
3. Compile `build/faruco_worker` on the machine where it will run.
4. Add the bridge checkout to your Python import path once, then import and use `faruco_bridge`.

### 1. Get the bridge source

For local testing, clone your bridge repository:

```bash
git clone <your-faruco-bridge-repository> faruco_bridge
cd faruco_bridge
```

For a host application that vendors the bridge as a submodule, run this from the host application's root:

```bash
git submodule add <your-faruco-bridge-repository> third_party/faruco_bridge
git submodule update --init --recursive
```

The URL is a placeholder until you publish your own bridge repository. ArUco2 itself is fetched separately in the next step; it is not included in this source archive.

### 2. Install native build dependencies

On Kubuntu/Ubuntu or Raspberry Pi OS/Debian:

```bash
sudo apt update
sudo apt install -y build-essential cmake git pkg-config libopencv-dev python3-numpy python3-opencv
```

The Python client requires NumPy. Python OpenCV (`cv2`) is needed only for the webcam example and `DetectionResult.annotate()`; the native worker links against the system's C++ OpenCV libraries.

If you use a virtual environment, make sure it can import NumPy. On Ubuntu-family systems, a development environment created with `python3 -m venv .venv --system-site-packages` can use the distribution-provided NumPy/OpenCV packages without installing this bridge as a package.

### 3. Fetch ArUco2 and compile the worker

Run these commands from the bridge checkout:

```bash
./scripts/bootstrap_aruco2.sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
```

The executable should be created at:

```text
build/faruco_worker
```

The bootstrap script clones ArUco2 into `third_party/aruco2/` and prints the exact commit it fetched. For reproducible deployment, record and pin a tested ArUco2 commit rather than relying on the latest upstream revision indefinitely.

If you already have an ArUco2 checkout, you can build against it instead:

```bash
cmake -S . -B build -DARUCO2_ROOT=/path/to/aruco2 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
```

**Do not run `pip install`, `pip install -e`, or `cmake --install` for the normal workflow.** The bridge Python code is imported from its checkout, and `faruco_bridge` finds `build/faruco_worker` relative to that checkout by default. For a non-standard build location, pass `worker_path=...` to `Detector` / `generate_marker`, or set the `FARUCO_WORKER` environment variable.

Build the executable on each target machine. An x86-64 Kubuntu executable cannot be copied to an ARM Raspberry Pi; compile it on the Pi or build it for the Pi's target architecture.

## Using the bridge from a host application

Suppose the host project contains this submodule:

```text
your-app/
├── src/
│   └── main.py
└── third_party/
    └── faruco_bridge/
        ├── build/faruco_worker
        └── faruco_bridge/
```

Add the submodule checkout to `sys.path` once near your application's entry point. Adapt the path to match your own project layout:

```python
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
BRIDGE_ROOT = APP_ROOT / "third_party" / "faruco_bridge"
sys.path.insert(0, str(BRIDGE_ROOT))

from faruco_bridge import Detector, generate_marker
```

If the bridge checkout is already on `PYTHONPATH`, or your application launches with the bridge root on Python's import path, no path setup is needed. No custom Python package is installed into the environment.

## Generate a marker

The bridge uses ArUco2's **existing Fractal ArUco generator**. It exposes the upstream generator's supported choices; it does not define a new marker format or add custom per-level geometry.

From the bridge checkout:

```bash
./build/faruco_worker generate \
  --type FRACTAL_3L_6 \
  --bit-size 20 \
  --output fractal_3l_6.png
```

Show the generation-specific command help at any time:

```bash
./build/faruco_worker generate --help
```

### Generator arguments

| Argument | Default | Meaning |
|---|---|---|
| `--type TYPE` | `FRACTAL_3L_6` | Select one of the four predefined ArUco2 fractal configurations: `FRACTAL_2L_6`, `FRACTAL_3L_6`, `FRACTAL_4L_6`, or `FRACTAL_5L_6`. |
| `--bit-size PIXELS` | `20` | Pixel side length of a bit cell in the **innermost marker**. Must be an integer of at least 2. This is a rasterization parameter, not the physical marker size. |
| `--output FILE` | `fractal_marker.png` | Destination filename for the generated grayscale PNG. |

### What the type names mean

All four current configurations use 6×6-bit marker designs. The `2L`, `3L`, `4L`, and `5L` portions select the number of nested levels (outer marker included):

- `FRACTAL_2L_6`: 2 levels (outer marker plus one inner level).
- `FRACTAL_3L_6`: 3 levels.
- `FRACTAL_4L_6`: 4 levels.
- `FRACTAL_5L_6`: 5 levels.

The preset determines the nested marker geometry and how the levels are arranged and scaled. The generator does **not** provide independent options for the number of levels, each level's bit-grid size, or the relative placement/scale of the nested markers. Those are part of the predefined configuration in ArUco2. To change them would require changing the upstream fractal definition/implementation; this bridge intentionally does not add that feature.

### How `--bit-size` affects the image

ArUco2 interprets `bit-size` as the desired pixel size of one bit cell in the innermost marker. The output image contains the complete outer marker and its nested markers, so the final PNG dimensions depend on both the chosen preset's geometry and `bit-size`; `bit-size` is not necessarily the number of pixels per bit in the outermost marker, and it does not set a physical print dimension. The generated image is square and grayscale. Check its actual pixel dimensions before printing, then set the print scale to obtain the physical side length required by your application.

### Marker ID

The upstream `getFractalMarkerImage()` signature currently includes an `id` argument, but the current implementation explicitly ignores it and its documentation notes that the fractal definitions have only one built-in ID. Therefore, the bridge does not expose `--id` or a Python `marker_id` argument. Passing other IDs would not generate different markers. This follows the current upstream behavior rather than pretending arbitrary IDs are supported.

### Keep generation and detection matched

Use the same `--type` for the worker that you used to generate the printed/rendered marker. For example, a marker generated with `FRACTAL_3L_6` should be detected by a worker configured with `--type FRACTAL_3L_6`. The bridge returns the detection ID reported by ArUco2; it does not let the generator assign arbitrary IDs.

The wrapper calls ArUco2's `getFractalMarkerImage()` directly. See the [upstream Fractal ArUco API](https://github.com/rmsalinas/aruco2/blob/main/opencv2/objdetect/aruco2.hpp) and [implementation](https://github.com/rmsalinas/aruco2/blob/main/src/aruco2_fractal.cpp) for the source definitions.

Python equivalent:

```python
from faruco_bridge import generate_marker

path = generate_marker(
    fractal_type="FRACTAL_3L_6",
    output="fractal_3l_6.png",
    bit_size=20,
)
print(path)
```

## Detect images or camera frames from Python

```python
import cv2
from faruco_bridge import Detector

with Detector(fractal_type="FRACTAL_3L_6", marker_size_m=0.10) as detector:
    frame = cv2.imread("camera_frame.png")
    result = detector.detect(frame)
    print("worker processing ms:", result.elapsed_ms)
    for marker in result.markers:
        print(marker.id, marker.corners)
        print(marker.object_points, marker.image_points)
    cv2.imwrite("annotated.png", result.annotate(frame))
```

For pose estimates, provide camera calibration when starting the worker:

```python
with Detector(
    fractal_type="FRACTAL_3L_6",
    marker_size_m=0.10,
    camera_matrix=camera_matrix,
    dist_coeffs=dist_coeffs,
) as detector:
    result = detector.detect(frame)
    for marker in result.markers:
        if marker.tvec is not None:
            print("translation:", marker.tvec.ravel())
            print("rotation vector:", marker.rvec.ravel())
            print("reprojection RMS (px):", marker.reprojection_error_px)
```

If `camera_matrix` is supplied but `dist_coeffs` is omitted, five zero distortion coefficients are used. `marker_size_m` sets the assumed physical outer side length in metres. Calibration must match the input image resolution and camera setup.

## Webcam example

Run from the bridge checkout after compiling the worker:

```bash
python3 examples/webcam_demo.py --camera 0 --type FRACTAL_3L_6
```

The example adds the bridge checkout to its own import path, so it can run directly from the checkout without installing the Python code. An optional NPZ calibration file can contain arrays named `camera_matrix` and `dist_coeffs`:

```bash
python3 examples/webcam_demo.py --camera 0 --calibration calibration.npz
```

Press `q` or Escape to exit.

## Tests

The client/protocol tests use a fake worker, so they do not require an ArUco2 checkout or a working native build. They do require `pytest` and NumPy. On Kubuntu/Ubuntu, `pytest` can be installed for development with `sudo apt install python3-pytest` (this is only a test dependency, not a bridge installation):

```bash
python3 -m pytest -q
```

A native smoke test must still be performed on a machine with ArUco2 and the OpenCV development libraries installed: generate a marker, detect it in a saved or rendered frame, and check corner and pose-point output.

## Worker protocol v1

The worker starts with `faruco_worker worker [options]` and prints a JSON readiness line. Each detection request consists of an ASCII header line followed by the indicated raw frame bytes:

```text
DETECT <request_id> <width> <height> <channels> <payload_bytes> <include_points>\n
<exactly payload_bytes raw bytes>
```

`channels` is `1` for grayscale or `3` for BGR, with interleaved row-major `uint8` pixels. Responses are one JSON object per line. `PING` and `QUIT` provide health-check and shutdown commands. Stdout is reserved for protocol messages; diagnostics go to stderr.

The protocol is synchronous: Python sends one frame and waits for its result before sending another. This avoids per-frame process startup and image re-encoding, while keeping the initial implementation simple. It does not yet implement asynchronous pipelining or shared-memory transport.

## Notes and limitations

- The C++ executable is built for the host architecture and links to that machine's OpenCV libraries.
- The executable being compiled confirms build success, not detection or pose accuracy. Validate with known images and the intended camera calibration.
- ArUco2's fractal point API behaves differently depending on the number of detected markers. Test single- and multi-marker cases before relying on full inner/outer correspondences for pose estimation.
