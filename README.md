# FArUco Bridge

A Python client for **ArUco2's Fractal ArUco (FArUco) implementation**. The Python package communicates with a persistent C++ worker over stdin/stdout. Detection and marker generation run in ArUco2's C++ implementation.

The worker is compiled automatically while the platform-specific Python wheel is built and is bundled inside `faruco_bridge/bin/`. You normally do not need to run CMake yourself or set `FARUCO_WORKER`.

## Install from source

These instructions target Kubuntu/Ubuntu and Raspberry Pi OS/Debian. The native worker is compiled on the machine where the package is installed, so build it on the target machine for Raspberry Pi/ARM deployments.

### 1. Get the source

```bash
git clone https://github.com/PladsElsker/faruco-bridge.git
cd faruco-bridge
```

ArUco2 is fetched from `https://github.com/PladsElsker/aruco2.git` automatically into the temporary build directory when the wheel is built. No ArUco2 checkout or Git submodule setup is needed in the project tree.

### 2. Install the native build dependencies

`pip` invokes CMake during wheel construction. The system therefore needs a C++17 compiler, CMake, and OpenCV development files before installing the Python package:

```bash
sudo apt update
sudo apt install -y build-essential cmake git pkg-config libopencv-dev python3-venv
```

`libopencv-dev` also installs the OpenCV shared libraries the worker needs at runtime. The wheel bundles `faruco_worker` and the ArUco2 license, but does not bundle OpenCV's shared libraries.

### 3. Install the Python package

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

The C++ build uses two parallel jobs by default to avoid exhausting memory on small Raspberry Pi boards. Increase the job count if desired, for example `FARUCO_BUILD_JOBS=4 python -m pip install .`.

During `pip install .`, the package build shallow-clones the default branch of the ArUco2 fork into the temporary build directory, configures CMake, compiles `faruco_worker`, and places the executable and ArUco2 license in the wheel. The project checkout itself does not contain ArUco2 as a submodule. The resulting wheel is platform-specific (OS/CPU architecture), and the build needs network access to GitHub. Python OpenCV is optional; install it only for `DetectionResult.annotate()` or the webcam example:

```bash
python -m pip install '.[opencv]'
```

The normal Python API locates the bundled worker automatically. You can still override it using `worker_path=...` or `FARUCO_WORKER` when testing another build.

### Installing as a submodule in another application

From the host application's root:

```bash
git submodule add https://github.com/PladsElsker/faruco-bridge.git third_party/faruco_bridge
sudo apt install -y build-essential cmake git pkg-config libopencv-dev
python -m pip install ./third_party/faruco_bridge
```

The `pip install` command fetches ArUco2, compiles and bundles the worker into the active Python environment. You do not need to initialize nested submodules, run separate CMake commands, or set `FARUCO_WORKER`.

## Lazy worker startup

Constructing a `Detector` validates its configuration but does not start a process or resolve the worker executable. The first valid `detect()` call starts the C++ worker and performs its protocol handshake; subsequent detections reuse that process. `close()` or leaving a `with Detector(...)` block before the first detection does not start the worker. `ping()` returns `False` until a worker has been started.

## Generate a marker

The bridge uses ArUco2's **existing Fractal ArUco generator**. It exposes the upstream generator's supported choices; it does not define a new marker format or add custom per-level geometry.

Generate a marker through the bundled worker using the Python API:

```python
from faruco_bridge import generate_marker

output = generate_marker(
    fractal_type="FRACTAL_3L_6",
    output="fractal_3l_6.png",
    bit_size=20,
)
print(output)
```

### Command-line usage
```sh
python -m generator \
    --type FRACTAL_3L_6 \
    --bit-size 20 \
    --output fractal_3l_6.png
```

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

After installing the package, run:

```bash
python examples/webcam.py --camera 0 --type FRACTAL_3L_6
```

The example expects the package to be installed; an editable install builds the worker into the checkout. An optional NPZ calibration file can contain arrays named `camera_matrix` and `dist_coeffs`:

```bash
python3 examples/webcam.py --camera 0 --calibration calibration.npz
```

Press `q` or Escape to exit.

## Tests

The client/protocol tests use a fake worker. From the repository root, install the Python test dependencies and run them without triggering the native wheel build:

```bash
python -m pip install numpy pytest
PYTHONPATH=. python -m pytest -q
```

To test a full editable installation, use `python -m pip install -e '.[test]'`; this also compiles the native worker and therefore requires the native build dependencies from step 2. A native smoke test should be performed on the target machine: generate a marker, detect it in a saved or rendered frame, and check corner and pose-point output.

## Worker protocol v1

The worker starts with `faruco_worker worker [options]` and prints a JSON readiness line. Each detection request consists of an ASCII header line followed by the indicated raw frame bytes:

```text
DETECT <request_id> <width> <height> <channels> <payload_bytes> <include_points>\n
<exactly payload_bytes raw bytes>
```

`channels` is `1` for grayscale or `3` for BGR, with interleaved row-major `uint8` pixels. Responses are one JSON object per line. `PING` and `QUIT` provide health-check and shutdown commands. Stdout is reserved for protocol messages; diagnostics go to stderr.

The bundled worker starts lazily on the first valid detection and is then reused. The protocol is synchronous: Python sends one frame and waits for its result before sending another. This avoids per-frame process startup and image re-encoding, while keeping the initial implementation simple. It does not yet implement asynchronous pipelining or shared-memory transport.

## Notes and limitations

- The wheel contains a worker built for the host architecture and links to that machine's OpenCV libraries. The wheel itself is not a portable, pure-Python wheel.
- The executable being compiled confirms build success, not detection or pose accuracy. Validate with known images and the intended camera calibration.
- ArUco2's fractal point API behaves differently depending on the number of detected markers. Test single- and multi-marker cases before relying on full inner/outer correspondences for pose estimation.
