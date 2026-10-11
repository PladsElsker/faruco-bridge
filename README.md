# FArUco Bridge

A Python client for **ArUco2's Fractal ArUco (FArUco) implementation**. The Python package communicates with a separately compiled C++ worker over a persistent stdin/stdout pipe. Detection and marker generation remain in ArUco2's C++ implementation.

The Python client is an installable package (`faruco-bridge`). The `faruco_worker` executable is built separately for the target machine; it is not bundled into the Python wheel.

## Install from source

These instructions are for Kubuntu/Ubuntu and Raspberry Pi OS/Debian. Build the native executable on each target machine, including the Raspberry Pi itself when targeting ARM.

### 1. Get the source and ArUco2 submodule

```bash
git clone --recurse-submodules https://github.com/PladsElsker/faruco-bridge.git
cd faruco-bridge
```

If you already cloned without submodules, initialize them:

```bash
git submodule update --init --recursive
```

### 2. Install native build dependencies

```bash
sudo apt update
sudo apt install -y build-essential cmake git pkg-config libopencv-dev
```

### 3. Compile the C++ worker

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
```

The executable should be at `build/faruco_worker`. CMake requires the ArUco2 submodule in `third_party/aruco2`. For a custom ArUco2 checkout, configure with `-DARUCO2_ROOT=/path/to/aruco2`.

### 4. Install the Python package

Create a virtual environment and install the project from the bridge root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

NumPy is installed as a Python dependency. Python OpenCV is optional and is only needed for the webcam example and `DetectionResult.annotate()`; install it with `python -m pip install '.[opencv]'` if required. The C++ build still requires the system OpenCV development package from step 2.

Tell the installed package where the separately compiled executable is located:

```bash
export FARUCO_WORKER="$PWD/build/faruco_worker"
```

Keep this environment variable set in the shell or service that runs your application. Alternatively, put `faruco_worker` on `PATH`, or pass `worker_path="/absolute/path/to/faruco_worker"` to `Detector` and `generate_marker`. The worker starts only when the first valid `Detector.detect()` call is made.

### Installing as a submodule in another application

From the host application's root:

```bash
git submodule add https://github.com/PladsElsker/faruco-bridge.git third_party/faruco_bridge
git submodule update --init --recursive
cmake -S third_party/faruco_bridge -B third_party/faruco_bridge/build -DCMAKE_BUILD_TYPE=Release
cmake --build third_party/faruco_bridge/build -j2
python -m pip install ./third_party/faruco_bridge
export FARUCO_WORKER="$PWD/third_party/faruco_bridge/build/faruco_worker"
```

This installs the Python package into the active Python environment while keeping the C++ worker and ArUco2 source within the submodule checkout. The executable must be compiled for the host application's target architecture.

## Lazy worker startup

Constructing a `Detector` validates its configuration but does not start a process or resolve the worker executable. The first valid `detect()` call starts the C++ worker and performs its protocol handshake; subsequent detections reuse that process. `close()` or leaving a `with Detector(...)` block before the first detection does not start the worker. `ping()` returns `False` until a worker has been started.

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

After compiling the worker and setting `FARUCO_WORKER`, run:

```bash
python examples/webcam_demo.py --camera 0 --type FRACTAL_3L_6
```

The example can also run directly from the checkout without installing the package. An optional NPZ calibration file can contain arrays named `camera_matrix` and `dist_coeffs`:

```bash
python3 examples/webcam_demo.py --camera 0 --calibration calibration.npz
```

Press `q` or Escape to exit.

## Tests

The client/protocol tests use a fake worker, so they do not require an ArUco2 checkout or a working native build. Install the development test dependency and run them with:

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

A native smoke test must still be performed on a machine with ArUco2 and the OpenCV development libraries installed: generate a marker, detect it in a saved or rendered frame, and check corner and pose-point output.

## Worker protocol v1

The worker starts with `faruco_worker worker [options]` and prints a JSON readiness line. Each detection request consists of an ASCII header line followed by the indicated raw frame bytes:

```text
DETECT <request_id> <width> <height> <channels> <payload_bytes> <include_points>\n
<exactly payload_bytes raw bytes>
```

`channels` is `1` for grayscale or `3` for BGR, with interleaved row-major `uint8` pixels. Responses are one JSON object per line. `PING` and `QUIT` provide health-check and shutdown commands. Stdout is reserved for protocol messages; diagnostics go to stderr.

The worker starts lazily on the first valid detection and is then reused. The protocol is synchronous: Python sends one frame and waits for its result before sending another. This avoids per-frame process startup and image re-encoding, while keeping the initial implementation simple. It does not yet implement asynchronous pipelining or shared-memory transport.

## Notes and limitations

- The C++ executable is built for the host architecture and links to that machine's OpenCV libraries.
- The executable being compiled confirms build success, not detection or pose accuracy. Validate with known images and the intended camera calibration.
- ArUco2's fractal point API behaves differently depending on the number of detected markers. Test single- and multi-marker cases before relying on full inner/outer correspondences for pose estimation.
