# FArUco Bridge

Generate and detect ArUco2 Fractal ArUco (FArUco) markers from Python. Requires Python 3.10 or newer. Installation builds the native worker and bundles it with the package. You do not need to clone ArUco2, manage its sources, or run CMake yourself.

## 1. Install

The commands below are for Ubuntu/Debian-based systems, including Raspberry Pi OS. On other Linux distributions, install the equivalent compiler, CMake, Git, and OpenCV development packages. Build the package on the machine where it will run; the resulting wheel is specific to that OS and CPU architecture.

### Install system dependencies

```bash
sudo apt update
sudo apt install -y build-essential cmake git pkg-config libopencv-dev python3-venv
```

The build needs network access to GitHub to fetch ArUco2. The native worker links to the system OpenCV libraries, so those must remain installed at runtime.

### Clone and create a virtual environment

```bash
git clone https://github.com/PladsElsker/faruco-bridge.git
cd faruco-bridge
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
```

### Install the package

For detection and marker generation:

```bash
python -m pip install .
```

For the examples and `DetectionResult.annotate()`, install the optional Python OpenCV dependency too:

```bash
python -m pip install '.[opencv]'
```

During installation, the build fetches the default branch of `https://github.com/PladsElsker/aruco2.git`, compiles `faruco_worker`, and bundles the executable and ArUco2 license into the wheel. There is no ArUco2 submodule or manual CMake step. The build uses two parallel jobs by default; on a larger machine, you can change that with `FARUCO_BUILD_JOBS=4 python -m pip install .`.

## 2. Detect markers

The detector accepts a NumPy `uint8` image in grayscale (`H × W`) or BGR (`H × W × 3`) format. Use `with Detector(...)` to make sure its worker is closed after use.

```python
import cv2
from faruco_bridge import Detector

frame = cv2.imread("camera_frame.png")
if frame is None:
    raise FileNotFoundError("camera_frame.png")

with Detector(fractal_type="FRACTAL_3L_6", marker_size_m=0.10) as detector:
    result = detector.detect(frame)
    print(f"Worker time: {result.elapsed_ms:.1f} ms")

    for marker in result.markers:
        print("ID:", marker.id)
        print("Corners:", marker.corners)
        print("Object points:", marker.object_points)
        print("Image points:", marker.image_points)

    cv2.imwrite("annotated.png", result.annotate(frame))
```

The native worker starts automatically on the first valid `detect()` call and is reused for subsequent frames. Creating a `Detector` does not start the worker. `close()` or exiting the `with` block shuts it down.

### Pose estimation

For pose estimates, pass camera calibration that matches the image source and resolution:

```python
import numpy as np
from faruco_bridge import Detector

with np.load("calibration.npz") as calibration:
    camera_matrix = calibration["camera_matrix"]
    dist_coeffs = calibration["dist_coeffs"] if "dist_coeffs" in calibration else None

with Detector(
    fractal_type="FRACTAL_3L_6",
    marker_size_m=0.10,  # physical side length of the outer marker, in metres
    camera_matrix=camera_matrix,
    dist_coeffs=dist_coeffs,
) as detector:
    result = detector.detect(frame)
    for marker in result.markers:
        if marker.tvec is not None:
            print("Translation:", marker.tvec.ravel())
            print("Rotation vector:", marker.rvec.ravel())
            print("Reprojection RMS (px):", marker.reprojection_error_px)
```

`camera_matrix` must be a 3 × 3 matrix. If it is provided and `dist_coeffs` is omitted, zero distortion coefficients are used. `marker_size_m` is the physical side length of the **outer marker**. Pose estimates are only as accurate as the supplied calibration and the image measurements.

## 3. Generate markers

The standalone helper at `examples/generator.py` provides a small Python API and command-line interface. It does not use the detection client, but it does use the bundled native worker for generation. Complete the installation steps above first.

### Generate a PNG

Run from the repository root with the virtual environment activated:

```bash
python examples/generator.py \
    --type FRACTAL_3L_6 \
    --bit-size 20 \
    --output fractal_3l_6.png
```

### Generator arguments

| Argument | Default | Meaning |
|---|---|---|
| `--type TYPE` | `FRACTAL_3L_6` | Preset to generate: `FRACTAL_2L_6`, `FRACTAL_3L_6`, `FRACTAL_4L_6`, or `FRACTAL_5L_6`. |
| `--bit-size PIXELS` | `20` | Pixel size of a bit cell in the **innermost marker**. Must be an integer of at least 2. It does not set the final image dimensions or print size. |
| `--output FILE` | `fractal_marker.png` | Destination path for the generated grayscale PNG. |

You can also import the helper from a Python script run from the repository root:

```python
from examples.generator import generate_marker

path = generate_marker(
    fractal_type="FRACTAL_3L_6",
    output="fractal_3l_6.png",
    bit_size=20,
)
print(path)
```

The presets specify the total number of nested levels, including the outer marker: `2L`, `3L`, `4L`, or `5L`. Each preset fixes the layout and relative scale of its levels; they cannot be configured independently. The `bit-size` value is the pixel size of a bit cell in the innermost marker. Check the generated PNG dimensions and set print scaling separately to obtain your required physical size.

The current upstream generator supports one built-in marker ID and ignores its `id` argument. Custom marker IDs are not available. Use the same fractal preset for generation and detection; the detector reports the ID returned by ArUco2.

## 4. Run the examples

Install the OpenCV extra during setup, or add Python OpenCV to the active environment:

```bash
python -m pip install 'opencv-python>=4.8'
```

### Webcam

```bash
python examples/webcam.py --camera 0 --type FRACTAL_3L_6
```

Use a different camera index if needed. To provide calibration from an NPZ file containing `camera_matrix` and optionally `dist_coeffs`:

```bash
python examples/webcam.py --camera 0 --calibration calibration.npz
```

Press `q` or Escape to exit.

### Screen/window capture

The screen demo uses PipeWire and your desktop's screen-sharing portal to let you select a screen or window. 

Install the extra dependency in the same virtual environment:

```bash
python -m pip install 'pipewire-capture>=0.3.0'
```

Then run:

```bash
python examples/screen.py --type FRACTAL_3L_6
```

Choose the source in the sharing dialog. By default, the first captured frame is saved as `monitor_screenshot.png`. Press `q` or Escape to quit, `s` to save the prediction view, or `m` to save the captured source. Use `--region X Y W H` to crop the captured source and `--preview-position X Y` to move the preview; this can keep the preview out of a full-screen capture and prevent visual feedback. The screen demo uses approximate virtual camera intrinsics by default, so its pose values are not a substitute for physical-camera calibration.

## 5. Tests

The client/protocol tests use a fake worker. They test the Python layer, not native detection accuracy. From the repository root:

```bash
python -m pip install numpy pytest
PYTHONPATH=. python -m pytest -q
```

Also perform a native smoke test on the target machine: generate a marker image, detect it in a known image or camera frame, and verify the returned corners and, when calibration is supplied, pose estimates.

## 6. Install inside another application

FArUco Bridge can be added to another project's repository as a Git submodule. From that project's root:

```bash
git submodule add https://github.com/PladsElsker/faruco-bridge.git third_party/faruco_bridge
sudo apt install -y build-essential cmake git pkg-config libopencv-dev
python -m pip install ./third_party/faruco_bridge
```

The package install fetches ArUco2 and builds the worker automatically. There is no nested ArUco2 submodule to initialize.

## Reference and limitations

- **Supported presets:** `FRACTAL_2L_6`, `FRACTAL_3L_6`, `FRACTAL_4L_6`, and `FRACTAL_5L_6`.
- **Image input:** NumPy `uint8` grayscale or BGR data; frames are processed synchronously.
- **Worker lifecycle:** one persistent worker per `Detector`, started on the first valid detection and stopped by `close()` or the context manager.
- **Runtime dependency:** the wheel includes the worker and ArUco2 license, but uses the system's OpenCV shared libraries.
- **Marker IDs:** custom IDs are not supported by the current upstream fractal generator.
- **Pose:** estimates require the outer marker's physical size and suitable camera calibration. Screen capture uses approximate virtual-camera parameters unless matching calibration is supplied.

### Maintainer reference: worker protocol

The worker is bundled in the wheel and normally does not need to be invoked directly. Protocol version 1 starts the worker in worker mode and exchanges line-oriented JSON messages over stdout, with raw image bytes following each `DETECT` request header:

```text
DETECT <request_id> <width> <height> <channels> <payload_bytes> <include_points>\n
<exactly payload_bytes raw bytes>
```

`channels` is `1` for grayscale or `3` for BGR, with interleaved row-major `uint8` pixels. `PING` and `QUIT` provide health checks and shutdown. Stdout is reserved for protocol messages; diagnostics are written to stderr. Requests are synchronous; asynchronous pipelining and shared-memory transport are not implemented.
