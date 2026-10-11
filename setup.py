"""Build the platform-specific Python wheel and its native C++ worker."""

from __future__ import annotations

import os
import shutil
import shlex
import subprocess
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.dist import Distribution
from setuptools.command.bdist_wheel import bdist_wheel

ROOT = Path(__file__).resolve().parent
WORKER_FILENAME = "faruco_worker.exe" if os.name == "nt" else "faruco_worker"


class BinaryDistribution(Distribution):
    """Tell wheel that the package contains a compiled, platform-specific worker."""

    def has_ext_modules(self) -> bool:
        # The native executable has no Python ABI dependency, but the wheel is
        # still OS/architecture-specific. BinaryWheel below gives it a py3-none tag.
        return True


class BinaryWheel(bdist_wheel):
    """Tag the bundled executable wheel by platform, not Python ABI."""

    def get_tag(self) -> tuple[str, str, str]:
        _python_tag, _abi_tag, platform_tag = super().get_tag()
        return "py3", "none", platform_tag


class BuildPyWithCMake(build_py):
    """Compile faruco_worker and place it inside the installed Python package."""

    def _is_editable(self) -> bool:
        return bool(getattr(self, "editable_mode", False))

    def _install_prefix(self) -> Path:
        # Editable installs need the executable beside the source package.
        # Regular wheel builds install it into build_lib, which becomes site-packages.
        if self._is_editable():
            return ROOT
        return Path(self.build_lib).resolve()

    def _worker_output(self) -> Path:
        return self._install_prefix() / "faruco_bridge" / "bin" / WORKER_FILENAME

    def _license_output(self) -> Path:
        return self._install_prefix() / "faruco_bridge" / "licenses" / "ARUCO2-LICENSE.txt"

    def run(self) -> None:
        super().run()

        cmake = os.environ.get("CMAKE") or shutil.which("cmake")
        if not cmake:
            raise RuntimeError(
                "CMake is required to build faruco-bridge. Install CMake and a C++17 compiler, "
                "then install the system OpenCV development package (for example, libopencv-dev)."
            )

        aruco_root = ROOT / "third_party" / "aruco2"
        required_sources = (
            aruco_root / "opencv2" / "objdetect" / "aruco2.hpp",
            aruco_root / "src" / "aruco2_fractal.cpp",
        )
        missing = [str(path) for path in required_sources if not path.is_file()]
        if missing:
            raise RuntimeError(
                "The ArUco2 source submodule is missing or incomplete. Clone the repository "
                "with --recurse-submodules, or run 'git submodule update --init --recursive'. "
                "Missing: " + ", ".join(missing)
            )

        build_command = self.get_finalized_command("build")
        cmake_build_dir = Path(build_command.build_temp).resolve() / "faruco-cmake"
        cmake_build_dir.mkdir(parents=True, exist_ok=True)
        install_prefix = self._install_prefix()
        install_prefix.mkdir(parents=True, exist_ok=True)

        configure = [
            cmake,
            "-S", str(ROOT),
            "-B", str(cmake_build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_INSTALL_PREFIX={install_prefix}",
        ]
        # Honour CMAKE_ARGS for custom toolchains or OpenCV locations without
        # requiring project-specific setup.py changes.
        extra_args = shlex.split(os.environ.get("CMAKE_ARGS", ""))
        configure.extend(extra_args)
        subprocess.check_call(configure, cwd=ROOT)

        parallelism = os.environ.get("FARUCO_BUILD_JOBS", "2")
        if not parallelism.isdigit() or int(parallelism) < 1:
            raise RuntimeError("FARUCO_BUILD_JOBS must be a positive integer")
        subprocess.check_call(
            [cmake, "--build", str(cmake_build_dir), "--config", "Release",
             "--target", "faruco_worker", "--parallel", parallelism],
            cwd=ROOT,
        )
        subprocess.check_call(
            [cmake, "--install", str(cmake_build_dir), "--config", "Release"],
            cwd=ROOT,
        )

        worker = self._worker_output()
        if not worker.is_file():
            raise RuntimeError(
                f"CMake completed but the installed worker was not found at {worker}. "
                "Check the CMake install rule for faruco_worker."
            )
        if os.name != "nt":
            worker.chmod(worker.stat().st_mode | 0o111)

        # Preserve the upstream license alongside the bundled native executable.
        license_source = aruco_root / "LICENSE"
        if not license_source.is_file():
            raise RuntimeError(f"The ArUco2 license file was not found: {license_source}")
        license_output = self._license_output()
        license_output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(license_source, license_output)

    def get_outputs(self, include_bytecode: bool = True) -> list[str]:
        outputs = list(super().get_outputs(include_bytecode=include_bytecode))
        for output in (self._worker_output(), self._license_output()):
            if output.is_file() and str(output) not in outputs:
                outputs.append(str(output))
        return outputs


setup(
    cmdclass={"build_py": BuildPyWithCMake, "bdist_wheel": BinaryWheel},
    distclass=BinaryDistribution,
    package_data={"faruco_bridge": ["bin/faruco_worker", "bin/faruco_worker.exe", "licenses/ARUCO2-LICENSE.txt"]},
    include_package_data=True,
    zip_safe=False,
)
