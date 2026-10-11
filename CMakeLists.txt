cmake_minimum_required(VERSION 3.16)
project(faruco_bridge VERSION 0.1.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(ARUCO2_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/aruco2" CACHE PATH
    "Path to the ArUco2 checkout")
if(NOT EXISTS "${ARUCO2_ROOT}/opencv2/objdetect/aruco2.hpp" OR
   NOT EXISTS "${ARUCO2_ROOT}/src/aruco2_fractal.cpp")
    message(FATAL_ERROR
        "ArUco2 source not found at '${ARUCO2_ROOT}'. Run:\n"
        "  ./scripts/bootstrap_aruco2.sh\n"
        "or configure with -DARUCO2_ROOT=/path/to/aruco2")
endif()

find_package(OpenCV REQUIRED)

# ArUco2's current upstream CMake compiles its core sources into each sample
# executable rather than exporting a reusable library. Build the exact source
# set needed by this worker and retain upstream's OpenCL kernel generation step.
set(FARUCO_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${FARUCO_GENERATED_DIR}")
set(ARUCO2_OPENCL_HPP "${FARUCO_GENERATED_DIR}/opencl_kernels_aruco2.hpp")
set(ARUCO2_OPENCL_CPP "${FARUCO_GENERATED_DIR}/opencl_kernels_aruco2.cpp")

add_custom_command(
    OUTPUT "${ARUCO2_OPENCL_HPP}" "${ARUCO2_OPENCL_CPP}"
    COMMAND "${CMAKE_COMMAND}"
        "-DCL_PATH=${ARUCO2_ROOT}/opencl/aruco2detect.cl"
        "-DHPP_PATH=${ARUCO2_OPENCL_HPP}"
        "-DCPP_PATH=${ARUCO2_OPENCL_CPP}"
        -P "${ARUCO2_ROOT}/utils/cl2cpp.cmake"
    DEPENDS
        "${ARUCO2_ROOT}/opencl/aruco2detect.cl"
        "${ARUCO2_ROOT}/utils/cl2cpp.cmake"
    COMMENT "Generating ArUco2 OpenCL kernel wrappers"
    VERBATIM)

set_source_files_properties("${ARUCO2_OPENCL_HPP}" PROPERTIES HEADER_FILE_ONLY TRUE)
add_executable(faruco_worker
    cpp/faruco_worker.cpp
    "${ARUCO2_ROOT}/src/aruco2.cpp"
    "${ARUCO2_ROOT}/src/aruco_dictionary.cpp"
    "${ARUCO2_ROOT}/src/aruco2_fractal.cpp"
    "${ARUCO2_OPENCL_HPP}"
    "${ARUCO2_OPENCL_CPP}")

target_include_directories(faruco_worker PRIVATE
    # Put the checked-out ArUco2 header ahead of any system OpenCV header with
    # the same include path.
    "${ARUCO2_ROOT}"
    "${ARUCO2_ROOT}/src"
    "${ARUCO2_ROOT}/opencv2/objdetect"
    "${FARUCO_GENERATED_DIR}"
    ${OpenCV_INCLUDE_DIRS})
target_link_libraries(faruco_worker PRIVATE ${OpenCV_LIBS})
target_compile_features(faruco_worker PRIVATE cxx_std_17)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(faruco_worker PRIVATE -Wall -Wextra -Wpedantic)
endif()


message(STATUS "FArUco bridge: OpenCV ${OpenCV_VERSION}")
message(STATUS "ArUco2 source: ${ARUCO2_ROOT}")
