#include "opencv2/objdetect/aruco2.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int kProtocolVersion = 1;
constexpr std::uint64_t kMaxFrameBytes = 256ull * 1024ull * 1024ull;

struct Options {
    cv::aruco2::FractalType type = cv::aruco2::FRACTAL_3L_6;
    int bitSize = 20;
    double markerSize = 0.10;
    bool hasCameraMatrix = false;
    cv::Mat cameraMatrix;
    cv::Mat distCoeffs;
};

std::string typeName(cv::aruco2::FractalType type) {
    switch (type) {
        case cv::aruco2::FRACTAL_2L_6: return "FRACTAL_2L_6";
        case cv::aruco2::FRACTAL_3L_6: return "FRACTAL_3L_6";
        case cv::aruco2::FRACTAL_4L_6: return "FRACTAL_4L_6";
        case cv::aruco2::FRACTAL_5L_6: return "FRACTAL_5L_6";
    }
    throw std::invalid_argument("Unsupported fractal type");
}

cv::aruco2::FractalType parseType(const std::string& value) {
    if (value == "FRACTAL_2L_6") return cv::aruco2::FRACTAL_2L_6;
    if (value == "FRACTAL_3L_6") return cv::aruco2::FRACTAL_3L_6;
    if (value == "FRACTAL_4L_6") return cv::aruco2::FRACTAL_4L_6;
    if (value == "FRACTAL_5L_6") return cv::aruco2::FRACTAL_5L_6;
    throw std::invalid_argument("Unknown fractal type '" + value + "'");
}

std::string argValue(int argc, char** argv, const std::string& name,
                     const std::string& fallback = "") {
    for (int i = 2; i < argc; ++i) {
        if (std::string(argv[i]) == name) {
            if (i + 1 >= argc) throw std::invalid_argument("Missing value for " + name);
            return argv[i + 1];
        }
    }
    return fallback;
}

std::vector<double> parseCsvDoubles(const std::string& value, const std::string& label) {
    std::vector<double> out;
    std::stringstream stream(value);
    std::string token;
    while (std::getline(stream, token, ',')) {
        if (token.empty()) throw std::invalid_argument("Empty value in " + label);
        std::size_t consumed = 0;
        const double v = std::stod(token, &consumed);
        if (consumed != token.size() || !std::isfinite(v))
            throw std::invalid_argument("Invalid numeric value in " + label);
        out.push_back(v);
    }
    if (out.empty()) throw std::invalid_argument(label + " must not be empty");
    return out;
}

Options parseOptions(int argc, char** argv) {
    Options opts;
    opts.type = parseType(argValue(argc, argv, "--type", "FRACTAL_3L_6"));
    opts.bitSize = std::stoi(argValue(argc, argv, "--bit-size", "20"));
    opts.markerSize = std::stod(argValue(argc, argv, "--marker-size", "0.10"));
    if (opts.bitSize < 2) throw std::invalid_argument("--bit-size must be >= 2");
    if (!(opts.markerSize > 0.0) || !std::isfinite(opts.markerSize))
        throw std::invalid_argument("--marker-size must be a finite positive number");

    const std::string camera = argValue(argc, argv, "--camera-matrix");
    const std::string distortion = argValue(argc, argv, "--dist-coeffs");
    if (!camera.empty()) {
        const auto values = parseCsvDoubles(camera, "--camera-matrix");
        if (values.size() != 9)
            throw std::invalid_argument("--camera-matrix needs 9 row-major values");
        opts.cameraMatrix = cv::Mat(3, 3, CV_64F);
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                opts.cameraMatrix.at<double>(r, c) = values[static_cast<std::size_t>(r * 3 + c)];
        if (opts.cameraMatrix.at<double>(0, 0) <= 0 || opts.cameraMatrix.at<double>(1, 1) <= 0)
            throw std::invalid_argument("Camera matrix focal lengths (fx, fy) must be positive");
        opts.hasCameraMatrix = true;
        std::vector<double> d = distortion.empty()
            ? std::vector<double>{0.0, 0.0, 0.0, 0.0, 0.0}
            : parseCsvDoubles(distortion, "--dist-coeffs");
        const std::vector<std::size_t> allowed{4, 5, 8, 12, 14};
        if (std::find(allowed.begin(), allowed.end(), d.size()) == allowed.end())
            throw std::invalid_argument("--dist-coeffs must contain 4, 5, 8, 12, or 14 values");
        opts.distCoeffs = cv::Mat(1, static_cast<int>(d.size()), CV_64F);
        for (std::size_t i = 0; i < d.size(); ++i)
            opts.distCoeffs.at<double>(0, static_cast<int>(i)) = d[i];
    } else if (!distortion.empty()) {
        throw std::invalid_argument("--dist-coeffs requires --camera-matrix");
    }
    return opts;
}

void writePoint2Array(std::ostream& out, const std::vector<cv::Point2f>& points) {
    out << '[';
    for (std::size_t i = 0; i < points.size(); ++i) {
        if (i) out << ',';
        out << '[' << points[i].x << ',' << points[i].y << ']';
    }
    out << ']';
}

void writePoint3Mat(std::ostream& out, const cv::Mat& points) {
    out << '[';
    for (int i = 0; i < points.rows; ++i) {
        if (i) out << ',';
        cv::Vec3f p;
        if (points.type() == CV_32FC3) p = points.at<cv::Vec3f>(i, 0);
        else {
            const cv::Vec3d pd = points.at<cv::Vec3d>(i, 0);
            p = cv::Vec3f(static_cast<float>(pd[0]), static_cast<float>(pd[1]), static_cast<float>(pd[2]));
        }
        out << '[' << p[0] << ',' << p[1] << ',' << p[2] << ']';
    }
    out << ']';
}

void writePoint2Mat(std::ostream& out, const cv::Mat& points) {
    out << '[';
    for (int i = 0; i < points.rows; ++i) {
        if (i) out << ',';
        cv::Vec2f p;
        if (points.type() == CV_32FC2) p = points.at<cv::Vec2f>(i, 0);
        else {
            const cv::Vec2d pd = points.at<cv::Vec2d>(i, 0);
            p = cv::Vec2f(static_cast<float>(pd[0]), static_cast<float>(pd[1]));
        }
        out << '[' << p[0] << ',' << p[1] << ']';
    }
    out << ']';
}

void writeDoubleVector(std::ostream& out, const cv::Mat& values) {
    out << '[';
    const cv::Mat contiguous = values.isContinuous() ? values : values.clone();
    const double* data = contiguous.ptr<double>();
    for (std::size_t i = 0; i < contiguous.total(); ++i) {
        if (i) out << ',';
        out << data[i];
    }
    out << ']';
}

std::string detectResponse(std::uint32_t requestId, const cv::Mat& image,
                           const Options& opts, bool includePoints) {
    const auto start = std::chrono::steady_clock::now();
    const auto markers = cv::aruco2::detectFractals(image, opts.type);

    std::ostringstream markerOut;
    markerOut << std::setprecision(9) << '[';
    for (std::size_t index = 0; index < markers.size(); ++index) {
        if (index) markerOut << ',';
        const auto& marker = markers[index];
        markerOut << "{\"id\":" << marker.id << ",\"type\":\"" << typeName(marker.type)
                  << "\",\"corners\":";
        writePoint2Array(markerOut, marker.corners);
        markerOut << ",\"object_points\":null,\"image_points\":null,\"pose\":null";

        cv::Mat objectPoints, imagePoints;
        if (includePoints || opts.hasCameraMatrix) {
            cv::aruco2::getSolvePnpPoints(marker, objectPoints, imagePoints,
                                          static_cast<float>(opts.markerSize));
            if (includePoints) {
                markerOut << ",\"object_points\":";
                writePoint3Mat(markerOut, objectPoints);
                markerOut << ",\"image_points\":";
                writePoint2Mat(markerOut, imagePoints);
            }
        }

        if (opts.hasCameraMatrix && !objectPoints.empty() && objectPoints.rows >= 4 &&
            objectPoints.rows == imagePoints.rows) {
            cv::Mat rvec, tvec;
            bool solved = false;
            try {
                solved = cv::solvePnP(objectPoints, imagePoints, opts.cameraMatrix,
                                       opts.distCoeffs, rvec, tvec, false, cv::SOLVEPNP_ITERATIVE);
            } catch (const cv::Exception&) {
                solved = false;
            }
            if (solved) {
                std::vector<cv::Point2f> projected;
                cv::projectPoints(objectPoints, rvec, tvec, opts.cameraMatrix,
                                  opts.distCoeffs, projected);
                double squared = 0.0;
                const int n = std::min(static_cast<int>(projected.size()), imagePoints.rows);
                for (int i = 0; i < n; ++i) {
                    const cv::Point2f observed = imagePoints.at<cv::Vec2f>(i, 0);
                    const cv::Point2f delta = projected[static_cast<std::size_t>(i)] - observed;
                    squared += delta.dot(delta);
                }
                const double rms = n > 0 ? std::sqrt(squared / n) : -1.0;
                markerOut << ",\"pose\":{\"rvec\":";
                cv::Mat r64, t64;
                rvec.convertTo(r64, CV_64F);
                tvec.convertTo(t64, CV_64F);
                writeDoubleVector(markerOut, r64);
                markerOut << ",\"tvec\":";
                writeDoubleVector(markerOut, t64);
                markerOut << ",\"reprojection_error_px\":" << rms << '}';
            }
        }
        markerOut << '}';
    }
    markerOut << ']';

    const auto end = std::chrono::steady_clock::now();
    const double elapsedMs = std::chrono::duration<double, std::milli>(end - start).count();
    std::ostringstream response;
    response << std::setprecision(9)
             << "{\"ok\":true,\"request_id\":" << requestId
             << ",\"elapsed_ms\":" << elapsedMs
             << ",\"marker_count\":" << markers.size()
             << ",\"markers\":" << markerOut.str() << '}';
    return response.str();
}
void printError(const std::string& message, std::uint32_t requestId = 0) {
    std::string safe = message;
    std::replace(safe.begin(), safe.end(), '\\', '/');
    std::replace(safe.begin(), safe.end(), '"', '\'');
    std::replace(safe.begin(), safe.end(), '\n', ' ');
    std::cout << "{\"ok\":false,\"request_id\":" << requestId
              << ",\"error\":\"" << safe << "\"}" << std::endl;
}

int runWorker(const Options& opts) {
    std::cout << "{\"ok\":true,\"event\":\"ready\",\"protocol_version\":"
              << kProtocolVersion << ",\"fractal_type\":\"" << typeName(opts.type)
              << "\",\"pose_enabled\":" << (opts.hasCameraMatrix ? "true" : "false")
              << "}" << std::endl;

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line == "QUIT") {
            std::cout << "{\"ok\":true,\"event\":\"stopped\"}" << std::endl;
            return 0;
        }
        if (line == "PING") {
            std::cout << "{\"ok\":true,\"event\":\"pong\"}" << std::endl;
            continue;
        }

        std::istringstream request(line);
        std::string command;
        request >> command;
        if (command != "DETECT") {
            printError("Unknown command");
            continue;
        }

        std::uint32_t requestId = 0;
        int width = 0, height = 0, channels = 0, includePointsRaw = 1;
        std::uint64_t payloadBytes = 0;
        if (!(request >> requestId >> width >> height >> channels >> payloadBytes >> includePointsRaw)) {
            printError("Malformed DETECT header");
            continue;
        }
        if (payloadBytes > kMaxFrameBytes) {
            printError("Frame exceeds the configured maximum size", requestId);
            return 2; // framing can't safely continue without reading an oversized payload
        }

        std::vector<unsigned char> pixels(static_cast<std::size_t>(payloadBytes));
        if (payloadBytes > 0) {
            std::cin.read(reinterpret_cast<char*>(pixels.data()), static_cast<std::streamsize>(payloadBytes));
            if (static_cast<std::uint64_t>(std::cin.gcount()) != payloadBytes) {
                printError("Unexpected end of input while reading frame payload", requestId);
                return 2;
            }
        }

        try {
            if (width <= 0 || height <= 0) throw std::invalid_argument("Frame width/height must be positive");
            if (channels != 1 && channels != 3) throw std::invalid_argument("Frame channels must be 1 (gray) or 3 (BGR)");
            const std::uint64_t expected = static_cast<std::uint64_t>(width) *
                                           static_cast<std::uint64_t>(height) *
                                           static_cast<std::uint64_t>(channels);
            if (expected != payloadBytes) throw std::invalid_argument("Frame payload size does not match dimensions");
            const int cvType = channels == 1 ? CV_8UC1 : CV_8UC3;
            cv::Mat frame(height, width, cvType, pixels.data());
            std::cout << detectResponse(requestId, frame, opts, includePointsRaw != 0) << std::endl;
        } catch (const cv::Exception& e) {
            printError(e.what(), requestId);
        } catch (const std::exception& e) {
            printError(e.what(), requestId);
        }
    }
    return 0;
}

int runGenerate(int argc, char** argv) {
    const auto opts = parseOptions(argc, argv);
    const std::string output = argValue(argc, argv, "--output", "fractal_marker.png");
    cv::Mat marker;
    // The current upstream Fractal ArUco implementation defines one built-in ID;
    // its id parameter is presently ignored by getFractalMarkerImage().
    cv::aruco2::getFractalMarkerImage(marker, opts.type, 0, opts.bitSize);
    if (!cv::imwrite(output, marker)) throw std::runtime_error("Could not write marker image to '" + output + "'");
    std::cout << "Generated " << typeName(opts.type) << " marker to " << output
              << " (" << marker.cols << "x" << marker.rows << ")\n";
    return 0;
}

bool hasHelpFlag(int argc, char** argv) {
    for (int i = 2; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--help" || arg == "-h") return true;
    }
    return false;
}

void validateOptions(int argc, char** argv,
                     const std::vector<std::string>& allowedOptions) {
    for (int i = 2; i < argc; ++i) {
        const std::string option(argv[i]);
        if (std::find(allowedOptions.begin(), allowedOptions.end(), option) == allowedOptions.end()) {
            throw std::invalid_argument("Unknown option: " + option + " (use --help for usage)");
        }
        if (i + 1 >= argc) {
            throw std::invalid_argument("Missing value for " + option);
        }
        ++i; // All currently supported options take a value.
    }
}

void printGenerateUsage() {
    std::cout
        << "Generate a predefined ArUco2 Fractal ArUco marker.\n\n"
        << "Usage:\n"
        << "  faruco_worker generate [--type TYPE] [--bit-size PIXELS] [--output FILE]\n\n"
        << "Options:\n"
        << "  --type TYPE       FRACTAL_2L_6, FRACTAL_3L_6, FRACTAL_4L_6, or FRACTAL_5L_6\n"
        << "                    (default: FRACTAL_3L_6)\n"
        << "  --bit-size PIXELS Pixel size of one bit cell in the innermost marker; integer >= 2\n"
        << "                    (default: 20)\n"
        << "  --output FILE     Output PNG path (default: fractal_marker.png)\n\n"
        << "The type selects a predefined nesting configuration; levels and bit-grid sizes are\n"
        << "not independently configurable. The current upstream implementation has one\n"
        << "built-in fractal ID; this command intentionally exposes no --id option.\n"
        << "Output pixel dimensions depend on the selected preset and bit size.\n";
}

void printWorkerUsage() {
    std::cout
        << "Start a persistent FArUco detection worker.\n\n"
        << "Usage:\n"
        << "  faruco_worker worker [--type TYPE] [--marker-size METRES]\n"
        << "      [--camera-matrix fx,0,cx,0,fy,cy,0,0,1] [--dist-coeffs k1,k2,p1,p2,k3]\n\n"
        << "The type must match the configuration used to generate the target marker.\n"
        << "Camera parameters are optional and enable pose estimation when supplied.\n";
}

void printUsage() {
    std::cout
        << "FArUco worker backed by ArUco2\n\n"
        << "Commands:\n"
        << "  faruco_worker generate --help   Show marker generation options\n"
        << "  faruco_worker worker --help     Show worker options\n\n"
        << "Use 'faruco_worker generate --help' or 'faruco_worker worker --help' for details.\n";
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2 || std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h") {
            printUsage();
            return argc < 2 ? 2 : 0;
        }
        const std::string mode(argv[1]);
        if (mode == "generate") {
            if (hasHelpFlag(argc, argv)) {
                printGenerateUsage();
                return 0;
            }
            validateOptions(argc, argv, {"--type", "--bit-size", "--output"});
            return runGenerate(argc, argv);
        }
        if (mode == "worker") {
            if (hasHelpFlag(argc, argv)) {
                printWorkerUsage();
                return 0;
            }
            validateOptions(argc, argv, {"--type", "--marker-size", "--camera-matrix", "--dist-coeffs"});
            return runWorker(parseOptions(argc, argv));
        }
        printUsage();
        throw std::invalid_argument("Unknown mode '" + mode + "'");
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV error: " << e.what() << '\n';
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
