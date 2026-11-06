// swift-tools-version: 6.0
import Foundation
import PackageDescription

let packageRoot = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
let candidateMLXRoots = [
    "../../apple-mlx",
    "../apple-mlx",
    "../../mlx",
]
let resolvedMLXRoot: String = candidateMLXRoots.first { relRoot in
    let cmakePath = packageRoot
        .appendingPathComponent("\(relRoot)/mlx-swift/Source/Cmlx/mlx/CMakeLists.txt")
        .standardizedFileURL
        .path
    return FileManager.default.fileExists(atPath: cmakePath)
} ?? "../../apple-mlx"

let mlxCmakePath = packageRoot
    .appendingPathComponent("\(resolvedMLXRoot)/mlx-swift/Source/Cmlx/mlx/CMakeLists.txt")
    .standardizedFileURL
    .path

let env = ProcessInfo.processInfo.environment
let isXcode = (env["__CFBundleIdentifier"]?.contains("Xcode") == true)
    || (env["XPC_SERVICE_NAME"]?.contains("Xcode") == true)
let enableNativeMLX: Bool = {
    if env["MBED_DISABLE_NATIVE_MLX"] == "1" || env["MBED_ENABLE_NATIVE_MLX"] == "0" {
        return false
    }
    if env["MBED_ENABLE_NATIVE_MLX"] == "1" || isXcode {
        return FileManager.default.fileExists(atPath: mlxCmakePath)
    }
    return false
}()

var packageDependencies: [Package.Dependency] = []
var coreTargetDependencies: [Target.Dependency] = []

if enableNativeMLX {
    packageDependencies.append(contentsOf: [
        .package(path: "\(resolvedMLXRoot)/mlx-swift"),
        .package(path: "\(resolvedMLXRoot)/mlx-swift-lm"),
        .package(
            url: "https://github.com/huggingface/swift-transformers",
            .upToNextMajor(from: "1.1.0")
        ),
    ])
    coreTargetDependencies.append(contentsOf: [
        .product(name: "MLX", package: "mlx-swift"),
        .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
        .product(name: "MLXVLM", package: "mlx-swift-lm"),
        .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
        .product(name: "Transformers", package: "swift-transformers"),
    ])
}

let package = Package(
    name: "mbed",
    platforms: [
        .macOS(.v15)
    ],
    products: [
        .library(name: "MbedCore", targets: ["MbedCore"]),
        .executable(name: "mbed-app", targets: ["MbedApp"]),
        .executable(name: "mbed-cli", targets: ["MbedCLI"]),
    ],
    dependencies: packageDependencies,
    targets: [
        .target(
            name: "MbedCore",
            dependencies: coreTargetDependencies,
            linkerSettings: [
                .linkedLibrary("sqlite3"),
                .linkedFramework("Accelerate"),
                .linkedFramework("AVFoundation"),
                .linkedFramework("CoreGraphics"),
                .linkedFramework("CoreML"),
                .linkedFramework("CoreServices"),
                .linkedFramework("ImageIO"),
                .linkedFramework("PDFKit"),
            ]
        ),
        .executableTarget(
            name: "MbedApp",
            dependencies: ["MbedCore"],
            linkerSettings: [
                .linkedFramework("AppKit"),
                .linkedFramework("AVKit"),
                .linkedFramework("SwiftUI"),
            ]
        ),
        .executableTarget(
            name: "MbedCLI",
            dependencies: ["MbedCore"]
        ),
        .testTarget(
            name: "MbedCoreTests",
            dependencies: ["MbedCore"]
        ),
    ]
)
