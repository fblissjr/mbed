// swift-tools-version: 6.0
import Foundation
import PackageDescription

let env = ProcessInfo.processInfo.environment
let isXcode = (env["__CFBundleIdentifier"]?.contains("Xcode") == true)
    || (env["XPC_SERVICE_NAME"]?.contains("Xcode") == true)
let enableNativeMLX: Bool = {
    if env["MBED_DISABLE_NATIVE_MLX"] == "1" || env["MBED_ENABLE_NATIVE_MLX"] == "0" {
        return false
    }
    return env["MBED_ENABLE_NATIVE_MLX"] == "1" || isXcode
}()

var packageDependencies: [Package.Dependency] = []
var coreTargetDependencies: [Target.Dependency] = []

if enableNativeMLX {
    packageDependencies.append(contentsOf: [
        .package(
            url: "https://github.com/ml-explore/mlx-swift",
            .upToNextMinor(from: "0.33.0")
        ),
        .package(
            url: "https://github.com/ml-explore/mlx-swift-lm",
            branch: "main"
        ),
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
