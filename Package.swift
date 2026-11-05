// swift-tools-version: 6.0
import Foundation
import PackageDescription

let enableNativeMLX = ProcessInfo.processInfo.environment["MBED_ENABLE_NATIVE_MLX"] == "1"

var packageDependencies: [Package.Dependency] = []
var coreTargetDependencies: [Target.Dependency] = []

if enableNativeMLX {
    packageDependencies.append(contentsOf: [
        .package(path: "../../mlx/mlx-swift"),
        .package(path: "../../mlx/mlx-swift-lm"),
    ])
    coreTargetDependencies.append(contentsOf: [
        .product(name: "MLX", package: "mlx-swift"),
        .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
        .product(name: "MLXVLM", package: "mlx-swift-lm"),
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
