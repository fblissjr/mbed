// swift-tools-version: 6.0
import PackageDescription

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
    targets: [
        .target(
            name: "MbedCore",
            linkerSettings: [
                .linkedLibrary("sqlite3"),
                .linkedFramework("Accelerate"),
                .linkedFramework("AVFoundation"),
                .linkedFramework("CoreGraphics"),
                .linkedFramework("ImageIO"),
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
