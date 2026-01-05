// swift-tools-version: 5.9
// FusionML - High-Performance ML Framework for Apple Silicon

import PackageDescription

let package = Package(
    name: "FusionML",
    platforms: [
        .macOS(.v13),
        .iOS(.v16)
    ],
    products: [
        // Main library
        .library(
            name: "FusionML",
            targets: ["FusionML"]
        ),
        
        // Examples
        .executable(
            name: "QuickStart",
            targets: ["QuickStart"]
        ),
        .executable(
            name: "TrainingExample",
            targets: ["TrainingExample"]
        ),
        .executable(
            name: "BenchmarkExample",
            targets: ["BenchmarkExample"]
        )
    ],
    targets: [
        // Main FusionML library
        .target(
            name: "FusionML",
            dependencies: [],
            path: "sources/FusionML",
            resources: [
                .process("Metal/Kernels.metal")
            ]
        ),
        
        // Examples
        .executableTarget(
            name: "QuickStart",
            dependencies: ["FusionML"],
            path: "examples/quickstart"
        ),
        .executableTarget(
            name: "TrainingExample",
            dependencies: ["FusionML"],
            path: "examples/training"
        ),
        .executableTarget(
            name: "BenchmarkExample",
            dependencies: ["FusionML"],
            path: "examples/benchmark"
        ),
        
        // Tests
        .testTarget(
            name: "FusionMLTests",
            dependencies: ["FusionML"],
            path: "tests/FusionMLTests"
        )
    ]
)
