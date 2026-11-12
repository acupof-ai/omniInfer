// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "ummmlx",
    platforms: [.macOS(.v14)],
    targets: [
        .executableTarget(name: "ummmlx"),
        .testTarget(name: "ummmlxTests", dependencies: ["ummmlx"])
    ]
)
