// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "hybrid-retrieval-kit",
    // Only iOS is declared because CI builds exactly two things: this package for
    // `generic/platform=iOS Simulator` via xcodebuild, and Linux via `swift build`/`swift test`
    // (Linux ignores the platforms list). Declaring platforms CI never builds would be
    // an unverified claim.
    platforms: [.iOS(.v17)],
    products: [
        .library(name: "HybridRetrieval", targets: ["HybridRetrieval"])
    ],
    targets: [
        .target(name: "HybridRetrieval"),
        .testTarget(name: "HybridRetrievalTests", dependencies: ["HybridRetrieval"])
    ]
)
