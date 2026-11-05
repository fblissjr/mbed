import Foundation
import Testing
@testable import MbedCore

/// Deterministic mock engine for testing `IndexCoordinator` and `VectorStore`.
actor MockEmbeddingEngine: EmbeddingEngine {
    let modelID: String = "embeddinggemma-2-MLX-BF16"
    let nativeDimensions: Int = Matryoshka.gemma2NativeDimensions
    let supportedDimensions: [Int] = Matryoshka.gemma2Dimensions

    func embedBatch(
        _ inputs: [EmbeddingInput],
        dimensions: Int? = nil
    ) async throws -> [EmbeddedOutput] {
        let targetDim = dimensions ?? nativeDimensions
        return inputs.enumerated().map { idx, input in
            var raw = [Float](repeating: 0.0, count: 768)
            var hasVideoWindows = false

            for part in input.parts {
                switch part {
                case .text(let s):
                    if s.localizedCaseInsensitiveContains("burst") || s.localizedCaseInsensitiveContains("flash") {
                        raw[0] = 1.0
                        raw[1] = 0.25
                    } else if s.localizedCaseInsensitiveContains("swift") || s.localizedCaseInsensitiveContains("code") {
                        raw[2] = 1.0
                        raw[3] = 0.5
                    } else {
                        raw[(idx + 5) % 16] = 1.0
                    }
                case let .media(kind, _, _, windowSampling):
                    raw[4] = 1.0
                    if kind == .video && windowSampling != .off {
                        hasVideoWindows = true
                    }
                }
            }

            let vec = Matryoshka.project(raw, dimensions: targetDim)
            var windows: [EmbeddedOutput.WindowVector] = []
            if hasVideoWindows {
                var win0 = [Float](repeating: 0.0, count: 768)
                win0[4] = 1.0
                var win1 = [Float](repeating: 0.0, count: 768)
                win1[0] = 1.0 // Matches "burst" / "flash" query
                win1[1] = 0.30
                windows = [
                    EmbeddedOutput.WindowVector(
                        start: 0.0,
                        end: 32.0,
                        frameTimes: [0.0, 1.0, 2.0],
                        vector: Matryoshka.project(win0, dimensions: targetDim)
                    ),
                    EmbeddedOutput.WindowVector(
                        start: 32.0,
                        end: 64.0,
                        frameTimes: [32.0, 33.0, 34.0],
                        vector: Matryoshka.project(win1, dimensions: targetDim)
                    ),
                ]
            }
            return EmbeddedOutput(vector: vec, windows: windows)
        }
    }
}

@Suite("MbedCore Tests")
struct MbedCoreTests {
    @Test("Matryoshka truncation and L2 normalization across 768/512/256/128")
    func testMatryoshkaProjection() {
        let raw = (0..<768).map { Float($0 + 1) }
        for dim in Matryoshka.gemma2Dimensions {
            let projected = Matryoshka.project(raw, dimensions: dim)
            #expect(projected.count == dim)
            let selfCos = Matryoshka.cosine(projected, projected, dimensions: dim)
            #expect(abs(selfCos - 1.0) < 1e-5)
        }
    }

    @Test("WindowPlanner pickContent scales frame budget with clip duration and catches brief events")
    func testWindowPlannerPickContentShortClip() {
        // 3-second clip decoded at 4 fps (12 frames), with a brief flash event at 1.25..1.75s (indices 5, 6).
        let times = (0..<12).map { Double($0) * 0.25 }
        let pixelCount = 16 * 16 * 3
        let thumbnails: [[Float]] = (0..<12).map { idx in
            let val: Float = (idx == 5 || idx == 6) ? 235.0 : 20.0
            return [Float](repeating: val, count: pixelCount)
        }

        let picked = WindowPlanner.pickContent(
            thumbnails: thumbnails,
            times: times,
            lo: 0.0,
            hi: 3.0,
            fps: 1.0,
            maxFrames: 32
        )

        // Duration-scaled budget: ceil(3.0 * 1.0) == 3 frames (not all 12 candidate frames).
        #expect(picked.count == 3)
        // Must catch at least one of the flash frames at 1.25s (idx 5) or 1.50s (idx 6).
        #expect(picked.contains(5) || picked.contains(6))
    }

    @Test("WindowPlanner plans consecutive 32s windows for long videos")
    func testWindowPlannerLongVideo() {
        let times = (0..<280).map { Double($0) * 0.25 }
        let windows = WindowPlanner.planWindows(
            times: times,
            duration: 70.0,
            windowSeconds: 32.0,
            sampling: .fps,
            fps: 1.0,
            maxFrames: 32
        )
        #expect(windows.count == 3)
        #expect(windows[0].start == 0.0 && windows[0].end == 32.0)
        #expect(windows[1].start == 32.0 && windows[1].end == 64.0)
        #expect(windows[2].start == 64.0 && windows[2].end == 70.0)
    }

    @Test("NativeMLXEmbeddingGemma2 trims padded patches, gathers 2D positions, and mean-pools")
    func testNativeMLXEmbeddingGemma2Math() {
        let positionIDs: [[[Int]]] = [
            [[0, 0], [1, 0], [0, 1], [-1, -1], [-1, -1]]
        ]
        let validCount = NativeMLXEmbeddingGemma2.activePatchCount(positionIDs: positionIDs)
        #expect(validCount == 3)

        let tableX: [[Float]] = [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ]
        let tableY: [[Float]] = [
            [0, 0, 0, 0],
            [10, 10, 10, 10],
        ]
        let gathered = NativeMLXEmbeddingGemma2.gatherPositionEmbeddings(
            tableX: tableX,
            tableY: tableY,
            patchPositions: [(x: 0, y: 0), (x: 1, y: 0), (x: 0, y: 1), (x: -1, y: -1)]
        )
        #expect(gathered.count == 4)
        #expect(gathered[0] == [1, 2, 3, 4])
        #expect(gathered[1] == [5, 6, 7, 8])
        #expect(gathered[2] == [11, 12, 13, 14])
        #expect(gathered[3] == [0, 0, 0, 0])

        let pooled = NativeMLXEmbeddingGemma2.meanPoolAndNormalize(
            tokenHiddenStates: Array(gathered.prefix(validCount)),
            attentionMask: [1, 1, 1],
            dimensions: 4
        )
        #expect(pooled.count == 4)
        #expect(abs(Matryoshka.cosine(pooled, pooled, dimensions: 4) - 1.0) < 1e-5)
    }

    @Test("NativeMLXEmbeddingGemma2 loads local EmbeddingGemma 2 checkpoint descriptor when present")
    func testCheckpointDescriptorLoad() throws {
        let checkpointURL = NativeMLXEmbeddingGemma2.defaultCheckpointURL
        guard FileManager.default.fileExists(atPath: checkpointURL.path) else {
            return
        }
        let descriptor = try NativeMLXEmbeddingGemma2.CheckpointDescriptor.load(from: checkpointURL)
        #expect(descriptor.modelType == "embedding_gemma2")
        #expect(descriptor.hiddenSize == 512)
        #expect(descriptor.embeddingDim == 768)
        #expect(descriptor.videoMaxFrames == 32)
        #expect(descriptor.videoMaxSoftTokens == 140)
        #expect(descriptor.imageMaxSoftTokens == 280)
        #expect(descriptor.prompts["query"] == "task: search result | query: ")
    }

    @Test("VectorStore and IndexCoordinator deduplicate windowed video hits and support incremental skip")
    func testVectorStoreAndCoordinatorDeduplication() async throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("mbed-test-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let dbURL = tempDir.appendingPathComponent("test.sqlite3")
        let store = try VectorStore(databaseURL: dbURL)
        let engine = MockEmbeddingEngine()
        let coordinator = IndexCoordinator(engine: engine, store: store)

        // Create a dummy video file and a Swift source file.
        let videoURL = tempDir.appendingPathComponent("demo_clip.mp4")
        try Data("fake-mp4-payload".utf8).write(to: videoURL)

        let codeURL = tempDir.appendingPathComponent("Sample.swift")
        let codeLines = (1...120).map { "let line\($0) = \"swift code line \($0)\"" }.joined(separator: "\n")
        try Data(codeLines.utf8).write(to: codeURL)

        let summary1 = try await coordinator.indexURLs([tempDir])
        #expect(summary1.indexed == 2)
        let stats = try store.stats()
        #expect(stats.artifactCount == 2)
        // Video has 1 whole + 2 temporal windows = 3 chunks; 120-line Swift file has 1 whole + 3 line chunks = 4 -> 7 total.
        #expect(stats.chunkCount == 7)

        // Second index pass skips unchanged fingerprints.
        let summary2 = try await coordinator.indexURLs([tempDir])
        #expect(summary2.indexed == 0)

        // Search for "burst" at 256d Matryoshka slice: should return the video once (deduplicated),
        // with its bestSpan pointing to the second temporal window (32s-64s).
        let hits = try await coordinator.search(queryText: "burst flash event", dimensions: 256, limit: 10)
        #expect(hits.count == 2)
        let topHit = try #require(hits.first)
        #expect(topHit.artifact.url.lastPathComponent == "demo_clip.mp4")
        #expect(topHit.bestSpan == .temporal(start: 32.0, end: 64.0, frameTimes: [32.0, 33.0, 34.0]))
        #expect(topHit.spanMatches.count == 3)
    }
}
