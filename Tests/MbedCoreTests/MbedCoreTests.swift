import AppKit
import AVFoundation
import Foundation
import PDFKit
import Testing
@testable import MbedCore

/// Deterministic mock engine for testing `IndexCoordinator` and `VectorStore`.
actor MockEmbeddingEngine: EmbeddingEngine {
    let modelID: String
    let nativeDimensions: Int = Matryoshka.gemma2NativeDimensions
    let supportedDimensions: [Int] = Matryoshka.gemma2Dimensions
    private(set) var recordedInputs: [EmbeddingInput] = []

    init(modelID: String = "embeddinggemma-2-MLX-BF16") {
        self.modelID = modelID
    }

    func embedBatch(
        _ inputs: [EmbeddingInput],
        dimensions: Int? = nil
    ) async throws -> [EmbeddedOutput] {
        recordedInputs.append(contentsOf: inputs)
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
                case let .media(kind, _, _, _, windowSampling):
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

    @Test("WindowPlanner plans overlapping 16s-stride windows by default and supports custom stride")
    func testWindowPlannerLongVideo() {
        let times = (0..<280).map { Double($0) * 0.25 }

        // Default overlapping stride = windowSeconds / 2 = 16.0s -> [0..32, 16..48, 32..64, 48..70]
        let overlapping = WindowPlanner.planWindows(
            times: times,
            duration: 70.0,
            windowSeconds: 32.0,
            sampling: .fps,
            fps: 1.0,
            maxFrames: 32
        )
        #expect(overlapping.count == 4)
        #expect(overlapping[0].start == 0.0 && overlapping[0].end == 32.0)
        #expect(overlapping[1].start == 16.0 && overlapping[1].end == 48.0)
        #expect(overlapping[2].start == 32.0 && overlapping[2].end == 64.0)
        #expect(overlapping[3].start == 48.0 && overlapping[3].end == 70.0)

        // Non-overlapping stride = 32.0s -> [0..32, 32..64, 64..70]
        let consecutive = WindowPlanner.planWindows(
            times: times,
            duration: 70.0,
            windowSeconds: 32.0,
            strideSeconds: 32.0,
            sampling: .fps,
            fps: 1.0,
            maxFrames: 32
        )
        #expect(consecutive.count == 3)
        #expect(consecutive[0].start == 0.0 && consecutive[0].end == 32.0)
        #expect(consecutive[1].start == 32.0 && consecutive[1].end == 64.0)
        #expect(consecutive[2].start == 64.0 && consecutive[2].end == 70.0)
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
        let configURL = checkpointURL.appendingPathComponent("config.json")
        guard FileManager.default.isReadableFile(atPath: configURL.path) else {
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

    @Test("Track 4.1: VectorStore invalidates cached artifacts when windowSampling or modelID changes")
    func testVectorStoreConfigInvalidation() async throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("mbed-config-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let dbURL = tempDir.appendingPathComponent("test.sqlite3")
        let store = try VectorStore(databaseURL: dbURL)
        let videoURL = tempDir.appendingPathComponent("clip.mp4")
        try Data("fake-video".utf8).write(to: videoURL)

        let engine1 = MockEmbeddingEngine(modelID: "embeddinggemma-2-MLX-BF16")
        let coord1 = IndexCoordinator(engine: engine1, store: store)

        let first = try await coord1.indexURLs([videoURL], videoWindowSampling: .fps)
        #expect(first.indexed == 1)

        // Same sampling and model -> skipped
        let second = try await coord1.indexURLs([videoURL], videoWindowSampling: .fps)
        #expect(second.indexed == 0)

        // Different windowSampling (.content) -> re-indexed
        let third = try await coord1.indexURLs([videoURL], videoWindowSampling: .content)
        #expect(third.indexed == 1)

        // Different modelID -> re-indexed
        let engine2 = MockEmbeddingEngine(modelID: "embeddinggemma-2-MLX-8bit")
        let coord2 = IndexCoordinator(engine: engine2, store: store)
        let fourth = try await coord2.indexURLs([videoURL], videoWindowSampling: .content)
        #expect(fourth.indexed == 1)
    }

    @Test("Track 4.2 & 4.3: PDFKit page chunking, PDF thumbnail generation, and codeRetrieval prompt routing")
    func testPDFChunkingAndCodeRetrievalRouting() async throws {
        #expect(EmbeddingTask.codeRetrieval.prefixText == "task: code retrieval | query: ")
        #expect(EmbeddingTask.questionAnswering.prefixText == "task: question answering | query: ")
        #expect(EmbeddingTask.factVerification.prefixText == "task: fact checking | query: ")

        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("mbed-pdf-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let pdfURL = tempDir.appendingPathComponent("manual.pdf")
        let pdfDoc = PDFDocument()
        for pageNum in 1...3 {
            let image = NSImage(size: NSSize(width: 200, height: 200), flipped: false) { rect in
                NSColor.white.setFill()
                rect.fill()
                let str = "Page \(pageNum) content about EmbeddingGemma \(pageNum)" as NSString
                str.draw(at: NSPoint(x: 16, y: 100), withAttributes: [.font: NSFont.systemFont(ofSize: 14)])
                return true
            }
            if let page = PDFPage(image: image) {
                pdfDoc.insert(page, at: pdfDoc.pageCount)
            }
        }
        #expect(pdfDoc.write(to: pdfURL))

        let thumb = await ArtifactProcessor.makeThumbnailJPEG(for: pdfURL, kind: .document)
        #expect(thumb != nil)
        let pdfChunks = try ArtifactProcessor.chunkDocument(at: pdfURL)
        // 3-page PDF -> 1 whole + 3 page spans = 4 chunks
        #expect(pdfChunks.count == 4)
        #expect(pdfChunks[1].span == .pages(startPage: 1, endPage: 1))
        #expect(pdfChunks[3].span == .pages(startPage: 3, endPage: 3))

        // Also test codeRetrieval task selection and document title propagation
        let dbURL = tempDir.appendingPathComponent("test.sqlite3")
        let store = try VectorStore(databaseURL: dbURL)
        let engine = MockEmbeddingEngine()
        let coordinator = IndexCoordinator(engine: engine, store: store)

        let codeURL = tempDir.appendingPathComponent("Engine.swift")
        try "func embed() {}".write(to: codeURL, atomically: true, encoding: .utf8)
        _ = try await coordinator.indexURLs([codeURL])

        let recordedAfterIndex = await engine.recordedInputs
        let indexedCodeInput = try #require(recordedAfterIndex.first)
        #expect(indexedCodeInput.title == "Engine.swift")
        #expect(indexedCodeInput.task == .document)

        _ = try await coordinator.search(queryText: "embed function", kinds: [.code], limit: 5)
        let recorded = await engine.recordedInputs
        let lastQuery = try #require(recorded.last)
        #expect(lastQuery.task == .codeRetrieval)
        #expect(lastQuery.task.prefixText == "task: code retrieval | query: ")
        #expect(lastQuery.task.promptName == "InstructionRetrieval")
    }

    @Test("Track 2.1: HeylookMLXEngine.planBatches enforces <= 16 items and <= 192 MiB raw media budget")
    func testHeylookPlanBatches() {
        // 1. 20 small text inputs -> split into [16, 4]
        let textInputs = (0..<20).map { i in
            EmbeddingInput(parts: [.text("item \(i)")])
        }
        let textBatches = HeylookMLXEngine.planBatches(textInputs)
        #expect(textBatches.count == 2)
        #expect(textBatches[0].count == 16)
        #expect(textBatches[1].count == 4)

        // 2. Media inputs exceeding custom byte budget (1_200 * 0.75 = 900 raw bytes max, 3 items of 450 bytes each -> [2, 1])
        let payload = Data(repeating: 0xAA, count: 450)
        let mediaInputs = (0..<3).map { _ in
            EmbeddingInput(parts: [.media(kind: .image, mimeType: "image/jpeg", data: payload)])
        }
        let mediaBatches = HeylookMLXEngine.planBatches(mediaInputs, maxItems: 16, maxRequestMediaBytes: 1_200)
        #expect(mediaBatches.count == 2)
        #expect(mediaBatches[0].count == 2)
        #expect(mediaBatches[1].count == 1)
    }

    @Test("Track 3.2 & 3.3: Live AVFoundation audio WAV synthesis, 30s window extraction, and video soundtrack parity")
    func testAudioWindowsAndVideoSoundtrackParity() async throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("mbed-audio-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        // Synthesize a 36-second 16kHz mono WAV file -> should produce 2 overlapping 30s windows ([0..30], [15..36])
        let sampleRate = 16_000
        let totalSamples = 36 * sampleRate
        let samples = (0..<totalSamples).map { i in
            Float(sin(2.0 * .pi * 440.0 * Double(i) / Double(sampleRate))) * 0.25
        }
        let wavData = ArtifactProcessor.encodeWAV16Mono(samples: samples, sampleRate: sampleRate)
        let wavURL = tempDir.appendingPathComponent("tone_36s.wav")
        try wavData.write(to: wavURL)

        let audioResult = try await ArtifactProcessor.extractAudioWindows(
            for: wavURL,
            windowSeconds: 30.0,
            strideSeconds: 15.0,
            sampleRate: sampleRate
        )
        #expect(audioResult.windows.count == 2)
        #expect(audioResult.windows[0].start == 0.0 && abs(audioResult.windows[0].end - 30.0) < 0.1)
        #expect(audioResult.windows[1].start == 15.0 && abs(audioResult.windows[1].end - 36.0) < 0.1)

        // Verify IndexCoordinator indexes the 36s audio file into 1 whole + 2 temporal spans = 3 chunks
        let dbURL = tempDir.appendingPathComponent("audio.sqlite3")
        let store = try VectorStore(databaseURL: dbURL)
        let engine = MockEmbeddingEngine()
        let coordinator = IndexCoordinator(engine: engine, store: store)

        let summary = try await coordinator.indexURLs([wavURL])
        #expect(summary.indexed == 1)
        let stats = try store.stats()
        #expect(stats.chunkCount == 3)
    }

    @Test("Track 5.5: VectorStore persists and removes watched folders")
    func testWatchedFoldersPersistence() throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("mbed-watched-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let dbURL = tempDir.appendingPathComponent("watched.sqlite3")
        let store = try VectorStore(databaseURL: dbURL)

        let dirA = tempDir.appendingPathComponent("ClipsA", isDirectory: true)
        let dirB = tempDir.appendingPathComponent("ClipsB", isDirectory: true)
        try store.addWatchedFolder(dirA, windowSampling: .content)
        try store.addWatchedFolder(dirB, windowSampling: .fps)

        let list1 = try store.listWatchedFolders()
        #expect(list1.count == 2)
        #expect(list1.first(where: { $0.url.lastPathComponent == "ClipsA" })?.windowSampling == .content)

        try store.removeWatchedFolder(dirA)
        let list2 = try store.listWatchedFolders()
        #expect(list2.count == 1)
        #expect(list2.first?.url.lastPathComponent == "ClipsB")
    }

    @Test("Track 2.2 & Prompt Parity: HeylookMLXEngine fetchCapabilities updates batch limits and formats document titles & pure-media wire payloads")
    func testHeylookMLXEngineHTTPWireAndCapabilities() async throws {
        MockHeylookURLProtocol.reset()
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [MockHeylookURLProtocol.self]
        let session = URLSession(configuration: config)

        let engine = HeylookMLXEngine(
            baseURL: URL(string: "http://127.0.0.1:8000")!,
            modelID: "embeddinggemma-2-MLX-BF16",
            session: session
        )

        // 1. embedBatch without prior fetchCapabilities() automatically probes GET /v1/models once
        // and applies limits.items = 2 (splitting 3 items into 2 POST /v1/embeddings calls: [2, 1]).
        let inputs: [EmbeddingInput] = [
            EmbeddingInput.text("func hello() {}", task: .document, title: "Hello.swift"),
            EmbeddingInput(
                parts: [.media(kind: .image, mimeType: "image/jpeg", data: Data([0xFF, 0xD8, 0xFF]))],
                task: .document
            ),
            EmbeddingInput.text("greeting function", task: .query),
        ]
        let outputs = try await engine.embedBatch(inputs, dimensions: 128)
        #expect(outputs.count == 3)
        let updatedMax = await engine.maxItemsPerRequest
        #expect(updatedMax == 2)

        let postBodies = MockHeylookURLProtocol.recordedEmbeddingBodies()
        #expect(postBodies.count == 2)

        let batch1Items = try #require(postBodies[0]["input"] as? [[String: Any]])
        #expect(batch1Items.count == 2)

        // Item 0: Document with title -> inline "title: Hello.swift | text: func hello() {}", no prompt_name
        let item0Blocks = try #require(batch1Items[0]["content"] as? [[String: Any]])
        #expect(item0Blocks.first?["text"] as? String == "title: Hello.swift | text: func hello() {}")
        #expect(batch1Items[0]["prompt_name"] == nil)

        // Item 1: Pure image (no text) -> no prompt_name (parity with EmbeddingGemma2Processor and mlx-swift-lm)
        #expect(batch1Items[1]["prompt_name"] == nil)

        // Batch 2, Item 0: Query text -> prompt_name == "query"
        let batch2Items = try #require(postBodies[1]["input"] as? [[String: Any]])
        #expect(batch2Items.count == 1)
        #expect(batch2Items[0]["prompt_name"] as? String == "query")
    }

    @Test("Track 3.1: Live AVAssetWriter MP4 synthesis, 32x32 RGB candidate extraction, frame decoding, and window thumbnails")
    func testLiveAVFoundationVideoCandidatesAndFrameDecoding() async throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("mbed-video-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let mp4URL = tempDir.appendingPathComponent("flash_clip.mp4")
        try Self.writeSyntheticMP4(to: mp4URL, width: 64, height: 64, fps: 4, totalFrames: 12, flashIndices: [5, 6])

        // 1. Metadata-only fast path (.fps) skips CGImage decoding -> thumbnails is empty
        let fastCandidates = try await ArtifactProcessor.extractVideoCandidates(
            for: mp4URL,
            sampling: .fps
        )
        #expect(fastCandidates.duration >= 2.5)
        #expect(!fastCandidates.times.isEmpty)
        #expect(fastCandidates.thumbnails.isEmpty)

        // 2. Content-aware path (.content) decodes 32x32 RGB thumbnails (32 * 32 * 3 = 3072 floats)
        let contentCandidates = try await ArtifactProcessor.extractVideoCandidates(
            for: mp4URL,
            sampling: .content
        )
        #expect(contentCandidates.thumbnails.count == contentCandidates.times.count)
        let firstThumb = try #require(contentCandidates.thumbnails.first)
        #expect(firstThumb.count == 32 * 32 * 3)

        let planned = WindowPlanner.planWindows(
            times: contentCandidates.times,
            duration: contentCandidates.duration,
            windowSeconds: 32.0,
            sampling: .content,
            fps: 1.0,
            maxFrames: 32,
            thumbnails: contentCandidates.thumbnails
        )
        let win = try #require(planned.first)
        #expect(win.frameTimes.count == 3)
        // Must pick at least one frame inside the 1.1s..1.7s flash window
        #expect(win.frameTimes.contains(where: { $0 >= 1.1 && $0 <= 1.7 }))

        // 3. decodeVideoFrames extracts full-resolution CGImages at the selected timestamps
        let decodedFrames = try await ArtifactProcessor.decodeVideoFrames(for: mp4URL, at: win.frameTimes)
        #expect(decodedFrames.count == win.frameTimes.count)
        #expect(decodedFrames.first?.image.width == 64)

        // 4. makeThumbnailJPEG at specific window timestamp produces valid JPEG bytes
        let windowThumb = await ArtifactProcessor.makeThumbnailJPEG(for: mp4URL, kind: .video, atSeconds: 1.25)
        #expect(windowThumb != nil)
    }

    @Test("Track 4.3 & 5.5: pruneMissingArtifacts removes deleted files, FolderWatcher detects changes, and search honors taskOverride")
    func testPruneMissingArtifactsAndTaskOverride() async throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("mbed-prune-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let watchedDir = tempDir.appendingPathComponent("Watched", isDirectory: true)
        try FileManager.default.createDirectory(at: watchedDir, withIntermediateDirectories: true)

        let fileA = watchedDir.appendingPathComponent("A.txt")
        let fileB = watchedDir.appendingPathComponent("B.txt")
        try "Document A".write(to: fileA, atomically: true, encoding: .utf8)
        try "Document B".write(to: fileB, atomically: true, encoding: .utf8)

        let dbURL = tempDir.appendingPathComponent("prune.sqlite3")
        let store = try VectorStore(databaseURL: dbURL)
        let engine = MockEmbeddingEngine()
        let coordinator = IndexCoordinator(engine: engine, store: store)

        // 1. Index directory with 2 files -> 2 artifacts in store
        let first = try await coordinator.indexURLs([watchedDir])
        #expect(first.indexed == 2)
        #expect(try store.stats().artifactCount == 2)

        // 2. Delete B.txt and re-index directory -> pruneMissingArtifacts removes B.txt from SQLite
        try FileManager.default.removeItem(at: fileB)
        _ = try await coordinator.indexURLs([watchedDir])
        let afterPruneStats = try store.stats()
        #expect(afterPruneStats.artifactCount == 1)

        // 3. Verify search(..., taskOverride: .questionAnswering) overrides the default query task
        _ = try await coordinator.search(
            queryText: "how does mbed work?",
            taskOverride: .questionAnswering
        )
        let recorded = await engine.recordedInputs
        let lastInput = try #require(recorded.last)
        #expect(lastInput.task == .questionAnswering)
        #expect(lastInput.task.prefixText == "task: question answering | query: ")

        // 4. Verify FolderWatcher fires onChange when a file is written in watchedDir
        let flag = LockedFlag()
        let watcher = FolderWatcher(debounceInterval: 0.1) { _ in
            flag.setTrue()
        }
        watcher.watch(urls: [watchedDir])
        try "Document C".write(to: watchedDir.appendingPathComponent("C.txt"), atomically: true, encoding: .utf8)
        for _ in 0..<25 {
            if flag.isTrue() { break }
            try await Task.sleep(nanoseconds: 100_000_000)
        }
        watcher.stop()
        #expect(flag.isTrue())
    }

    @Test("Track 1.1 & 1.2: Standalone in-process MLXSwiftLMEmbeddingGemma2Engine embeds text & titled documents on Metal GPU")
    func testInProcessMLXSwiftLMEmbeddingGemma2Engine() async throws {
        #if canImport(MLXVLM)
        #expect(NativeMLXEmbeddingGemma2.isAvailableInProcess)
        let checkpointURL = NativeMLXEmbeddingGemma2.defaultCheckpointURL
        guard FileManager.default.fileExists(atPath: checkpointURL.appendingPathComponent("config.json").path) else {
            return
        }
        let engine = MLXSwiftLMEmbeddingGemma2Engine(checkpointURL: checkpointURL)
        let outputs = try await engine.embedBatch(
            [
                .text("Swift actor concurrency and Metal shaders", task: .document, title: "Concurrency.md"),
                .text("how do actors work in Swift?", task: .query),
                .text("lasagna recipe with ricotta cheese", task: .document, title: "Cooking.md"),
            ],
            dimensions: 768
        )
        #expect(outputs.count == 3)
        #expect(outputs[0].vector.count == 768)
        let simRelevant = Matryoshka.cosine(outputs[1].vector, outputs[0].vector)
        let simIrrelevant = Matryoshka.cosine(outputs[1].vector, outputs[2].vector)
        #expect(simRelevant > simIrrelevant)
        #else
        #expect(!NativeMLXEmbeddingGemma2.isAvailableInProcess)
        #endif
    }

    private static func writeSyntheticMP4(
        to url: URL,
        width: Int,
        height: Int,
        fps: Int32,
        totalFrames: Int,
        flashIndices: Set<Int>
    ) throws {
        let writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
        let settings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
        ]
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
        input.expectsMediaDataInRealTime = false
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: input,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32ARGB),
                kCVPixelBufferWidthKey as String: width,
                kCVPixelBufferHeightKey as String: height,
            ]
        )
        writer.add(input)
        writer.startWriting()
        writer.startSession(atSourceTime: .zero)

        for frameIdx in 0..<totalFrames {
            while !input.isReadyForMoreMediaData {
                Thread.sleep(forTimeInterval: 0.005)
            }
            var pixelBuffer: CVPixelBuffer?
            CVPixelBufferCreate(
                kCFAllocatorDefault,
                width,
                height,
                kCVPixelFormatType_32ARGB,
                nil,
                &pixelBuffer
            )
            if let pb = pixelBuffer {
                CVPixelBufferLockBaseAddress(pb, [])
                if let base = CVPixelBufferGetBaseAddress(pb) {
                    let bytesPerRow = CVPixelBufferGetBytesPerRow(pb)
                    let isFlash = flashIndices.contains(frameIdx)
                    let luma: UInt8 = isFlash ? 245 : 18
                    memset(base, Int32(luma), bytesPerRow * height)
                }
                CVPixelBufferUnlockBaseAddress(pb, [])
                let time = CMTime(value: CMTimeValue(frameIdx), timescale: fps)
                adaptor.append(pb, withPresentationTime: time)
            }
        }

        input.markAsFinished()
        let sema = DispatchSemaphore(value: 0)
        writer.finishWriting { sema.signal() }
        sema.wait()
    }
}

private final class LockedFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var value = false

    func setTrue() {
        lock.lock()
        value = true
        lock.unlock()
    }

    func isTrue() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return value
    }
}

final class MockHeylookURLProtocol: URLProtocol, @unchecked Sendable {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var bodies: [[String: Any]] = []

    static func reset() {
        lock.lock()
        bodies.removeAll()
        lock.unlock()
    }

    static func recordedEmbeddingBodies() -> [[String: Any]] {
        lock.lock()
        defer { lock.unlock() }
        return bodies
    }

    override class func canInit(with request: URLRequest) -> Bool {
        true
    }

    override class func canonicalRequest(for request: URLRequest) -> URLRequest {
        request
    }

    override func startLoading() {
        guard let url = request.url else { return }
        if url.path.hasSuffix("/v1/models") {
            let payload: [String: Any] = [
                "data": [
                    [
                        "id": "embeddinggemma-2-MLX-BF16",
                        "embedding": [
                            "limits": [
                                "items": 2,
                                "request_media_bytes": 256 * 1024 * 1024,
                            ],
                            "video_windows": [
                                "default_seconds": 32.0,
                                "sampling": ["fps", "content"],
                            ],
                            "prompts": [
                                "query": "task: search result | query: ",
                                "document": "title: none | text: ",
                            ],
                        ],
                    ]
                ]
            ]
            let data = (try? JSONSerialization.data(withJSONObject: payload)) ?? Data()
            let resp = HTTPURLResponse(url: url, statusCode: 200, httpVersion: "HTTP/1.1", headerFields: nil)!
            client?.urlProtocol(self, didReceive: resp, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: data)
            client?.urlProtocolDidFinishLoading(self)
            return
        }

        if url.path.hasSuffix("/v1/embeddings") {
            let bodyData: Data
            if let d = request.httpBody {
                bodyData = d
            } else if let stream = request.httpBodyStream {
                stream.open()
                var buf = [UInt8](repeating: 0, count: 65_536)
                var acc = Data()
                while stream.hasBytesAvailable {
                    let read = stream.read(&buf, maxLength: buf.count)
                    if read > 0 { acc.append(buf, count: read) } else { break }
                }
                stream.close()
                bodyData = acc
            } else {
                bodyData = Data()
            }

            let parsed = (try? JSONSerialization.jsonObject(with: bodyData) as? [String: Any]) ?? [:]
            Self.lock.lock()
            Self.bodies.append(parsed)
            Self.lock.unlock()

            let items = (parsed["input"] as? [[String: Any]]) ?? []
            let dim = (parsed["dimensions"] as? Int) ?? 128
            let rows: [[String: Any]] = items.enumerated().map { idx, _ in
                var vec = [Float](repeating: 0.0, count: dim)
                vec[idx % max(1, dim)] = 1.0
                return [
                    "index": idx,
                    "embedding": vec,
                ]
            }
            let responseObj: [String: Any] = [
                "model": "embeddinggemma-2-MLX-BF16",
                "dimensions": dim,
                "data": rows,
            ]
            let data = (try? JSONSerialization.data(withJSONObject: responseObj)) ?? Data()
            let resp = HTTPURLResponse(url: url, statusCode: 200, httpVersion: "HTTP/1.1", headerFields: nil)!
            client?.urlProtocol(self, didReceive: resp, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: data)
            client?.urlProtocolDidFinishLoading(self)
        }
    }

    override func stopLoading() {}
}
