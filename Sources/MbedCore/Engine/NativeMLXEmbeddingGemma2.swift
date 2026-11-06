import Accelerate
import AVFoundation
import CoreImage
import Foundation

#if canImport(MLXVLM)
import MLXHuggingFace
import MLXLMCommon
import MLXVLM
import Tokenizers
#endif

/// Standalone MLX Swift integration for `EmbeddingGemma 2` (`embedding_gemma2`).
///
/// The local checkout at `/Users/fredbliss/mlx/mlx-swift-lm` includes first-party Apple support for
/// `EmbeddingGemma 2`:
/// - `MLXLMCommon.EmbeddingGemma2` & `EmbeddingGemma2Configuration` (`Libraries/MLXLMCommon/Models/EmbeddingGemma2.swift`)
/// - `MLXLMCommon.Gemma4VisionModel` & `Gemma4AudioModel` (`Libraries/MLXLMCommon/Models/Gemma4Encoders.swift`),
///   which already uses `take(positionEmbeddingTable[0], xIndices, axis: 0)` and exact aspect-preserving
///   patch grids without 2,520-patch padding.
/// - `MLXVLM.EmbeddingGemma2Embedding` (`Libraries/MLXVLM/EmbeddingGemma2.swift`), an actor conforming to
///   `MLXVLM.MultimodalEmbedding` with on-demand vision/audio component loading (`componentLoading: .onFirstUse`),
///   a custom Metal bicubic resize kernel (`embedding_gemma2_resize_axis`), and `UserInput.Video.frames(...)`
///   support that allows `mbed`'s `WindowPlanner` (`fps` or duration-scaled `content`) to pass pre-selected
///   window frames directly into the vision tower.
public enum NativeMLXEmbeddingGemma2 {

    public static let defaultCheckpointURL = URL(
        fileURLWithPath: "/Users/fredbliss/Storage/llms/google/embeddinggemma-2-MLX-BF16"
    )

    public struct CheckpointDescriptor: Codable, Sendable, Equatable {
        public let modelType: String
        public let hiddenSize: Int
        public let embeddingDim: Int
        public let numHiddenLayers: Int
        public let maxInputTokens: Int
        public let videoFPS: Double
        public let videoMaxFrames: Int
        public let videoMaxSoftTokens: Int
        public let imageMaxSoftTokens: Int
        public let prompts: [String: String]
        public let poolingModeMeanTokens: Bool

        public var maxWindowSeconds: Double {
            videoFPS > 0 ? Double(videoMaxFrames) / videoFPS : 32.0
        }

        public static func load(from directory: URL = defaultCheckpointURL) throws -> CheckpointDescriptor {
            let configURL = directory.appendingPathComponent("config.json")
            guard FileManager.default.fileExists(atPath: configURL.path) else {
                throw EmbeddingEngineError.checkpointNotFound(directory)
            }
            let rawConfig = try JSONSerialization.jsonObject(
                with: Data(contentsOf: configURL)
            ) as? [String: Any] ?? [:]

            let textConfig = rawConfig["text_config"] as? [String: Any] ?? rawConfig
            let modelType = (rawConfig["model_type"] as? String) ?? "embedding_gemma2"
            let hiddenSize = (textConfig["hidden_size"] as? Int) ?? 512
            let embeddingDim = (textConfig["embedding_dim"] as? Int) ?? 768
            let numHiddenLayers = (textConfig["num_hidden_layers"] as? Int) ?? 24

            let procURL = directory.appendingPathComponent("processor_config.json")
            let rawProc = (try? JSONSerialization.jsonObject(with: Data(contentsOf: procURL)) as? [String: Any]) ?? [:]
            let imageProc = rawProc["image_processor"] as? [String: Any] ?? [:]
            let videoProc = rawProc["video_processor"] as? [String: Any] ?? [:]
            let fps = (videoProc["fps"] as? Double) ?? Double((videoProc["fps"] as? Int) ?? 1)
            let maxFrames = (videoProc["max_frames"] as? Int) ?? 32
            let videoSoftTokens = (videoProc["max_soft_tokens"] as? Int) ?? 140
            let imageSoftTokens = (imageProc["max_soft_tokens"] as? Int) ?? 280

            let stURL = directory.appendingPathComponent("config_sentence_transformers.json")
            let rawST = (try? JSONSerialization.jsonObject(with: Data(contentsOf: stURL)) as? [String: Any]) ?? [:]
            let prompts = (rawST["prompts"] as? [String: String]) ?? [:]

            let poolURL = directory.appendingPathComponent("1_Pooling/config.json")
            let rawPool = (try? JSONSerialization.jsonObject(with: Data(contentsOf: poolURL)) as? [String: Any]) ?? [:]
            let meanPool = (rawPool["pooling_mode_mean_tokens"] as? Bool) ?? true

            return CheckpointDescriptor(
                modelType: modelType,
                hiddenSize: hiddenSize,
                embeddingDim: embeddingDim,
                numHiddenLayers: numHiddenLayers,
                maxInputTokens: 8192,
                videoFPS: fps,
                videoMaxFrames: maxFrames,
                videoMaxSoftTokens: videoSoftTokens,
                imageMaxSoftTokens: imageSoftTokens,
                prompts: prompts,
                poolingModeMeanTokens: meanPool
            )
        }
    }

    /// Reference implementation of `trim_visual_padding`: finds the maximum active patch index
    /// across a batch where `positionIDs != -1` so trailing padding slots are stripped before attention.
    public static func activePatchCount(positionIDs: [[[Int]]]) -> Int {
        var maxValid = 0
        for grid in positionIDs {
            var validInGrid = 0
            for coord in grid where coord.count == 2 && (coord[0] != -1 || coord[1] != -1) {
                validInGrid += 1
            }
            if validInGrid > maxValid {
                maxValid = validInGrid
            }
        }
        return maxValid
    }

    /// Reference implementation of `gathered_position_embeddings`:
    /// Direct 2D position table row gather (`tableX[posX] + tableY[posY]`) with `-1` padding masked to zero,
    /// matching `Gemma4VisionPatchEmbedder` in `mlx-swift-lm/Libraries/MLXLMCommon/Models/Gemma4Encoders.swift`.
    public static func gatherPositionEmbeddings(
        tableX: [[Float]],
        tableY: [[Float]],
        patchPositions: [(x: Int, y: Int)]
    ) -> [[Float]] {
        let dim = tableX.first?.count ?? 0
        return patchPositions.map { pos in
            if pos.x < 0 || pos.y < 0 {
                return [Float](repeating: 0, count: dim)
            }
            let rx = tableX[min(max(0, pos.x), tableX.count - 1)]
            let ry = tableY[min(max(0, pos.y), tableY.count - 1)]
            var out = [Float](repeating: 0, count: dim)
            vDSP_vadd(rx, 1, ry, 1, &out, 1, vDSP_Length(dim))
            return out
        }
    }

    /// Attention-mask weighted mean pooling (`1_Pooling`) followed by Matryoshka truncation + L2 normalization (`2_Normalize`).
    public static func meanPoolAndNormalize(
        tokenHiddenStates: [[Float]],
        attentionMask: [Int],
        dimensions: Int = Matryoshka.gemma2NativeDimensions
    ) -> [Float] {
        guard let hiddenWidth = tokenHiddenStates.first?.count, hiddenWidth > 0 else {
            return []
        }
        var sum = [Float](repeating: 0, count: hiddenWidth)
        var activeCount: Float = 0
        for (row, mask) in zip(tokenHiddenStates, attentionMask) where mask != 0 {
            vDSP_vadd(sum, 1, row, 1, &sum, 1, vDSP_Length(hiddenWidth))
            activeCount += Float(mask)
        }
        if activeCount > 0 {
            var inv = 1.0 / activeCount
            vDSP_vsmul(sum, 1, &inv, &sum, 1, vDSP_Length(hiddenWidth))
        }
        return Matryoshka.project(sum, dimensions: dimensions)
    }
    public static var isAvailableInProcess: Bool {
        #if canImport(MLXVLM) && canImport(MLXLMCommon) && canImport(MLXHuggingFace)
        return true
        #else
        return false
        #endif
    }
}

#if canImport(MLXVLM)
/// In-process `EmbeddingEngine` backed directly by `mlx-swift-lm`'s `MLXVLM.EmbeddingGemma2Embedding`.
///
/// When `mbed` is linked with local `/Users/fredbliss/mlx/mlx-swift` and `/Users/fredbliss/mlx/mlx-swift-lm`
/// (`MBED_ENABLE_NATIVE_MLX=1`), this engine runs `embeddinggemma-2-MLX-BF16` completely in-process
/// with zero Python or HTTP overhead, and uses `WindowPlanner` + `UserInput.Video.frames(...)` to
/// embed 32-second video windows (`16s` overlapping stride, `fps` or duration-scaled `content`).
public actor MLXSwiftLMEmbeddingGemma2Engine: EmbeddingEngine {
    public nonisolated let modelID: String
    public nonisolated let checkpointURL: URL
    public nonisolated let computeOptions: ModelComputeOptions
    public nonisolated let nativeDimensions: Int = Matryoshka.gemma2NativeDimensions
    public nonisolated let supportedDimensions: [Int] = Matryoshka.gemma2Dimensions

    private let descriptor: NativeMLXEmbeddingGemma2.CheckpointDescriptor
    private let customTokenizerLoader: (any TokenizerLoader)?
    private let aneRuntime: ANECoreMLRuntime
    private var cachedTokenizer: (any Tokenizer)?
    private var cachedEmbedder: MLXVLM.EmbeddingGemma2Embedding?

    /// Synchronous initializer used by `MbedApp` and `MbedCLI`; loads the model weights and
    /// tokenizer lazily on the first `embedBatch` call (`componentLoading: .onFirstUse`).
    public init(
        checkpointURL: URL = NativeMLXEmbeddingGemma2.defaultCheckpointURL,
        computeOptions: ModelComputeOptions = .allMLXGPU,
        tokenizerLoader: (any TokenizerLoader)? = nil
    ) {
        self.checkpointURL = checkpointURL
        self.computeOptions = computeOptions
        self.modelID = checkpointURL.lastPathComponent
        self.customTokenizerLoader = tokenizerLoader
        self.aneRuntime = ANECoreMLRuntime(checkpointDirectory: checkpointURL)
        self.descriptor = (try? NativeMLXEmbeddingGemma2.CheckpointDescriptor.load(from: checkpointURL))
            ?? Self.fallbackDescriptor
    }

    /// Eager async initializer when the caller wants to pre-load `MLXVLM.EmbeddingGemma2Embedding` upfront.
    public init(
        modelDirectory: URL = NativeMLXEmbeddingGemma2.defaultCheckpointURL,
        computeOptions: ModelComputeOptions = .allMLXGPU,
        tokenizerLoader: any TokenizerLoader
    ) async throws {
        self.checkpointURL = modelDirectory
        self.computeOptions = computeOptions
        self.modelID = modelDirectory.lastPathComponent
        self.customTokenizerLoader = tokenizerLoader
        self.aneRuntime = ANECoreMLRuntime(checkpointDirectory: modelDirectory)
        self.descriptor = (try? NativeMLXEmbeddingGemma2.CheckpointDescriptor.load(from: modelDirectory))
            ?? Self.fallbackDescriptor
        self.cachedEmbedder = try await MLXVLM.EmbeddingGemma2Embedding(
            modelDirectory: modelDirectory,
            tokenizerLoader: tokenizerLoader,
            componentLoading: .onFirstUse
        )
    }

    private func resolvedTokenizer() async throws -> any Tokenizer {
        if let cachedTokenizer {
            return cachedTokenizer
        }
        let loader: any TokenizerLoader = customTokenizerLoader ?? #huggingFaceTokenizerLoader()
        let tok = try await loader.load(from: checkpointURL)
        self.cachedTokenizer = tok
        return tok
    }

    private func resolvedEmbedder() async throws -> MLXVLM.EmbeddingGemma2Embedding {
        if let cachedEmbedder {
            return cachedEmbedder
        }
        let loader: any TokenizerLoader = customTokenizerLoader ?? #huggingFaceTokenizerLoader()
        let loaded = try await MLXVLM.EmbeddingGemma2Embedding(
            modelDirectory: checkpointURL,
            tokenizerLoader: loader,
            componentLoading: .onFirstUse
        )
        self.cachedEmbedder = loaded
        return loaded
    }

    private static let fallbackDescriptor = NativeMLXEmbeddingGemma2.CheckpointDescriptor(
        modelType: "embedding_gemma2",
        hiddenSize: 512,
        embeddingDim: 768,
        numHiddenLayers: 24,
        maxInputTokens: 8192,
        videoFPS: 1.0,
        videoMaxFrames: 32,
        videoMaxSoftTokens: 140,
        imageMaxSoftTokens: 280,
        prompts: [:],
        poolingModeMeanTokens: true
    )

    private enum ResolvedPart: Sendable {
        case text(String)
        case image(URL)
        case video(URL)
        case audio(URL)
    }

    nonisolated private static func buildVLMInput(
        parts: [ResolvedPart],
        title: String?
    ) -> sending MLXVLM.EmbeddingInput {
        let vlmParts: [MLXVLM.EmbeddingInput.Part] = parts.map { part in
            switch part {
            case .text(let s): return .text(s)
            case .image(let url): return .image(.url(url))
            case .video(let url): return .video(.url(url))
            case .audio(let url): return .audio(.url(url))
            }
        }
        return MLXVLM.EmbeddingInput(vlmParts, title: title)
    }

    nonisolated private static func buildWindowVLMInput(
        frames: [(time: Double, image: CGImage)],
        audioWAVURL: URL?
    ) -> sending MLXVLM.EmbeddingInput {
        let sRGB = CGColorSpace(name: CGColorSpace.sRGB)
        let ciOptions: [CIImageOption: Any] = sRGB.map { [.colorSpace: $0] } ?? [:]
        let videoFrames = frames.map { frame in
            UserInput.VideoFrame(
                image: .ciImage(CIImage(cgImage: frame.image, options: ciOptions)),
                timeStamp: CMTime(seconds: frame.time, preferredTimescale: 90_000)
            )
        }
        var winParts: [MLXVLM.EmbeddingInput.Part] = [.video(.frames(videoFrames))]
        if let audioWAVURL {
            winParts.append(.text(" "))
            winParts.append(.audio(.url(audioWAVURL)))
        }
        return MLXVLM.EmbeddingInput(winParts)
    }

    public func embedBatch(
        _ inputs: [EmbeddingInput],
        dimensions: Int? = nil
    ) async throws -> [EmbeddedOutput] {
        var outputs: [EmbeddedOutput] = []
        outputs.reserveCapacity(inputs.count)

        for item in inputs {
            try Task.checkCancellation()
            let vlmTask = Self.mapTask(item.task)
            var resolvedParts: [ResolvedPart] = []
            var tempURLs: [URL] = []
            defer {
                for url in tempURLs {
                    try? FileManager.default.removeItem(at: url)
                }
            }

            var windowedVideoRequest: (url: URL, sampling: VideoWindowSampling)?
            var hasCompanionAudio = false

            for part in item.parts {
                switch part {
                case .text(let s):
                    resolvedParts.append(.text(s))
                case let .media(kind, _, data, url, windowSampling):
                    let resolvedURL: URL
                    if let url {
                        resolvedURL = url
                    } else if let data {
                        let tempURL = FileManager.default.temporaryDirectory
                            .appendingPathComponent("mbed-\(UUID().uuidString).\(Self.ext(for: kind))")
                        try data.write(to: tempURL)
                        tempURLs.append(tempURL)
                        resolvedURL = tempURL
                    } else {
                        continue
                    }

                    switch kind {
                    case .image:
                        resolvedParts.append(.image(resolvedURL))
                    case .video:
                        resolvedParts.append(.video(resolvedURL))
                        if windowSampling != .off && windowedVideoRequest == nil {
                            windowedVideoRequest = (resolvedURL, windowSampling)
                        }
                    case .audio:
                        resolvedParts.append(.audio(resolvedURL))
                        hasCompanionAudio = true
                    default:
                        break
                    }
                }
            }

            let primary: [Float]
            if windowedVideoRequest == nil,
               let aneVec = try await tryEmbedOnANE(
                   parts: resolvedParts,
                   item: item,
                   dimensions: dimensions ?? nativeDimensions
               ) {
                primary = aneVec
            } else {
                let embedder = try await resolvedEmbedder()
                primary = try await embedder.embed(
                    Self.buildVLMInput(parts: resolvedParts, title: item.title),
                    task: vlmTask,
                    dimensions: dimensions
                )
            }

            var windowVectors: [EmbeddedOutput.WindowVector] = []
            if let (videoURL, sampling) = windowedVideoRequest {
                let embedder = try await resolvedEmbedder()
                windowVectors = try await embedVideoWindows(
                    embedder: embedder,
                    videoURL: videoURL,
                    sampling: sampling,
                    includeAudio: hasCompanionAudio,
                    task: vlmTask,
                    dimensions: dimensions
                )
            }
            outputs.append(EmbeddedOutput(vector: primary, windows: windowVectors))
        }
        return outputs
    }

    /// Attempts to run text, image, audio, and video inputs through `ANECoreMLRuntime` when the user
    /// has assigned `textBackbone` (and `visionEncoder` / `audioEncoder` for media inputs) to
    /// `.cpuAndNeuralEngine`, `.cpuAndGPU`, or `.cpuOnly`. Returns `nil` if the CoreML bundles are not
    /// exported or the input exceeds the compiled ANE sequence buckets, seamlessly falling back to MLX.
    private func tryEmbedOnANE(
        parts: [ResolvedPart],
        item: EmbeddingInput,
        dimensions: Int
    ) async throws -> [Float]? {
        let hasImageOrVideo = parts.contains {
            if case .image = $0 { return true }
            if case .video = $0 { return true }
            return false
        }
        let hasAudio = parts.contains {
            if case .audio = $0 { return true }
            return false
        }
        let usesANEMedia = (hasImageOrVideo && computeOptions.visionEncoder != .mlxGPU)
            || (hasAudio && computeOptions.audioEncoder != .mlxGPU)

        guard computeOptions.textBackbone != .mlxGPU || usesANEMedia else { return nil }
        for part in parts {
            switch part {
            case .text:
                break
            case .image, .video:
                guard computeOptions.visionEncoder != .mlxGPU else { return nil }
            case .audio:
                guard computeOptions.audioEncoder != .mlxGPU else { return nil }
            }
        }

        let tokenizer = try await resolvedTokenizer()
        let bosID = tokenizer.bosToken.flatMap({ tokenizer.convertTokenToId($0) }) ?? 2
        let eosID = tokenizer.eosTokenId ?? tokenizer.convertTokenToId("<eos>") ?? 1

        var tokenIDs: [Int] = [bosID]
        var softTokens: [Float] = []
        var didPrependPrompt = false

        for part in parts {
            switch part {
            case .text(let rawText):
                let formatted: String
                if !didPrependPrompt && item.hasNonWhitespaceText {
                    didPrependPrompt = true
                    if item.task == .document || item.task == .retrievalDocument {
                        formatted = EmbeddingTask.documentPrompt(title: item.title, text: rawText)
                    } else {
                        formatted = item.task.promptPrefix + rawText
                    }
                } else {
                    formatted = rawText
                }
                if !formatted.isEmpty {
                    tokenIDs.append(contentsOf: tokenizer.encode(text: formatted, addSpecialTokens: false))
                }
            case .image(let imageURL):
                guard let prep = ANECoreMLRuntime.prepareImagePixels(from: imageURL, forVideoBudget: false),
                      let encoded = try await aneRuntime.encodeVision(
                          pixels: prep.pixels,
                          height: prep.height,
                          width: prep.width,
                          unit: computeOptions.visionEncoder
                      )
                else {
                    return nil
                }
                tokenIDs.append(255999) // <boi>
                tokenIDs.append(contentsOf: repeatElement(258880, count: encoded.tokenCount))
                tokenIDs.append(258882) // <eoi>
                softTokens.append(contentsOf: encoded.softTokens)
            case .audio(let audioURL):
                let pcm = try await ArtifactProcessor.decodeMonoPCM(for: audioURL)
                guard let mel = ANECoreMLRuntime.computeAudioLogMel(pcmSamples: pcm),
                      let encoded = try await aneRuntime.encodeAudio(
                          melFeatures: mel.features,
                          frameCount: mel.frameCount,
                          unit: computeOptions.audioEncoder
                      )
                else {
                    return nil
                }
                tokenIDs.append(256000) // <boa>
                tokenIDs.append(contentsOf: repeatElement(258881, count: encoded.tokenCount))
                tokenIDs.append(258883) // <eoa>
                softTokens.append(contentsOf: encoded.softTokens)
            case .video(let videoURL):
                let candidates = try await ArtifactProcessor.extractVideoCandidates(
                    for: videoURL,
                    sampling: .fps
                )
                let frameTimes = WindowPlanner.pickFPS(
                    times: candidates.times,
                    lo: 0,
                    hi: min(candidates.duration, descriptor.maxWindowSeconds),
                    fps: descriptor.videoFPS,
                    maxFrames: descriptor.videoMaxFrames
                )
                let decoded = try await ArtifactProcessor.decodeVideoFrames(
                    for: videoURL,
                    at: frameTimes
                )
                guard !decoded.isEmpty else { return nil }
                for frame in decoded {
                    guard let prep = ANECoreMLRuntime.prepareImagePixels(from: frame.image, forVideoBudget: true),
                          let encoded = try await aneRuntime.encodeVision(
                              pixels: prep.pixels,
                              height: prep.height,
                              width: prep.width,
                              unit: computeOptions.visionEncoder
                          )
                    else {
                        return nil
                    }
                    tokenIDs.append(255999) // <boi>
                    tokenIDs.append(contentsOf: repeatElement(258884, count: encoded.tokenCount))
                    tokenIDs.append(258882) // <eoi>
                    softTokens.append(contentsOf: encoded.softTokens)
                }
            }
        }
        tokenIDs.append(eosID)

        let textFusionUnit: TowerComputeUnit = (computeOptions.textBackbone == .mlxGPU)
            ? .cpuAndGPU
            : computeOptions.textBackbone

        return try await aneRuntime.encodeTextAndFuse(
            tokenIDs: tokenIDs,
            softTokens: softTokens.isEmpty ? nil : softTokens,
            unit: textFusionUnit,
            dimensions: dimensions
        )
    }

    private func embedVideoWindows(
        embedder: MLXVLM.EmbeddingGemma2Embedding,
        videoURL: URL,
        sampling: VideoWindowSampling,
        includeAudio: Bool,
        task: MLXVLM.EmbeddingTask,
        dimensions: Int?
    ) async throws -> [EmbeddedOutput.WindowVector] {
        let candidates = try await ArtifactProcessor.extractVideoCandidates(
            for: videoURL,
            sampling: sampling
        )
        let planned = WindowPlanner.planWindows(
            times: candidates.times,
            duration: candidates.duration,
            windowSeconds: descriptor.maxWindowSeconds,
            strideSeconds: descriptor.maxWindowSeconds / 2.0,
            sampling: sampling,
            fps: descriptor.videoFPS,
            maxFrames: descriptor.videoMaxFrames,
            thumbnails: candidates.thumbnails
        )
        guard planned.count > 1 else { return [] }

        let pcmSamples: [Float] = includeAudio
            ? ((try? await ArtifactProcessor.decodeMonoPCM(for: videoURL)) ?? [])
            : []

        var out: [EmbeddedOutput.WindowVector] = []
        out.reserveCapacity(planned.count)

        for win in planned {
            try Task.checkCancellation()
            let decoded = try await ArtifactProcessor.decodeVideoFrames(
                for: videoURL,
                at: win.frameTimes
            )
            guard !decoded.isEmpty else { continue }
            var tempAudioURL: URL?
            defer {
                if let tempAudioURL {
                    try? FileManager.default.removeItem(at: tempAudioURL)
                }
            }
            if !pcmSamples.isEmpty,
               let sliceWAV = ArtifactProcessor.encodeAudioSliceWAV(
                   samples: pcmSamples,
                   sampleRate: ArtifactProcessor.audioSampleRate,
                   startSeconds: win.start,
                   endSeconds: min(win.end, win.start + ArtifactProcessor.audioWindowSeconds)
               ) {
                let wavURL = FileManager.default.temporaryDirectory
                    .appendingPathComponent("mbed-win-audio-\(UUID().uuidString).wav")
                if (try? sliceWAV.write(to: wavURL)) != nil {
                    tempAudioURL = wavURL
                }
            }
            let vec = try await embedder.embed(
                Self.buildWindowVLMInput(frames: decoded, audioWAVURL: tempAudioURL),
                task: task,
                dimensions: dimensions
            )
            out.append(
                EmbeddedOutput.WindowVector(
                    start: win.start,
                    end: win.end,
                    frameTimes: decoded.map { ($0.time * 1000).rounded() / 1000 },
                    vector: vec
                )
            )
        }
        return out
    }

    private static func mapTask(_ task: EmbeddingTask) -> MLXVLM.EmbeddingTask {
        switch task {
        case .query, .retrievalQuery: return .searchQuery
        case .document, .retrievalDocument: return .document
        case .questionAnswering: return .questionAnswering
        case .factVerification: return .factChecking
        case .sts: return .sentenceSimilarity
        case .clustering: return .clustering
        case .classification: return .classification
        case .codeRetrieval: return .codeRetrieval
        case .none: return .none
        }
    }

    private static func ext(for kind: ArtifactKind) -> String {
        switch kind {
        case .image: return "jpg"
        case .video: return "mp4"
        case .audio: return "wav"
        default: return "dat"
        }
    }
}
#endif
