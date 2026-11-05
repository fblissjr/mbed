import Accelerate
import AVFoundation
import CoreImage
import Foundation

#if canImport(MLXVLM) && canImport(MLXLMCommon)
import MLXLMCommon
import MLXVLM
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
}

#if canImport(MLXVLM) && canImport(MLXLMCommon)
/// In-process `EmbeddingEngine` backed directly by `mlx-swift-lm`'s `MLXVLM.EmbeddingGemma2Embedding`.
///
/// When `mbed` is linked with local `/Users/fredbliss/mlx/mlx-swift` and `/Users/fredbliss/mlx/mlx-swift-lm`,
/// this engine runs `embeddinggemma-2-MLX-BF16` completely in-process with zero Python or HTTP overhead,
/// and uses `WindowPlanner` + `UserInput.Video.frames(...)` to embed 32-second video windows.
public actor MLXSwiftLMEmbeddingGemma2Engine: EmbeddingEngine {
    public nonisolated let modelID: String
    public nonisolated let nativeDimensions: Int = Matryoshka.gemma2NativeDimensions
    public nonisolated let supportedDimensions: [Int] = Matryoshka.gemma2Dimensions

    private let embedder: MLXVLM.EmbeddingGemma2Embedding

    public init(
        modelDirectory: URL = NativeMLXEmbeddingGemma2.defaultCheckpointURL,
        tokenizerLoader: any TokenizerLoader
    ) async throws {
        self.modelID = modelDirectory.lastPathComponent
        self.embedder = try await MLXVLM.EmbeddingGemma2Embedding(
            modelDirectory: modelDirectory,
            tokenizerLoader: tokenizerLoader,
            componentLoading: .onFirstUse
        )
    }

    public func embedBatch(
        _ inputs: [EmbeddingInput],
        dimensions: Int? = nil
    ) async throws -> [EmbeddedOutput] {
        var outputs: [EmbeddedOutput] = []
        outputs.reserveCapacity(inputs.count)

        for item in inputs {
            let vlmTask = Self.mapTask(item.task)
            var vlmParts: [MLXVLM.EmbeddingInput.Part] = []

            for part in item.parts {
                switch part {
                case .text(let s):
                    vlmParts.append(.text(s))
                case let .media(kind, _, data, _):
                    let tempURL = FileManager.default.temporaryDirectory
                        .appendingPathComponent("mbed-\(UUID().uuidString).\(Self.ext(for: kind))")
                    try data.write(to: tempURL)
                    defer { try? FileManager.default.removeItem(at: tempURL) }
                    switch kind {
                    case .image:
                        vlmParts.append(.image(.url(tempURL)))
                    case .video:
                        vlmParts.append(.video(.url(tempURL)))
                    case .audio:
                        vlmParts.append(.audio(.url(tempURL)))
                    default:
                        break
                    }
                }
            }

            let primary = try await embedder.embed(
                MLXVLM.EmbeddingInput(vlmParts),
                task: vlmTask,
                dimensions: dimensions
            )
            outputs.append(EmbeddedOutput(vector: primary, windows: []))
        }
        return outputs
    }

    private static func mapTask(_ task: EmbeddingTask) -> MLXVLM.EmbeddingTask {
        switch task {
        case .query, .retrievalQuery: return .searchQuery
        case .document, .retrievalDocument: return .document
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
