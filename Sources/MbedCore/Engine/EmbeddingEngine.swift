import Accelerate
import Foundation

/// Task prompts published by EmbeddingGemma 2 (`config_sentence_transformers.json`).
public enum EmbeddingTask: String, Codable, CaseIterable, Sendable {
    case query
    case document
    case retrievalQuery = "Retrieval-query"
    case retrievalDocument = "Retrieval-document"
    case sts = "STS"
    case clustering = "Clustering"
    case classification = "Classification"
    case codeRetrieval = "InstructionRetrieval"
    case none = ""

    public var promptName: String? {
        rawValue.isEmpty ? nil : rawValue
    }

    /// Prefix string from `embeddinggemma-2-MLX-BF16/config_sentence_transformers.json`,
    /// used when running the tokenizer/model in-process.
    public var prefixText: String {
        switch self {
        case .query, .retrievalQuery, .codeRetrieval:
            return "task: search result | query: "
        case .document, .retrievalDocument:
            return "title: none | text: "
        case .clustering:
            return "task: clustering | query: "
        case .classification:
            return "task: classification | query: "
        case .sts:
            return "task: sentence similarity | query: "
        case .none:
            return ""
        }
    }
}

/// Window sampling strategy for temporal artifacts (video/audio).
public enum VideoWindowSampling: String, Codable, CaseIterable, Sendable {
    case off = ""
    case fps = "fps"
    case content = "content"

    public var label: String {
        switch self {
        case .off: return "Whole video only"
        case .fps: return "Fixed 1 fps (32s windows)"
        case .content: return "Content-aware (32s windows)"
        }
    }
}

/// One ordered block inside a multimodal `EmbeddingInput`.
public enum EmbeddingPart: Sendable, Equatable {
    case text(String)
    case media(kind: ArtifactKind, mimeType: String, data: Data, windowSampling: VideoWindowSampling = .off)
}

/// One item submitted for embedding. Produces one primary vector (and optional window vectors).
public struct EmbeddingInput: Sendable, Equatable {
    public let parts: [EmbeddingPart]
    public let task: EmbeddingTask

    public init(parts: [EmbeddingPart], task: EmbeddingTask = .document) {
        self.parts = parts
        self.task = task
    }

    public static func text(_ value: String, task: EmbeddingTask) -> EmbeddingInput {
        EmbeddingInput(parts: [.text(value)], task: task)
    }
}

/// Output for one `EmbeddingInput`.
public struct EmbeddedOutput: Sendable {
    public struct WindowVector: Sendable {
        public let start: Double
        public let end: Double
        public let frameTimes: [Double]
        public let vector: [Float]

        public init(start: Double, end: Double, frameTimes: [Double], vector: [Float]) {
            self.start = start
            self.end = end
            self.frameTimes = frameTimes
            self.vector = vector
        }
    }

    public let vector: [Float]
    public let windows: [WindowVector]

    public init(vector: [Float], windows: [WindowVector] = []) {
        self.vector = vector
        self.windows = windows
    }
}

/// Protocol for an EmbeddingGemma 2 MLX embedding backend.
/// Implemented by `HeylookMLXEngine` (HTTP client to local `heylookllm`) and
/// `NativeMLXEmbeddingGemma2` (standalone in-process MLX Swift architecture).
public protocol EmbeddingEngine: Sendable {
    var modelID: String { get }
    var nativeDimensions: Int { get }
    var supportedDimensions: [Int] { get }

    func embedBatch(
        _ inputs: [EmbeddingInput],
        dimensions: Int?
    ) async throws -> [EmbeddedOutput]
}

/// Matryoshka dimension truncation and hardware-accelerated L2 math (`Accelerate` / `vDSP`).
/// Storing the full 768d EmbeddingGemma 2 vector once allows querying at 128, 256, 512, or 768
/// dimensions without ever re-indexing files.
public enum Matryoshka {
    public static let gemma2Dimensions: [Int] = [128, 256, 512, 768]
    public static let gemma2NativeDimensions: Int = 768

    /// Truncate to `dimensions` and L2-normalize using `vDSP`.
    public static func project(_ vector: [Float], dimensions: Int) -> [Float] {
        let d = min( max(1, dimensions), vector.count )
        var slice = Array(vector.prefix(d))
        var sumSq: Float = 0
        vDSP_svesq(slice, 1, &sumSq, vDSP_Length(d))
        let norm = sqrt(sumSq)
        guard norm > 0, norm.isFinite else { return slice }
        var scale = 1.0 / norm
        vDSP_vsmul(slice, 1, &scale, &slice, 1, vDSP_Length(d))
        return slice
    }

    /// Cosine similarity between two vectors (truncated & L2-normalized to `dimensions`).
    public static func cosine(_ a: [Float], _ b: [Float], dimensions: Int? = nil) -> Float {
        let d = min(a.count, b.count, dimensions ?? Int.max)
        guard d > 0 else { return 0 }
        let pa = d == a.count ? a : project(a, dimensions: d)
        let pb = d == b.count ? b : project(b, dimensions: d)
        var dot: Float = 0
        vDSP_dotpr(pa, 1, pb, 1, &dot, vDSP_Length(d))
        var na: Float = 0
        var nb: Float = 0
        vDSP_svesq(pa, 1, &na, vDSP_Length(d))
        vDSP_svesq(pb, 1, &nb, vDSP_Length(d))
        let denom = sqrt(na * nb)
        return denom > 0 ? dot / denom : 0
    }
}
