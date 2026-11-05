import Foundation

/// Extensible modality taxonomy. Supports media (image, video, audio) today
/// and non-media artifacts (text, code, document) under the same indexing contract.
public enum ArtifactKind: String, Codable, CaseIterable, Sendable {
    case image
    case video
    case audio
    case text
    case code
    case document

    public var isMedia: Bool {
        switch self {
        case .image, .video, .audio:
            return true
        case .text, .code, .document:
            return false
        }
    }

    public static func detect(from url: URL) -> ArtifactKind? {
        switch url.pathExtension.lowercased() {
        case "png", "jpg", "jpeg", "webp", "gif", "bmp", "heic", "tiff":
            return .image
        case "mp4", "mov", "m4v", "webm", "mkv":
            return .video
        case "wav", "mp3", "m4a", "flac", "ogg", "aac", "opus":
            return .audio
        case "txt", "md", "rst", "org":
            return .text
        case "swift", "py", "js", "ts", "c", "cpp", "h", "rs", "go", "json", "toml", "yaml", "yml":
            return .code
        case "pdf", "rtf", "html":
            return .document
        default:
            return nil
        }
    }

    public static func mimeType(for url: URL) -> String {
        switch url.pathExtension.lowercased() {
        case "png": return "image/png"
        case "jpg", "jpeg": return "image/jpeg"
        case "webp": return "image/webp"
        case "gif": return "image/gif"
        case "bmp": return "image/bmp"
        case "mp4", "m4v": return "video/mp4"
        case "mov": return "video/quicktime"
        case "webm": return "video/webm"
        case "mkv": return "video/x-matroska"
        case "wav": return "audio/wav"
        case "mp3": return "audio/mpeg"
        case "m4a", "aac": return "audio/mp4"
        case "flac": return "audio/flac"
        case "ogg", "opus": return "audio/ogg"
        default: return "application/octet-stream"
        }
    }
}

/// Fast change-detection stamp for incremental indexing without hashing multi-hundred-MB videos.
public struct ContentFingerprint: Codable, Hashable, Sendable {
    public let byteCount: Int64
    public let modifiedUnixMs: Int64

    public init(byteCount: Int64, modifiedUnixMs: Int64) {
        self.byteCount = byteCount
        self.modifiedUnixMs = modifiedUnixMs
    }

    public var stampKey: String {
        "\(byteCount):\(modifiedUnixMs)"
    }
}

/// Identifies a sub-span inside an Artifact across modalities:
/// - `.whole`: the entire file or item
/// - `.temporal`: a time window `[start, end)` in seconds (video or audio)
/// - `.lines`: a line range `[startLine, endLine]` (code or text)
/// - `.pages`: a page range `[startPage, endPage]` (documents)
public enum ArtifactSpan: Codable, Hashable, Sendable {
    case whole
    case temporal(start: Double, end: Double, frameTimes: [Double] = [])
    case lines(startLine: Int, endLine: Int)
    case pages(startPage: Int, endPage: Int)

    public var displayBadge: String? {
        switch self {
        case .whole:
            return nil
        case let .temporal(start, end, _):
            return "\(Self.formatSeconds(start))-\(Self.formatSeconds(end))"
        case let .lines(startLine, endLine):
            return "L\(startLine)-L\(endLine)"
        case let .pages(startPage, endPage):
            return "p.\(startPage)-\(endPage)"
        }
    }

    public var startTimeSeconds: Double? {
        if case let .temporal(start, _, _) = self { return start }
        return nil
    }

    private static func formatSeconds(_ s: Double) -> String {
        let total = max(0, Int(s.rounded(.down)))
        return String(format: "%d:%02d", total / 60, total % 60)
    }
}

/// An indexable source artifact (media file, text note, code file, or document).
public struct Artifact: Codable, Identifiable, Hashable, Sendable {
    public let id: String
    public let collectionID: String
    public let kind: ArtifactKind
    public let url: URL
    public let title: String
    public let fingerprint: ContentFingerprint
    public let attributes: [String: String]

    public init(
        id: String,
        collectionID: String,
        kind: ArtifactKind,
        url: URL,
        title: String,
        fingerprint: ContentFingerprint,
        attributes: [String: String] = [:]
    ) {
        self.id = id
        self.collectionID = collectionID
        self.kind = kind
        self.url = url
        self.title = title
        self.fingerprint = fingerprint
        self.attributes = attributes
    }
}

/// A single embedded unit belonging to an `Artifact`.
/// One artifact can produce a `.whole` chunk plus zero or more sub-span chunks
/// (such as 32s video windows or code/text sections).
public struct EmbeddedChunk: Codable, Identifiable, Sendable {
    public let id: String
    public let artifactID: String
    public let span: ArtifactSpan
    public let vector: [Float]
    public let thumbnailJPEG: Data?
    public let previewText: String?

    public init(
        id: String,
        artifactID: String,
        span: ArtifactSpan,
        vector: [Float],
        thumbnailJPEG: Data? = nil,
        previewText: String? = nil
    ) {
        self.id = id
        self.artifactID = artifactID
        self.span = span
        self.vector = vector
        self.thumbnailJPEG = thumbnailJPEG
        self.previewText = previewText
    }
}

/// A deduplicated search result grouped by parent `Artifact`.
/// Preserves the best-scoring span so the UI never shows duplicate cards for the same file,
/// while allowing instant jump-to-timestamp or jump-to-line on every matching span.
public struct SearchHit: Identifiable, Hashable, Sendable {
    public struct SpanMatch: Sendable, Hashable {
        public let chunkID: String
        public let span: ArtifactSpan
        public let score: Float
        public let previewText: String?

        public init(chunkID: String, span: ArtifactSpan, score: Float, previewText: String? = nil) {
            self.chunkID = chunkID
            self.span = span
            self.score = score
            self.previewText = previewText
        }
    }

    public var id: String { artifact.id }
    public let artifact: Artifact
    public let bestScore: Float
    public let bestSpan: ArtifactSpan
    public let spanMatches: [SpanMatch]
    public let thumbnailJPEG: Data?
    public let previewText: String?

    public init(
        artifact: Artifact,
        bestScore: Float,
        bestSpan: ArtifactSpan,
        spanMatches: [SpanMatch],
        thumbnailJPEG: Data?,
        previewText: String?
    ) {
        self.artifact = artifact
        self.bestScore = bestScore
        self.bestSpan = bestSpan
        self.spanMatches = spanMatches
        self.thumbnailJPEG = thumbnailJPEG
        self.previewText = previewText
    }
}
