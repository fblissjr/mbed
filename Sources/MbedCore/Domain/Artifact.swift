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

    public var endTimeSeconds: Double? {
        if case let .temporal(_, end, _) = self { return end }
        return nil
    }

    public var frameTimes: [Double] {
        if case let .temporal(_, _, times) = self { return times }
        return []
    }

    public static func formatSeconds(_ s: Double) -> String {
        let total = max(0, Int(s.rounded(.down)))
        return String(format: "%d:%02d", total / 60, total % 60)
    }

    public static func formatTimestampMillis(_ s: Double) -> String {
        let clamped = max(0, s)
        let mins = Int(clamped) / 60
        let secs = clamped - Double(mins * 60)
        return String(format: "%d:%04.1f", mins, secs)
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
    public let windowSamplingRaw: String
    public let modelID: String
    public let attributes: [String: String]

    public init(
        id: String,
        collectionID: String,
        kind: ArtifactKind,
        url: URL,
        title: String,
        fingerprint: ContentFingerprint,
        windowSamplingRaw: String = "",
        modelID: String = "",
        attributes: [String: String] = [:]
    ) {
        self.id = id
        self.collectionID = collectionID
        self.kind = kind
        self.url = url
        self.title = title
        self.fingerprint = fingerprint
        self.windowSamplingRaw = windowSamplingRaw
        self.modelID = modelID
        self.attributes = attributes
    }

    public var includesAudioTrack: Bool {
        modelID.hasSuffix("+audio")
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

/// Describes the specific multimodal feature / tower that produced a scored chunk vector.
public struct ScoredFeatureSummary: Sendable, Hashable {
    public let title: String
    public let subtitle: String
    public let systemImage: String
    public let isAudioTrack: Bool

    public init(title: String, subtitle: String, systemImage: String, isAudioTrack: Bool = false) {
        self.title = title
        self.subtitle = subtitle
        self.systemImage = systemImage
        self.isAudioTrack = isAudioTrack
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
        public let scoresByDimension: [Int: Float]

        public init(
            chunkID: String,
            span: ArtifactSpan,
            score: Float,
            previewText: String? = nil,
            scoresByDimension: [Int: Float] = [:]
        ) {
            self.chunkID = chunkID
            self.span = span
            self.score = score
            self.previewText = previewText
            self.scoresByDimension = scoresByDimension
        }

        public var isAudioTrackChunk: Bool {
            chunkID.contains("#audio-t=")
        }

        /// Summarizes which modality tower and sampling strategy produced this chunk's embedding.
        public func scoredFeature(for artifact: Artifact) -> ScoredFeatureSummary {
            let samplingBadge: String
            switch artifact.windowSamplingRaw {
            case "content": samplingBadge = "Content Novelty"
            case "fps": samplingBadge = "Fixed 1 fps"
            default: samplingBadge = "Whole"
            }

            switch artifact.kind {
            case .video:
                if isAudioTrackChunk {
                    return ScoredFeatureSummary(
                        title: "Audio Soundtrack",
                        subtitle: "16 kHz Mono WAV · Conformer Tower",
                        systemImage: "waveform",
                        isAudioTrack: true
                    )
                }
                if span == .whole {
                    if artifact.includesAudioTrack {
                        return ScoredFeatureSummary(
                            title: "Whole Video + Audio",
                            subtitle: "Vision (<=140 tokens/f) + 16 kHz Audio",
                            systemImage: "film.stack"
                        )
                    }
                    return ScoredFeatureSummary(
                        title: "Whole Video",
                        subtitle: "Vision Tower (<=140 soft tokens/frame)",
                        systemImage: "film"
                    )
                }
                let frameCount = span.frameTimes.count
                let countLabel = frameCount > 0 ? "\(frameCount) frames" : "32s window"
                return ScoredFeatureSummary(
                    title: "Video Window (\(countLabel))",
                    subtitle: "\(samplingBadge) · Vision Tower",
                    systemImage: "film"
                )

            case .audio:
                let title = (span == .whole) ? "Whole Audio (<=30s)" : "Audio Window (30s)"
                return ScoredFeatureSummary(
                    title: title,
                    subtitle: "16 kHz Mono PCM · Conformer Tower",
                    systemImage: "waveform",
                    isAudioTrack: true
                )

            case .image:
                return ScoredFeatureSummary(
                    title: "Image Patch Grid",
                    subtitle: "Aspect-Preserving Vision (<=280 soft tokens)",
                    systemImage: "photo"
                )

            case .document:
                let hasText = !(previewText ?? "").trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                if hasText {
                    return ScoredFeatureSummary(
                        title: span == .whole ? "Whole Document Text" : "PDF Page Text",
                        subtitle: "title: \(artifact.title) | text: ...",
                        systemImage: "doc.text"
                    )
                }
                return ScoredFeatureSummary(
                    title: "Rendered Page Image",
                    subtitle: "PDFKit Page Raster · Vision Tower",
                    systemImage: "doc.richtext"
                )

            case .code:
                return ScoredFeatureSummary(
                    title: span == .whole ? "Whole Source File" : "Code Line Span",
                    subtitle: "title: \(artifact.title) | text: ...",
                    systemImage: "chevron.left.forwardslash.chevron.right"
                )

            case .text:
                return ScoredFeatureSummary(
                    title: span == .whole ? "Whole Text Note" : "Text Line Span",
                    subtitle: "title: \(artifact.title) | text: ...",
                    systemImage: "doc.plaintext"
                )
            }
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

    public var bestMatch: SpanMatch? {
        spanMatches.first
    }

    public var bestScoredFeature: ScoredFeatureSummary {
        if let first = spanMatches.first {
            return first.scoredFeature(for: artifact)
        }
        return SpanMatch(chunkID: "\(artifact.id)#whole", span: bestSpan, score: bestScore, previewText: previewText)
            .scoredFeature(for: artifact)
    }
}
