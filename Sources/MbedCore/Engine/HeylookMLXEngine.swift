import Foundation

public enum EmbeddingEngineError: LocalizedError, Sendable {
    case invalidResponse(status: Int, detail: String)
    case vectorCountMismatch(expected: Int, actual: Int)
    case checkpointNotFound(URL)

    public var errorDescription: String? {
        switch self {
        case let .invalidResponse(status, detail):
            return "Embedding server error (\(status)): \(detail)"
        case let .vectorCountMismatch(expected, actual):
            return "Expected \(expected) vectors from engine, received \(actual)"
        case let .checkpointNotFound(url):
            return "EmbeddingGemma 2 MLX checkpoint not found at \(url.path)"
        }
    }
}

/// Capabilities reported by `heylookllm`'s `GET /v1/models` endpoint for an embedding model.
public struct HeylookServerCapabilities: Codable, Sendable, Equatable {
    public let modelID: String
    public let maxItems: Int
    public let maxRequestMediaBytes: Int
    public let defaultWindowSeconds: Double
    public let videoSamplingModes: [String]
    public let prompts: [String: String]

    public init(
        modelID: String,
        maxItems: Int = 16,
        maxRequestMediaBytes: Int = 256 * 1024 * 1024,
        defaultWindowSeconds: Double = 32.0,
        videoSamplingModes: [String] = ["fps", "content"],
        prompts: [String: String] = [:]
    ) {
        self.modelID = modelID
        self.maxItems = maxItems
        self.maxRequestMediaBytes = maxRequestMediaBytes
        self.defaultWindowSeconds = defaultWindowSeconds
        self.videoSamplingModes = videoSamplingModes
        self.prompts = prompts
    }
}

/// Option A Engine: talks to `heylookllm`'s `/v1/embeddings` endpoint pinned to
/// MLX `embeddinggemma-2-MLX-BF16`, while `mbed` owns native macOS media extraction,
/// duration-scaled windowing, and SQLite/Accelerate Matryoshka indexing.
public actor HeylookMLXEngine: EmbeddingEngine {
    public static let defaultMaxItemsPerRequest: Int = 16
    public static let defaultMaxRequestMediaBytes: Int = 256 * 1024 * 1024

    public nonisolated let baseURL: URL
    public nonisolated let modelID: String
    public nonisolated let nativeDimensions: Int = Matryoshka.gemma2NativeDimensions
    public nonisolated let supportedDimensions: [Int] = Matryoshka.gemma2Dimensions
    public private(set) var maxItemsPerRequest: Int
    public private(set) var maxRequestMediaBytes: Int
    private var didFetchCapabilities: Bool = false
    private let session: URLSession

    public init(
        baseURL: URL = URL(string: "http://127.0.0.1:8000")!,
        modelID: String = "embeddinggemma-2-MLX-BF16",
        maxItemsPerRequest: Int = HeylookMLXEngine.defaultMaxItemsPerRequest,
        maxRequestMediaBytes: Int = HeylookMLXEngine.defaultMaxRequestMediaBytes,
        session: URLSession = .shared
    ) {
        self.baseURL = baseURL
        self.modelID = modelID
        self.maxItemsPerRequest = max(1, maxItemsPerRequest)
        self.maxRequestMediaBytes = max(1, maxRequestMediaBytes)
        self.session = session
    }

    /// Splits a list of `EmbeddingInput` items into bounded sub-batches respecting `heylook`'s
    /// `limits.items` (`16`) and `limits.request_media_bytes` (`256 MiB`, with `0.75` headroom
    /// = `192 MiB` raw media bytes to account for 4/3 base64 expansion and JSON framing),
    /// matching `frontend/js/embed-kit.js` `planBatches`.
    public static func planBatches(
        _ inputs: [EmbeddingInput],
        maxItems: Int = defaultMaxItemsPerRequest,
        maxRequestMediaBytes: Int = defaultMaxRequestMediaBytes
    ) -> [[EmbeddingInput]] {
        guard !inputs.isEmpty else { return [] }
        let capItems = max(1, maxItems)
        let maxRawBytes = max(1, Int(Double(maxRequestMediaBytes) * 0.75))

        var batches = [[EmbeddingInput]]()
        var current = [EmbeddingInput]()
        var currentBytes = 0

        for item in inputs {
            let itemBytes = item.parts.reduce(0) { $0 + $1.estimatedBytes }
            if !current.isEmpty && (current.count >= capItems || currentBytes + itemBytes > maxRawBytes) {
                batches.append(current)
                current = []
                currentBytes = 0
            }
            current.append(item)
            currentBytes += itemBytes
        }
        if !current.isEmpty {
            batches.append(current)
        }
        return batches
    }

    /// Queries `GET /v1/models` on `heylookllm` to discover server-side embedding limits,
    /// video window capabilities, and registered prompt names.
    public func fetchCapabilities() async throws -> HeylookServerCapabilities {
        let url = baseURL.appendingPathComponent("v1/models")
        let (data, response) = try await session.data(from: url)
        let status = (response as? HTTPURLResponse)?.statusCode ?? 200
        guard (200..<300).contains(status) else {
            throw EmbeddingEngineError.invalidResponse(
                status: status,
                detail: String(data: data, encoding: .utf8) ?? "HTTP \(status)"
            )
        }
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let rows = root["data"] as? [[String: Any]]
        else {
            return HeylookServerCapabilities(modelID: modelID)
        }
        let matching = rows.first { ($0["id"] as? String) == modelID && $0["embedding"] != nil }
            ?? rows.first { $0["embedding"] != nil }
        guard let matching,
              let emb = matching["embedding"] as? [String: Any]
        else {
            return HeylookServerCapabilities(modelID: modelID)
        }
        let resolvedID = (matching["id"] as? String) ?? modelID
        let limits = emb["limits"] as? [String: Any] ?? [:]
        let maxItems = (limits["items"] as? Int) ?? Self.defaultMaxItemsPerRequest
        let maxMediaBytes = (limits["request_media_bytes"] as? Int) ?? Self.defaultMaxRequestMediaBytes
        let videoWin = emb["video_windows"] as? [String: Any] ?? [:]
        let winSec = (videoWin["default_seconds"] as? Double)
            ?? Double((videoWin["default_seconds"] as? Int) ?? 32)
        let samplingModes = (videoWin["sampling"] as? [String]) ?? ["fps", "content"]
        let prompts = (emb["prompts"] as? [String: String]) ?? [:]

        let caps = HeylookServerCapabilities(
            modelID: resolvedID,
            maxItems: maxItems,
            maxRequestMediaBytes: maxMediaBytes,
            defaultWindowSeconds: winSec,
            videoSamplingModes: samplingModes,
            prompts: prompts
        )
        self.maxItemsPerRequest = max(1, caps.maxItems)
        self.maxRequestMediaBytes = max(1, caps.maxRequestMediaBytes)
        self.didFetchCapabilities = true
        return caps
    }

    public func embedBatch(
        _ inputs: [EmbeddingInput],
        dimensions: Int? = nil
    ) async throws -> [EmbeddedOutput] {
        guard !inputs.isEmpty else { return [] }
        if !didFetchCapabilities {
            self.didFetchCapabilities = true
            _ = try? await fetchCapabilities()
        }
        let batches = Self.planBatches(
            inputs,
            maxItems: maxItemsPerRequest,
            maxRequestMediaBytes: maxRequestMediaBytes
        )
        guard !batches.isEmpty else { return [] }

        var allOutputs = [EmbeddedOutput]()
        allOutputs.reserveCapacity(inputs.count)
        for batch in batches {
            try Task.checkCancellation()
            let batchOutputs = try await embedSingleBatch(batch, dimensions: dimensions)
            allOutputs.append(contentsOf: batchOutputs)
        }
        return allOutputs
    }

    private func embedSingleBatch(
        _ inputs: [EmbeddingInput],
        dimensions: Int? = nil
    ) async throws -> [EmbeddedOutput] {
        guard !inputs.isEmpty else { return [] }
        let endpoint = baseURL.appendingPathComponent("v1/embeddings")
        let reqID = "mbed-\(UUID().uuidString.lowercased())"

        let wireItems: [[String: Any]] = try inputs.map { item in
            let hasText = item.hasNonWhitespaceText
            let isDocumentWithTitle = hasText
                && (item.task == .document || item.task == .retrievalDocument)
                && (item.title?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false)
            var appliedTitlePrefix = false

            let blocks: [[String: Any]] = try item.parts.map { part in
                switch part {
                case let .text(text):
                    if isDocumentWithTitle && !appliedTitlePrefix
                        && !text.hasPrefix("title:") && !text.hasPrefix("task:") {
                        appliedTitlePrefix = true
                        let formatted = EmbeddingTask.documentPrompt(title: item.title, text: text)
                        return ["type": "text", "text": formatted]
                    }
                    return ["type": "text", "text": text]
                case let .media(kind, mimeType, _, _, windowSampling):
                    let rawData = try part.resolvedData()
                    var block: [String: Any] = [
                        "type": kind.rawValue,
                        "source": [
                            "type": "base64",
                            "media_type": mimeType,
                            "data": rawData.base64EncodedString(),
                        ],
                    ]
                    if kind == .video && windowSampling != .off {
                        block["windows"] = ["sampling": windowSampling.rawValue]
                    }
                    return block
                }
            }
            var entry: [String: Any] = ["content": blocks]
            // Upstream EmbeddingGemma 2 (`processing_embedding_gemma2.py` and `mlx-swift-lm`)
            // only prepends task prompts when the input contains text; pure-media inputs encode
            // bare `<bos><|media|><eos>`. Also omit `prompt_name` when `title: <title> | text: ...`
            // was already formatted inline.
            if hasText, !appliedTitlePrefix, let promptName = item.task.promptName {
                entry["prompt_name"] = promptName
            }
            return entry
        }

        var body: [String: Any] = [
            "model": modelID,
            "dimensions": dimensions ?? nativeDimensions,
            "input": wireItems,
        ]
        if let dimensions {
            body["dimensions"] = dimensions
        }

        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue(reqID, forHTTPHeaderField: "X-Request-ID")
        request.httpBody = try JSONSerialization.data(withJSONObject: body)

        let maxAttempts = 4
        for attempt in 0..<maxAttempts {
            do {
                let (data, response) = try await session.data(for: request)
                let http = response as? HTTPURLResponse
                let status = http?.statusCode ?? 200
                if status == 429 || status == 503 {
                    if attempt + 1 < maxAttempts {
                        try await Task.sleep(nanoseconds: UInt64(1_500_000_000 * (attempt + 1)))
                        continue
                    }
                }
                guard (200..<300).contains(status) else {
                    let detail = (try? JSONDecoder().decode(ErrorEnvelope.self, from: data).detail)
                        ?? String(data: data, encoding: .utf8)
                        ?? "HTTP \(status)"
                    throw EmbeddingEngineError.invalidResponse(status: status, detail: detail)
                }
                let decoded = try JSONDecoder().decode(WireEmbeddingResult.self, from: data)
                guard decoded.data.count == inputs.count else {
                    throw EmbeddingEngineError.vectorCountMismatch(
                        expected: inputs.count,
                        actual: decoded.data.count
                    )
                }
                return decoded.data.sorted(by: { $0.index < $1.index }).map { row in
                    let wins = (row.windows ?? []).map {
                        EmbeddedOutput.WindowVector(
                            start: $0.start,
                            end: $0.end,
                            frameTimes: $0.frame_times,
                            vector: $0.embedding
                        )
                    }
                    return EmbeddedOutput(vector: row.embedding, windows: wins)
                }
            } catch is CancellationError {
                await cancelRemoteRequest(id: reqID)
                throw CancellationError()
            }
        }
        throw EmbeddingEngineError.invalidResponse(status: 503, detail: "Retries exhausted")
    }

    private func cancelRemoteRequest(id: String) async {
        let cancelURL = baseURL
            .appendingPathComponent("v1/requests")
            .appendingPathComponent(id)
        var req = URLRequest(url: cancelURL)
        req.httpMethod = "DELETE"
        _ = try? await session.data(for: req)
    }
}

private struct ErrorEnvelope: Decodable {
    let detail: String?
}

private struct WireEmbeddingResult: Decodable {
    struct WindowRow: Decodable {
        let start: Double
        let end: Double
        let frame_times: [Double]
        let embedding: [Float]
    }

    struct VectorRow: Decodable {
        let index: Int
        let embedding: [Float]
        let windows: [WindowRow]?
    }

    let model: String
    let dimensions: Int
    let data: [VectorRow]
    let input_tokens: Int?
    let elapsed_ms: Double?
}
