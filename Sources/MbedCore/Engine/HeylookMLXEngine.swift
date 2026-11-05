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

/// Option A Engine: talks to `heylookllm`'s `/v1/embeddings` endpoint pinned to
/// MLX `embeddinggemma-2-MLX-BF16`, while `mbed` owns native macOS media extraction,
/// duration-scaled windowing, and SQLite/Accelerate Matryoshka indexing.
public struct HeylookMLXEngine: EmbeddingEngine, Sendable {
    public let baseURL: URL
    public let modelID: String
    public let nativeDimensions: Int = Matryoshka.gemma2NativeDimensions
    public let supportedDimensions: [Int] = Matryoshka.gemma2Dimensions
    private let session: URLSession

    public init(
        baseURL: URL = URL(string: "http://127.0.0.1:8000")!,
        modelID: String = "embeddinggemma-2-MLX-BF16",
        session: URLSession = .shared
    ) {
        self.baseURL = baseURL
        self.modelID = modelID
        self.session = session
    }

    public func embedBatch(
        _ inputs: [EmbeddingInput],
        dimensions: Int? = nil
    ) async throws -> [EmbeddedOutput] {
        guard !inputs.isEmpty else { return [] }
        let endpoint = baseURL.appendingPathComponent("v1/embeddings")
        let reqID = "mbed-\(UUID().uuidString.lowercased())"

        let wireItems: [[String: Any]] = inputs.map { item in
            let blocks: [[String: Any]] = item.parts.map { part in
                switch part {
                case let .text(text):
                    return ["type": "text", "text": text]
                case let .media(kind, mimeType, data, windowSampling):
                    var block: [String: Any] = [
                        "type": kind.rawValue,
                        "source": [
                            "type": "base64",
                            "media_type": mimeType,
                            "data": data.base64EncodedString(),
                        ],
                    ]
                    if kind == .video && windowSampling != .off {
                        block["windows"] = ["sampling": windowSampling.rawValue]
                    }
                    return block
                }
            }
            var entry: [String: Any] = ["content": blocks]
            if let promptName = item.task.promptName {
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
