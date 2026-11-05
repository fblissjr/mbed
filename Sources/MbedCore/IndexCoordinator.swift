import Foundation

/// Coordinates indexing and multimodal search across folders and files using an `EmbeddingEngine`
/// and a persistent `VectorStore`.
public final class IndexCoordinator: Sendable {
    public let engine: any EmbeddingEngine
    public let store: VectorStore

    public init(engine: any EmbeddingEngine, store: VectorStore) {
        self.engine = engine
        self.store = store
    }

    public struct IndexProgress: Sendable {
        public let processed: Int
        public let total: Int
        public let currentTitle: String
        public let skipped: [String]
    }

    /// Recursively scans `urls` (files or directories), skips unchanged files via `ContentFingerprint`,
    /// embeds new or updated media and text/code artifacts at native 768d, and stores them in `VectorStore`.
    public func indexURLs(
        _ urls: [URL],
        collectionID: String = "default",
        videoWindowSampling: VideoWindowSampling = .fps,
        includeNonMedia: Bool = true,
        onProgress: (@Sendable (IndexProgress) -> Void)? = nil
    ) async throws -> (indexed: Int, skipped: [String]) {
        let candidates = discoverFiles(in: urls, includeNonMedia: includeNonMedia)
        var skipped = [String]()
        var indexed = 0

        for (idx, fileURL) in candidates.enumerated() {
            try Task.checkCancellation()
            guard let kind = ArtifactKind.detect(from: fileURL) else { continue }
            guard let fingerprint = Self.fingerprint(for: fileURL) else {
                skipped.append("\(fileURL.lastPathComponent): unreadable file attributes")
                continue
            }

            let artifactID = "\(collectionID)|\(fileURL.standardizedFileURL.path)"
            if (try? store.isUpToDate(artifactID: artifactID, fingerprint: fingerprint)) == true {
                onProgress?(IndexProgress(
                    processed: idx + 1,
                    total: candidates.count,
                    currentTitle: fileURL.lastPathComponent,
                    skipped: skipped
                ))
                continue
            }

            onProgress?(IndexProgress(
                processed: idx,
                total: candidates.count,
                currentTitle: fileURL.lastPathComponent,
                skipped: skipped
            ))

            do {
                let artifact = Artifact(
                    id: artifactID,
                    collectionID: collectionID,
                    kind: kind,
                    url: fileURL,
                    title: fileURL.lastPathComponent,
                    fingerprint: fingerprint
                )
                let chunks = try await buildChunks(
                    for: artifact,
                    videoWindowSampling: videoWindowSampling
                )
                try store.upsert(artifact: artifact, chunks: chunks)
                indexed += 1
            } catch is CancellationError {
                throw CancellationError()
            } catch {
                skipped.append("\(fileURL.lastPathComponent): \(error.localizedDescription)")
            }

            onProgress?(IndexProgress(
                processed: idx + 1,
                total: candidates.count,
                currentTitle: fileURL.lastPathComponent,
                skipped: skipped
            ))
        }

        return (indexed, skipped)
    }

    /// Executes a text, media, or hybrid (text + example file) query against the index.
    public func search(
        queryText: String = "",
        exampleFileURL: URL? = nil,
        dimensions: Int = Matryoshka.gemma2NativeDimensions,
        kinds: Set<ArtifactKind>? = nil,
        collectionID: String? = nil,
        limit: Int = 24
    ) async throws -> [SearchHit] {
        var parts = [EmbeddingPart]()
        let trimmed = queryText.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty {
            parts.append(.text(trimmed))
        }
        if let fileURL = exampleFileURL, let kind = ArtifactKind.detect(from: fileURL) {
            if kind.isMedia {
                let data = try Data(contentsOf: fileURL)
                parts.append(.media(
                    kind: kind,
                    mimeType: ArtifactKind.mimeType(for: fileURL),
                    data: data,
                    windowSampling: .off
                ))
            } else {
                let text = try String(contentsOf: fileURL, encoding: .utf8)
                parts.append(.text(String(text.prefix(8000))))
            }
        }
        guard !parts.isEmpty else { return [] }

        let outputs = try await engine.embedBatch(
            [EmbeddingInput(parts: parts, task: .query)],
            dimensions: engine.nativeDimensions
        )
        guard let queryVec = outputs.first?.vector else { return [] }
        return try store.search(
            queryVector: queryVec,
            dimensions: dimensions,
            kinds: kinds,
            collectionID: collectionID,
            limit: limit
        )
    }

    private func buildChunks(
        for artifact: Artifact,
        videoWindowSampling: VideoWindowSampling
    ) async throws -> [EmbeddedChunk] {
        if artifact.kind.isMedia {
            let data = try Data(contentsOf: artifact.url)
            let mime = ArtifactKind.mimeType(for: artifact.url)
            let winMode: VideoWindowSampling = artifact.kind == .video ? videoWindowSampling : .off
            let input = EmbeddingInput(
                parts: [.media(kind: artifact.kind, mimeType: mime, data: data, windowSampling: winMode)],
                task: .document
            )
            async let thumbTask = ArtifactProcessor.makeThumbnailJPEG(for: artifact.url, kind: artifact.kind)
            let outputs = try await engine.embedBatch([input], dimensions: engine.nativeDimensions)
            let thumb = await thumbTask
            guard let first = outputs.first else { return [] }

            var chunks = [
                EmbeddedChunk(
                    id: "\(artifact.id)#whole",
                    artifactID: artifact.id,
                    span: .whole,
                    vector: first.vector,
                    thumbnailJPEG: thumb,
                    previewText: nil
                )
            ]
            // Only add separate window chunks when there are 2+ windows (so a short single-window
            // video never creates redundant duplicate vectors alongside `#whole`).
            if first.windows.count > 1 {
                for w in first.windows {
                    chunks.append(
                        EmbeddedChunk(
                            id: "\(artifact.id)#t=\(w.start)-\(w.end)",
                            artifactID: artifact.id,
                            span: .temporal(start: w.start, end: w.end, frameTimes: w.frameTimes),
                            vector: w.vector,
                            thumbnailJPEG: thumb,
                            previewText: nil
                        )
                    )
                }
            }
            return chunks
        } else {
            let text = try String(contentsOf: artifact.url, encoding: .utf8)
            let textChunks = ArtifactProcessor.chunkTextDocument(text)
            guard !textChunks.isEmpty else { return [] }
            let inputs = textChunks.map { EmbeddingInput.text($0.text, task: .document) }
            let outputs = try await engine.embedBatch(inputs, dimensions: engine.nativeDimensions)
            return zip(textChunks, outputs).enumerated().map { idx, pair in
                let (spanAndText, out) = pair
                return EmbeddedChunk(
                    id: "\(artifact.id)#chunk-\(idx)",
                    artifactID: artifact.id,
                    span: spanAndText.span,
                    vector: out.vector,
                    thumbnailJPEG: nil,
                    previewText: String(spanAndText.text.prefix(280))
                )
            }
        }
    }

    private func discoverFiles(in urls: [URL], includeNonMedia: Bool) -> [URL] {
        var result = [URL]()
        let fm = FileManager.default
        let keys: [URLResourceKey] = [.isRegularFileKey, .isHiddenKey]

        for root in urls {
            var isDir: ObjCBool = false
            guard fm.fileExists(atPath: root.path, isDirectory: &isDir) else { continue }
            if !isDir.boolValue {
                if let kind = ArtifactKind.detect(from: root), (kind.isMedia || includeNonMedia) {
                    result.append(root)
                }
                continue
            }
            guard let enumerator = fm.enumerator(
                at: root,
                includingPropertiesForKeys: keys,
                options: [.skipsHiddenFiles, .skipsPackageDescendants]
            ) else { continue }
            for case let fileURL as URL in enumerator {
                guard let vals = try? fileURL.resourceValues(forKeys: Set(keys)),
                      vals.isRegularFile == true,
                      let kind = ArtifactKind.detect(from: fileURL),
                      (kind.isMedia || includeNonMedia)
                else { continue }
                result.append(fileURL)
            }
        }
        return result.sorted { $0.path < $1.path }
    }

    public static func fingerprint(for url: URL) -> ContentFingerprint? {
        guard let attrs = try? FileManager.default.attributesOfItem(atPath: url.path),
              let size = attrs[.size] as? NSNumber,
              let mtime = attrs[.modificationDate] as? Date
        else { return nil }
        return ContentFingerprint(
            byteCount: size.int64Value,
            modifiedUnixMs: Int64(mtime.timeIntervalSince1970 * 1000)
        )
    }
}
