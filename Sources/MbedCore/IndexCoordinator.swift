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

    /// Recursively scans `urls` (files or directories), skips unchanged files via `ContentFingerprint`
    /// + `(windowSampling, modelID)`, embeds new or updated media, text, code, and PDF artifacts
    /// at native 768d, and stores them in `VectorStore`.
    public func indexURLs(
        _ urls: [URL],
        collectionID: String = "default",
        videoWindowSampling: VideoWindowSampling = .fps,
        loadAudioFromVideo: Bool = false,
        includeNonMedia: Bool = true,
        onProgress: (@Sendable (IndexProgress) -> Void)? = nil
    ) async throws -> (indexed: Int, skipped: [String]) {
        for root in urls {
            var isDir: ObjCBool = false
            if FileManager.default.fileExists(atPath: root.path, isDirectory: &isDir), isDir.boolValue {
                try? store.addWatchedFolder(
                    root,
                    collectionID: collectionID,
                    windowSampling: videoWindowSampling
                )
                _ = try? store.pruneMissingArtifacts(under: root, collectionID: collectionID)
            }
        }

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

            let effectiveSampling: VideoWindowSampling =
                (kind == .video || kind == .audio) ? videoWindowSampling : .off
            let effectiveModelID: String =
                (kind == .video && loadAudioFromVideo) ? "\(engine.modelID)+audio" : engine.modelID

            let artifactID = "\(collectionID)|\(fileURL.standardizedFileURL.path)"
            if (try? store.isUpToDate(
                artifactID: artifactID,
                fingerprint: fingerprint,
                windowSampling: effectiveSampling,
                modelID: effectiveModelID
            )) == true {
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
                    url: fileURL.standardizedFileURL,
                    title: fileURL.lastPathComponent,
                    fingerprint: fingerprint,
                    windowSamplingRaw: effectiveSampling.rawValue,
                    modelID: effectiveModelID
                )
                let chunks = try await buildChunks(
                    for: artifact,
                    videoWindowSampling: videoWindowSampling,
                    loadAudioFromVideo: loadAudioFromVideo
                )
                guard !chunks.isEmpty else {
                    skipped.append("\(fileURL.lastPathComponent): no embeddable content")
                    continue
                }
                try store.upsert(
                    artifact: artifact,
                    chunks: chunks,
                    windowSampling: effectiveSampling,
                    modelID: effectiveModelID
                )
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
        taskOverride: EmbeddingTask? = nil,
        collectionID: String? = nil,
        limit: Int = 24
    ) async throws -> [SearchHit] {
        var parts = [EmbeddingPart]()
        let trimmed = queryText.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty {
            parts.append(.text(trimmed))
        }
        if let fileURL = exampleFileURL, let kind = ArtifactKind.detect(from: fileURL) {
            switch kind {
            case .image, .video, .audio:
                parts.append(.mediaURL(kind: kind, url: fileURL, windowSampling: .off))
            case .document:
                let plans = try ArtifactProcessor.chunkDocument(at: fileURL)
                if let firstText = plans.first(where: { !$0.text.isEmpty })?.text {
                    parts.append(.text(String(firstText.prefix(8000))))
                } else if let firstThumb = plans.first(where: { $0.thumbnailJPEG != nil })?.thumbnailJPEG {
                    parts.append(.media(kind: .image, mimeType: "image/jpeg", data: firstThumb))
                }
            case .text, .code:
                let text = try String(contentsOf: fileURL, encoding: .utf8)
                parts.append(.text(String(text.prefix(8000))))
            }
        }
        guard !parts.isEmpty else { return [] }

        let queryTask: EmbeddingTask = taskOverride ?? ((kinds == [.code]) ? .codeRetrieval : .query)
        let outputs = try await engine.embedBatch(
            [EmbeddingInput(parts: parts, task: queryTask)],
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
        videoWindowSampling: VideoWindowSampling,
        loadAudioFromVideo: Bool
    ) async throws -> [EmbeddedChunk] {
        switch artifact.kind {
        case .image:
            let input = EmbeddingInput(
                parts: [.mediaURL(kind: .image, url: artifact.url, windowSampling: .off)],
                task: .document
            )
            async let thumbTask = ArtifactProcessor.makeThumbnailJPEG(for: artifact.url, kind: .image)
            let outputs = try await engine.embedBatch([input], dimensions: engine.nativeDimensions)
            let thumb = await thumbTask
            guard let first = outputs.first else { return [] }
            return [
                EmbeddedChunk(
                    id: "\(artifact.id)#whole",
                    artifactID: artifact.id,
                    span: .whole,
                    vector: first.vector,
                    thumbnailJPEG: thumb,
                    previewText: nil
                )
            ]

        case .video:
            var wholeParts: [EmbeddingPart] = [
                .mediaURL(kind: .video, url: artifact.url, windowSampling: videoWindowSampling)
            ]
            var pcmSamples: [Float] = []
            if loadAudioFromVideo {
                pcmSamples = (try? await ArtifactProcessor.decodeMonoPCM(for: artifact.url)) ?? []
                if let wholeWAV = ArtifactProcessor.encodeAudioSliceWAV(
                    samples: pcmSamples,
                    sampleRate: ArtifactProcessor.audioSampleRate,
                    startSeconds: 0.0,
                    endSeconds: ArtifactProcessor.audioWindowSeconds
                ) {
                    // Upstream `EmbeddingGemma2Processor` `load_audio_from_video=True` parity:
                    // join `<|video|>` and `<|audio|>` with a space (`" ".join(...)`).
                    wholeParts.append(.text(" "))
                    wholeParts.append(
                        .media(kind: .audio, mimeType: "audio/wav", data: wholeWAV, windowSampling: .off)
                    )
                }
            }

            async let thumbTask = ArtifactProcessor.makeThumbnailJPEG(for: artifact.url, kind: .video)
            let outputs = try await engine.embedBatch(
                [EmbeddingInput(parts: wholeParts, task: .document)],
                dimensions: engine.nativeDimensions
            )
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
                    let repTime = w.frameTimes.isEmpty
                        ? (w.start + w.end) / 2.0
                        : w.frameTimes[w.frameTimes.count / 2]
                    let winThumb = await ArtifactProcessor.makeThumbnailJPEG(
                        for: artifact.url,
                        kind: .video,
                        atSeconds: repTime
                    ) ?? thumb
                    chunks.append(
                        EmbeddedChunk(
                            id: "\(artifact.id)#t=\(w.start)-\(w.end)",
                            artifactID: artifact.id,
                            span: .temporal(start: w.start, end: w.end, frameTimes: w.frameTimes),
                            vector: w.vector,
                            thumbnailJPEG: winThumb,
                            previewText: nil
                        )
                    )
                }

                // When `loadAudioFromVideo` is enabled, slice the soundtrack along the exact same
                // `[w.start, w.end]` window boundaries (capped at 30s per `Gemma4AudioModel` forward)
                // so speech/music hits highlight the identical video window and sampled frames.
                if loadAudioFromVideo && !pcmSamples.isEmpty {
                    var alignedSlices: [(win: EmbeddedOutput.WindowVector, wav: Data)] = []
                    for w in first.windows {
                        if let wav = ArtifactProcessor.encodeAudioSliceWAV(
                            samples: pcmSamples,
                            sampleRate: ArtifactProcessor.audioSampleRate,
                            startSeconds: w.start,
                            endSeconds: min(w.end, w.start + ArtifactProcessor.audioWindowSeconds)
                        ) {
                            alignedSlices.append((w, wav))
                        }
                    }
                    if !alignedSlices.isEmpty {
                        let audioInputs = alignedSlices.map { pair in
                            EmbeddingInput(
                                parts: [.media(kind: .audio, mimeType: "audio/wav", data: pair.wav, windowSampling: .off)],
                                task: .document
                            )
                        }
                        let audioOutputs = try await engine.embedBatch(audioInputs, dimensions: engine.nativeDimensions)
                        for (pair, out) in zip(alignedSlices, audioOutputs) {
                            let w = pair.win
                            chunks.append(
                                EmbeddedChunk(
                                    id: "\(artifact.id)#audio-t=\(w.start)-\(w.end)",
                                    artifactID: artifact.id,
                                    span: .temporal(start: w.start, end: w.end, frameTimes: w.frameTimes),
                                    vector: out.vector,
                                    thumbnailJPEG: thumb,
                                    previewText: "Audio track (\(ArtifactSpan.formatSeconds(w.start))-\(ArtifactSpan.formatSeconds(w.end)))"
                                )
                            )
                        }
                    }
                }
            }
            return chunks

        case .audio:
            // Decode 16 kHz mono PCM once upfront so `#whole` uses the compact 30s-capped WAV
            // (`<= 960 KB`) rather than base64-encoding an entire multi-minute audio file over HTTP.
            let extracted = try? await ArtifactProcessor.extractAudioWindows(for: artifact.url)
            let wholePart: EmbeddingPart
            if let wholeWAV = extracted?.wholeWAV {
                wholePart = .media(kind: .audio, mimeType: "audio/wav", data: wholeWAV, windowSampling: .off)
            } else {
                wholePart = .mediaURL(kind: .audio, url: artifact.url, windowSampling: .off)
            }
            let wholeInput = EmbeddingInput(parts: [wholePart], task: .document)
            let outputs = try await engine.embedBatch([wholeInput], dimensions: engine.nativeDimensions)
            guard let first = outputs.first else { return [] }

            var chunks = [
                EmbeddedChunk(
                    id: "\(artifact.id)#whole",
                    artifactID: artifact.id,
                    span: .whole,
                    vector: first.vector,
                    thumbnailJPEG: nil,
                    previewText: nil
                )
            ]
            if videoWindowSampling != .off,
               let extracted,
               extracted.windows.count > 1 {
                let winInputs = extracted.windows.map { slice in
                    EmbeddingInput(
                        parts: [.media(kind: .audio, mimeType: "audio/wav", data: slice.wavData, windowSampling: .off)],
                        task: .document
                    )
                }
                let winOutputs = try await engine.embedBatch(winInputs, dimensions: engine.nativeDimensions)
                for (slice, out) in zip(extracted.windows, winOutputs) {
                    chunks.append(
                        EmbeddedChunk(
                            id: "\(artifact.id)#t=\(slice.start)-\(slice.end)",
                            artifactID: artifact.id,
                            span: .temporal(start: slice.start, end: slice.end, frameTimes: []),
                            vector: out.vector,
                            thumbnailJPEG: nil,
                            previewText: nil
                        )
                    )
                }
            }
            return chunks

        case .document:
            let plans = try ArtifactProcessor.chunkDocument(at: artifact.url)
            guard !plans.isEmpty else { return [] }
            let inputs: [EmbeddingInput] = plans.compactMap { plan in
                if !plan.text.isEmpty {
                    return EmbeddingInput.text(plan.text, task: .document, title: artifact.title)
                } else if let pageJPEG = plan.thumbnailJPEG {
                    return EmbeddingInput(
                        parts: [.media(kind: .image, mimeType: "image/jpeg", data: pageJPEG)],
                        task: .document
                    )
                }
                return nil
            }
            guard inputs.count == plans.count else { return [] }
            let outputs = try await engine.embedBatch(inputs, dimensions: engine.nativeDimensions)
            let fallbackThumb = plans.first(where: { $0.thumbnailJPEG != nil })?.thumbnailJPEG
            return zip(plans, outputs).enumerated().map { idx, pair in
                let (plan, out) = pair
                return EmbeddedChunk(
                    id: "\(artifact.id)#doc-\(idx)",
                    artifactID: artifact.id,
                    span: plan.span,
                    vector: out.vector,
                    thumbnailJPEG: plan.thumbnailJPEG ?? fallbackThumb,
                    previewText: plan.text.isEmpty ? nil : String(plan.text.prefix(280))
                )
            }

        case .text, .code:
            let text = try String(contentsOf: artifact.url, encoding: .utf8)
            let textChunks = ArtifactProcessor.chunkTextDocument(text)
            guard !textChunks.isEmpty else { return [] }
            let inputs = textChunks.map { EmbeddingInput.text($0.text, task: .document, title: artifact.title) }
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
