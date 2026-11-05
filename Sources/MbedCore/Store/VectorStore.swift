import Accelerate
import Foundation
import SQLite3

private let SQLITE_TRANSIENT = unsafeBitCast(-1, to: sqlite3_destructor_type.self)

public enum VectorStoreError: LocalizedError, Sendable {
    case openFailed(String)
    case sqlFailed(String)

    public var errorDescription: String? {
        switch self {
        case let .openFailed(msg): return "Cannot open mbed SQLite store: \(msg)"
        case let .sqlFailed(msg): return "SQLite error: \(msg)"
        }
    }
}

/// Persistent SQLite3 + Accelerate (`vDSP`) vector store for `mbed`.
///
/// Key properties:
/// - Stores native 768d `Float32` vectors once per chunk.
/// - Supports on-the-fly Matryoshka dimension slicing (`128`, `256`, `512`, `768`) at query time
///   via `Matryoshka.cosine`, so changing dimensions never requires re-indexing files.
/// - Deduplicates search results by parent `Artifact`, returning one `SearchHit` per file with
///   its highest-scoring span (e.g. `[0:32-1:04]` or `L1-L60`) and all matched sub-spans attached.
public final class VectorStore: @unchecked Sendable {
    private var db: OpaquePointer?
    private let lock = NSLock()

    public init(databaseURL: URL) throws {
        try FileManager.default.createDirectory(
            at: databaseURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        var handle: OpaquePointer?
        if sqlite3_open(databaseURL.path, &handle) != SQLITE_OK {
            let msg = handle.flatMap { String(cString: sqlite3_errmsg($0)) } ?? "unknown"
            sqlite3_close(handle)
            throw VectorStoreError.openFailed(msg)
        }
        self.db = handle
        try execute("""
        PRAGMA journal_mode = WAL;
        PRAGMA synchronous = NORMAL;
        PRAGMA foreign_keys = ON;

        CREATE TABLE IF NOT EXISTS artifacts (
            id TEXT PRIMARY KEY,
            collection_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            url TEXT NOT NULL,
            title TEXT NOT NULL,
            byte_count INTEGER NOT NULL,
            modified_ms INTEGER NOT NULL,
            attributes_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            artifact_id TEXT NOT NULL REFERENCES artifacts(id) ON DELETE CASCADE,
            span_json TEXT NOT NULL,
            vector_blob BLOB NOT NULL,
            thumbnail_jpeg BLOB,
            preview_text TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_chunks_artifact ON chunks(artifact_id);
        CREATE INDEX IF NOT EXISTS idx_artifacts_collection ON artifacts(collection_id);
        """)
    }

    deinit {
        if let db { sqlite3_close(db) }
    }

    /// Returns whether an artifact with the exact same `id` and `fingerprint` is already indexed.
    public func isUpToDate(artifactID: String, fingerprint: ContentFingerprint) throws -> Bool {
        lock.lock()
        defer { lock.unlock() }
        let sql = "SELECT byte_count, modified_ms FROM artifacts WHERE id = ? LIMIT 1;"
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw lastError()
        }
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_text(stmt, 1, artifactID, -1, SQLITE_TRANSIENT)
        if sqlite3_step(stmt) == SQLITE_ROW {
            let bytes = sqlite3_column_int64(stmt, 0)
            let mtime = sqlite3_column_int64(stmt, 1)
            return bytes == fingerprint.byteCount && mtime == fingerprint.modifiedUnixMs
        }
        return false
    }

    /// Atomically upserts an `Artifact` and replaces its `EmbeddedChunk` rows.
    public func upsert(artifact: Artifact, chunks: [EmbeddedChunk]) throws {
        lock.lock()
        defer { lock.unlock() }
        try execute("BEGIN IMMEDIATE TRANSACTION;")
        do {
            try deleteChunksAndArtifactUnlocked(artifactID: artifact.id)
            let attrData = try JSONEncoder().encode(artifact.attributes)
            let attrString = String(data: attrData, encoding: .utf8) ?? "{}"

            let insertArtifact = """
            INSERT INTO artifacts (id, collection_id, kind, url, title, byte_count, modified_ms, attributes_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """
            var aStmt: OpaquePointer?
            guard sqlite3_prepare_v2(db, insertArtifact, -1, &aStmt, nil) == SQLITE_OK else {
                throw lastError()
            }
            defer { sqlite3_finalize(aStmt) }
            sqlite3_bind_text(aStmt, 1, artifact.id, -1, SQLITE_TRANSIENT)
            sqlite3_bind_text(aStmt, 2, artifact.collectionID, -1, SQLITE_TRANSIENT)
            sqlite3_bind_text(aStmt, 3, artifact.kind.rawValue, -1, SQLITE_TRANSIENT)
            sqlite3_bind_text(aStmt, 4, artifact.url.absoluteString, -1, SQLITE_TRANSIENT)
            sqlite3_bind_text(aStmt, 5, artifact.title, -1, SQLITE_TRANSIENT)
            sqlite3_bind_int64(aStmt, 6, artifact.fingerprint.byteCount)
            sqlite3_bind_int64(aStmt, 7, artifact.fingerprint.modifiedUnixMs)
            sqlite3_bind_text(aStmt, 8, attrString, -1, SQLITE_TRANSIENT)
            guard sqlite3_step(aStmt) == SQLITE_DONE else { throw lastError() }

            let insertChunk = """
            INSERT INTO chunks (id, artifact_id, span_json, vector_blob, thumbnail_jpeg, preview_text)
            VALUES (?, ?, ?, ?, ?, ?);
            """
            for chunk in chunks {
                var cStmt: OpaquePointer?
                guard sqlite3_prepare_v2(db, insertChunk, -1, &cStmt, nil) == SQLITE_OK else {
                    throw lastError()
                }
                defer { sqlite3_finalize(cStmt) }
                let spanData = try JSONEncoder().encode(chunk.span)
                let spanJSON = String(data: spanData, encoding: .utf8) ?? "{}"
                let vecData = chunk.vector.withUnsafeBufferPointer { Data(buffer: $0) }

                sqlite3_bind_text(cStmt, 1, chunk.id, -1, SQLITE_TRANSIENT)
                sqlite3_bind_text(cStmt, 2, chunk.artifactID, -1, SQLITE_TRANSIENT)
                sqlite3_bind_text(cStmt, 3, spanJSON, -1, SQLITE_TRANSIENT)
                _ = vecData.withUnsafeBytes { raw in
                    sqlite3_bind_blob(cStmt, 4, raw.baseAddress, Int32(vecData.count), SQLITE_TRANSIENT)
                }
                if let thumb = chunk.thumbnailJPEG {
                    _ = thumb.withUnsafeBytes { raw in
                        sqlite3_bind_blob(cStmt, 5, raw.baseAddress, Int32(thumb.count), SQLITE_TRANSIENT)
                    }
                } else {
                    sqlite3_bind_null(cStmt, 5)
                }
                if let preview = chunk.previewText {
                    sqlite3_bind_text(cStmt, 6, preview, -1, SQLITE_TRANSIENT)
                } else {
                    sqlite3_bind_null(cStmt, 6)
                }
                guard sqlite3_step(cStmt) == SQLITE_DONE else { throw lastError() }
            }

            try execute("COMMIT;")
        } catch {
            _ = try? execute("ROLLBACK;")
            throw error
        }
    }

    /// Vector search with on-the-fly Matryoshka truncation (`dimensions`) and parent-artifact grouping.
    /// Every indexed file appears at most once in the returned `[SearchHit]`, ranked by its best chunk
    /// (whether `.whole` or a specific `.temporal` / `.lines` window), with all sub-span matches attached.
    public func search(
        queryVector: [Float],
        dimensions: Int = Matryoshka.gemma2NativeDimensions,
        kinds: Set<ArtifactKind>? = nil,
        collectionID: String? = nil,
        limit: Int = 24
    ) throws -> [SearchHit] {
        lock.lock()
        defer { lock.unlock() }

        let projectedQuery = Matryoshka.project(queryVector, dimensions: dimensions)
        let sql = """
        SELECT
            a.id, a.collection_id, a.kind, a.url, a.title, a.byte_count, a.modified_ms, a.attributes_json,
            c.id, c.span_json, c.vector_blob, c.thumbnail_jpeg, c.preview_text
        FROM chunks c
        JOIN artifacts a ON a.id = c.artifact_id;
        """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw lastError()
        }
        defer { sqlite3_finalize(stmt) }

        struct GroupedAccumulator {
            let artifact: Artifact
            var bestScore: Float
            var bestSpan: ArtifactSpan
            var spanMatches: [SearchHit.SpanMatch]
            var thumbnailJPEG: Data?
            var previewText: String?
        }

        var groups = [String: GroupedAccumulator]()
        let decoder = JSONDecoder()

        while sqlite3_step(stmt) == SQLITE_ROW {
            let aID = String(cString: sqlite3_column_text(stmt, 0))
            let colID = String(cString: sqlite3_column_text(stmt, 1))
            if let filterCol = collectionID, filterCol != colID { continue }

            let kindRaw = String(cString: sqlite3_column_text(stmt, 2))
            guard let kind = ArtifactKind(rawValue: kindRaw) else { continue }
            if let filterKinds = kinds, !filterKinds.contains(kind) { continue }

            let urlStr = String(cString: sqlite3_column_text(stmt, 3))
            let title = String(cString: sqlite3_column_text(stmt, 4))
            let bytes = sqlite3_column_int64(stmt, 5)
            let mtime = sqlite3_column_int64(stmt, 6)
            let attrStr = String(cString: sqlite3_column_text(stmt, 7))

            let chunkID = String(cString: sqlite3_column_text(stmt, 8))
            let spanStr = String(cString: sqlite3_column_text(stmt, 9))
            let span = (try? decoder.decode(ArtifactSpan.self, from: Data(spanStr.utf8))) ?? .whole

            let blobPtr = sqlite3_column_blob(stmt, 10)
            let blobBytes = Int(sqlite3_column_bytes(stmt, 10))
            let floatCount = blobBytes / MemoryLayout<Float>.size
            guard let blobPtr, floatCount > 0 else { continue }
            let floatPtr = blobPtr.bindMemory(to: Float.self, capacity: floatCount)
            let rawVector = Array(UnsafeBufferPointer(start: floatPtr, count: floatCount))

            let score = Matryoshka.cosine(projectedQuery, rawVector, dimensions: dimensions)

            var thumb: Data?
            if let tPtr = sqlite3_column_blob(stmt, 11) {
                let tLen = Int(sqlite3_column_bytes(stmt, 11))
                if tLen > 0 { thumb = Data(bytes: tPtr, count: tLen) }
            }
            let preview: String? = sqlite3_column_type(stmt, 12) == SQLITE_NULL
                ? nil
                : String(cString: sqlite3_column_text(stmt, 12))

            let match = SearchHit.SpanMatch(chunkID: chunkID, span: span, score: score, previewText: preview)

            if var existing = groups[aID] {
                existing.spanMatches.append(match)
                // Prefer the highest-scoring span; when a temporal/line window ties or beats `.whole`,
                // surface the specific window so the user can jump directly to the timestamp or line.
                if score > existing.bestScore || (abs(score - existing.bestScore) < 1e-4 && existing.bestSpan == .whole && span != .whole) {
                    existing.bestScore = score
                    existing.bestSpan = span
                    if let preview { existing.previewText = preview }
                }
                if existing.thumbnailJPEG == nil, let thumb {
                    existing.thumbnailJPEG = thumb
                }
                groups[aID] = existing
            } else {
                let attrs = (try? decoder.decode([String: String].self, from: Data(attrStr.utf8))) ?? [:]
                let artifact = Artifact(
                    id: aID,
                    collectionID: colID,
                    kind: kind,
                    url: URL(string: urlStr) ?? URL(fileURLWithPath: urlStr),
                    title: title,
                    fingerprint: ContentFingerprint(byteCount: bytes, modifiedUnixMs: mtime),
                    attributes: attrs
                )
                groups[aID] = GroupedAccumulator(
                    artifact: artifact,
                    bestScore: score,
                    bestSpan: span,
                    spanMatches: [match],
                    thumbnailJPEG: thumb,
                    previewText: preview
                )
            }
        }

        return groups.values
            .sorted { $0.bestScore > $1.bestScore }
            .prefix(limit)
            .map { group in
                SearchHit(
                    artifact: group.artifact,
                    bestScore: group.bestScore,
                    bestSpan: group.bestSpan,
                    spanMatches: group.spanMatches.sorted { $0.score > $1.score },
                    thumbnailJPEG: group.thumbnailJPEG,
                    previewText: group.previewText
                )
            }
    }

    public struct StoreStats: Sendable, Equatable {
        public let artifactCount: Int
        public let chunkCount: Int
        public let countsByKind: [ArtifactKind: Int]
    }

    public func stats() throws -> StoreStats {
        lock.lock()
        defer { lock.unlock() }
        var byKind = [ArtifactKind: Int]()
        var totalArtifacts = 0
        var stmt: OpaquePointer?
        if sqlite3_prepare_v2(db, "SELECT kind, COUNT(*) FROM artifacts GROUP BY kind;", -1, &stmt, nil) == SQLITE_OK {
            defer { sqlite3_finalize(stmt) }
            while sqlite3_step(stmt) == SQLITE_ROW {
                let kStr = String(cString: sqlite3_column_text(stmt, 0))
                let count = Int(sqlite3_column_int64(stmt, 1))
                totalArtifacts += count
                if let k = ArtifactKind(rawValue: kStr) {
                    byKind[k] = count
                }
            }
        }
        var totalChunks = 0
        if sqlite3_prepare_v2(db, "SELECT COUNT(*) FROM chunks;", -1, &stmt, nil) == SQLITE_OK {
            defer { sqlite3_finalize(stmt) }
            if sqlite3_step(stmt) == SQLITE_ROW {
                totalChunks = Int(sqlite3_column_int64(stmt, 0))
            }
        }
        return StoreStats(artifactCount: totalArtifacts, chunkCount: totalChunks, countsByKind: byKind)
    }

    public func clearAll() throws {
        lock.lock()
        defer { lock.unlock() }
        try execute("DELETE FROM chunks; DELETE FROM artifacts;")
    }

    private func deleteChunksAndArtifactUnlocked(artifactID: String) throws {
        for sql in ["DELETE FROM chunks WHERE artifact_id = ?;", "DELETE FROM artifacts WHERE id = ?;"] {
            var stmt: OpaquePointer?
            guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { throw lastError() }
            defer { sqlite3_finalize(stmt) }
            sqlite3_bind_text(stmt, 1, artifactID, -1, SQLITE_TRANSIENT)
            guard sqlite3_step(stmt) == SQLITE_DONE else { throw lastError() }
        }
    }

    private func execute(_ sql: String) throws {
        var errMsg: UnsafeMutablePointer<CChar>?
        if sqlite3_exec(db, sql, nil, nil, &errMsg) != SQLITE_OK {
            let msg = errMsg.map { String(cString: $0) } ?? "SQLite error"
            sqlite3_free(errMsg)
            throw VectorStoreError.sqlFailed(msg)
        }
    }

    private func lastError() -> VectorStoreError {
        let msg = db.flatMap { String(cString: sqlite3_errmsg($0)) } ?? "SQLite error"
        return .sqlFailed(msg)
    }
}
