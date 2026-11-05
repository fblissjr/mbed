import Foundation
import MbedCore

@main
struct MbedCLI {
    static func main() async {
        let args = Array(CommandLine.arguments.dropFirst())
        guard let subcommand = args.first else {
            printUsage()
            return
        }

        let serverURL = URL(string: ProcessInfo.processInfo.environment["HEYLOOK_URL"] ?? "http://127.0.0.1:8000")!
        let dbURL = URL(fileURLWithPath: ProcessInfo.processInfo.environment["MBED_DB"] ?? defaultDatabasePath())

        do {
            let engine = makeEngine(args: args, serverURL: serverURL)

            switch subcommand {
            case "status":
                let store = try VectorStore(databaseURL: dbURL)
                let stats = try store.stats()
                let watched = try store.listWatchedFolders()
                print("Database:          \(dbURL.path)")
                print("Indexed artifacts: \(stats.artifactCount)")
                print("Embedded chunks:   \(stats.chunkCount)")
                print("Watched folders:   \(watched.count)")
                print("Model ID:          \(engine.modelID) (\(engine.nativeDimensions)d)")
                print("Native MLX linked: \(NativeMLXEmbeddingGemma2.isAvailableInProcess ? "yes" : "no (build with MBED_ENABLE_NATIVE_MLX=1)")")

            case "watched":
                let store = try VectorStore(databaseURL: dbURL)
                let watched = try store.listWatchedFolders()
                if watched.isEmpty {
                    print("No watched folders.")
                } else {
                    for folder in watched {
                        print("\(folder.url.path) (sampling=\(folder.windowSampling.rawValue))")
                    }
                }

            case "index":
                guard args.count >= 2 else {
                    fputs("Usage: mbed-cli index <path> [--sampling fps|content|off] [--audio-from-video] [--engine heylook|native] [--checkpoint <dir>]\n", stderr)
                    exit(1)
                }
                let targetURL = URL(fileURLWithPath: args[1])
                var sampling: VideoWindowSampling = .fps
                if let idx = args.firstIndex(of: "--sampling"), idx + 1 < args.count {
                    let val = args[idx + 1]
                    sampling = val == "off" ? .off : (VideoWindowSampling(rawValue: val) ?? .fps)
                }
                let loadAudioFromVideo = args.contains("--audio-from-video")
                let store = try VectorStore(databaseURL: dbURL)
                var isDir: ObjCBool = false
                if FileManager.default.fileExists(atPath: targetURL.path, isDirectory: &isDir), isDir.boolValue {
                    try? store.addWatchedFolder(targetURL, windowSampling: sampling)
                }
                let coordinator = IndexCoordinator(engine: engine, store: store)

                let result = try await coordinator.indexURLs(
                    [targetURL],
                    videoWindowSampling: sampling,
                    loadAudioFromVideo: loadAudioFromVideo
                ) { progress in
                    print("[\(progress.processed)/\(progress.total)] \(progress.currentTitle)")
                }
                print("Indexed \(result.indexed) artifacts (\(result.skipped.count) skipped).")

            case "search":
                guard args.count >= 2 else {
                    fputs("Usage: mbed-cli search <query> [--dim 768|512|256|128] [--kind video|image|audio|text|code|document] [--task query|code|qa|fact|sts|clustering|classification] [--engine heylook|native]\n", stderr)
                    exit(1)
                }
                let query = args[1]
                var dim = 768
                if let idx = args.firstIndex(of: "--dim"), idx + 1 < args.count,
                   let parsed = Int(args[idx + 1]) {
                    dim = parsed
                }
                var kinds: Set<ArtifactKind>?
                if let idx = args.firstIndex(of: "--kind"), idx + 1 < args.count,
                   let kind = ArtifactKind(rawValue: args[idx + 1].lowercased()) {
                    kinds = [kind]
                }
                var taskOverride: EmbeddingTask?
                if let idx = args.firstIndex(of: "--task"), idx + 1 < args.count {
                    taskOverride = parseTask(args[idx + 1])
                }
                let store = try VectorStore(databaseURL: dbURL)
                let coordinator = IndexCoordinator(engine: engine, store: store)
                let hits = try await coordinator.search(
                    queryText: query,
                    dimensions: dim,
                    kinds: kinds,
                    taskOverride: taskOverride,
                    limit: 15
                )
                for (idx, hit) in hits.enumerated() {
                    let badge = hit.bestSpan.displayBadge ?? "Whole"
                    print(String(format: "%2d. [%.4f] %@ (%@) -> %@", idx + 1, hit.bestScore, hit.artifact.title, badge, hit.artifact.url.path))
                }

            case "inspect-checkpoint":
                let checkpointURL = args.count >= 2 && !args[1].hasPrefix("--")
                    ? URL(fileURLWithPath: args[1])
                    : NativeMLXEmbeddingGemma2.defaultCheckpointURL
                let desc = try NativeMLXEmbeddingGemma2.CheckpointDescriptor.load(from: checkpointURL)
                print("Checkpoint:  \(checkpointURL.path)")
                print("Model type:  \(desc.modelType)")
                print("Backbone:    \(desc.hiddenSize)d hidden -> \(desc.embeddingDim)d projection (\(desc.numHiddenLayers) layers)")
                print("Vision:      image \(desc.imageMaxSoftTokens) soft tokens, video max \(desc.videoMaxFrames) frames @ \(desc.videoFPS) fps (\(desc.videoMaxSoftTokens) soft tokens/frame)")
                print("Task prompts: \(desc.prompts.keys.sorted().joined(separator: ", "))")

            default:
                printUsage()
            }
        } catch {
            fputs("Error: \(error.localizedDescription)\n", stderr)
            exit(1)
        }
    }

    private static func makeEngine(args: [String], serverURL: URL) -> any EmbeddingEngine {
        var backend: EngineBackend = NativeMLXEmbeddingGemma2.isAvailableInProcess ? .nativeMLX : .heylook
        if let idx = args.firstIndex(of: "--engine"), idx + 1 < args.count {
            let val = args[idx + 1].lowercased()
            if val == "native" || val == "native-mlx" || val == "mlx" {
                backend = .nativeMLX
            } else if val == "heylook" || val == "http" {
                backend = .heylook
            }
        }
        var checkpointURL = NativeMLXEmbeddingGemma2.defaultCheckpointURL
        if let idx = args.firstIndex(of: "--checkpoint"), idx + 1 < args.count {
            checkpointURL = URL(fileURLWithPath: args[idx + 1])
        }

        switch backend {
        case .heylook:
            return HeylookMLXEngine(baseURL: serverURL)
        case .nativeMLX:
            #if canImport(MLXVLM)
            return MLXSwiftLMEmbeddingGemma2Engine(checkpointURL: checkpointURL)
            #else
            fputs("Warning: Native MLX not linked (rebuild with MBED_ENABLE_NATIVE_MLX=1); falling back to heylook HTTP.\n", stderr)
            return HeylookMLXEngine(baseURL: serverURL)
            #endif
        }
    }

    private static func parseTask(_ value: String) -> EmbeddingTask? {
        switch value.lowercased() {
        case "query", "search", "retrieval-query":
            return .query
        case "code", "coderetrieval", "instructionretrieval":
            return .codeRetrieval
        case "qa", "questionanswering":
            return .questionAnswering
        case "fact", "factchecking", "factverification":
            return .factVerification
        case "sts", "similarity":
            return .sts
        case "clustering", "cluster":
            return .clustering
        case "classification", "classify":
            return .classification
        case "document", "doc", "retrieval-document":
            return .document
        case "none", "off":
            return EmbeddingTask.none
        default:
            return EmbeddingTask(rawValue: value)
        }
    }

    private static func defaultDatabasePath() -> String {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let dir = appSupport.appendingPathComponent("mbed", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent("mbed.sqlite3").path
    }

    private static func printUsage() {
        print("""
        mbed-cli — Multimodal EmbeddingGemma 2 CLI
        Usage:
          mbed-cli status [--engine heylook|native]
          mbed-cli watched
          mbed-cli index <file-or-dir> [--sampling fps|content|off] [--audio-from-video] [--engine heylook|native] [--checkpoint <dir>]
          mbed-cli search <query> [--dim 768|512|256|128] [--kind video|image|audio|text|code|document] [--task query|code|qa|fact|sts|clustering|classification] [--engine heylook|native]
          mbed-cli inspect-checkpoint [checkpoint-dir]
        """)
    }
}
