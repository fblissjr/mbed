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
            switch subcommand {
            case "status":
                let store = try VectorStore(databaseURL: dbURL)
                let stats = try store.stats()
                let engine = HeylookMLXEngine(baseURL: serverURL)
                print("Database:          \(dbURL.path)")
                print("Indexed artifacts: \(stats.artifactCount)")
                print("Embedded chunks:   \(stats.chunkCount)")
                print("Model ID:          \(engine.modelID) (\(engine.nativeDimensions)d)")

            case "index":
                guard args.count >= 2 else {
                    fputs("Usage: mbed-cli index <path> [--sampling fps|content|off]\n", stderr)
                    exit(1)
                }
                let targetURL = URL(fileURLWithPath: args[1])
                var sampling: VideoWindowSampling = .fps
                if let idx = args.firstIndex(of: "--sampling"), idx + 1 < args.count {
                    let val = args[idx + 1]
                    sampling = val == "off" ? .off : (VideoWindowSampling(rawValue: val) ?? .fps)
                }
                let store = try VectorStore(databaseURL: dbURL)
                let engine = HeylookMLXEngine(baseURL: serverURL)
                let coordinator = IndexCoordinator(engine: engine, store: store)

                let result = try await coordinator.indexURLs(
                    [targetURL],
                    videoWindowSampling: sampling
                ) { progress in
                    print("[\(progress.processed)/\(progress.total)] \(progress.currentTitle)")
                }
                print("Indexed \(result.indexed) artifacts (\(result.skipped.count) skipped).")

            case "search":
                guard args.count >= 2 else {
                    fputs("Usage: mbed-cli search <query> [--dim 768|512|256|128]\n", stderr)
                    exit(1)
                }
                let query = args[1]
                var dim = 768
                if let idx = args.firstIndex(of: "--dim"), idx + 1 < args.count,
                   let parsed = Int(args[idx + 1]) {
                    dim = parsed
                }
                let store = try VectorStore(databaseURL: dbURL)
                let engine = HeylookMLXEngine(baseURL: serverURL)
                let coordinator = IndexCoordinator(engine: engine, store: store)
                let hits = try await coordinator.search(queryText: query, dimensions: dim, limit: 15)
                for (idx, hit) in hits.enumerated() {
                    let badge = hit.bestSpan.displayBadge ?? "Whole"
                    print(String(format: "%2d. [%.4f] %@ (%@) -> %@", idx + 1, hit.bestScore, hit.artifact.title, badge, hit.artifact.url.path))
                }

            case "inspect-checkpoint":
                let checkpointURL = args.count >= 2
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
          mbed-cli status
          mbed-cli index <file-or-dir> [--sampling fps|content|off]
          mbed-cli search <query> [--dim 768|512|256|128]
          mbed-cli inspect-checkpoint [checkpoint-dir]
        """)
    }
}
