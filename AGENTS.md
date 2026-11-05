# AGENTS.md

Operating guide for `mbed` — a native macOS Swift 6 package and desktop application for indexing and searching multimodal artifacts (video, images, audio, text, code, and documents) using **MLX `EmbeddingGemma 2`** (`embeddinggemma-2-MLX-BF16`).

## Where to Look First

- [docs/project/CURRENT.md](./docs/project/CURRENT.md): Graded status (`SOLID` vs `HALF-BAKED`) of every subsystem.
- [docs/project/TODO.md](./docs/project/TODO.md): 5-track backlog and `EmbeddingGemma 2` parity checklist from the `heylook` -> `mbed` migration.

## Design Principles

1. **Extensible Artifact & Span Model (`Sources/MbedCore/Domain/Artifact.swift`)**:
   - `mbed` starts with media (video, image, audio) and extends naturally to text, source code, and multi-page documents without schema rewrites.
   - Every indexed item is an `Artifact` (`ArtifactKind`: `.video`, `.image`, `.audio`, `.text`, `.code`, `.document`) with one or more `EmbeddedChunk` rows (`ArtifactSpan`: `.whole`, `.temporal(start:end:frameTimes:)`, `.lines(startLine:endLine:)`, `.pages(startPage:endPage:)`).
   - Keep `ArtifactProcessor` and `VectorStore` span-agnostic: new modalities only add a span planner in `ArtifactProcessor` and an `EmbeddingPart` mapping in `EmbeddingEngine`.
2. **Parent-Artifact Deduplication (`Sources/MbedCore/Store/VectorStore.swift`)**:
   - Search results group by `artifact.id`, never by raw chunk/window row. A 5-minute video with 19 windows or a code file with 30 line spans appears **once** in results, carrying its top-scoring `bestSpan` and ordered `spanMatches`.
3. **Matryoshka Representation Learning (MRL) Without Re-indexing**:
   - `VectorStore` stores full 768-dimensional float32 vectors once in SQLite (`WAL` mode) and slices/L2-normalizes to `768`, `512`, `256`, or `128` dimensions at query time using Apple `Accelerate` (`vDSP`).
   - `VectorStore.isUpToDate` tracks `(byte_count, modified_ms, window_sampling, model_id)` so changing video sampling (`fps` vs `content`), toggling `loadAudioFromVideo`, or switching checkpoints automatically invalidates stale embeddings.
4. **MLX `EmbeddingGemma 2` Only (`Sources/MbedCore/Engine/`)**:
   - Stick with MLX `EmbeddingGemma 2` (`embeddinggemma-2-MLX-BF16`, `hidden_size = 512` -> `embedding_dim = 768`, `imageMaxSoftTokens = 280`, `videoMaxSoftTokens = 140`).
   - Maintain exact prompt parity with `EmbeddingGemma2Processor` / `sentence-transformers`: support all 10 task prompts (`EmbeddingTask`), format titled documents inline as `"title: <filename> | text: <content>"`, and never attach text task prefixes to pure-media inputs (`!item.hasNonWhitespaceText`).
   - **Option A (`HeylookMLXEngine`)**: Actor HTTP client targeting `heylookllm`'s `POST /v1/embeddings`, `GET /v1/models` (`fetchCapabilities`), client-side sub-batching (`planBatches`: `<= 16` items, `<= 192 MiB` raw media bytes), and `DELETE /v1/requests/{id}` cancellation.
   - **Option B (`NativeMLXEmbeddingGemma2` & `MLXSwiftLMEmbeddingGemma2Engine`)**: Direct in-process Swift bridge to Apple's `MLXVLM.EmbeddingGemma2Embedding` (`mlx-swift-lm/Libraries/MLXVLM/EmbeddingGemma2.swift`) and `MLXLMCommon.EmbeddingGemma2` (gated behind `MBED_ENABLE_NATIVE_MLX=1` in `Package.swift`), which loads vision/audio towers on first use (`componentLoading: .onFirstUse`) and uses a direct `take(positionEmbeddingTable[0], xIndices, axis: 0)` gather and Metal bicubic resize kernel.
5. **Duration-Scaled Window Frame Budgets & Audio Windowing (`Sources/MbedCore/Pipeline/ArtifactProcessor.swift`)**:
   - `WindowPlanner.pickContent` scales its frame budget to the window duration (`min(maxFrames, max(2, ceil((hi - lo) * fps)))`) and falls back to all candidate frames in the window when `< 3` anchors exist, preventing short clips or trailing windows from feeding 4x dense frames into a 1 fps vision processor.
   - `ArtifactProcessor.extractVideoCandidates` uses `32x32` RGB thumbnails at exact frame-center timestamps `(Double(frameIdx) + 0.5) / nativeFPS` (and a metadata-only fast path when `sampling == .fps`).
   - All `AVAssetImageGenerator` callers must set `appliesPreferredTrackTransform = true` so portrait iPhone clips are decoded upright.
   - `.audio` artifacts decode 16 kHz mono PCM upfront (`decodeMonoPCM`) and slice `30s` windows with `15s` stride (`extractAudioWindows`, `encodeAudioSliceWAV`).

## Repository Layout

| Path | Role |
| :--- | :--- |
| `Package.swift` | Swift 6.0 package manifest (`macOS(.v14)`): `MbedCore` library, `MbedApp` SwiftUI macOS executable, `MbedCLI` CLI executable, and `MbedCoreTests` (with optional `MBED_ENABLE_NATIVE_MLX=1` linking of `mlx-swift` and `mlx-swift-lm`). |
| `scripts/build_native_mlx.sh` | Helper script verifying `mlx-swift` submodules and building `mbed` with `MBED_ENABLE_NATIVE_MLX=1` via `xcodebuild` (so the Metal shader bundle `default.metallib` is compiled). |
| `Sources/MbedCore/Domain/Artifact.swift` | `ArtifactKind`, `ArtifactSpan`, `Artifact`, `EmbeddedChunk`, `ContentFingerprint`, and `SearchHit`. |
| `Sources/MbedCore/Engine/EmbeddingEngine.swift` | `EmbeddingEngine` protocol, `EmbeddingTask` (all 10 `EmbeddingGemma 2` prompts + `documentPrompt(title:text:)`), `EngineBackend`, `VideoWindowSampling`, zero-copy `EmbeddingPart.mediaURL`, and `Matryoshka` (`vDSP` L2 norm + cosine). |
| `Sources/MbedCore/Engine/HeylookMLXEngine.swift` | `HeylookMLXEngine` actor talking to `heylookllm` `/v1/embeddings` and `/v1/models` (`fetchCapabilities`), with `planBatches` (`<= 16` items, `<= 192 MiB` media) and request-ID cancellation. |
| `Sources/MbedCore/Engine/NativeMLXEmbeddingGemma2.swift` | `NativeMLXEmbeddingGemma2` checkpoint inspector/planner and `#if canImport(MLXVLM)` `MLXSwiftLMEmbeddingGemma2Engine` actor. |
| `Sources/MbedCore/Pipeline/ArtifactProcessor.swift` | `ArtifactProcessor` and `WindowPlanner` (`16s`-stride overlapping `32s` video windows, `32x32` RGB candidate extractor, upright frame decoding, per-window thumbnails, 16 kHz 30s/15s audio windowing, `PDFKit` page chunking, and text/code line-span planning). |
| `Sources/MbedCore/Pipeline/FolderWatcher.swift` | `FSEventStream` directory watcher (`FolderWatcher`) for auto-indexing watched folders on file changes. |
| `Sources/MbedCore/Store/VectorStore.swift` | SQLite (`WAL`) vector store with `(byte_count, modified_ms, window_sampling, model_id)` invalidation, MRL slicing, parent-artifact deduplication, `watched_folders` persistence, and `pruneMissingArtifacts`. |
| `Sources/MbedCore/IndexCoordinator.swift` | Coordinator driving folder indexing, deleted-file pruning, opt-in video soundtrack indexing (`loadAudioFromVideo`), and hybrid text + example-artifact search with `taskOverride`. |
| `Sources/MbedApp/MbedApp.swift` | Native SwiftUI macOS app with engine switcher, query task picker, **Stop** button, watched folders, artifact grid, and inspector with `AVPlayerView` for video/audio and clickable `frameTimes` jump pills. |
| `Sources/MbedCLI/main.swift` | CLI for `status`, `inspect-checkpoint`, `index`, `search` (`--task`, `--dim`, `--engine`), and `watched`. |
| `Tests/MbedCoreTests/MbedCoreTests.swift` | 14 unit tests covering `Matryoshka`, `WindowPlanner`, live `AVAssetWriter` MP4 candidate extraction & frame decoding, live 16 kHz WAV synthesis & 30s audio windowing, `PDFKit` page chunking, `HeylookMLXEngine` batching & HTTP wire parity, `VectorStore` invalidation & pruning, `FolderWatcher`, and `CheckpointDescriptor`. |

## Commands

- Build: `swift build`
- Test: `swift test`
- Inspect checkpoint: `swift run mbed-cli inspect-checkpoint`
- Run CLI: `swift run mbed-cli status | index <dir> | search <query> | watched`
- Run macOS app: `swift run mbed-app`
- Build with in-process MLX Swift + Metal shaders: `./scripts/build_native_mlx.sh`
