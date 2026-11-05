# AGENTS.md

Operating guide for `mbed` — a native macOS Swift package and desktop application for indexing and searching multimodal artifacts (video, images, audio, text, code, and documents) using **MLX `EmbeddingGemma 2`** (`embeddinggemma-2-MLX-BF16`).

## Where to Look First

- [docs/project/CURRENT.md](./docs/project/CURRENT.md): Graded status (`SOLID` vs `HALF-BAKED`) of every subsystem.
- [docs/project/TODO.md](./docs/project/TODO.md): Prioritized backlog and incomplete items from the `heylook` -> `mbed` migration.

## Design Principles

1. **Extensible Artifact & Span Model (`Sources/MbedCore/Domain/Artifact.swift`)**:
   - `mbed` starts with media (video, image, audio) and extends naturally to text, source code, and multi-page documents without schema rewrites.
   - Every indexed item is an `Artifact` (`ArtifactKind`: `.video`, `.image`, `.audio`, `.text`, `.code`, `.document`) with one or more `ArtifactUnit` rows (`ArtifactSpan`: `.whole`, `.temporal(start:end:frameTimes:)`, `.lines(startLine:endLine:)`, `.pages(startPage:endPage:)`).
   - Keep `ArtifactProcessor` and `VectorStore` span-agnostic: new modalities only add a span planner in `ArtifactProcessor` and an `EmbeddingPayload` mapping in `EmbeddingEngine`.
2. **Parent-Artifact Deduplication (`Sources/MbedCore/Store/VectorStore.swift`)**:
   - Search results group by `artifact.id`, never by raw chunk/window row. A 5-minute video with 19 windows or a code file with 30 line spans appears **once** in results, carrying its top-scoring `bestSpan` and ordered `spanMatches`.
3. **Matryoshka Representation Learning (MRL) Without Re-indexing**:
   - `VectorStore` stores full 768-dimensional float32 vectors once in SQLite (`WAL` mode) and slices/L2-normalizes to `768`, `512`, `256`, or `128` dimensions at query time using Apple `Accelerate` (`vDSP`).
4. **MLX `EmbeddingGemma 2` Only (`Sources/MbedCore/Engine/`)**:
   - Stick with MLX `EmbeddingGemma 2` (`embeddinggemma-2-MLX-BF16`, `hidden_size = 512` -> `embedding_dim = 768`, `imageMaxSoftTokens = 280`, `videoMaxSoftTokens = 140`).
   - **Option A (`HeylookMLXEngine`)**: HTTP client targeting `heylookllm`'s `POST /v1/embeddings` and `DELETE /v1/requests/{id}` cancellation endpoint.
   - **Option B (`NativeMLXEmbeddingGemma2` & `MLXSwiftLMEmbeddingGemma2Engine`)**: Direct in-process Swift bridge to Apple's `MLXVLM.EmbeddingGemma2Embedding` (`mlx-swift-lm/Libraries/MLXVLM/EmbeddingGemma2.swift`) and `MLXLMCommon.EmbeddingGemma2`, which loads vision/audio towers on first use (`componentLoading: .onFirstUse`) and uses a direct `take(positionEmbeddingTable[0], xIndices, axis: 0)` gather and Metal bicubic resize kernel.
5. **Duration-Scaled Window Frame Budgets (`Sources/MbedCore/Pipeline/ArtifactProcessor.swift`)**:
   - `WindowPlanner.pickContent` scales its frame budget to the window duration (`min(maxFrames, max(2, ceil((hi - lo) * fps)))`) and falls back to all candidate frames in the window when `< 3` anchors exist, preventing short clips or trailing windows from feeding 4x dense frames into a 1 fps vision processor.
   - All `AVAssetImageGenerator` callers must set `appliesPreferredTrackTransform = true` so portrait iPhone clips are decoded upright.

## Repository Layout

| Path | Role |
| :--- | :--- |
| `Package.swift` | Swift 6.0 package manifest (`macOS(.v14)`): `MbedCore` library, `MbedApp` SwiftUI macOS executable, `MbedCLI` CLI executable, and `MbedCoreTests`. |
| `Sources/MbedCore/Domain/Artifact.swift` | `ArtifactKind`, `ArtifactSpan`, `Artifact`, `ArtifactUnit`, `SpanMatch`, and `ArtifactSearchHit`. |
| `Sources/MbedCore/Engine/EmbeddingEngine.swift` | `EmbeddingEngine` protocol, `EmbeddingTaskRole` (`query`/`document` prefixes), `MatryoshkaDimension`, and `MatryoshkaMath` (`vDSP` L2 norm + cosine). |
| `Sources/MbedCore/Engine/HeylookMLXEngine.swift` | `HeylookMLXEngine` actor talking to `heylookllm` `/v1/embeddings` with request-ID cancellation. |
| `Sources/MbedCore/Engine/NativeMLXEmbeddingGemma2.swift` | `NativeMLXEmbeddingGemma2` checkpoint inspector/planner and `#if canImport(MLXVLM)` `MLXSwiftLMEmbeddingGemma2Engine` actor. |
| `Sources/MbedCore/Pipeline/ArtifactProcessor.swift` | `ArtifactProcessor` and `WindowPlanner` (`fps` default + duration-scaled `content` outlier frame selection, upright thumbnail extraction, and extensible text/code line-span planning). |
| `Sources/MbedCore/Store/VectorStore.swift` | Actor-isolated SQLite (`WAL`) vector store with SHA-256 + mtime incremental change detection, MRL slicing, and parent-artifact deduplication. |
| `Sources/MbedCore/IndexCoordinator.swift` | `@MainActor` coordinator driving folder indexing, cancellation, and hybrid text + example-artifact search. |
| `Sources/MbedApp/MbedApp.swift` | Native SwiftUI macOS app with sidebar controls, artifact grid, and inspector sheet with `AVPlayerView` timestamp markers (`seek(to:)`). |
| `Sources/MbedCLI/main.swift` | CLI for `status`, `inspect-checkpoint`, `index`, and `search`. |
| `Tests/MbedCoreTests/MbedCoreTests.swift` | Unit tests covering `MatryoshkaMath`, `WindowPlanner` short-clip budget invariant, `VectorStore` parent deduplication, incremental text/code span planning, and live `embeddinggemma-2-MLX-BF16` checkpoint inspection. |

## Commands

- Build: `swift build`
- Test: `swift test`
- Inspect checkpoint: `swift run mbed-cli inspect-checkpoint`
- Run CLI: `swift run mbed-cli status | index <dir> | search <query>`
- Run macOS app: `swift run mbed-app`
