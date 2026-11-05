# mbed — Native macOS Multimodal Embedding Finder (`EmbeddingGemma 2`)

`mbed` is a native macOS Swift package and desktop application for indexing and searching multimodal artifacts (video, images, audio, text, code, and documents) using **MLX `EmbeddingGemma 2`** (`embeddinggemma-2-MLX-BF16`).

## Highlights

- **Extensible Domain Model (`MbedCore`)**:
  - `ArtifactKind`: `.video`, `.image`, `.audio`, `.text`, `.code`, `.document`.
  - `ArtifactSpan`: `.whole`, `.temporal(start:end:frameTimes:)`, `.lines(startLine:endLine:)`, `.pages(startPage:endPage:)`.
  - Parent-artifact deduplication in `VectorStore`: a long video with 10 temporal windows or a source file with 20 line spans appears **once** in search results, annotated with its highest-scoring `bestSpan` and full `spanMatches` list.
- **Matryoshka Representation Learning (MRL)**:
  - Stores full 768-dimensional vectors once in SQLite (`WAL` mode) and slices/L2-normalizes on the fly to `768`, `512`, `256`, or `128` dimensions via Apple `Accelerate` (`vDSP`).
- **Dual MLX Execution Architecture**:
  - **Option A (`HeylookMLXEngine`)**: Connects to `heylookllm`'s `/v1/embeddings` endpoint (`embeddinggemma-2-MLX-BF16`), with native `AVFoundation`/`ImageIO` display-transform-aware thumbnails, duration-scaled content-aware frame picking (`WindowPlanner`), and request cancellation (`DELETE /v1/requests/{id}`).
  - **Option B (`NativeMLXEmbeddingGemma2` & `MLXSwiftLMEmbeddingGemma2Engine`)**: Bridges directly to `/Users/fredbliss/mlx/mlx-swift-lm`'s first-party `MLXVLM.EmbeddingGemma2Embedding` (`Libraries/MLXVLM/EmbeddingGemma2.swift`) and `MLXLMCommon.EmbeddingGemma2` (`Libraries/MLXLMCommon/Models/EmbeddingGemma2.swift` + `Gemma4Encoders.swift`), which includes on-demand vision/audio component loading (`componentLoading: .onFirstUse`), a Metal bicubic resize kernel (`embedding_gemma2_resize_axis`), direct 2D position table gather (`take(positionEmbeddingTable[0], xIndices, axis: 0)`), and `UserInput.Video.frames(...)` support for window-level frame sets.

## Commands

```bash
# Build all targets (MbedCore, MbedApp, MbedCLI)
swift build

# Run the unit test suite
swift test

# Inspect the local EmbeddingGemma 2 MLX checkpoint
swift run mbed-cli inspect-checkpoint /Users/fredbliss/Storage/llms/google/embeddinggemma-2-MLX-BF16

# Run the CLI against heylookllm
swift run mbed-cli status
swift run mbed-cli index /path/to/media --sampling fps
swift run mbed-cli search "person walking across street" --dim 256

# Launch the native SwiftUI macOS desktop app
swift run mbed-app
```

## Project Status & Backlog

- [docs/project/CURRENT.md](./docs/project/CURRENT.md): Subsystem grading (`SOLID` vs `HALF-BAKED`).
- [docs/project/TODO.md](./docs/project/TODO.md): Remaining tasks from the `heylook` -> `mbed` extraction.
