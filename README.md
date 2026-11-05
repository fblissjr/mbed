# mbed — Native macOS Multimodal Embedding Finder (`EmbeddingGemma 2`)

`mbed` is a native macOS Swift 6 package and desktop application for indexing and searching multimodal artifacts (video, images, audio, text, code, and PDF/RTF documents) using **MLX `EmbeddingGemma 2`** (`embeddinggemma-2-MLX-BF16`).

## Highlights

- **Extensible Domain & Span Model (`MbedCore`)**:
  - `ArtifactKind`: `.video`, `.image`, `.audio`, `.text`, `.code`, `.document`.
  - `ArtifactSpan`: `.whole`, `.temporal(start:end:frameTimes:)`, `.lines(startLine:endLine:)`, `.pages(startPage:endPage:)`.
  - Parent-artifact deduplication in `VectorStore`: a long video with 10 temporal windows, a 36s audio file with overlapping 30s spans, a PDF with 20 pages, or a source file with 20 line spans appears **once** in search results, annotated with its highest-scoring `bestSpan` and full `spanMatches` list.
- **Matryoshka Representation Learning (MRL) & Task Prompt Parity**:
  - Stores full 768-dimensional vectors once in SQLite (`WAL` mode) and slices/L2-normalizes on the fly to `768`, `512`, `256`, or `128` dimensions via Apple `Accelerate` (`vDSP`).
  - Supports all 10 `EmbeddingGemma 2` task prompts (`query`, `document`, `Retrieval-query`, `Retrieval-document`, `QuestionAnswering`, `FactVerification`, `STS`, `Clustering`, `Classification`, and `InstructionRetrieval` for code retrieval), inline document title formatting (`title: <filename> | text: ...`), and pure-media prompt omission.
- **Multimodal & Temporal Windowing (`ArtifactProcessor`)**:
  - **Video**: `16s`-stride overlapping `32s` windows with fixed `1 fps` (`WindowPlanner.pickFPS`) or duration-scaled content-aware outlier selection (`WindowPlanner.pickContent` on `4 fps` `32x32` RGB candidate frames at exact frame-center timestamps), upright `AVAssetImageGenerator` decoding (`appliesPreferredTrackTransform = true`), per-window representative thumbnails, and opt-in video soundtrack indexing (`--audio-from-video` / `Include Video Soundtrack`).
  - **Audio**: Upfront 16 kHz mono PCM decoding (`AVAssetReader`) with a compact `<= 30s` `#whole` WAV and overlapping `30s` (`15s` stride) temporal windows.
  - **Documents & Code**: `PDFKit` page extraction + page thumbnail rendering (`.pages(startPage:endPage:)`), `NSAttributedString` rich-text extraction, and line-addressed text/code chunking (`.lines(startLine:endLine:)`).
- **Dual MLX Execution Architecture (Self-Contained In-Process MLX Default)**:
  - **Option B (`NativeMLXEmbeddingGemma2` & `MLXSwiftLMEmbeddingGemma2Engine`, default when built with `MLXVLM`)**: Runs 100% in-process on Apple Silicon GPU via `mlx-swift-lm`'s `MLXVLM.EmbeddingGemma2Embedding` and `MLXLMCommon.EmbeddingGemma2` (`./scripts/build_native_mlx.sh`), with on-demand vision/audio component loading (`componentLoading: .onFirstUse`), Metal bicubic resize (`embedding_gemma2_resize_axis`), direct 2D position table gather, and `UserInput.Video.frames(...)` window embedding — no external server required.
  - **Option A (`HeylookMLXEngine`)**: Actor targeting `heylookllm`'s `POST /v1/embeddings`, `GET /v1/models` capability discovery (`fetchCapabilities`), client-side batch planning (`planBatches`: `<= 16` items, `<= 192 MiB` raw media bytes), and request cancellation (`DELETE /v1/requests/{id}`).
- **Native macOS Desktop Frontend (`MbedApp` / `mbed-app`) & CLI (`MbedCLI` / `mbed-cli`)**:
  - **Desktop App (`MbedApp`)**: 3-column SwiftUI `NavigationSplitView`:
    - *Left Sidebar*: Engine switcher (`In-Process MLX` default vs `Heylook HTTP`), **Index Files / Folders…** button, **Stop** button for in-flight indexing cancellation, persisted watched folders with `FSEvents` auto-indexing (`FolderWatcher`) and deleted-file pruning (`pruneMissingArtifacts`), video sampling picker (`1 fps`, `Content (4 fps)`, `Off`), **Include Video Soundtrack** toggle, modality filter, **Query Task** picker, and Matryoshka dimension picker (`768d`, `512d`, `256d`, `128d`).
    - *Center Grid*: Search bar with optional **Attach Example…** (hybrid text + image/video/audio query), drag-and-drop indexing, and deduplicated artifact cards showing top similarity score and matched span badge.
    - *Right Inspector*: Inline `AVPlayerView` for `.video` and `.audio` hits with automatic seeking to the matched window, clickable millisecond `frameTimes` jump pills (`01:04.5`), ranked `spanMatches` list, **Reveal in Finder**, and **Open**.
  - **CLI (`mbed-cli`)**: `status`, `watched`, `inspect-checkpoint`, `index` (`--sampling`, `--audio-from-video`, `--engine`, `--checkpoint`), and `search` (`--dim`, `--kind`, `--task`, `--engine`).

## Commands

```bash
# Build & test with in-process MLX Swift + Metal shader bundle (standalone default)
./scripts/build_native_mlx.sh build
./scripts/build_native_mlx.sh test

# Launch the standalone native SwiftUI macOS desktop app (MbedApp)
./scripts/build_native_mlx.sh app
# (Or open Package.swift in Xcode, select the mbed-app scheme, and press Cmd+R)

# Run the standalone in-process MLX CLI (MbedCLI)
./scripts/build_native_mlx.sh cli status
./scripts/build_native_mlx.sh cli inspect-checkpoint
./scripts/build_native_mlx.sh cli index /path/to/media --sampling content --audio-from-video
./scripts/build_native_mlx.sh cli search "person walking across street" --dim 256 --task query
./scripts/build_native_mlx.sh cli watched

# Lightweight HTTP-only SwiftPM build & test (skips Metal shader compilation)
swift build
swift test
```

## Project Status & Backlog

- [docs/project/CURRENT.md](./docs/project/CURRENT.md): Subsystem grading (all subsystems `SOLID`).
- [docs/project/TODO.md](./docs/project/TODO.md): Completed 5-track backlog and `EmbeddingGemma 2` parity record.
