# Backlog & Completed Items (`heylook` -> `mbed`)

This document tracks the 5-track backlog from the extraction of Media Finder from `heylookitsanllm` into `mbed`.

---

## Track 1: In-Process `mlx-swift-lm` Engine (`NativeMLXEmbeddingGemma2`)

- [x] **1.1 Link `mlx-swift` and `mlx-swift-lm` in `Package.swift`**
  - Configured [`Package.swift`](../../Package.swift) to resolve upstream GitHub packages `https://github.com/ml-explore/mlx-swift` (`.upToNextMinor(from: "0.33.0")`), `https://github.com/ml-explore/mlx-swift-lm` (`branch: "main"`), and `https://github.com/huggingface/swift-transformers` (`.upToNextMajor(from: "1.1.0")`) whenever built via [`scripts/build_native_mlx.sh`](../../scripts/build_native_mlx.sh) (`MBED_ENABLE_NATIVE_MLX=1`) or opened in Xcode (`isXcode`), while keeping bare `swift build` / `swift test` fast and offline.
  - Verified full compilation of `mlx-swift_Cmlx.bundle` (`default.metallib` + `embedding_gemma2_resize_axis`) and live GPU inference via [`scripts/build_native_mlx.sh`](../../scripts/build_native_mlx.sh) (`testInProcessMLXSwiftLMEmbeddingGemma2Engine`), and made `.nativeMLX` the default backend in `MbedApp` and `MbedCLI` whenever `MLXVLM` is linked.
- [x] **1.4 Argmax Playground-Style Per-Tower ANE Compute Placement (`ComputePlacement.swift` & `scripts/export_ane_coreml.py`)**
  - Implemented `ModelComputeOptions`, `ANETowerInspector` (`MLComputePlan`), and `ANECoreMLRuntime` in [`ComputePlacement.swift`](../../Sources/MbedCore/Engine/ComputePlacement.swift), plus [`scripts/export_ane_coreml.py`](../../scripts/export_ane_coreml.py) exporting and compiling `EmbeddingGemma2VisionANE.mlmodelc` (**100% ANE**, `1,716/1,716` ops), `EmbeddingGemma2AudioANE.mlmodelc` (**99% ANE**, `1,714/1,738` ops), and `EmbeddingGemma2TextANE.mlmodelc` (**100% ANE**, `2,712/2,712` ops).
- [x] **1.2 Wire windowed video frame extraction into `MLXSwiftLMEmbeddingGemma2Engine`**
  - Updated [`MLXSwiftLMEmbeddingGemma2Engine.embedBatch`](../../Sources/MbedCore/Engine/NativeMLXEmbeddingGemma2.swift) to run `ArtifactProcessor.extractVideoCandidates`, `WindowPlanner.planWindows`, and `ArtifactProcessor.decodeVideoFrames` when `windowSampling != .off`, passing `UserInput.Video.frames([UserInput.VideoFrame])` (and optional aligned soundtrack WAV slices when `loadAudioFromVideo` is enabled) into `MLXVLM.EmbeddingGemma2Embedding` and populating `EmbeddedOutput.windows`.
- [x] **1.3 Pass file `URL`s directly in `EmbeddingPart` instead of round-tripping `Data` through `/tmp`**
  - Added `url: URL?` and `EmbeddingPart.mediaURL(...)` to [`EmbeddingEngine.swift`](../../Sources/MbedCore/Engine/EmbeddingEngine.swift) so in-process MLX reads directly from disk with zero copies and `HeylookMLXEngine` resolves `Data` lazily per sub-batch.

---

## Track 2: `HeylookMLXEngine` & Client-Side Batching Parity

- [x] **2.1 Batch `embedBatch` calls to respect `heylook`'s `16`-item & `256 MiB` request limits**
  - Implemented `HeylookMLXEngine.planBatches(_:maxItems:maxRequestMediaBytes:)` (`16` items, `0.75 * 256 MiB = 192 MiB` raw media bytes) in [`HeylookMLXEngine.swift`](../../Sources/MbedCore/Engine/HeylookMLXEngine.swift) and updated `embedBatch` to execute sub-batches sequentially with cooperative cancellation (`testHeylookPlanBatches`).
- [x] **2.2 Query `GET /v1/models` on connect in `HeylookMLXEngine`**
  - Added `HeylookServerCapabilities` and `fetchCapabilities()` (`GET /v1/models`) to [`HeylookMLXEngine.swift`](../../Sources/MbedCore/Engine/HeylookMLXEngine.swift), and wired `embedBatch` to automatically probe `fetchCapabilities()` once on first use so server-advertised `limits.items` and `limits.request_media_bytes` take effect automatically (`testHeylookMLXEngineHTTPWireAndCapabilities`).

---

## Track 3: Video Windowing, Audio Windowing & Soundtrack Extraction

- [x] **3.1 Overlapping video window stride (`16s` stride on `32s` windows)**
  - Added `strideSeconds: Double? = nil` (defaulting to `windowSeconds / 2.0 = 16.0s`) to `WindowPlanner.planWindows` in [`ArtifactProcessor.swift`](../../Sources/MbedCore/Pipeline/ArtifactProcessor.swift) (`testWindowPlannerLongVideo`).
  - *Note on Option A (`HeylookMLXEngine`)*: When `HeylookMLXEngine` sends a whole video with `video_window_sampling: "fps" | "content"`, windowing happens server-side in `heylookllm` (`src/heylook_llm/providers/video_windows.py`), which currently uses non-overlapping `32s` stride (`lo += seconds`). Local `WindowPlanner.planWindows` (used by Option B and `MbedCore`) uses `16s` overlapping stride.
- [x] **3.2 Live `AVFoundation` candidate extractor for `"content"` mode & per-window thumbnails**
  - Implemented `ArtifactProcessor.extractVideoCandidates` (`32x32` RGB thumbnails at exact frame-center timestamps `(idx + 0.5) / nativeFPS`, with a metadata-only fast path for `.fps`), `ArtifactProcessor.decodeVideoFrames` (`appliesPreferredTrackTransform = true`), and `ArtifactProcessor.makeThumbnailJPEG(..., atSeconds:)` in [`ArtifactProcessor.swift`](../../Sources/MbedCore/Pipeline/ArtifactProcessor.swift) (`testLiveAVFoundationVideoCandidatesAndFrameDecoding`).
- [x] **3.3 Audio windowing (`30s` windows) and video soundtrack indexing**
  - Implemented `ArtifactProcessor.decodeMonoPCM`, `ArtifactProcessor.encodeWAV16Mono`, `ArtifactProcessor.encodeAudioSliceWAV`, and `ArtifactProcessor.extractAudioWindows` (`30s` windows with `15s` stride at `16 kHz`) in [`ArtifactProcessor.swift`](../../Sources/MbedCore/Pipeline/ArtifactProcessor.swift), plus opt-in `loadAudioFromVideo` parity in [`IndexCoordinator.swift`](../../Sources/MbedCore/IndexCoordinator.swift) matching upstream `EmbeddingGemma 2` (`<|video|> <|audio|>` separated by `.text(" ")` + aligned `[w.start, w.end]` soundtrack windows) (`testAudioWindowsAndVideoSoundtrackParity`).

---

## Track 4: `VectorStore` Configuration Tracking & Extensible Modalities (`.document` / `.code`)

- [x] **4.1 Include `video_window_sampling` (and `model_id`) in `VectorStore.isUpToDate`**
  - Added `window_sampling` and `model_id` columns (with idempotent `ALTER TABLE` migrations) to `artifacts` in [`VectorStore.swift`](../../Sources/MbedCore/Store/VectorStore.swift) and required them to match in `isUpToDate` (`testVectorStoreConfigInvalidation`).
- [x] **4.2 PDF & Rich Document Extraction (`PDFKit` -> `ArtifactSpan.pages`)**
  - Implemented `ArtifactProcessor.chunkDocument(at:)` using macOS `PDFKit` (`PDFDocument`) and `NSAttributedString`, emitting `.pages(startPage:endPage:)` spans and rendered page thumbnails (`testPDFChunkingAndCodeRetrievalRouting`).
- [x] **4.3 Use `.codeRetrieval` task prompt for `.code` artifacts & expose query task overrides**
  - Fixed `EmbeddingTask.codeRetrieval.prefixText` to `"task: code retrieval | query: "` (and added `.questionAnswering` and `.factVerification`) in [`EmbeddingEngine.swift`](../../Sources/MbedCore/Engine/EmbeddingEngine.swift), forwarded `title: artifact.title` (`title: <filename> | text: ...`), routed `.code`-filtered searches to `.codeRetrieval`, and added `taskOverride: EmbeddingTask?` to `IndexCoordinator.search`, `MbedApp`, and `mbed-cli search --task` (`testPDFChunkingAndCodeRetrievalRouting`, `testPruneMissingArtifactsAndTaskOverride`).

---

## Track 5: Native macOS App (`MbedApp`) UX Completeness

- [x] **5.1 Indexing cancellation (`Stop` button)**
  - Stored `indexTask: Task<Void, Never>?` on `MbedViewModel` and added a destructive **Stop** button in [`MbedApp.swift`](../../Sources/MbedApp/MbedApp.swift).
- [x] **5.2 Sampled `frameTimes` jump buttons in `InspectorView`**
  - Rendered clickable millisecond timestamp pills (`ArtifactSpan.formatTimestampMillis`) beneath `MacVideoPlayerView` in [`MbedApp.swift`](../../Sources/MbedApp/MbedApp.swift) that seek `AVPlayer` directly to the sampled frame.
- [x] **5.3 Audio player in `InspectorView`**
  - Enabled `MacVideoPlayerView` (`AVPlayerView`) for `.audio` hits in [`MbedApp.swift`](../../Sources/MbedApp/MbedApp.swift) with automatic seeking to `startTimeSeconds`.
- [x] **5.4 Engine switcher (In-Process `mlx-swift-lm` vs `heylookllm` HTTP)**
  - Added an Engine picker, Query Task picker, and `Include Video Soundtrack` toggle in [`MbedApp.swift`](../../Sources/MbedApp/MbedApp.swift) and `--engine heylook|native`, `--checkpoint <dir>`, `--audio-from-video`, and `--task <task>` flags in [`main.swift`](../../Sources/MbedCLI/main.swift).
- [x] **5.5 Persisted watch folders, `FSEvents` auto-indexing & deleted-file pruning**
  - Added `watched_folders` table and `pruneMissingArtifacts(under:collectionID:)` in [`VectorStore.swift`](../../Sources/MbedCore/Store/VectorStore.swift), created [`FolderWatcher.swift`](../../Sources/MbedCore/Pipeline/FolderWatcher.swift) (`FSEventStream`), and wired watched folder management into `MbedApp` and `mbed-cli watched` (`testWatchedFoldersPersistence`, `testPruneMissingArtifactsAndTaskOverride`).
