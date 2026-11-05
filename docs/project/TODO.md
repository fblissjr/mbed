# Backlog & Incomplete Items (`heylook` -> `mbed`)

This document tracks everything that was left incomplete, half-baked, or stubbed during the initial extraction of Media Finder from `heylookitsanllm` into `mbed`.

---

## Track 1: In-Process `mlx-swift-lm` Engine (`NativeMLXEmbeddingGemma2`)

- [ ] **1.1 Link `mlx-swift` and `mlx-swift-lm` in `Package.swift`**
  - Add local package dependencies on `/Users/fredbliss/mlx/mlx-swift` and `/Users/fredbliss/mlx/mlx-swift-lm` (`MLXVLM`, `MLXLMCommon`, `MLXHuggingFace`) so `#if canImport(MLXVLM) && canImport(MLXLMCommon)` in [`NativeMLXEmbeddingGemma2.swift`](../../Sources/MbedCore/Engine/NativeMLXEmbeddingGemma2.swift) is active in `MbedCore`, `MbedApp`, and `MbedCLI`.
  - Build and run with `xcodebuild` (or SwiftPM + Metal shader bundle) so `MLXVLM`'s custom `embedding_gemma2_resize_axis` Metal bicubic resize kernel compiles into the binary.
- [ ] **1.2 Wire windowed video frame extraction into `MLXSwiftLMEmbeddingGemma2Engine`**
  - Currently [`MLXSwiftLMEmbeddingGemma2Engine.embedBatch`](../../Sources/MbedCore/Engine/NativeMLXEmbeddingGemma2.swift#L180-L221) embeds only the whole video (`vlmParts.append(.video(.url(tempURL)))`) and returns `EmbeddedOutput(vector: primary, windows: [])`.
  - When `windowSampling != .off` (`fps` or `content`):
    1. Decode frames using `AVAssetImageGenerator` (`appliesPreferredTrackTransform = true`).
    2. Run `WindowPlanner.planWindows` (`fps` at 1 fps, or `content` over 4 fps `32x32` luma thumbnails with duration-scaled budget `min(32, max(2, ceil((hi - lo) * 1.0)))`).
    3. For each planned window (when `windows.count > 1`), pass `UserInput.Video.frames([UserInput.VideoFrame])` into `MLXVLM.EmbeddingGemma2Embedding` and populate `EmbeddedOutput.windows`.
- [ ] **1.3 Pass file `URL`s directly in `EmbeddingPart` instead of round-tripping `Data` through `/tmp`**
  - `EmbeddingPart.media` currently holds raw `Data`. For a 200 MB video, `IndexCoordinator` reads the entire file into `Data`, and `MLXSwiftLMEmbeddingGemma2Engine` writes that `Data` back out to a temporary file (`tempURL`) just to hand a `URL` to `MLXVLM`.
  - Add `.mediaURL(kind: ArtifactKind, mimeType: String, url: URL, windowSampling: VideoWindowSampling)` (or carry `url: URL?` on `.media`) so in-process MLX reads the file directly from disk with zero copies, and only `HeylookMLXEngine` reads/base64-encodes `Data` when talking over HTTP.

---

## Track 2: `HeylookMLXEngine` & Client-Side Batching Parity

- [ ] **2.1 Batch `embedBatch` calls to respect `heylook`'s `16`-item & `256 MiB` request limits**
  - In `heylookitsanllm`, `/v1/embeddings` enforces `limits.items = 16` and `limits.request_media_bytes = 256 MiB` (which `frontend/js/embed-kit.js` handled via `planBatches`).
  - In `mbed`, [`IndexCoordinator.buildChunks`](../../Sources/MbedCore/IndexCoordinator.swift#L180-L197) sends all line chunks of a text/code document in a single `engine.embedBatch(inputs)` call. Any file with `> 16` chunks (`> ~810` lines) will fail with HTTP 422 against `heylookllm`.
  - Add batch chunking (`maxItems = 16`, `maxBytes = 192 MiB`) inside `HeylookMLXEngine.embedBatch`.
- [ ] **2.2 Query `GET /v1/models` on connect in `HeylookMLXEngine`**
  - Read `engine.embedding.limits`, `engine.embedding.video_windows`, and `engine.embedding.prompts` from `heylookllm`'s `/v1/models` endpoint instead of assuming static defaults.

---

## Track 3: Video Windowing, Audio Windowing & Soundtrack Extraction

- [ ] **3.1 Overlapping video window stride (`16s` stride on `32s` windows)**
  - [`WindowPlanner.planWindows`](../../Sources/MbedCore/Pipeline/ArtifactProcessor.swift#L127-L162) currently advances `lo += windowSeconds` (`32s` non-overlapping steps: `0–32s`, `32–64s`).
  - Add `strideSeconds: Double = 16.0` (matching the 32s window / 16s stride design) so an action that crosses the 32.0s boundary is captured cleanly inside an overlapping `[16s–48s]` window.
- [ ] **3.2 Live `AVFoundation` 4 fps candidate extractor for `"content"` mode**
  - Implement `ArtifactProcessor.extractCandidateThumbnails(for videoURL: URL, candidateFPS: Double = 4.0) async throws -> (times: [Double], thumbnails: [[Float]], frames: [CIImage])` using `AVAssetReader` or `AVAssetImageGenerator` (`appliesPreferredTrackTransform = true`, downsampled to `32x32` luma) so `WindowPlanner.pickContent` can run natively in Swift.
- [ ] **3.3 Audio windowing (`30s` windows) and video soundtrack indexing**
  - `EmbeddingGemma 2`'s `Gemma4AudioModel` encodes up to 30 seconds of audio per forward, and video blocks only embed visual frames (ignoring the audio track).
  - Slice `.audio` files longer than 30s into `ArtifactSpan.temporal(start:end:frameTimes:)` chunks, and optionally extract + embed the audio track of `.video` files so speech/music queries match video segments too.

---

## Track 4: `VectorStore` Configuration Tracking & Extensible Modalities (`.document` / `.code`)

- [ ] **4.1 Include `video_window_sampling` (and `model_id`) in `VectorStore.isUpToDate`**
  - Currently [`VectorStore.isUpToDate`](../../Sources/MbedCore/Store/VectorStore.swift#L78-L94) only compares `byte_count` and `modified_ms`.
  - If a user indexes a video folder under `Whole Video Only` (`.off`) and then switches to `Fixed FPS (32s)` (`.fps`) or `Content (32s)` (`.content`), re-indexing skips the already-indexed videos instead of generating their window chunks.
  - Store `window_sampling TEXT` on `artifacts` (for `.video` rows) and require it to match in `isUpToDate`.
- [ ] **4.2 PDF & Rich Document Extraction (`PDFKit` -> `ArtifactSpan.pages`)**
  - `ArtifactKind.detect` classifies `.pdf` and `.rtf` as `.document`, and `ArtifactSpan.pages(startPage:endPage:)` is defined, but [`IndexCoordinator.buildChunks`](../../Sources/MbedCore/IndexCoordinator.swift#L180-L197) currently calls `String(contentsOf: artifact.url, encoding: .utf8)` for all non-media files, which fails on binary PDFs.
  - Use macOS `PDFKit` (`PDFDocument(url:)`) in `ArtifactProcessor` to extract per-page text (and/or rendered page thumbnails for scanned PDFs) and emit `.pages(startPage: p, endPage: p)` chunks.
- [ ] **4.3 Use `.codeRetrieval` task prompt for `.code` artifacts**
  - `EmbeddingGemma 2` includes a dedicated `CodeRetrieval` prompt in `config_sentence_transformers.json`. Wire `.code` artifacts to use `.codeRetrieval` when appropriate or expose task-prompt selection.

---

## Track 5: Native macOS App (`MbedApp`) UX Completeness

- [ ] **5.1 Indexing cancellation (`Stop` button)**
  - Store the active indexing `Task<Void, Never>?` on `MbedViewModel` and render a **Stop** button next to the progress indicator in [`MbedApp.swift`](../../Sources/MbedApp/MbedApp.swift) that calls `indexTask?.cancel()`, triggering `HeylookMLXEngine.cancelRemoteRequest` (`DELETE /v1/requests/{id}`) or cooperative in-process cancellation.
- [ ] **5.2 Sampled `frameTimes` jump buttons in `InspectorView`**
  - `ArtifactSpan.temporal` already stores `frameTimes: [Double]`, and `heylook` `#/embeddings` Compare now shows clickable frame-timestamp jump buttons inside its video window preview dialog.
  - Render clickable timestamp pills for `match.span.frameTimes` beneath `MacVideoPlayerView` in [`InspectorView`](../../Sources/MbedApp/MbedApp.swift#L429-L486) so clicking any sampled frame seeks `AVPlayer` directly to that frame.
- [ ] **5.3 Audio player in `InspectorView`**
  - Add an inline `AVPlayerView` (or audio scrubber) when `hit.artifact.kind == .audio` so `.audio` search hits can be played back directly in the inspector.
- [ ] **5.4 Engine switcher (In-Process `mlx-swift-lm` vs `heylookllm` HTTP)**
  - Add a sidebar picker in `MbedApp` and a `--engine native|heylook` flag in `mbed-cli` once Track 1.1 is linked.
- [ ] **5.5 Persisted watch folders & `FSEvents` auto-indexing**
  - Persist added root folder paths in a `collections` table in `mbed.sqlite3`, show them in the sidebar with per-folder refresh/remove actions, and optionally watch them with `FSEventStream` for automatic background re-indexing.
