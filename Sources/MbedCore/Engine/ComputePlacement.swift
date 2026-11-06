import Accelerate
import CoreGraphics
import CoreML
import Foundation
import ImageIO

/// Compute device target for an individual `EmbeddingGemma 2` tower, inspired by
/// Argmax Playground / WhisperKit's per-encoder `ModelComputeOptions`.
public enum TowerComputeUnit: String, Codable, CaseIterable, Sendable {
    case mlxGPU = "mlx_gpu"
    case cpuAndNeuralEngine = "ane"
    case cpuAndGPU = "coreml_gpu"
    case cpuOnly = "cpu"

    public var label: String {
        switch self {
        case .mlxGPU: return "MLX GPU (Metal)"
        case .cpuAndNeuralEngine: return "Neural Engine (ANE)"
        case .cpuAndGPU: return "CoreML GPU"
        case .cpuOnly: return "CPU Only"
        }
    }

    public var shortBadge: String {
        switch self {
        case .mlxGPU: return "MLX GPU"
        case .cpuAndNeuralEngine: return "ANE"
        case .cpuAndGPU: return "CoreML GPU"
        case .cpuOnly: return "CPU"
        }
    }

    public var systemImage: String {
        switch self {
        case .mlxGPU, .cpuAndGPU: return "gpu"
        case .cpuAndNeuralEngine: return "brain.head.profile"
        case .cpuOnly: return "cpu"
        }
    }

    public var mlComputeUnits: MLComputeUnits {
        switch self {
        case .cpuAndNeuralEngine: return .cpuAndNeuralEngine
        case .cpuAndGPU, .mlxGPU: return .cpuAndGPU
        case .cpuOnly: return .cpuOnly
        }
    }
}

/// Individual sub-models inside multimodal `EmbeddingGemma 2` (`744.1M` total parameters)
/// that can be independently routed to ANE, GPU, or CPU.
public enum EmbeddingComponent: String, Codable, CaseIterable, Sendable, Identifiable {
    case visionEncoder = "vision_tower"
    case audioEncoder = "audio_tower"
    case textBackbone = "language_model"

    public var id: String { rawValue }

    public var title: String {
        switch self {
        case .visionEncoder: return "Vision Encoder"
        case .audioEncoder: return "Audio Encoder"
        case .textBackbone: return "Text & Fusion Backbone"
        }
    }

    public var architectureSummary: String {
        switch self {
        case .visionEncoder:
            return "Gemma4Vision 16L · d=768 · <=280/140 soft tokens"
        case .audioEncoder:
            return "Gemma4Audio Conformer 12L · d=1024 · 30s (750 tokens)"
        case .textBackbone:
            return "Gemma4Text 24L + PLE · d=512 -> 768d Matryoshka"
        }
    }

    public var parameterMillions: Double {
        switch self {
        case .visionEncoder: return 167.7
        case .audioEncoder: return 305.4
        case .textBackbone: return 271.0
        }
    }

    public var estimatedMemoryMB: Int {
        switch self {
        case .visionEncoder: return 335
        case .audioEncoder: return 611
        case .textBackbone: return 542
        }
    }

    public var coreMLBundleName: String {
        switch self {
        case .visionEncoder: return "EmbeddingGemma2VisionANE"
        case .audioEncoder: return "EmbeddingGemma2AudioANE"
        case .textBackbone: return "EmbeddingGemma2TextANE"
        }
    }
}

/// High-level preset for `ModelComputeOptions`.
public enum ComputePreset: String, Codable, CaseIterable, Sendable {
    case allMLXGPU = "all_mlx_gpu"
    case hybridANETowers = "hybrid_ane_towers"
    case allNeuralEngine = "all_ane"
    case custom = "custom"

    public var label: String {
        switch self {
        case .allMLXGPU: return "All MLX GPU (Default)"
        case .hybridANETowers: return "Hybrid (ANE Media + GPU Fusion)"
        case .allNeuralEngine: return "100% Neural Engine (All ANE)"
        case .custom: return "Custom Per-Tower Split"
        }
    }
}

/// Per-component compute assignment for `EmbeddingGemma 2`, modeled after Argmax Playground / WhisperKit.
public struct ModelComputeOptions: Codable, Sendable, Equatable, Hashable {
    public var visionEncoder: TowerComputeUnit
    public var audioEncoder: TowerComputeUnit
    public var textBackbone: TowerComputeUnit

    public init(
        visionEncoder: TowerComputeUnit = .mlxGPU,
        audioEncoder: TowerComputeUnit = .mlxGPU,
        textBackbone: TowerComputeUnit = .mlxGPU
    ) {
        self.visionEncoder = visionEncoder
        self.audioEncoder = audioEncoder
        self.textBackbone = textBackbone
    }

    public static let allMLXGPU = ModelComputeOptions(
        visionEncoder: .mlxGPU,
        audioEncoder: .mlxGPU,
        textBackbone: .mlxGPU
    )

    /// Offloads the heavy Vision (`167.7M`) and Audio (`305.4M`) encoders to the Apple Neural Engine
    /// while keeping the 24-layer dynamic-sequence multimodal fusion backbone (`271.0M`) on MLX GPU.
    public static let hybridANETowers = ModelComputeOptions(
        visionEncoder: .cpuAndNeuralEngine,
        audioEncoder: .cpuAndNeuralEngine,
        textBackbone: .mlxGPU
    )

    /// Routes all three towers (`744.1M` params) to CoreML `.cpuAndNeuralEngine` using enumerated
    /// shape buckets and FP16 pre-scaled RMSNorm (`alpha = 1/16`) for zero-GPU wired memory inference.
    public static let allNeuralEngine = ModelComputeOptions(
        visionEncoder: .cpuAndNeuralEngine,
        audioEncoder: .cpuAndNeuralEngine,
        textBackbone: .cpuAndNeuralEngine
    )

    public var preset: ComputePreset {
        if self == .allMLXGPU { return .allMLXGPU }
        if self == .hybridANETowers { return .hybridANETowers }
        if self == .allNeuralEngine { return .allNeuralEngine }
        return .custom
    }

    public subscript(component: EmbeddingComponent) -> TowerComputeUnit {
        get {
            switch component {
            case .visionEncoder: return visionEncoder
            case .audioEncoder: return audioEncoder
            case .textBackbone: return textBackbone
            }
        }
        set {
            switch component {
            case .visionEncoder: visionEncoder = newValue
            case .audioEncoder: audioEncoder = newValue
            case .textBackbone: textBackbone = newValue
            }
        }
    }

    public var usesANE: Bool {
        visionEncoder == .cpuAndNeuralEngine
            || audioEncoder == .cpuAndNeuralEngine
            || textBackbone == .cpuAndNeuralEngine
    }

    public var isFullANE: Bool {
        visionEncoder == .cpuAndNeuralEngine
            && audioEncoder == .cpuAndNeuralEngine
            && textBackbone == .cpuAndNeuralEngine
    }

    /// Memory breakdown (in MB) across ANE, GPU, and CPU according to the selected tower assignment.
    public var memorySplitMB: (aneMB: Int, gpuMB: Int, cpuMB: Int) {
        var ane = 0
        var gpu = 0
        var cpu = 0
        for comp in EmbeddingComponent.allCases {
            let mb = comp.estimatedMemoryMB
            switch self[comp] {
            case .cpuAndNeuralEngine: ane += mb
            case .mlxGPU, .cpuAndGPU: gpu += mb
            case .cpuOnly: cpu += mb
            }
        }
        return (ane, gpu, cpu)
    }

    public var shortSummaryBadge: String {
        if isFullANE {
            return "100% ANE (Vision · Audio · Text)"
        }
        return "V:\(visionEncoder.shortBadge) · A:\(audioEncoder.shortBadge) · T:\(textBackbone.shortBadge)"
    }
}

/// Inspects CoreML `.mlmodelc` / `.mlpackage` ANE bundles for `EmbeddingGemma 2` and reports
/// operation residency across Neural Engine, GPU, and CPU via `MLComputePlan`.
public enum ANETowerInspector {
    public struct ComponentComputeReport: Sendable, Equatable, Identifiable {
        public var id: String { component.id }
        public let component: EmbeddingComponent
        public let assignedUnit: TowerComputeUnit
        public let bundleURL: URL?
        public let isCompiledBundlePresent: Bool
        public let aneOperationCount: Int
        public let gpuOperationCount: Int
        public let cpuOperationCount: Int

        public var totalOperations: Int {
            aneOperationCount + gpuOperationCount + cpuOperationCount
        }

        public var aneResidencyFraction: Double {
            guard totalOperations > 0 else { return 0 }
            return Double(aneOperationCount) / Double(totalOperations)
        }
    }

    /// Locates `.mlmodelc` or `.mlpackage` for a given `EmbeddingComponent` inside `checkpointDirectory`
    /// (or its `coreml/` subdirectory).
    public static func locateBundle(
        for component: EmbeddingComponent,
        in checkpointDirectory: URL = NativeMLXEmbeddingGemma2.defaultCheckpointURL
    ) -> URL? {
        let fm = FileManager.default
        let baseName = component.coreMLBundleName
        let candidateDirs = [
            checkpointDirectory,
            checkpointDirectory.appendingPathComponent("coreml", isDirectory: true),
            checkpointDirectory.appendingPathComponent("ane", isDirectory: true),
        ]
        for dir in candidateDirs {
            for ext in ["mlmodelc", "mlpackage"] {
                let url = dir.appendingPathComponent("\(baseName).\(ext)")
                if fm.fileExists(atPath: url.path) {
                    return url
                }
            }
        }
        return nil
    }

    /// Synchronously checks which CoreML ANE tower bundles exist on disk for `checkpointDirectory`.
    public static func statusSnapshot(
        options: ModelComputeOptions,
        checkpointDirectory: URL = NativeMLXEmbeddingGemma2.defaultCheckpointURL
    ) -> [ComponentComputeReport] {
        EmbeddingComponent.allCases.map { comp in
            let bundle = locateBundle(for: comp, in: checkpointDirectory)
            return ComponentComputeReport(
                component: comp,
                assignedUnit: options[comp],
                bundleURL: bundle,
                isCompiledBundlePresent: bundle != nil,
                aneOperationCount: 0,
                gpuOperationCount: 0,
                cpuOperationCount: 0
            )
        }
    }

    /// Uses macOS `MLComputePlan` to inspect the hardware device assignment of every operation
    /// in a compiled `.mlmodelc` bundle.
    public static func inspectComputePlan(
        for component: EmbeddingComponent,
        assignedUnit: TowerComputeUnit = .cpuAndNeuralEngine,
        in checkpointDirectory: URL = NativeMLXEmbeddingGemma2.defaultCheckpointURL
    ) async throws -> ComponentComputeReport {
        guard let bundleURL = locateBundle(for: component, in: checkpointDirectory) else {
            return ComponentComputeReport(
                component: component,
                assignedUnit: assignedUnit,
                bundleURL: nil,
                isCompiledBundlePresent: false,
                aneOperationCount: 0,
                gpuOperationCount: 0,
                cpuOperationCount: 0
            )
        }

        let compiledURL: URL
        if bundleURL.pathExtension == "mlpackage" {
            compiledURL = try await MLModel.compileModel(at: bundleURL)
        } else {
            compiledURL = bundleURL
        }

        let config = MLModelConfiguration()
        config.computeUnits = assignedUnit.mlComputeUnits
        let plan = try await MLComputePlan.load(contentsOf: compiledURL, configuration: config)

        var aneOps = 0
        var gpuOps = 0
        var cpuOps = 0

        if case let .program(program) = plan.modelStructure {
            for (_, function) in program.functions {
                for op in function.block.operations {
                    if let usage = plan.deviceUsage(for: op) {
                        switch usage.preferred {
                        case .neuralEngine:
                            aneOps += 1
                        case .gpu:
                            gpuOps += 1
                        case .cpu:
                            cpuOps += 1
                        @unknown default:
                            cpuOps += 1
                        }
                    }
                }
            }
        }

        return ComponentComputeReport(
            component: component,
            assignedUnit: assignedUnit,
            bundleURL: compiledURL,
            isCompiledBundlePresent: true,
            aneOperationCount: aneOps,
            gpuOperationCount: gpuOps,
            cpuOperationCount: cpuOps
        )
    }
}

/// Executes exported CoreML `.mlmodelc` / `.mlpackage` towers (`VisionEncoder`, `AudioEncoder`,
/// `TextBackbone`) on the user-selected `TowerComputeUnit` (`.cpuAndNeuralEngine`, `.cpuAndGPU`,
/// or `.cpuOnly`), enabling both Hybrid (ANE media + MLX GPU fusion) and 100% ANE execution.
public actor ANECoreMLRuntime {
    public static let supportedVisionBuckets: Set<[Int]> = [
        [480, 672],
        [672, 480],
        [528, 528],
        [672, 960],
        [960, 672],
        [768, 768],
    ]
    public static let supportedTextSequenceBuckets: [Int] = [512]
    public static let audioMelFrames: Int = 2999
    public static let audioMelBins: Int = 128
    public static let hiddenSize: Int = 512
    public static let embeddingScale: Float = 22.625 // Float(sqrt(512.0)) rounded to BF16

    private let checkpointDirectory: URL
    private var loadedModels: [String: MLModel] = [:]
    private var mappedTokenTable: Data?

    public init(checkpointDirectory: URL = NativeMLXEmbeddingGemma2.defaultCheckpointURL) {
        self.checkpointDirectory = checkpointDirectory
    }

    private func model(for component: EmbeddingComponent, unit: TowerComputeUnit) async throws -> MLModel? {
        guard unit != .mlxGPU else { return nil }
        let cacheKey = "\(component.rawValue):\(unit.rawValue)"
        if let existing = loadedModels[cacheKey] {
            return existing
        }
        guard let bundleURL = ANETowerInspector.locateBundle(for: component, in: checkpointDirectory) else {
            return nil
        }
        let compiledURL: URL
        if bundleURL.pathExtension == "mlpackage" {
            compiledURL = try await MLModel.compileModel(at: bundleURL)
        } else {
            compiledURL = bundleURL
        }
        let config = MLModelConfiguration()
        config.computeUnits = unit.mlComputeUnits
        let loaded = try MLModel(contentsOf: compiledURL, configuration: config)
        loadedModels[cacheKey] = loaded
        return loaded
    }

    private func tokenEmbeddingTable() -> Data? {
        if let mappedTokenTable { return mappedTokenTable }
        let candidates = [
            checkpointDirectory.appendingPathComponent("coreml/embed_tokens_fp16.bin"),
            checkpointDirectory.appendingPathComponent("ane/embed_tokens_fp16.bin"),
            checkpointDirectory.appendingPathComponent("embed_tokens_fp16.bin"),
        ]
        for url in candidates where FileManager.default.fileExists(atPath: url.path) {
            if let data = try? Data(contentsOf: url, options: .mappedIfSafe) {
                self.mappedTokenTable = data
                return data
            }
        }
        return nil
    }

    /// Runs `EmbeddingGemma2VisionANE` on `[1, 3, height, width]` pixels in `[0, 1]`.
    /// Returns flattened `[softTokenCount * 512]` floats and `softTokenCount`, or `nil` if the
    /// bundle is absent or `(height, width)` is outside the enumerated ANE buckets.
    public func encodeVision(
        pixels: [Float],
        height: Int,
        width: Int,
        unit: TowerComputeUnit
    ) async throws -> (softTokens: [Float], tokenCount: Int)? {
        guard Self.supportedVisionBuckets.contains([height, width]),
              pixels.count == 3 * height * width,
              let visionModel = try await model(for: .visionEncoder, unit: unit)
        else {
            return nil
        }
        let shape: [NSNumber] = [1, 3, NSNumber(value: height), NSNumber(value: width)]
        let inputArray = try MLMultiArray(shape: shape, dataType: .float32)
        let dstPtr = inputArray.dataPointer.bindMemory(to: Float.self, capacity: pixels.count)
        pixels.withUnsafeBufferPointer { src in
            guard let base = src.baseAddress else { return }
            dstPtr.update(from: base, count: pixels.count)
        }
        let provider = try MLDictionaryFeatureProvider(dictionary: ["pixels": inputArray])
        let outFeatures = try Self.runSyncPrediction(model: visionModel, provider: provider)
        guard let outArray = outFeatures.featureValue(for: "soft_tokens")?.multiArrayValue else {
            return nil
        }
        let floats = Self.extractFloatArray(from: outArray)
        let tokenCount = floats.count / Self.hiddenSize
        return (floats, tokenCount)
    }

    /// Selects the closest aspect-preserving `(height, width)` bucket in `supportedVisionBuckets`
    /// and decodes upright sRGB `[1, 3, height, width]` pixels in `[0, 1]`.
    public static func prepareImagePixels(
        from imageURL: URL,
        forVideoBudget: Bool = false
    ) -> (pixels: [Float], height: Int, width: Int)? {
        guard let src = CGImageSourceCreateWithURL(imageURL as CFURL, nil),
              let cgImage = CGImageSourceCreateImageAtIndex(src, 0, nil)
        else {
            return nil
        }
        return prepareImagePixels(from: cgImage, forVideoBudget: forVideoBudget)
    }

    public static func prepareImagePixels(
        from cgImage: CGImage,
        forVideoBudget: Bool = false
    ) -> (pixels: [Float], height: Int, width: Int)? {
        let srcW = max(1, cgImage.width)
        let srcH = max(1, cgImage.height)
        let aspect = Double(srcW) / Double(srcH)

        let (targetH, targetW): (Int, Int)
        if forVideoBudget {
            if aspect > 1.15 {
                (targetH, targetW) = (480, 672)
            } else if aspect < 0.87 {
                (targetH, targetW) = (672, 480)
            } else {
                (targetH, targetW) = (528, 528)
            }
        } else {
            if aspect > 1.15 {
                (targetH, targetW) = (672, 960)
            } else if aspect < 0.87 {
                (targetH, targetW) = (960, 672)
            } else {
                (targetH, targetW) = (768, 768)
            }
        }

        var rgba = [UInt8](repeating: 0, count: targetW * targetH * 4)
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        rgba.withUnsafeMutableBytes { rawPtr in
            guard let base = rawPtr.baseAddress,
                  let ctx = CGContext(
                      data: base,
                      width: targetW,
                      height: targetH,
                      bitsPerComponent: 8,
                      bytesPerRow: targetW * 4,
                      space: colorSpace,
                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
                  )
            else { return }
            ctx.interpolationQuality = .high
            ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: targetW, height: targetH))
        }

        let planeSize = targetH * targetW
        var chw = [Float](repeating: 0, count: 3 * planeSize)
        let inv255: Float = 1.0 / 255.0
        for i in 0..<planeSize {
            chw[i] = Float(rgba[i * 4 + 0]) * inv255
            chw[planeSize + i] = Float(rgba[i * 4 + 1]) * inv255
            chw[2 * planeSize + i] = Float(rgba[i * 4 + 2]) * inv255
        }
        return (chw, targetH, targetW)
    }

    /// Computes `[frameCount, 128]` 16 kHz HTK log-mel spectrogram features matching `Gemma4AudioProcessor`
    /// (`frameLength = 400`, `hopLength = 160`, `fftLength = 512`, `125..7600 Hz`, `preemphasis = 0.97`,
    /// periodic Hann window, `log(mel + 1e-6)`), capped at `30s` (`480,000` samples -> `2,999` frames).
    public static func computeAudioLogMel(
        pcmSamples: [Float]
    ) -> (features: [Float], frameCount: Int)? {
        let maxSamples = 480_000 // 30s at 16 kHz
        let frameLength = 400
        let hopLength = 160
        let fftLength = 512
        let specBins = fftLength / 2 + 1 // 257
        let melBins = Self.audioMelBins  // 128

        let count = min(pcmSamples.count, maxSamples)
        guard count >= frameLength else { return nil }
        let frameCount = min(Self.audioMelFrames, 1 + (count - frameLength) / hopLength)
        guard frameCount > 0 else { return nil }

        // Periodic Hann window of length 400
        var window = [Float](repeating: 0, count: frameLength)
        let twoPiOverN = 2.0 * Float.pi / Float(frameLength)
        for i in 0..<frameLength {
            window[i] = 0.5 - 0.5 * cos(twoPiOverN * Float(i))
        }

        // HTK mel filterbank [specBins=257, melBins=128] over 125..7600 Hz
        func hzToMel(_ f: Float) -> Float { 2595.0 * log10(1.0 + f / 700.0) }
        func melToHz(_ m: Float) -> Float { 700.0 * (pow(10.0, m / 2595.0) - 1.0) }
        let melMin = hzToMel(125.0)
        let melMax = hzToMel(7600.0)
        let edges = (0...(melBins + 1)).map { i -> Float in
            melToHz(melMin + (melMax - melMin) * Float(i) / Float(melBins + 1))
        }
        let binFreqs = (0..<specBins).map { k in Float(k) * 16_000.0 / Float(fftLength) }

        var filters = [Float](repeating: 0, count: specBins * melBins)
        for m in 0..<melBins {
            let fLeft = edges[m]
            let fCenter = edges[m + 1]
            let fRight = edges[m + 2]
            for k in 0..<specBins {
                let f = binFreqs[k]
                if f >= fLeft && f <= fCenter && fCenter > fLeft {
                    filters[k * melBins + m] = (f - fLeft) / (fCenter - fLeft)
                } else if f > fCenter && f <= fRight && fRight > fCenter {
                    filters[k * melBins + m] = (fRight - f) / (fRight - fCenter)
                }
            }
        }

        guard let fftSetup = vDSP_create_fftsetup(vDSP_Length(9), FFTRadix(kFFTRadix2)) else {
            return nil
        }
        defer { vDSP_destroy_fftsetup(fftSetup) }

        var powerSpec = [Float](repeating: 0, count: frameCount * specBins)
        var realPart = [Float](repeating: 0, count: fftLength / 2)
        var imagPart = [Float](repeating: 0, count: fftLength / 2)
        var paddedFrame = [Float](repeating: 0, count: fftLength)

        for frameIdx in 0..<frameCount {
            let start = frameIdx * hopLength
            for i in 0..<frameLength {
                let cur = pcmSamples[start + i]
                let prev = (i == 0) ? cur : pcmSamples[start + i - 1]
                paddedFrame[i] = (cur - 0.97 * prev) * window[i]
            }
            for i in frameLength..<fftLength {
                paddedFrame[i] = 0
            }

            realPart.withUnsafeMutableBufferPointer { rBuf in
                imagPart.withUnsafeMutableBufferPointer { iBuf in
                    var split = DSPSplitComplex(realp: rBuf.baseAddress!, imagp: iBuf.baseAddress!)
                    paddedFrame.withUnsafeBufferPointer { fBuf in
                        fBuf.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: fftLength / 2) { cPtr in
                            vDSP_ctoz(cPtr, 2, &split, 1, vDSP_Length(fftLength / 2))
                        }
                    }
                    vDSP_fft_zrip(fftSetup, &split, 1, vDSP_Length(9), FFTDirection(kFFTDirection_Forward))
                    // Unpack vDSP packed real FFT (scaled by 0.5 inside vDSP_fft_zrip -> multiply by 0.25 for power)
                    let rowBase = frameIdx * specBins
                    powerSpec[rowBase + 0] = (rBuf[0] * rBuf[0]) * 0.25
                    powerSpec[rowBase + fftLength / 2] = (iBuf[0] * iBuf[0]) * 0.25
                    for k in 1..<(fftLength / 2) {
                        powerSpec[rowBase + k] = (rBuf[k] * rBuf[k] + iBuf[k] * iBuf[k]) * 0.25
                    }
                }
            }
        }

        var melOut = [Float](repeating: 0, count: frameCount * melBins)
        vDSP_mmul(
            powerSpec, 1,
            filters, 1,
            &melOut, 1,
            vDSP_Length(frameCount),
            vDSP_Length(melBins),
            vDSP_Length(specBins)
        )
        for i in 0..<melOut.count {
            melOut[i] = log(max(0, melOut[i]) + 1e-6)
        }
        return (melOut, frameCount)
    }

    /// Runs `EmbeddingGemma2AudioANE` on `[frameCount, 128]` log-mel features (padded/capped to `2999` frames).
    public func encodeAudio(
        melFeatures: [Float],
        frameCount: Int,
        unit: TowerComputeUnit
    ) async throws -> (softTokens: [Float], tokenCount: Int)? {
        guard frameCount > 0,
              melFeatures.count == frameCount * Self.audioMelBins,
              let audioModel = try await model(for: .audioEncoder, unit: unit)
        else {
            return nil
        }
        let shape: [NSNumber] = [1, NSNumber(value: Self.audioMelFrames), NSNumber(value: Self.audioMelBins)]
        let inputArray = try MLMultiArray(shape: shape, dataType: .float32)
        let totalFloats = Self.audioMelFrames * Self.audioMelBins
        let dstPtr = inputArray.dataPointer.bindMemory(to: Float.self, capacity: totalFloats)
        dstPtr.initialize(repeating: 0, count: totalFloats)
        let copyFrames = min(frameCount, Self.audioMelFrames)
        let copyCount = copyFrames * Self.audioMelBins
        melFeatures.withUnsafeBufferPointer { src in
            guard let base = src.baseAddress else { return }
            dstPtr.update(from: base, count: copyCount)
        }
        let provider = try MLDictionaryFeatureProvider(dictionary: ["mel_features": inputArray])
        let outFeatures = try Self.runSyncPrediction(model: audioModel, provider: provider)
        guard let outArray = outFeatures.featureValue(for: "soft_tokens")?.multiArrayValue else {
            return nil
        }
        let allFloats = Self.extractFloatArray(from: outArray)
        // Two stride-2 Conv2d layers with pad=1 reduce T -> ((T + 1) / 2 + 1) / 2
        let afterConv0 = (copyFrames + 1) / 2
        let validTokens = min(750, max(1, (afterConv0 + 1) / 2))
        let sliced = Array(allFloats.prefix(validTokens * Self.hiddenSize))
        return (sliced, validTokens)
    }

    /// Runs `EmbeddingGemma2TextANE` on `tokenIDs` (splicing in any `softTokens` at `softTokenIDs`),
    /// returning the L2-normalized embedding truncated to `dimensions`.
    public func encodeTextAndFuse(
        tokenIDs: [Int],
        softTokens: [Float]? = nil,
        softTokenIDs: Set<Int> = [258880, 258884, 258881],
        unit: TowerComputeUnit,
        dimensions: Int = Matryoshka.gemma2NativeDimensions
    ) async throws -> [Float]? {
        guard !tokenIDs.isEmpty,
              let bucketLen = Self.supportedTextSequenceBuckets.first(where: { tokenIDs.count <= $0 }),
              let tokData = tokenEmbeddingTable(),
              let textModel = try await model(for: .textBackbone, unit: unit)
        else {
            return nil
        }

        let packedWidth = Self.hiddenSize + 1 // 512 embeds + 1 attention_mask
        let packedShape: [NSNumber] = [1, NSNumber(value: bucketLen), NSNumber(value: packedWidth)]
        let packedArray = try MLMultiArray(shape: packedShape, dataType: .float32)

        let totalPackedFloats = bucketLen * packedWidth
        let packedPtr = packedArray.dataPointer.bindMemory(to: Float.self, capacity: totalPackedFloats)
        packedPtr.initialize(repeating: 0, count: totalPackedFloats)

        let rowBytes = Self.hiddenSize * MemoryLayout<UInt16>.size
        var softCursor = 0
        let totalSoftRows = (softTokens?.count ?? 0) / Self.hiddenSize

        tokData.withUnsafeBytes { rawTokBuffer in
            guard let fp16Base = rawTokBuffer.baseAddress?.assumingMemoryBound(to: Float16.self) else {
                return
            }
            let maxVocab = rawTokBuffer.count / rowBytes
            for (pos, tokID) in tokenIDs.enumerated() {
                let dstRow = packedPtr.advanced(by: pos * packedWidth)
                dstRow[Self.hiddenSize] = 1.0 // attention_mask = 1.0
                if softTokenIDs.contains(tokID), let softTokens, softCursor < totalSoftRows {
                    let srcOffset = softCursor * Self.hiddenSize
                    for d in 0..<Self.hiddenSize {
                        dstRow[d] = softTokens[srcOffset + d]
                    }
                    softCursor += 1
                } else if tokID >= 0 && tokID < maxVocab {
                    let srcRow = fp16Base.advanced(by: tokID * Self.hiddenSize)
                    for d in 0..<Self.hiddenSize {
                        dstRow[d] = Float(srcRow[d]) * Self.embeddingScale
                    }
                }
            }
        }

        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "packed_input": packedArray
        ])
        let outFeatures = try Self.runSyncPrediction(model: textModel, provider: provider)
        guard let outArray = outFeatures.featureValue(for: "embedding")?.multiArrayValue else {
            return nil
        }
        let rawVec = Self.extractFloatArray(from: outArray)
        return Matryoshka.project(rawVec, dimensions: dimensions)
    }

    nonisolated private static func runSyncPrediction(
        model: MLModel,
        provider: MLDictionaryFeatureProvider
    ) throws -> any MLFeatureProvider {
        try model.prediction(from: provider)
    }

    private static func extractFloatArray(from multiArray: MLMultiArray) -> [Float] {
        let count = multiArray.count
        guard count > 0 else { return [] }
        switch multiArray.dataType {
        case .float32:
            let ptr = multiArray.dataPointer.bindMemory(to: Float.self, capacity: count)
            return Array(UnsafeBufferPointer(start: ptr, count: count))
        case .float16:
            let ptr = multiArray.dataPointer.bindMemory(to: Float16.self, capacity: count)
            var out = [Float](repeating: 0, count: count)
            for i in 0..<count {
                out[i] = Float(ptr[i])
            }
            return out
        default:
            return (0..<count).map { multiArray[$0].floatValue }
        }
    }
}
