import Accelerate
import AppKit
@preconcurrency import AVFoundation
import CoreGraphics
import Foundation
import ImageIO
import PDFKit
import UniformTypeIdentifiers

/// Duration-scaled video window frame planner (`fps` default vs `content` novelty).
/// Fixes the short-clip over-sampling bug by scaling each window's frame budget
/// to `min(maxFrames, max(2, ceil((hi - lo) * fps)))`.
public enum WindowPlanner {
    public static let candidateFPS: Double = 4.0
    public static let suppressionRadius: Int = 1

    public struct PlannedWindow: Sendable, Equatable {
        public let start: Double
        public let end: Double
        public let selectedIndices: [Int]
        public let frameTimes: [Double]

        public init(start: Double, end: Double, selectedIndices: [Int], frameTimes: [Double]) {
            self.start = start
            self.end = end
            self.selectedIndices = selectedIndices
            self.frameTimes = frameTimes
        }
    }

    /// Fixed-rate (`1/fps`) frame selection inside `[lo, hi)`.
    public static func pickFPS(
        times: [Double],
        lo: Double,
        hi: Double,
        fps: Double = 1.0,
        maxFrames: Int = 32
    ) -> [Int] {
        guard !times.isEmpty, fps > 0, hi > lo else { return [] }
        let step = 1.0 / fps
        var chosen = Set<Int>()
        var t = lo
        var count = 0
        while t < hi && count < maxFrames {
            var bestIdx = 0
            var bestDist = Double.infinity
            for (i, time) in times.enumerated() {
                let d = abs(time - t)
                if d < bestDist {
                    bestDist = d
                    bestIdx = i
                }
            }
            chosen.insert(bestIdx)
            t += step
            count += 1
        }
        return chosen.sorted()
    }

    /// Content-aware frame selection inside `[lo, hi)`:
    /// - Scales the frame budget to `min(maxFrames, max(2, ceil((hi - lo) * fps)))` so short clips
    ///   never dump all 4 fps candidates into the vision tower.
    /// - Keeps half the budget as evenly spaced anchors and fills the rest with candidates
    ///   having the highest L1 novelty from the anchor median.
    public static func pickContent(
        thumbnails: [[Float]],
        times: [Double],
        lo: Double,
        hi: Double,
        fps: Double = 1.0,
        maxFrames: Int = 32
    ) -> [Int] {
        let inWindow = times.enumerated().compactMap { i, t in
            (t >= lo && t < hi && i < thumbnails.count) ? i : nil
        }
        guard !inWindow.isEmpty else { return [] }
        let budget = min(maxFrames, max(2, Int(ceil((hi - lo) * fps))))
        if inWindow.count <= budget {
            return inWindow
        }

        let anchorCount = max(1, budget / 2)
        var anchors = [Int]()
        if anchorCount == 1 {
            anchors = [0]
        } else {
            for k in 0..<anchorCount {
                let pos = Double(k) * Double(inWindow.count - 1) / Double(anchorCount - 1)
                anchors.append(Int(pos.rounded()))
            }
        }

        let pixelCount = thumbnails[inWindow[0]].count
        let referencePool = anchors.count >= 3 ? anchors.map { thumbnails[inWindow[$0]] } : inWindow.map { thumbnails[$0] }
        var background = [Float](repeating: 0, count: pixelCount)
        for p in 0..<pixelCount {
            var col = referencePool.map { $0[p] }
            col.sort()
            background[p] = col[col.count / 2]
        }

        var novelty = [(offset: Int, score: Float)]()
        novelty.reserveCapacity(inWindow.count)
        for (offset, globalIdx) in inWindow.enumerated() {
            let thumb = thumbnails[globalIdx]
            var diffSum: Float = 0
            for p in 0..<pixelCount {
                diffSum += abs(thumb[p] - background[p])
            }
            novelty.append((offset, diffSum / Float(max(1, pixelCount))))
        }
        novelty.sort { $0.score > $1.score }

        var chosen = Set(anchors)
        var blocked = Set<Int>()
        for item in novelty {
            if chosen.count >= budget { break }
            if chosen.contains(item.offset) || blocked.contains(item.offset) { continue }
            chosen.insert(item.offset)
            for r in (item.offset - suppressionRadius)...(item.offset + suppressionRadius) {
                blocked.insert(r)
            }
        }
        return chosen.sorted().map { inWindow[$0] }
    }

    /// Plans windows of `windowSeconds` across `duration` advancing by `strideSeconds`
    /// (defaults to `windowSeconds / 2.0`, i.e. `16.0s` stride for `32.0s` windows).
    public static func planWindows(
        times: [Double],
        duration: Double,
        windowSeconds: Double = 32.0,
        strideSeconds: Double? = nil,
        sampling: VideoWindowSampling = .fps,
        fps: Double = 1.0,
        maxFrames: Int = 32,
        thumbnails: [[Float]] = []
    ) -> [PlannedWindow] {
        guard sampling != .off, duration > 0, windowSeconds > 0 else { return [] }
        let stride = max(0.5, strideSeconds ?? (windowSeconds / 2.0))
        var out = [PlannedWindow]()
        var lo = 0.0
        while lo < duration {
            let hi = min(lo + windowSeconds, duration)
            let sel: [Int]
            if sampling == .content && !thumbnails.isEmpty {
                sel = pickContent(thumbnails: thumbnails, times: times, lo: lo, hi: hi, fps: fps, maxFrames: maxFrames)
            } else {
                sel = pickFPS(times: times, lo: lo, hi: hi, fps: fps, maxFrames: maxFrames)
            }
            let filtered = sel.filter { times[$0] >= lo && times[$0] < hi }
            let finalSel = filtered.isEmpty ? sel : filtered
            if !finalSel.isEmpty {
                out.append(
                    PlannedWindow(
                        start: (lo * 1000).rounded() / 1000,
                        end: (hi * 1000).rounded() / 1000,
                        selectedIndices: finalSel,
                        frameTimes: finalSel.map { (times[$0] * 1000).rounded() / 1000 }
                    )
                )
            }
            if hi >= duration { break }
            lo += stride
        }
        return out
    }
}

/// Extensible artifact processor: generates thumbnails for images/videos/PDFs (`AVFoundation` + `ImageIO` + `PDFKit`)
/// with display transforms applied (`appliesPreferredTrackTransform = true`), extracts 4 fps candidate thumbnails
/// and 30s audio windows, and splits text, code, and PDF documents into line- or page-addressed chunks.
public enum ArtifactProcessor {
    public static let maxThumbnailEdge: Int = 240
    public static let audioSampleRate: Int = 16_000
    public static let audioWindowSeconds: Double = 30.0
    public static let audioStrideSeconds: Double = 15.0

    public struct VideoCandidateSet: Sendable {
        public let duration: Double
        public let times: [Double]
        public let thumbnails: [[Float]]

        public init(duration: Double, times: [Double], thumbnails: [[Float]]) {
            self.duration = duration
            self.times = times
            self.thumbnails = thumbnails
        }
    }

    public struct AudioWindowSlice: Sendable {
        public let start: Double
        public let end: Double
        public let samples: [Float]
        public let wavData: Data
        public let rmsEnergy: Float

        public init(start: Double, end: Double, samples: [Float], wavData: Data, rmsEnergy: Float) {
            self.start = start
            self.end = end
            self.samples = samples
            self.wavData = wavData
            self.rmsEnergy = rmsEnergy
        }
    }

    public struct DocumentChunkPlan: Sendable {
        public let span: ArtifactSpan
        public let text: String
        public let thumbnailJPEG: Data?

        public init(span: ArtifactSpan, text: String, thumbnailJPEG: Data? = nil) {
            self.span = span
            self.text = text
            self.thumbnailJPEG = thumbnailJPEG
        }
    }

    /// Creates a compact JPEG thumbnail for an image, video, or PDF document using native macOS
    /// ImageIO / AVFoundation / PDFKit.
    public static func makeThumbnailJPEG(
        for url: URL,
        kind: ArtifactKind,
        atSeconds: Double = 0.1
    ) async -> Data? {
        switch kind {
        case .image:
            guard let src = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
            let options: [CFString: Any] = [
                kCGImageSourceCreateThumbnailFromImageAlways: true,
                kCGImageSourceCreateThumbnailWithTransform: true,
                kCGImageSourceThumbnailMaxPixelSize: maxThumbnailEdge,
            ]
            guard let cgImage = CGImageSourceCreateThumbnailAtIndex(src, 0, options as CFDictionary) else {
                return nil
            }
            return encodeJPEG(cgImage)

        case .video:
            let asset = AVURLAsset(url: url)
            let generator = AVAssetImageGenerator(asset: asset)
            generator.appliesPreferredTrackTransform = true
            generator.maximumSize = CGSize(width: maxThumbnailEdge, height: maxThumbnailEdge)
            let time = CMTime(seconds: max(0, atSeconds), preferredTimescale: 600)
            guard let (cgImage, _) = try? await generator.image(at: time) else {
                return nil
            }
            return encodeJPEG(cgImage)

        case .document:
            guard url.pathExtension.lowercased() == "pdf",
                  let doc = PDFDocument(url: url),
                  let page = doc.page(at: 0)
            else { return nil }
            return renderPDFPageJPEG(page, maxEdge: maxThumbnailEdge)

        case .audio, .text, .code:
            return nil
        }
    }

    /// Live `AVFoundation` candidate extractor for `WindowPlanner.planWindows`.
    /// For `.content` mode, decodes `32x32` RGB thumbnails at `candidateFPS` (`4.0 fps`) with
    /// `appliesPreferredTrackTransform = true` and `.zero` time tolerance at the center of each
    /// frame's interval. For `.fps` mode, computes candidate timestamps from track metadata
    /// without decoding intermediate thumbnails.
    public static func extractVideoCandidates(
        for videoURL: URL,
        sampling: VideoWindowSampling = .content,
        candidateFPS: Double = WindowPlanner.candidateFPS,
        thumbEdge: Int = 32
    ) async throws -> VideoCandidateSet {
        let asset = AVURLAsset(url: videoURL)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            return VideoCandidateSet(duration: 0, times: [], thumbnails: [])
        }
        let (timeRange, nominalFPS) = try await track.load(.timeRange, .nominalFrameRate)
        let startSeconds = max(0, timeRange.start.seconds.isFinite ? timeRange.start.seconds : 0)
        let duration = max(0, timeRange.duration.seconds)
        let nativeFPS = Double(nominalFPS) > 0 ? Double(nominalFPS) : 30.0
        guard duration > 0 else {
            return VideoCandidateSet(duration: 0, times: [], thumbnails: [])
        }

        let rate = sampling == .content ? candidateFPS : 1.0
        let stepFrames = max(1, Int((nativeFPS / rate).rounded()))
        let totalFrames = max(1, Int((duration * nativeFPS).rounded()))
        var candidateTimes = [Double]()
        var queryCMTimes = [CMTime]()
        var frameIdx = 0
        while frameIdx < totalFrames {
            let logicalTime = Double(frameIdx) / nativeFPS
            if logicalTime < duration {
                candidateTimes.append(logicalTime)
                let centerSeconds = startSeconds + (Double(frameIdx) + 0.5) / nativeFPS
                queryCMTimes.append(CMTime(seconds: centerSeconds, preferredTimescale: 90_000))
            }
            frameIdx += stepFrames
        }
        if candidateTimes.isEmpty {
            candidateTimes = [0.0]
            queryCMTimes = [CMTime(seconds: startSeconds, preferredTimescale: 90_000)]
        }

        guard sampling == .content else {
            return VideoCandidateSet(duration: duration, times: candidateTimes, thumbnails: [])
        }

        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.maximumSize = CGSize(width: thumbEdge * 2, height: thumbEdge * 2)
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero

        var decodedTimes = [Double]()
        var thumbnails = [[Float]]()
        decodedTimes.reserveCapacity(queryCMTimes.count)
        thumbnails.reserveCapacity(queryCMTimes.count)

        var idx = 0
        for await result in generator.images(for: queryCMTimes) {
            let logicalTime = idx < candidateTimes.count ? candidateTimes[idx] : 0.0
            idx += 1
            if case let .success(_, cgImage, _) = result {
                decodedTimes.append((logicalTime * 1000).rounded() / 1000)
                thumbnails.append(rasterizeRGBThumb(cgImage, edge: thumbEdge))
            }
        }
        return VideoCandidateSet(duration: duration, times: decodedTimes, thumbnails: thumbnails)
    }

    /// Decodes upright full-resolution `CGImage` frames at the requested timestamps (`appliesPreferredTrackTransform = true`).
    public static func decodeVideoFrames(
        for videoURL: URL,
        at times: [Double]
    ) async throws -> [(time: Double, image: CGImage)] {
        guard !times.isEmpty else { return [] }
        let asset = AVURLAsset(url: videoURL)
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero

        let cmTimes = times.map { CMTime(seconds: $0, preferredTimescale: 90_000) }
        var out = [(Double, CGImage)]()
        out.reserveCapacity(cmTimes.count)
        for await result in generator.images(for: cmTimes) {
            if case let .success(requested, cgImage, _) = result {
                out.append((max(0, requested.seconds), cgImage))
            }
        }
        return out
    }

    /// Decodes mono 16 kHz `Float32` PCM from an `.audio` or `.video` file and slices it into
    /// 30-second windows (`EmbeddingGemma 2`'s 480,000-sample audio tower context).
    public static func extractAudioWindows(
        for url: URL,
        windowSeconds: Double = audioWindowSeconds,
        strideSeconds: Double = audioStrideSeconds,
        sampleRate: Int = audioSampleRate,
        minRMSEnergy: Float = 1e-4
    ) async throws -> (wholeWAV: Data?, duration: Double, windows: [AudioWindowSlice]) {
        let samples = try await decodeMonoPCM(for: url, sampleRate: sampleRate)
        guard !samples.isEmpty else {
            return (nil, 0, [])
        }
        let duration = Double(samples.count) / Double(sampleRate)
        let maxWholeSamples = min(samples.count, Int(windowSeconds * Double(sampleRate)))
        let wholeSlice = Array(samples.prefix(maxWholeSamples))
        var wholeRMS: Float = 0
        vDSP_rmsqv(wholeSlice, 1, &wholeRMS, vDSP_Length(wholeSlice.count))
        let wholeWAV = wholeRMS >= minRMSEnergy ? encodeWAV16Mono(samples: wholeSlice, sampleRate: sampleRate) : nil

        guard duration > windowSeconds else {
            return (wholeWAV, duration, [])
        }

        let winSamples = max(1, Int(windowSeconds * Double(sampleRate)))
        let stepSamples = max(1, Int(strideSeconds * Double(sampleRate)))
        var windows = [AudioWindowSlice]()
        var startSample = 0
        while startSample < samples.count {
            let endSample = min(samples.count, startSample + winSamples)
            let chunk = Array(samples[startSample..<endSample])
            var rms: Float = 0
            if !chunk.isEmpty {
                vDSP_rmsqv(chunk, 1, &rms, vDSP_Length(chunk.count))
            }
            let startSec = (Double(startSample) / Double(sampleRate) * 1000).rounded() / 1000
            let endSec = (Double(endSample) / Double(sampleRate) * 1000).rounded() / 1000
            if rms >= minRMSEnergy, chunk.count >= sampleRate / 2 {
                let wav = encodeWAV16Mono(samples: chunk, sampleRate: sampleRate)
                windows.append(
                    AudioWindowSlice(
                        start: startSec,
                        end: endSec,
                        samples: chunk,
                        wavData: wav,
                        rmsEnergy: rms
                    )
                )
            }
            if endSample >= samples.count { break }
            startSample += stepSamples
        }
        return (wholeWAV, duration, windows)
    }

    /// Decodes mono `Float32` PCM at `sampleRate` from the first audio track of `url`.
    public static func decodeMonoPCM(for url: URL, sampleRate: Int = audioSampleRate) async throws -> [Float] {
        let asset = AVURLAsset(url: url)
        let tracks = try await asset.loadTracks(withMediaType: .audio)
        guard let track = tracks.first else { return [] }

        let settings: [String: Any] = [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: Double(sampleRate),
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 32,
            AVLinearPCMIsFloatKey: true,
            AVLinearPCMIsNonInterleaved: false,
        ]
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: settings)
        let reader = try AVAssetReader(asset: asset)
        guard reader.canAdd(output) else { return [] }
        reader.add(output)
        guard reader.startReading() else { return [] }

        var samples = [Float]()
        while let sampleBuffer = output.copyNextSampleBuffer() {
            guard let blockBuffer = CMSampleBufferGetDataBuffer(sampleBuffer) else { continue }
            let byteCount = CMBlockBufferGetDataLength(blockBuffer)
            let count = byteCount / MemoryLayout<Float>.size
            guard count > 0 else { continue }
            var chunk = [Float](repeating: 0, count: count)
            CMBlockBufferCopyDataBytes(
                blockBuffer,
                atOffset: 0,
                dataLength: byteCount,
                destination: &chunk
            )
            samples.append(contentsOf: chunk)
        }
        return samples
    }

    /// Encodes mono `[-1, 1]` `Float32` samples as a standard 44-byte RIFF `audio/wav` (16-bit PCM) `Data`.
    public static func encodeWAV16Mono(samples: [Float], sampleRate: Int = audioSampleRate) -> Data {
        let numSamples = samples.count
        let dataBytes = UInt32(numSamples * 2)
        let riffChunkSize = 36 + dataBytes
        let byteRate = UInt32(sampleRate * 2)
        let blockAlign: UInt16 = 2
        let bitsPerSample: UInt16 = 16

        var data = Data()
        data.reserveCapacity(44 + Int(dataBytes))
        data.append(contentsOf: "RIFF".utf8)
        withUnsafeBytes(of: riffChunkSize.littleEndian) { data.append(contentsOf: $0) }
        data.append(contentsOf: "WAVEfmt ".utf8)
        withUnsafeBytes(of: UInt32(16).littleEndian) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: UInt16(1).littleEndian) { data.append(contentsOf: $0) } // PCM = 1
        withUnsafeBytes(of: UInt16(1).littleEndian) { data.append(contentsOf: $0) } // Mono = 1
        withUnsafeBytes(of: UInt32(sampleRate).littleEndian) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: byteRate.littleEndian) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: blockAlign.littleEndian) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: bitsPerSample.littleEndian) { data.append(contentsOf: $0) }
        data.append(contentsOf: "data".utf8)
        withUnsafeBytes(of: dataBytes.littleEndian) { data.append(contentsOf: $0) }

        var pcm16 = [Int16](repeating: 0, count: numSamples)
        for i in 0..<numSamples {
            let clamped = max(-1.0, min(1.0, samples[i]))
            pcm16[i] = Int16((clamped * 32767.0).rounded()).littleEndian
        }
        pcm16.withUnsafeBytes { data.append(contentsOf: $0) }
        return data
    }

    /// Slices `[startSeconds ..< endSeconds]` from a decoded mono PCM array and encodes it as a
    /// 16-bit mono WAV if its RMS energy exceeds `minRMSEnergy`.
    public static func encodeAudioSliceWAV(
        samples: [Float],
        sampleRate: Int = audioSampleRate,
        startSeconds: Double,
        endSeconds: Double,
        minRMSEnergy: Float = 1e-4
    ) -> Data? {
        guard !samples.isEmpty, endSeconds > startSeconds else { return nil }
        let startIdx = max(0, min(samples.count, Int((startSeconds * Double(sampleRate)).rounded())))
        let endIdx = max(startIdx, min(samples.count, Int((endSeconds * Double(sampleRate)).rounded())))
        let count = endIdx - startIdx
        guard count >= sampleRate / 2 else { return nil }
        let slice = Array(samples[startIdx..<endIdx])
        var rms: Float = 0
        vDSP_rmsqv(slice, 1, &rms, vDSP_Length(slice.count))
        guard rms >= minRMSEnergy else { return nil }
        return encodeWAV16Mono(samples: slice, sampleRate: sampleRate)
    }

    /// Extracts page-addressed chunks (`ArtifactSpan.pages`) from PDFs using `PDFKit`
    /// or rich text documents (`.rtf`, `.html`) using `NSAttributedString`.
    public static func chunkDocument(at url: URL) throws -> [DocumentChunkPlan] {
        let ext = url.pathExtension.lowercased()
        if ext == "pdf" {
            guard let pdf = PDFDocument(url: url), pdf.pageCount > 0 else {
                return []
            }
            var pagePlans = [DocumentChunkPlan]()
            var allTexts = [String]()
            var firstThumb: Data?

            for idx in 0..<pdf.pageCount {
                guard let page = pdf.page(at: idx) else { continue }
                let pageNumber = idx + 1
                let pageThumb = renderPDFPageJPEG(page, maxEdge: maxThumbnailEdge)
                if firstThumb == nil { firstThumb = pageThumb }
                let rawText = (page.string ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
                if !rawText.isEmpty {
                    allTexts.append(rawText)
                }
                if !rawText.isEmpty || pageThumb != nil {
                    pagePlans.append(
                        DocumentChunkPlan(
                            span: .pages(startPage: pageNumber, endPage: pageNumber),
                            text: rawText,
                            thumbnailJPEG: pageThumb
                        )
                    )
                }
            }

            let wholeText = allTexts.joined(separator: "\n\n")
            if pdf.pageCount == 1, let single = pagePlans.first {
                return [
                    DocumentChunkPlan(
                        span: .whole,
                        text: single.text,
                        thumbnailJPEG: single.thumbnailJPEG
                    )
                ]
            }

            var out = [DocumentChunkPlan]()
            if !wholeText.isEmpty || firstThumb != nil {
                out.append(
                    DocumentChunkPlan(
                        span: .whole,
                        text: String(wholeText.prefix(8000)),
                        thumbnailJPEG: firstThumb
                    )
                )
            }
            out.append(contentsOf: pagePlans)
            return out
        }

        // RTF or HTML document: decode via NSAttributedString, falling back to UTF-8 text.
        let text: String
        if let attr = try? NSAttributedString(url: url, options: [:], documentAttributes: nil),
           !attr.string.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            text = attr.string
        } else {
            text = try String(contentsOf: url, encoding: .utf8)
        }
        return chunkTextDocument(text).map {
            DocumentChunkPlan(span: $0.span, text: $0.text, thumbnailJPEG: nil)
        }
    }

    /// Splits a text or code document into line-addressed spans (`ArtifactSpan.lines`)
    /// so `mbed` can index notes, source code, and markdown alongside media.
    public static func chunkTextDocument(
        _ text: String,
        maxLinesPerChunk: Int = 60,
        overlapLines: Int = 10
    ) -> [(span: ArtifactSpan, text: String)] {
        let lines = text.components(separatedBy: .newlines)
        let nonEmpty = lines.contains { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
        guard nonEmpty else { return [] }
        if lines.count <= maxLinesPerChunk {
            return [(.whole, text)]
        }

        var chunks: [(ArtifactSpan, String)] = [(.whole, String(text.prefix(8000)))]
        let step = max(1, maxLinesPerChunk - overlapLines)
        var start = 0
        while start < lines.count {
            let end = min(lines.count, start + maxLinesPerChunk)
            let slice = lines[start..<end].joined(separator: "\n")
            if !slice.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                chunks.append((.lines(startLine: start + 1, endLine: end), slice))
            }
            if end == lines.count { break }
            start += step
        }
        return chunks
    }

    private static func renderPDFPageJPEG(_ page: PDFPage, maxEdge: Int) -> Data? {
        let nsImage = page.thumbnail(of: CGSize(width: maxEdge, height: maxEdge), for: .cropBox)
        var rect = CGRect(origin: .zero, size: nsImage.size)
        guard let cgImage = nsImage.cgImage(forProposedRect: &rect, context: nil, hints: nil) else {
            return nil
        }
        return encodeJPEG(cgImage)
    }

    private static func rasterizeRGBThumb(_ cgImage: CGImage, edge: Int) -> [Float] {
        let width = max(1, edge)
        let height = max(1, edge)
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        rgba.withUnsafeMutableBytes { ptr in
            guard let base = ptr.baseAddress,
                  let ctx = CGContext(
                      data: base,
                      width: width,
                      height: height,
                      bitsPerComponent: 8,
                      bytesPerRow: width * 4,
                      space: colorSpace,
                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
                  )
            else { return }
            ctx.interpolationQuality = .low
            ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        }
        var rgb = [Float](repeating: 0, count: width * height * 3)
        for i in 0..<(width * height) {
            rgb[i * 3 + 0] = Float(rgba[i * 4 + 0])
            rgb[i * 3 + 1] = Float(rgba[i * 4 + 1])
            rgb[i * 3 + 2] = Float(rgba[i * 4 + 2])
        }
        return rgb
    }

    private static func encodeJPEG(_ image: CGImage, quality: Double = 0.78) -> Data? {
        let data = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(
            data as CFMutableData,
            UTType.jpeg.identifier as CFString,
            1,
            nil
        ) else {
            return nil
        }
        let props: [CFString: Any] = [kCGImageDestinationLossyCompressionQuality: quality]
        CGImageDestinationAddImage(dest, image, props as CFDictionary)
        guard CGImageDestinationFinalize(dest) else { return nil }
        return data as Data
    }
}
