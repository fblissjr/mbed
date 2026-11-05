import Accelerate
import AVFoundation
import CoreGraphics
import Foundation
import ImageIO
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

    /// Plans consecutive windows of `windowSeconds` across `duration`.
    public static func planWindows(
        times: [Double],
        duration: Double,
        windowSeconds: Double = 32.0,
        sampling: VideoWindowSampling = .fps,
        fps: Double = 1.0,
        maxFrames: Int = 32,
        thumbnails: [[Float]] = []
    ) -> [PlannedWindow] {
        guard sampling != .off, duration > 0, windowSeconds > 0 else { return [] }
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
            lo += windowSeconds
        }
        return out
    }
}

/// Extensible artifact processor: generates thumbnails for images/videos (`AVFoundation` + `ImageIO`)
/// with display transforms applied (`appliesPreferredTrackTransform = true`), and splits text/code
/// documents into line-addressed chunks.
public enum ArtifactProcessor {
    public static let maxThumbnailEdge: Int = 240

    /// Creates a compact JPEG thumbnail for an image or video using native macOS ImageIO / AVFoundation.
    public static func makeThumbnailJPEG(for url: URL, kind: ArtifactKind) async -> Data? {
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
            let time = CMTime(seconds: 0.1, preferredTimescale: 600)
            guard let (cgImage, _) = try? await generator.image(at: time) else {
                return nil
            }
            return encodeJPEG(cgImage)

        case .audio, .text, .code, .document:
            return nil
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
