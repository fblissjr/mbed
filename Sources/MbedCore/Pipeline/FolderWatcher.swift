import CoreServices
import Foundation

/// Lightweight macOS `FSEventStream` watcher that monitors registered folder paths
/// and fires a debounced callback when files are added, modified, or renamed.
public final class FolderWatcher: @unchecked Sendable {
    private let lock = NSLock()
    private var streamRef: FSEventStreamRef?
    private let queue = DispatchQueue(label: "mbed.folder-watcher", qos: .utility)
    private var debounceWorkItem: DispatchWorkItem?
    private let debounceInterval: TimeInterval
    private let onChange: @Sendable ([URL]) -> Void
    private var watchedPaths: [String] = []

    public init(
        debounceInterval: TimeInterval = 1.5,
        onChange: @escaping @Sendable ([URL]) -> Void
    ) {
        self.debounceInterval = debounceInterval
        self.onChange = onChange
    }

    deinit {
        stop()
    }

    public func watch(urls: [URL]) {
        let paths = Array(Set(urls.map { $0.standardizedFileURL.path })).sorted()
        lock.lock()
        if paths == watchedPaths && streamRef != nil {
            lock.unlock()
            return
        }
        stopUnlocked()
        watchedPaths = paths
        guard !paths.isEmpty else {
            lock.unlock()
            return
        }

        var context = FSEventStreamContext(
            version: 0,
            info: Unmanaged.passUnretained(self).toOpaque(),
            retain: nil,
            release: nil,
            copyDescription: nil
        )
        let flags = FSEventStreamCreateFlags(
            kFSEventStreamCreateFlagUseCFTypes
                | kFSEventStreamCreateFlagFileEvents
                | kFSEventStreamCreateFlagNoDefer
        )
        if let stream = FSEventStreamCreate(
            kCFAllocatorDefault,
            Self.eventCallback,
            &context,
            paths as CFArray,
            FSEventStreamEventId(kFSEventStreamEventIdSinceNow),
            0.5,
            flags
        ) {
            self.streamRef = stream
            FSEventStreamSetDispatchQueue(stream, queue)
            FSEventStreamStart(stream)
        }
        lock.unlock()
    }

    public func stop() {
        lock.lock()
        stopUnlocked()
        lock.unlock()
    }

    private func stopUnlocked() {
        debounceWorkItem?.cancel()
        debounceWorkItem = nil
        if let stream = streamRef {
            FSEventStreamStop(stream)
            FSEventStreamInvalidate(stream)
            FSEventStreamRelease(stream)
            streamRef = nil
        }
    }

    private static let eventCallback: FSEventStreamCallback = {
        _, clientCallBackInfo, _, _, _, _ in
        guard let info = clientCallBackInfo else { return }
        let watcher = Unmanaged<FolderWatcher>.fromOpaque(info).takeUnretainedValue()
        watcher.scheduleNotify()
    }

    private func scheduleNotify() {
        lock.lock()
        debounceWorkItem?.cancel()
        let currentURLs = watchedPaths.map { URL(fileURLWithPath: $0) }
        let callback = onChange
        let item = DispatchWorkItem {
            callback(currentURLs)
        }
        debounceWorkItem = item
        let delay = debounceInterval
        lock.unlock()
        queue.asyncAfter(deadline: .now() + delay, execute: item)
    }
}
