import AVKit
import AppKit
import MbedCore
import SwiftUI
import UniformTypeIdentifiers

@MainActor
@Observable
final class MbedViewModel {
    var serverURLString: String = "http://127.0.0.1:8000"
    var queryText: String = ""
    var queryAttachmentURL: URL?
    var selectedDimensions: Int = 768
    var selectedSampling: VideoWindowSampling = .fps
    var selectedKindFilter: ArtifactKind?
    var hits: [SearchHit] = []
    var selectedHit: SearchHit?
    var selectedSpanIndex: Int = 0
    var statusMessage: String = "Ready"
    var isBusy: Bool = false
    var artifactCount: Int = 0
    var chunkCount: Int = 0

    private var store: VectorStore?
    private var coordinator: IndexCoordinator?

    init() {
        setupCoordinator()
    }

    func setupCoordinator() {
        do {
            let appSupport = try FileManager.default.url(
                for: .applicationSupportDirectory,
                in: .userDomainMask,
                appropriateFor: nil,
                create: true
            ).appendingPathComponent("mbed", isDirectory: true)
            try FileManager.default.createDirectory(at: appSupport, withIntermediateDirectories: true)
            let dbURL = appSupport.appendingPathComponent("mbed.sqlite3")
            let vectorStore = try VectorStore(databaseURL: dbURL)
            let baseURL = URL(string: serverURLString) ?? URL(string: "http://127.0.0.1:8000")!
            let engine = HeylookMLXEngine(baseURL: baseURL)
            self.store = vectorStore
            self.coordinator = IndexCoordinator(engine: engine, store: vectorStore)
            refreshCounts()
        } catch {
            statusMessage = "Store init failed: \(error.localizedDescription)"
        }
    }

    func refreshCounts() {
        guard let store, let stats = try? store.stats() else { return }
        artifactCount = stats.artifactCount
        chunkCount = stats.chunkCount
    }

    func indexURLs(_ urls: [URL]) async {
        guard let coordinator else { return }
        isBusy = true
        defer {
            isBusy = false
            refreshCounts()
        }

        do {
            let result = try await coordinator.indexURLs(
                urls,
                videoWindowSampling: selectedSampling
            ) { progress in
                Task { @MainActor in
                    self.statusMessage = "Indexing (\(progress.processed)/\(progress.total)): \(progress.currentTitle)"
                }
            }
            statusMessage = "Indexed \(result.indexed) artifacts (\(result.skipped.count) skipped)."
        } catch {
            statusMessage = "Index error: \(error.localizedDescription)"
        }
    }

    func runSearch() async {
        guard let coordinator else { return }
        let trimmed = queryText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty || queryAttachmentURL != nil else { return }

        isBusy = true
        statusMessage = "Searching (\(selectedDimensions)d)..."
        defer { isBusy = false }

        do {
            let kinds: Set<ArtifactKind>? = selectedKindFilter.map { [$0] }
            let results = try await coordinator.search(
                queryText: trimmed,
                exampleFileURL: queryAttachmentURL,
                dimensions: selectedDimensions,
                kinds: kinds,
                limit: 30
            )
            hits = results
            selectedHit = results.first
            selectedSpanIndex = 0
            statusMessage = "Found \(results.count) results (\(selectedDimensions)d Matryoshka slice)."
        } catch {
            statusMessage = "Search failed: \(error.localizedDescription)"
        }
    }

    func clearIndex() {
        guard let store else { return }
        do {
            try store.clearAll()
            hits = []
            selectedHit = nil
            refreshCounts()
            statusMessage = "Index cleared."
        } catch {
            statusMessage = "Clear failed: \(error.localizedDescription)"
        }
    }
}

@main
struct MbedApp: App {
    @State private var viewModel = MbedViewModel()

    var body: some Scene {
        WindowGroup("mbed — Multimodal Embedding Finder") {
            ContentView(viewModel: viewModel)
                .frame(minWidth: 1020, minHeight: 660)
        }
    }
}

struct ContentView: View {
    @Bindable var viewModel: MbedViewModel

    var body: some View {
        NavigationSplitView {
            SidebarView(viewModel: viewModel)
                .navigationSplitViewColumnWidth(min: 230, ideal: 260, max: 310)
        } content: {
            SearchAndResultsView(viewModel: viewModel)
                .navigationSplitViewColumnWidth(min: 420, ideal: 520)
        } detail: {
            InspectorView(viewModel: viewModel)
                .navigationSplitViewColumnWidth(min: 320, ideal: 400)
        }
    }
}

struct SidebarView: View {
    @Bindable var viewModel: MbedViewModel
    @State private var isTargetingDrop = false

    var body: some View {
        List {
            Section("Modality Filter") {
                Button {
                    viewModel.selectedKindFilter = nil
                } label: {
                    Label("All Artifacts", systemImage: "square.grid.2x2")
                        .fontWeight(viewModel.selectedKindFilter == nil ? .semibold : .regular)
                }
                .buttonStyle(.plain)

                ForEach(ArtifactKind.allCases, id: \.self) { kind in
                    Button {
                        viewModel.selectedKindFilter = kind
                    } label: {
                        Label(kind.rawValue.capitalized, systemImage: iconName(for: kind))
                            .fontWeight(viewModel.selectedKindFilter == kind ? .semibold : .regular)
                    }
                    .buttonStyle(.plain)
                }
            }

            Section("MLX EmbeddingGemma 2") {
                Picker("Matryoshka Dim", selection: $viewModel.selectedDimensions) {
                    ForEach(Matryoshka.gemma2Dimensions, id: \.self) { dim in
                        Text("\(dim)d").tag(dim)
                    }
                }
                Picker("Video Sampling", selection: $viewModel.selectedSampling) {
                    Text("Fixed FPS (32s)").tag(VideoWindowSampling.fps)
                    Text("Content (32s)").tag(VideoWindowSampling.content)
                    Text("Whole Video Only").tag(VideoWindowSampling.off)
                }
                HStack {
                    Text("Artifacts")
                    Spacer()
                    Text("\(viewModel.artifactCount)")
                        .foregroundStyle(.secondary)
                }
                HStack {
                    Text("Embedded Chunks")
                    Spacer()
                    Text("\(viewModel.chunkCount)")
                        .foregroundStyle(.secondary)
                }
            }

            Section("Index") {
                Button("Add Files or Folders...") {
                    presentOpenPanel()
                }
                Button("Clear Index", role: .destructive) {
                    viewModel.clearIndex()
                }
            }

            Section("Server") {
                TextField("heylook URL", text: $viewModel.serverURLString)
                    .textFieldStyle(.roundedBorder)
                    .onSubmit {
                        viewModel.setupCoordinator()
                    }
            }
        }
        .overlay {
            if isTargetingDrop {
                RoundedRectangle(cornerRadius: 10)
                    .stroke(Color.accentColor, lineWidth: 2)
                    .padding(6)
            }
        }
        .onDrop(of: [.fileURL], isTargeted: $isTargetingDrop) { providers in
            handleDroppedFiles(providers)
        }
    }

    private func iconName(for kind: ArtifactKind) -> String {
        switch kind {
        case .image: return "photo"
        case .video: return "film"
        case .audio: return "waveform"
        case .text: return "doc.plaintext"
        case .code: return "chevron.left.forwardslash.chevron.right"
        case .document: return "doc.richtext"
        }
    }

    private func presentOpenPanel() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = true
        if panel.runModal() == .OK {
            let urls = panel.urls
            Task {
                await viewModel.indexURLs(urls)
            }
        }
    }

    private func handleDroppedFiles(_ providers: [NSItemProvider]) -> Bool {
        for provider in providers {
            provider.loadItem(forTypeIdentifier: UTType.fileURL.identifier, options: nil) { item, _ in
                if let data = item as? Data,
                   let url = URL(dataRepresentation: data, relativeTo: nil) {
                    Task { @MainActor in
                        await viewModel.indexURLs([url])
                    }
                }
            }
        }
        return true
    }
}

struct SearchAndResultsView: View {
    @Bindable var viewModel: MbedViewModel

    var body: some View {
        VStack(spacing: 0) {
            VStack(alignment: .leading, spacing: 8) {
                HStack(spacing: 8) {
                    TextField("Search across video windows, images, audio, text, and code...", text: $viewModel.queryText)
                        .textFieldStyle(.roundedBorder)
                        .onSubmit {
                            Task { await viewModel.runSearch() }
                        }

                    Button("Attach Query Media...") {
                        pickQueryAttachment()
                    }

                    Button("Search") {
                        Task { await viewModel.runSearch() }
                    }
                    .keyboardShortcut(.defaultAction)
                    .disabled(viewModel.isBusy)
                }

                if let attachment = viewModel.queryAttachmentURL {
                    HStack(spacing: 6) {
                        Image(systemName: "paperclip")
                        Text("Query attachment: \(attachment.lastPathComponent)")
                            .font(.caption)
                        Button("Remove") {
                            viewModel.queryAttachmentURL = nil
                        }
                        .font(.caption)
                        .buttonStyle(.borderless)
                    }
                    .foregroundStyle(.secondary)
                }

                HStack {
                    Text(viewModel.statusMessage)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                    if viewModel.isBusy {
                        ProgressView()
                            .controlSize(.small)
                    }
                }
            }
            .padding(12)

            Divider()

            if viewModel.hits.isEmpty {
                ContentUnavailableView(
                    "No Matching Artifacts",
                    systemImage: "sparkle.magnifyingglass",
                    description: Text("Drop folders or files in the sidebar to index with EmbeddingGemma 2, then run a text or multimodal search.")
                )
            } else {
                List(viewModel.hits, selection: $viewModel.selectedHit) { hit in
                    HitRowView(hit: hit)
                        .tag(hit)
                }
                .onChange(of: viewModel.selectedHit) { _, _ in
                    viewModel.selectedSpanIndex = 0
                }
            }
        }
    }

    private func pickQueryAttachment() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        if panel.runModal() == .OK {
            viewModel.queryAttachmentURL = panel.url
        }
    }
}

struct HitRowView: View {
    let hit: SearchHit

    var body: some View {
        HStack(spacing: 12) {
            ThumbnailBadge(hit: hit)
                .frame(width: 64, height: 48)
                .clipShape(RoundedRectangle(cornerRadius: 6))

            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text(hit.artifact.title)
                        .font(.headline)
                        .lineLimit(1)
                    Spacer()
                    Text(String(format: "%.3f", hit.bestScore))
                        .font(.system(.caption, design: .monospaced))
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(Color.accentColor.opacity(0.15))
                        .clipShape(Capsule())
                }

                HStack(spacing: 8) {
                    Text(hit.artifact.kind.rawValue.uppercased())
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    Text(hit.bestSpan.displayBadge ?? "Whole")
                        .font(.caption)
                        .foregroundStyle(.primary)
                    if hit.spanMatches.count > 1 {
                        Text("+\(hit.spanMatches.count - 1) spans")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                }
            }
        }
        .padding(.vertical, 4)
    }
}

struct ThumbnailBadge: View {
    let hit: SearchHit

    var body: some View {
        if let data = hit.thumbnailJPEG, let nsImage = NSImage(data: data) {
            Image(nsImage: nsImage)
                .resizable()
                .scaledToFill()
        } else {
            ZStack {
                Color.secondary.opacity(0.15)
                Image(systemName: hit.artifact.kind == .code ? "chevron.left.forwardslash.chevron.right" : "doc.text")
                    .foregroundStyle(.secondary)
            }
        }
    }
}

struct InspectorView: View {
    @Bindable var viewModel: MbedViewModel
    @State private var player: AVPlayer?

    var body: some View {
        if let hit = viewModel.selectedHit {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    Text(hit.artifact.title)
                        .font(.title3)
                        .fontWeight(.semibold)

                    Text(hit.artifact.url.path)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)

                    if hit.artifact.kind == .video {
                        MacVideoPlayerView(player: player)
                            .frame(height: 240)
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                            .onAppear {
                                configurePlayer(for: hit, spanIndex: viewModel.selectedSpanIndex)
                            }
                            .onChange(of: hit.id) { _, _ in
                                configurePlayer(for: hit, spanIndex: viewModel.selectedSpanIndex)
                            }
                            .onChange(of: viewModel.selectedSpanIndex) { _, newIdx in
                                configurePlayer(for: hit, spanIndex: newIdx)
                            }
                    } else if hit.artifact.kind == .image,
                              let nsImage = NSImage(contentsOf: hit.artifact.url) {
                        Image(nsImage: nsImage)
                            .resizable()
                            .scaledToFit()
                            .frame(maxHeight: 260)
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                    } else if let snippet = hit.spanMatches[safe: viewModel.selectedSpanIndex]?.previewText ?? hit.previewText {
                        ScrollView(.horizontal) {
                            Text(snippet)
                                .font(.system(.caption, design: .monospaced))
                                .textSelection(.enabled)
                                .padding(10)
                        }
                        .background(Color.secondary.opacity(0.1))
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                    }

                    VStack(alignment: .leading, spacing: 6) {
                        Text("Matching Spans (\(hit.spanMatches.count))")
                            .font(.subheadline)
                            .fontWeight(.medium)

                        ForEach(Array(hit.spanMatches.enumerated()), id: \.offset) { idx, match in
                            Button {
                                viewModel.selectedSpanIndex = idx
                            } label: {
                                HStack {
                                    Text(match.span.displayBadge ?? "Whole")
                                        .font(.caption)
                                    Spacer()
                                    Text(String(format: "%.3f", match.score))
                                        .font(.system(.caption, design: .monospaced))
                                }
                                .padding(6)
                                .background(
                                    viewModel.selectedSpanIndex == idx
                                        ? Color.accentColor.opacity(0.2)
                                        : Color.secondary.opacity(0.08)
                                )
                                .clipShape(RoundedRectangle(cornerRadius: 6))
                            }
                            .buttonStyle(.plain)
                        }
                    }

                    Button("Reveal in Finder") {
                        NSWorkspace.shared.activateFileViewerSelecting([hit.artifact.url])
                    }
                }
                .padding(16)
            }
        } else {
            ContentUnavailableView(
                "No Selection",
                systemImage: "sidebar.right",
                description: Text("Select a search result to inspect its matching spans or preview media.")
            )
        }
    }

    private func configurePlayer(for hit: SearchHit, spanIndex: Int) {
        let avPlayer = AVPlayer(url: hit.artifact.url)
        let match = hit.spanMatches[safe: spanIndex]
        if let start = match?.span.startTimeSeconds ?? hit.bestSpan.startTimeSeconds, start > 0 {
            let target = CMTime(seconds: start, preferredTimescale: 600)
            avPlayer.seek(to: target, toleranceBefore: .zero, toleranceAfter: .zero)
        }
        self.player = avPlayer
    }
}

struct MacVideoPlayerView: NSViewRepresentable {
    let player: AVPlayer?

    func makeNSView(context: Context) -> AVPlayerView {
        let view = AVPlayerView()
        view.controlsStyle = .inline
        view.player = player
        return view
    }

    func updateNSView(_ nsView: AVPlayerView, context: Context) {
        if nsView.player !== player {
            nsView.player = player
        }
    }
}

private extension Collection {
    subscript(safe index: Index) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}
