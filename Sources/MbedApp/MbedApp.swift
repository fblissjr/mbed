import AVKit
import AppKit
import MbedCore
import SwiftUI
import UniformTypeIdentifiers

@MainActor
@Observable
final class MbedViewModel {
    var selectedBackend: EngineBackend = NativeMLXEmbeddingGemma2.isAvailableInProcess ? .nativeMLX : .heylook
    var computeOptions: ModelComputeOptions = .allMLXGPU
    var componentComputeReports: [ANETowerInspector.ComponentComputeReport] = []
    var serverURLString: String = "http://127.0.0.1:8000"
    var checkpointPath: String = NativeMLXEmbeddingGemma2.defaultCheckpointURL.path
    var serverCapabilities: HeylookServerCapabilities?
    var queryText: String = ""
    var queryAttachmentURL: URL?
    var selectedDimensions: Int = 768
    var selectedSampling: VideoWindowSampling = .fps
    var loadAudioFromVideo: Bool = false
    var selectedKindFilter: ArtifactKind?
    var selectedTaskOverride: EmbeddingTask?
    var hits: [SearchHit] = []
    var selectedHit: SearchHit?
    var selectedSpanIndex: Int = 0
    var statusMessage: String = "Ready"
    var isBusy: Bool = false
    var isIndexing: Bool = false
    var artifactCount: Int = 0
    var chunkCount: Int = 0
    var countsByKind: [ArtifactKind: Int] = [:]
    var watchedFolders: [VectorStore.WatchedFolder] = []

    var hasActiveQuery: Bool {
        !queryText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || queryAttachmentURL != nil
    }

    var effectiveQueryTask: EmbeddingTask {
        selectedTaskOverride ?? (selectedKindFilter == .code ? .codeRetrieval : .query)
    }

    var activeQueryTowersLabel: String {
        let hasText = !queryText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        let attachmentKind = queryAttachmentURL.flatMap { ArtifactKind.detect(from: $0) }
        switch (hasText, attachmentKind) {
        case (true, nil):
            return "Text Tower"
        case (true, .some(.image)):
            return "Text + Vision (Image <=280 tokens)"
        case (true, .some(.video)):
            return "Text + Vision (Video <=140 tokens/f)"
        case (true, .some(.audio)):
            return "Text + Audio (16 kHz Conformer)"
        case (true, .some(let k)):
            return "Text + \(k.rawValue.capitalized)"
        case (false, .some(.image)):
            return "Vision Tower (Image <=280 tokens)"
        case (false, .some(.video)):
            return "Vision Tower (Video <=140 tokens/f)"
        case (false, .some(.audio)):
            return "Audio Tower (16 kHz Conformer)"
        case (false, .some(let k)):
            return "Text Tower (\(k.rawValue.capitalized))"
        case (false, nil):
            return "Text / Vision / Audio"
        }
    }

    var activeQueryPromptLabel: String {
        let hasText = !queryText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        if !hasText, let url = queryAttachmentURL, let kind = ArtifactKind.detect(from: url), kind.isMedia {
            return "Pure Media (<bos><|media|><eos>)"
        }
        return effectiveQueryTask.prefixText.trimmingCharacters(in: .whitespaces)
    }

    var activeWindowFeaturesLabel: String {
        let samplingText: String
        switch selectedSampling {
        case .fps: samplingText = "Video: Fixed 1 fps (32s)"
        case .content: samplingText = "Video: Content Novelty (32s)"
        case .off: samplingText = "Video: Whole Only"
        }
        return loadAudioFromVideo ? "\(samplingText) + 16 kHz Soundtrack" : samplingText
    }

    private var store: VectorStore?
    private var engine: (any EmbeddingEngine)?
    private var coordinator: IndexCoordinator?
    private var indexTask: Task<Void, Never>?
    private var folderWatcher: FolderWatcher?

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
            let vectorStore = try store ?? VectorStore(databaseURL: dbURL)
            self.store = vectorStore

            let activeEngine: any EmbeddingEngine
            switch selectedBackend {
            case .heylook:
                let baseURL = URL(string: serverURLString) ?? URL(string: "http://127.0.0.1:8000")!
                let heylook = HeylookMLXEngine(baseURL: baseURL)
                activeEngine = heylook
                Task {
                    self.serverCapabilities = try? await heylook.fetchCapabilities()
                }
            case .nativeMLX:
                #if canImport(MLXVLM)
                activeEngine = MLXSwiftLMEmbeddingGemma2Engine(
                    checkpointURL: URL(fileURLWithPath: checkpointPath),
                    computeOptions: computeOptions
                )
                #else
                let baseURL = URL(string: serverURLString) ?? URL(string: "http://127.0.0.1:8000")!
                activeEngine = HeylookMLXEngine(baseURL: baseURL)
                statusMessage = "Native MLX not linked (build with MBED_ENABLE_NATIVE_MLX=1); using heylook HTTP."
                #endif
            }

            self.engine = activeEngine
            self.coordinator = IndexCoordinator(engine: activeEngine, store: vectorStore)
            refreshCounts()
            refreshWatchedFolders()
            refreshComputeReports()
        } catch {
            statusMessage = "Store init failed: \(error.localizedDescription)"
        }
    }

    func refreshComputeReports() {
        let dir = URL(fileURLWithPath: checkpointPath)
        componentComputeReports = ANETowerInspector.statusSnapshot(
            options: computeOptions,
            checkpointDirectory: dir
        )
        Task {
            var updated: [ANETowerInspector.ComponentComputeReport] = []
            for comp in EmbeddingComponent.allCases {
                let unit = computeOptions[comp]
                if let report = try? await ANETowerInspector.inspectComputePlan(
                    for: comp,
                    assignedUnit: unit,
                    in: dir
                ) {
                    updated.append(report)
                }
            }
            if updated.count == EmbeddingComponent.allCases.count {
                self.componentComputeReports = updated
            }
        }
    }

    func applyComputePreset(_ preset: ComputePreset) {
        switch preset {
        case .allMLXGPU:
            computeOptions = .allMLXGPU
        case .hybridANETowers:
            computeOptions = .hybridANETowers
        case .allNeuralEngine:
            computeOptions = .allNeuralEngine
        case .custom:
            break
        }
        setupCoordinator()
    }

    func refreshCounts() {
        guard let store, let stats = try? store.stats() else { return }
        artifactCount = stats.artifactCount
        chunkCount = stats.chunkCount
        countsByKind = stats.countsByKind
    }

    func refreshWatchedFolders() {
        guard let store, let folders = try? store.listWatchedFolders() else { return }
        watchedFolders = folders
        let folderURLs = folders.map(\.url)
        if folderWatcher == nil {
            folderWatcher = FolderWatcher { [weak self] changedDirs in
                Task { @MainActor [weak self] in
                    guard let self, !self.isIndexing else { return }
                    self.startIndexing(changedDirs, persistFolders: false)
                }
            }
        }
        folderWatcher?.watch(urls: folderURLs)
    }

    func removeWatchedFolder(_ folder: VectorStore.WatchedFolder) {
        guard let store else { return }
        try? store.removeWatchedFolder(folder.url)
        refreshWatchedFolders()
    }

    func reindexWatchedFolders() {
        let urls = watchedFolders.map(\.url)
        guard !urls.isEmpty else { return }
        startIndexing(urls, persistFolders: false)
    }

    func startIndexing(_ urls: [URL], persistFolders: Bool = true) {
        indexTask?.cancel()
        if persistFolders, let store {
            for url in urls {
                var isDir: ObjCBool = false
                if FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir), isDir.boolValue {
                    try? store.addWatchedFolder(url, windowSampling: selectedSampling)
                }
            }
            refreshWatchedFolders()
        }
        indexTask = Task {
            await indexURLs(urls)
        }
    }

    func cancelIndexing() {
        indexTask?.cancel()
        indexTask = nil
        isIndexing = false
        isBusy = false
        statusMessage = "Indexing cancelled."
    }

    func indexURLs(_ urls: [URL]) async {
        guard let coordinator else { return }
        isBusy = true
        isIndexing = true
        defer {
            isBusy = false
            isIndexing = false
            refreshCounts()
            refreshWatchedFolders()
        }

        do {
            let result = try await coordinator.indexURLs(
                urls,
                videoWindowSampling: selectedSampling,
                loadAudioFromVideo: loadAudioFromVideo
            ) { progress in
                Task { @MainActor in
                    self.statusMessage = "Indexing (\(progress.processed)/\(progress.total)): \(progress.currentTitle)"
                }
            }
            statusMessage = "Indexed \(result.indexed) artifacts (\(result.skipped.count) skipped)."
        } catch is CancellationError {
            statusMessage = "Indexing cancelled."
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
                taskOverride: selectedTaskOverride,
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

    init() {
        NSApplication.shared.setActivationPolicy(.regular)
        NSApplication.shared.activate(ignoringOtherApps: true)
    }

    var body: some Scene {
        WindowGroup("mbed — Multimodal Embedding Finder") {
            ContentView(viewModel: viewModel)
                .frame(minWidth: 1020, minHeight: 660)
                .onAppear {
                    NSApplication.shared.activate(ignoringOtherApps: true)
                }
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
                    HStack {
                        Label("All Artifacts", systemImage: "square.grid.2x2")
                            .fontWeight(viewModel.selectedKindFilter == nil ? .semibold : .regular)
                        Spacer()
                        Text("\(viewModel.artifactCount)")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                }
                .buttonStyle(.plain)

                ForEach(ArtifactKind.allCases, id: \.self) { kind in
                    Button {
                        viewModel.selectedKindFilter = kind
                    } label: {
                        HStack {
                            Label(kind.rawValue.capitalized, systemImage: iconName(for: kind))
                                .fontWeight(viewModel.selectedKindFilter == kind ? .semibold : .regular)
                            Spacer()
                            if let count = viewModel.countsByKind[kind], count > 0 {
                                Text("\(count)")
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }
                    .buttonStyle(.plain)
                }
            }
            .onChange(of: viewModel.selectedKindFilter) { _, _ in
                if viewModel.hasActiveQuery {
                    Task { await viewModel.runSearch() }
                }
            }

            Section("MLX EmbeddingGemma 2") {
                Picker("Engine", selection: $viewModel.selectedBackend) {
                    ForEach(EngineBackend.allCases, id: \.self) { backend in
                        Text(backend.label).tag(backend)
                    }
                }
                .onChange(of: viewModel.selectedBackend) { _, _ in
                    viewModel.setupCoordinator()
                }

                Picker("Matryoshka Dim", selection: $viewModel.selectedDimensions) {
                    ForEach(Matryoshka.gemma2Dimensions, id: \.self) { dim in
                        Text("\(dim)d").tag(dim)
                    }
                }
                .onChange(of: viewModel.selectedDimensions) { _, _ in
                    if viewModel.hasActiveQuery {
                        Task { await viewModel.runSearch() }
                    }
                }

                Picker("Query Task", selection: $viewModel.selectedTaskOverride) {
                    Text("Auto (Search / Code)").tag(Optional<EmbeddingTask>.none)
                    Text("Search Result").tag(Optional<EmbeddingTask>.some(.query))
                    Text("Code Retrieval").tag(Optional<EmbeddingTask>.some(.codeRetrieval))
                    Text("Question Answering").tag(Optional<EmbeddingTask>.some(.questionAnswering))
                    Text("Fact Verification").tag(Optional<EmbeddingTask>.some(.factVerification))
                    Text("Sentence Similarity").tag(Optional<EmbeddingTask>.some(.sts))
                    Text("Clustering").tag(Optional<EmbeddingTask>.some(.clustering))
                    Text("Classification").tag(Optional<EmbeddingTask>.some(.classification))
                }
                .onChange(of: viewModel.selectedTaskOverride) { _, _ in
                    if viewModel.hasActiveQuery {
                        Task { await viewModel.runSearch() }
                    }
                }
                Picker("Video Sampling", selection: $viewModel.selectedSampling) {
                    Text("Fixed FPS (32s)").tag(VideoWindowSampling.fps)
                    Text("Content (32s)").tag(VideoWindowSampling.content)
                    Text("Whole Video Only").tag(VideoWindowSampling.off)
                }
                Toggle("Include Video Soundtrack", isOn: $viewModel.loadAudioFromVideo)

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

            if viewModel.selectedBackend == .nativeMLX {
                Section("Compute Placement (ANE / GPU / CPU)") {
                    Picker(
                        "Preset",
                        selection: Binding(
                            get: { viewModel.computeOptions.preset },
                            set: { viewModel.applyComputePreset($0) }
                        )
                    ) {
                        ForEach(ComputePreset.allCases, id: \.self) { preset in
                            Text(preset.label).tag(preset)
                        }
                    }

                    ForEach(EmbeddingComponent.allCases) { component in
                        VStack(alignment: .leading, spacing: 2) {
                            Picker(
                                component.title,
                                selection: Binding(
                                    get: { viewModel.computeOptions[component] },
                                    set: { newUnit in
                                        viewModel.computeOptions[component] = newUnit
                                        viewModel.setupCoordinator()
                                    }
                                )
                            ) {
                                ForEach(TowerComputeUnit.allCases, id: \.self) { unit in
                                    Text(unit.shortBadge).tag(unit)
                                }
                            }
                            HStack(spacing: 4) {
                                Text("\(Int(component.parameterMillions.rounded()))M · \(component.estimatedMemoryMB) MB")
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                                Spacer()
                                if let report = viewModel.componentComputeReports.first(where: { $0.component == component }),
                                   viewModel.computeOptions[component] != .mlxGPU {
                                    if report.isCompiledBundlePresent {
                                        let pct = Int((report.aneResidencyFraction * 100).rounded())
                                        Text(report.totalOperations > 0 ? "CoreML Ready (\(pct)% ANE ops)" : "CoreML Bundle Ready")
                                            .font(.caption2)
                                            .foregroundStyle(.green)
                                    } else {
                                        Text("Needs CoreML export (MLX fallback)")
                                            .font(.caption2)
                                            .foregroundStyle(.orange)
                                    }
                                }
                            }
                        }
                        .padding(.vertical, 2)
                    }

                    ComputeMemorySplitBar(options: viewModel.computeOptions)
                }
            }

            Section("Index") {
                Button("Add Files or Folders...") {
                    presentOpenPanel()
                }
                if !viewModel.watchedFolders.isEmpty {
                    Button("Re-scan Watched Folders") {
                        viewModel.reindexWatchedFolders()
                    }
                }
                Button("Clear Index", role: .destructive) {
                    viewModel.clearIndex()
                }
            }

            if !viewModel.watchedFolders.isEmpty {
                Section("Watched Folders") {
                    ForEach(viewModel.watchedFolders) { folder in
                        HStack {
                            Image(systemName: "folder")
                                .foregroundStyle(.secondary)
                            VStack(alignment: .leading, spacing: 2) {
                                Text(folder.url.lastPathComponent)
                                    .font(.caption)
                                    .lineLimit(1)
                                Text(folder.url.path)
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                                    .lineLimit(1)
                            }
                            Spacer()
                            Button {
                                viewModel.removeWatchedFolder(folder)
                            } label: {
                                Image(systemName: "xmark.circle.fill")
                                    .foregroundStyle(.secondary)
                            }
                            .buttonStyle(.plain)
                        }
                    }
                }
            }

            Section(viewModel.selectedBackend == .heylook ? "heylook Server" : "Local MLX Checkpoint") {
                if viewModel.selectedBackend == .heylook {
                    TextField("heylook URL", text: $viewModel.serverURLString)
                        .textFieldStyle(.roundedBorder)
                        .onSubmit {
                            viewModel.setupCoordinator()
                        }
                } else {
                    TextField("Checkpoint Path", text: $viewModel.checkpointPath)
                        .textFieldStyle(.roundedBorder)
                        .onSubmit {
                            viewModel.setupCoordinator()
                        }
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
            viewModel.startIndexing(panel.urls)
        }
    }

    private func handleDroppedFiles(_ providers: [NSItemProvider]) -> Bool {
        for provider in providers {
            provider.loadItem(forTypeIdentifier: UTType.fileURL.identifier, options: nil) { item, _ in
                if let data = item as? Data,
                   let url = URL(dataRepresentation: data, relativeTo: nil) {
                    Task { @MainActor in
                        viewModel.startIndexing([url])
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

                ScoringPipelineBanner(viewModel: viewModel)

                HStack(spacing: 8) {
                    Text(viewModel.statusMessage)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                    if viewModel.isIndexing {
                        Button("Stop", role: .destructive) {
                            viewModel.cancelIndexing()
                        }
                        .controlSize(.small)
                    }
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
                    HitRowView(hit: hit, activeDimensions: viewModel.selectedDimensions)
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

struct ScoringPipelineBanner: View {
    let viewModel: MbedViewModel

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 6) {
                FeaturePill(
                    icon: "cpu",
                    label: "Query: \(viewModel.activeQueryTowersLabel)",
                    tint: .blue
                )
                if viewModel.selectedBackend == .nativeMLX {
                    FeaturePill(
                        icon: viewModel.computeOptions.usesANE ? "brain.head.profile" : "gpu",
                        label: viewModel.computeOptions.shortSummaryBadge,
                        tint: viewModel.computeOptions.usesANE ? .green : .indigo
                    )
                }
                FeaturePill(
                    icon: "text.quote",
                    label: viewModel.activeQueryPromptLabel,
                    tint: .purple,
                    monospaced: true
                )
                FeaturePill(
                    icon: "slider.horizontal.below.rectangle",
                    label: "Matryoshka: \(viewModel.selectedDimensions)d Cosine",
                    tint: .teal
                )
                FeaturePill(
                    icon: "film.stack",
                    label: viewModel.activeWindowFeaturesLabel,
                    tint: .orange
                )
            }
        }
    }
}

struct ComputeMemorySplitBar: View {
    let options: ModelComputeOptions

    var body: some View {
        let split = options.memorySplitMB
        let total = max(1, Double(split.aneMB + split.gpuMB + split.cpuMB))
        VStack(alignment: .leading, spacing: 5) {
            GeometryReader { geo in
                HStack(spacing: 2) {
                    if split.aneMB > 0 {
                        RoundedRectangle(cornerRadius: 3)
                            .fill(Color.green)
                            .frame(width: max(6, geo.size.width * (Double(split.aneMB) / total)))
                    }
                    if split.gpuMB > 0 {
                        RoundedRectangle(cornerRadius: 3)
                            .fill(Color.blue)
                            .frame(width: max(6, geo.size.width * (Double(split.gpuMB) / total)))
                    }
                    if split.cpuMB > 0 {
                        RoundedRectangle(cornerRadius: 3)
                            .fill(Color.secondary)
                            .frame(width: max(6, geo.size.width * (Double(split.cpuMB) / total)))
                    }
                }
            }
            .frame(height: 6)

            HStack(spacing: 8) {
                if split.aneMB > 0 {
                    Label("ANE: \(split.aneMB) MB", systemImage: "brain.head.profile")
                        .font(.caption2)
                        .foregroundStyle(.green)
                }
                if split.gpuMB > 0 {
                    Label("GPU: \(split.gpuMB) MB", systemImage: "gpu")
                        .font(.caption2)
                        .foregroundStyle(.blue)
                }
                if split.cpuMB > 0 {
                    Label("CPU: \(split.cpuMB) MB", systemImage: "cpu")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }
        }
        .padding(.top, 4)
    }
}

struct FeaturePill: View {
    let icon: String
    let label: String
    var tint: Color = .accentColor
    var monospaced: Bool = false

    var body: some View {
        HStack(spacing: 4) {
            Image(systemName: icon)
                .font(.caption2)
            Text(label)
                .font(monospaced ? .system(.caption2, design: .monospaced) : .caption2)
                .lineLimit(1)
        }
        .padding(.horizontal, 7)
        .padding(.vertical, 3)
        .foregroundStyle(tint)
        .background(tint.opacity(0.12))
        .clipShape(Capsule())
    }
}

struct HitRowView: View {
    let hit: SearchHit
    let activeDimensions: Int

    var body: some View {
        let bestFeature = hit.bestScoredFeature
        let hasAudioMatch = hit.spanMatches.contains(where: \.isAudioTrackChunk)
        let hasVisualWindowMatch = hit.spanMatches.contains { !$0.isAudioTrackChunk && $0.span != .whole }

        HStack(spacing: 12) {
            ThumbnailBadge(hit: hit)
                .frame(width: 64, height: 48)
                .clipShape(RoundedRectangle(cornerRadius: 6))

            VStack(alignment: .leading, spacing: 5) {
                HStack(spacing: 8) {
                    Text(hit.artifact.title)
                        .font(.headline)
                        .lineLimit(1)
                    Spacer()
                    ScoreBadgeView(score: hit.bestScore, dimensions: activeDimensions)
                }

                HStack(spacing: 6) {
                    Text(hit.artifact.kind.rawValue.uppercased())
                        .font(.caption2)
                        .fontWeight(.semibold)
                        .foregroundStyle(.secondary)

                    Text(hit.bestSpan.displayBadge ?? "Whole")
                        .font(.system(.caption, design: .monospaced))
                        .padding(.horizontal, 5)
                        .padding(.vertical, 1)
                        .background(Color.secondary.opacity(0.14))
                        .clipShape(RoundedRectangle(cornerRadius: 4))

                    HStack(spacing: 4) {
                        Image(systemName: bestFeature.systemImage)
                            .font(.caption2)
                        Text(bestFeature.title)
                            .font(.caption)
                            .lineLimit(1)
                    }
                    .foregroundStyle(bestFeature.isAudioTrack ? Color.orange : Color.accentColor)

                    if hit.artifact.kind == .video && hasAudioMatch && hasVisualWindowMatch {
                        Text("+ Audio & Vision")
                            .font(.caption2)
                            .padding(.horizontal, 5)
                            .padding(.vertical, 1)
                            .background(Color.orange.opacity(0.14))
                            .foregroundStyle(.orange)
                            .clipShape(Capsule())
                    }

                    Spacer()

                    if hit.spanMatches.count > 1 {
                        Text("\(hit.spanMatches.count) scored spans")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                }
            }
        }
        .padding(.vertical, 4)
    }
}

struct ScoreBadgeView: View {
    let score: Float
    let dimensions: Int

    var body: some View {
        let normalized = Double(max(0, min(1, score)))
        HStack(spacing: 6) {
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    Capsule()
                        .fill(Color.secondary.opacity(0.18))
                    Capsule()
                        .fill(Color.accentColor)
                        .frame(width: max(4, geo.size.width * normalized))
                }
            }
            .frame(width: 38, height: 6)

            Text(String(format: "%.3f", score))
                .font(.system(.caption, design: .monospaced))
                .fontWeight(.medium)

            Text("\(dimensions)d")
                .font(.system(.caption2, design: .monospaced))
                .foregroundStyle(.secondary)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 3)
        .background(Color.accentColor.opacity(0.12))
        .clipShape(Capsule())
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
                Image(systemName: iconName)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private var iconName: String {
        switch hit.artifact.kind {
        case .audio: return "waveform"
        case .code: return "chevron.left.forwardslash.chevron.right"
        case .document: return "doc.richtext"
        case .image: return "photo"
        case .video: return "film"
        case .text: return "doc.text"
        }
    }
}

struct InspectorView: View {
    @Bindable var viewModel: MbedViewModel
    @State private var player: AVPlayer?

    var body: some View {
        if let hit = viewModel.selectedHit {
            let activeMatch = hit.spanMatches[safe: viewModel.selectedSpanIndex] ?? hit.bestMatch
            let activeSpan = activeMatch?.span ?? hit.bestSpan
            let activeFeature = activeMatch?.scoredFeature(for: hit.artifact) ?? hit.bestScoredFeature
            let activeFrameTimes = activeSpan.frameTimes
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(hit.artifact.title)
                            .font(.title3)
                            .fontWeight(.semibold)

                        Text(hit.artifact.url.path)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .textSelection(.enabled)
                    }

                    ScoredFeatureBreakdownCard(
                        hit: hit,
                        activeMatch: activeMatch,
                        activeFeature: activeFeature,
                        activeSpan: activeSpan,
                        selectedDimensions: $viewModel.selectedDimensions
                    )

                    if hit.artifact.kind == .video || hit.artifact.kind == .audio {
                        MacVideoPlayerView(player: player)
                            .frame(height: hit.artifact.kind == .audio ? 64 : 240)
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

                        if !activeFrameTimes.isEmpty {
                            VStack(alignment: .leading, spacing: 6) {
                                HStack {
                                    Text("Sampled Vision Frames (\(activeFrameTimes.count))")
                                        .font(.caption)
                                        .fontWeight(.medium)
                                        .foregroundStyle(.secondary)
                                    Spacer()
                                    Text(samplingModeLabel(for: hit.artifact))
                                        .font(.caption2)
                                        .foregroundStyle(.secondary)
                                }

                                ScrollView(.horizontal, showsIndicators: false) {
                                    HStack(spacing: 6) {
                                        ForEach(activeFrameTimes, id: \.self) { frameTime in
                                            Button {
                                                seekPlayer(to: frameTime)
                                            } label: {
                                                Text(ArtifactSpan.formatTimestampMillis(frameTime))
                                                    .font(.system(.caption2, design: .monospaced))
                                                    .padding(.horizontal, 8)
                                                    .padding(.vertical, 4)
                                                    .background(Color.secondary.opacity(0.14))
                                                    .clipShape(Capsule())
                                            }
                                            .buttonStyle(.plain)
                                        }
                                    }
                                }
                            }
                        }
                    } else if hit.artifact.kind == .image,
                              let nsImage = NSImage(contentsOf: hit.artifact.url) {
                        Image(nsImage: nsImage)
                            .resizable()
                            .scaledToFit()
                            .frame(maxHeight: 260)
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                    }

                    if let snippet = activeMatch?.previewText ?? hit.previewText {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Embedded Text / Transcript Preview")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            ScrollView(.horizontal) {
                                Text(snippet)
                                    .font(.system(.caption, design: .monospaced))
                                    .textSelection(.enabled)
                                    .padding(10)
                            }
                            .background(Color.secondary.opacity(0.1))
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                        }
                    }

                    VStack(alignment: .leading, spacing: 6) {
                        HStack {
                            Text("Scored Spans & Features (\(hit.spanMatches.count))")
                                .font(.subheadline)
                                .fontWeight(.medium)
                            Spacer()
                            Text("\(viewModel.selectedDimensions)d Cosine")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }

                        ForEach(Array(hit.spanMatches.enumerated()), id: \.offset) { idx, match in
                            let feature = match.scoredFeature(for: hit.artifact)
                            let isSelected = viewModel.selectedSpanIndex == idx
                            Button {
                                viewModel.selectedSpanIndex = idx
                            } label: {
                                HStack(spacing: 8) {
                                    Image(systemName: feature.systemImage)
                                        .font(.caption)
                                        .foregroundStyle(feature.isAudioTrack ? Color.orange : Color.accentColor)
                                        .frame(width: 16)

                                    VStack(alignment: .leading, spacing: 2) {
                                        HStack(spacing: 6) {
                                            Text(match.span.displayBadge ?? "Whole")
                                                .font(.system(.caption, design: .monospaced))
                                                .fontWeight(.semibold)
                                            Text(feature.title)
                                                .font(.caption)
                                                .lineLimit(1)
                                        }
                                        Text(feature.subtitle)
                                            .font(.caption2)
                                            .foregroundStyle(.secondary)
                                            .lineLimit(1)
                                    }

                                    Spacer()

                                    ScoreBadgeView(score: match.score, dimensions: viewModel.selectedDimensions)
                                }
                                .padding(8)
                                .background(
                                    isSelected
                                        ? Color.accentColor.opacity(0.18)
                                        : Color.secondary.opacity(0.07)
                                )
                                .clipShape(RoundedRectangle(cornerRadius: 8))
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
                description: Text("Select a search result to inspect its scored features, Matryoshka dimension breakdown, and matching spans.")
            )
        }
    }

    private func samplingModeLabel(for artifact: Artifact) -> String {
        switch artifact.windowSamplingRaw {
        case "content": return "Content Novelty (4 fps -> 1 fps budget)"
        case "fps": return "Fixed 1 fps (32s window)"
        default: return "Whole clip"
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

    private func seekPlayer(to seconds: Double) {
        guard let player else { return }
        let target = CMTime(seconds: seconds, preferredTimescale: 600)
        player.seek(to: target, toleranceBefore: .zero, toleranceAfter: .zero)
    }
}

struct ScoredFeatureBreakdownCard: View {
    let hit: SearchHit
    let activeMatch: SearchHit.SpanMatch?
    let activeFeature: ScoredFeatureSummary
    let activeSpan: ArtifactSpan
    @Binding var selectedDimensions: Int

    var body: some View {
        let activeScore = activeMatch?.score ?? hit.bestScore
        let dimScores = activeMatch?.scoresByDimension ?? [:]

        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .top, spacing: 10) {
                Image(systemName: activeFeature.systemImage)
                    .font(.title3)
                    .foregroundStyle(activeFeature.isAudioTrack ? Color.orange : Color.accentColor)
                    .frame(width: 28, height: 28)
                    .background(
                        (activeFeature.isAudioTrack ? Color.orange : Color.accentColor).opacity(0.14)
                    )
                    .clipShape(RoundedRectangle(cornerRadius: 6))

                VStack(alignment: .leading, spacing: 2) {
                    HStack(spacing: 6) {
                        Text(activeFeature.title)
                            .font(.subheadline)
                            .fontWeight(.semibold)
                        Text(activeSpan.displayBadge ?? "Whole")
                            .font(.system(.caption2, design: .monospaced))
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(Color.secondary.opacity(0.15))
                            .clipShape(Capsule())
                    }
                    Text(activeFeature.subtitle)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }

                Spacer()

                VStack(alignment: .trailing, spacing: 2) {
                    Text(String(format: "%.4f", activeScore))
                        .font(.system(.subheadline, design: .monospaced))
                        .fontWeight(.bold)
                    Text("\(selectedDimensions)d cosine")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }

            if !dimScores.isEmpty {
                Divider()
                VStack(alignment: .leading, spacing: 6) {
                    Text("Matryoshka Prefix Slice Scores (click to switch)")
                        .font(.caption2)
                        .foregroundStyle(.secondary)

                    HStack(spacing: 6) {
                        ForEach(Matryoshka.gemma2Dimensions, id: \.self) { dim in
                            let val = dimScores[dim] ?? activeScore
                            let isCurrent = (dim == selectedDimensions)
                            Button {
                                selectedDimensions = dim
                            } label: {
                                VStack(alignment: .leading, spacing: 3) {
                                    HStack {
                                        Text("\(dim)d")
                                            .font(.system(.caption2, design: .monospaced))
                                            .fontWeight(isCurrent ? .bold : .regular)
                                        Spacer()
                                        Text(String(format: "%.3f", val))
                                            .font(.system(.caption2, design: .monospaced))
                                    }
                                    GeometryReader { geo in
                                        ZStack(alignment: .leading) {
                                            Capsule()
                                                .fill(Color.secondary.opacity(0.16))
                                            Capsule()
                                                .fill(isCurrent ? Color.accentColor : Color.secondary.opacity(0.6))
                                                .frame(width: max(3, geo.size.width * Double(max(0, min(1, val)))))
                                        }
                                    }
                                    .frame(height: 4)
                                }
                                .padding(6)
                                .background(
                                    isCurrent
                                        ? Color.accentColor.opacity(0.16)
                                        : Color.secondary.opacity(0.08)
                                )
                                .clipShape(RoundedRectangle(cornerRadius: 6))
                            }
                            .buttonStyle(.plain)
                        }
                    }
                }
            }

            Divider()

            HStack(spacing: 12) {
                Label(
                    hit.artifact.modelID.isEmpty ? "embeddinggemma-2-MLX-BF16" : hit.artifact.modelID,
                    systemImage: "cube"
                )
                .font(.caption2)
                .foregroundStyle(.secondary)
                .lineLimit(1)

                Spacer()

                Text(ByteCountFormatter.string(fromByteCount: hit.artifact.fingerprint.byteCount, countStyle: .file))
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(12)
        .background(Color.secondary.opacity(0.07))
        .clipShape(RoundedRectangle(cornerRadius: 10))
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
