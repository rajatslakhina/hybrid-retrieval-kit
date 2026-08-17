/// Thin composition facade: one object that wires chunker → embedder → `IndexStore` →
/// built-in sources (+ any app-provided sources) → fusion → orchestrator.
///
/// Deliberately *thin*: it adds no behavior of its own, only default wiring, so an app
/// with opinions composes the parts directly and loses nothing. (The facade exists
/// because "assemble five objects correctly" is exactly the kind of README friction
/// that makes teams fork examples instead of using the seams.)
public struct HybridSearchEngine: Sendable {
    public let store: IndexStore
    private let orchestrator: RetrievalOrchestrator

    /// - Parameters:
    ///   - embedder: the semantic seam; defaults to the deterministic hashing fallback.
    ///   - extraSources: app-provided sources (a Spotlight adapter, an FAQ table, a
    ///     remote endpoint) fanned out alongside the built-in lexical + vector sources.
    public init(
        chunker: any Chunker = SentenceWindowChunker(),
        embedder: any EmbeddingProvider = HashingEmbedder(),
        vectorByteBudget: Int = 4 << 20,
        minimumVectorSimilarity: Double = 0.2,
        extraSources: [any RetrievalSource] = [],
        fuser: any RankFuser = ReciprocalRankFusion(),
        configuration: OrchestratorConfiguration = OrchestratorConfiguration()
    ) {
        let store = IndexStore(
            chunker: chunker,
            embedder: embedder,
            vectorByteBudget: vectorByteBudget,
            minimumVectorSimilarity: minimumVectorSimilarity
        )
        self.store = store
        var sources: [any RetrievalSource] = [
            LexicalSource(store: store),
            VectorSource(store: store, embedder: embedder)
        ]
        sources.append(contentsOf: extraSources)
        self.orchestrator = RetrievalOrchestrator(
            sources: sources,
            fuser: fuser,
            configuration: configuration
        )
    }

    /// Feed the change stream (or one-off changes) through to the index.
    @discardableResult
    public func apply(_ change: DocumentChange) async throws -> ApplyResult {
        try await store.apply(change)
    }

    /// Fan out, enforce, fuse, report.
    public func search(_ query: RetrievalQuery) async -> RetrievalResponse {
        await orchestrator.retrieve(query)
    }
}
