/// A pluggable retrieval backend. **This is the seam that makes Spotlight "just one
/// source".** The orchestrator fans out to any number of these — the built-in lexical
/// and vector sources below, a `CSSearchQuery`-backed Spotlight source on device, a
/// remote endpoint, an app-specific FAQ table — with per-source deadlines and
/// per-source trust boundaries handled uniformly above this protocol.
///
/// Contract:
/// - Return chunks ordered best-first (the orchestrator re-sorts defensively anyway).
/// - Respect `query.maxTier`. The orchestrator ALSO enforces it after the fact
///   (defense in depth) and reports any source that violated it.
/// - Be cancellation-responsive. A source that ignores cancellation is charged as
///   `.timedOut` and keeps running only until its own body finishes.
public protocol RetrievalSource: Sendable {
    var id: SourceID { get }
    func retrieve(_ query: RetrievalQuery) async throws -> [ScoredChunk]
}

/// BM25 source over the shared `IndexStore`.
public struct LexicalSource: RetrievalSource {
    public let id: SourceID
    private let store: IndexStore

    public init(id: SourceID = "lexical.bm25", store: IndexStore) {
        self.id = id
        self.store = store
    }

    public func retrieve(_ query: RetrievalQuery) async throws -> [ScoredChunk] {
        await store.lexicalSearch(query.text, maxTier: query.maxTier, limit: query.perSourceLimit)
    }
}

/// Cosine-similarity source over the shared `IndexStore`.
///
/// The query is embedded *here*, outside the store's isolation domain — the store is
/// never blocked on a model, and a slow embedder only slows this one source, which the
/// orchestrator's per-source deadline then contains.
public struct VectorSource: RetrievalSource {
    public let id: SourceID
    private let store: IndexStore
    private let embedder: any EmbeddingProvider

    public init(id: SourceID = "vector.cosine", store: IndexStore, embedder: any EmbeddingProvider) {
        self.id = id
        self.store = store
        self.embedder = embedder
    }

    public func retrieve(_ query: RetrievalQuery) async throws -> [ScoredChunk] {
        let vector = try await embedder.embed(query.text)
        return await store.vectorSearch(vector, maxTier: query.maxTier, limit: query.perSourceLimit)
    }
}
