/// Result of applying one change-feed entry. These are *results*, not errors: stale and
/// superseded deliveries are normal operation for any change feed, and callers (sync
/// engines, tests, dashboards) need to observe them.
public enum ApplyResult: Equatable, Sendable {
    /// The change was committed.
    case applied(chunksIndexed: Int, vectorRefusals: Int)
    /// The change's sequence number is not newer than what the store already holds.
    case staleIgnored
    /// The change was newest when it arrived, but a newer change for the same document
    /// committed while this one was awaiting embeddings. Its work was discarded.
    case supersededDuringEmbedding
    /// The document was removed (tombstone recorded at the change's sequence).
    case deleted
}

/// The single owner of all index state: lexical index, vector index, canonical chunk
/// text, and per-document sequence numbers.
///
/// ## Why an actor (micro-architecture)
/// Every mutation and read goes through one isolation domain, so `LexicalIndex` and
/// `VectorIndex` can stay lock-free single-threaded structures. Queries fan out from
/// *sources* (see `LexicalSource`/`VectorSource`), which hop through the actor only for
/// the in-memory search itself — query embedding happens outside, so the store is never
/// blocked on a model.
///
/// ## Ordering under reentrancy (macro-system correctness)
/// `apply(_:)` awaits the embedder, and actors are reentrant across `await`: a change
/// arriving mid-embedding can commit first. Correctness comes from a
/// **check → await → re-check → commit** protocol on the per-document sequence number:
///
/// 1. Reject if `sequence <= latest(document)` (stale delivery).
/// 2. Chunk synchronously, then await embeddings (actor free; others may commit).
/// 3. Re-check: if a newer sequence committed meanwhile, discard this work
///    (`.supersededDuringEmbedding`) — never overwrite newer state with older data.
/// 4. Commit synchronously (no suspension inside the mutation), so a commit is atomic.
///
/// This is last-writer-wins *by sequence number*, not by wall-clock completion order —
/// the difference between a sync engine that converges and one that corrupts itself
/// under slow embedding. Proven by `IndexStoreTests.testSlowEmbedCannotOverwriteNewerCommit`.
public actor IndexStore {
    private struct StoredChunk {
        let text: String
        let tier: PrivacyTier
    }

    private let chunker: any Chunker
    private let embedder: any EmbeddingProvider
    private let lexical: LexicalIndex
    private let vectors: VectorIndex

    private var chunkContent: [ChunkID: StoredChunk] = [:]
    private var documentChunks: [DocumentID: [ChunkID]] = [:]
    /// Latest sequence seen per document, including delete tombstones. Grows with the
    /// number of *distinct documents ever seen* (not with churn) — bounded by corpus
    /// cardinality, which is the correct retention for exactly-once semantics.
    private var latestSequence: [DocumentID: UInt64] = [:]

    public init(
        chunker: any Chunker = SentenceWindowChunker(),
        embedder: any EmbeddingProvider = HashingEmbedder(),
        lexicalParameters: LexicalIndex.Parameters = .init(),
        vectorByteBudget: Int = 4 << 20,
        minimumVectorSimilarity: Double = 0.2
    ) {
        self.chunker = chunker
        self.embedder = embedder
        self.lexical = LexicalIndex(parameters: lexicalParameters)
        self.vectors = VectorIndex(
            dimensions: embedder.dimensions,
            byteBudget: vectorByteBudget,
            minimumSimilarity: minimumVectorSimilarity
        )
    }

    // MARK: - Ingest

    /// Applies one change-feed entry. See the type documentation for the ordering
    /// protocol. Throws only if the embedder throws — in which case *nothing* was
    /// committed and the document's previous state is intact.
    @discardableResult
    public func apply(_ change: DocumentChange) async throws -> ApplyResult {
        switch change.kind {
        case .delete(let id):
            guard isNewer(change.sequence, for: id) else { return .staleIgnored }
            removeChunks(of: id)
            latestSequence[id] = change.sequence
            return .deleted

        case .upsert(let document):
            guard isNewer(change.sequence, for: document.id) else { return .staleIgnored }
            let chunks = chunker.chunk(document.text, document: document.id)

            // Await embeddings with the actor free. Others may commit in this window.
            var embedded: [(Chunk, [Float])] = []
            embedded.reserveCapacity(chunks.count)
            for chunk in chunks {
                let vector = try await embedder.embed(chunk.text)
                embedded.append((chunk, vector))
            }

            // Re-check after suspension: newer state must never be overwritten by older.
            guard isNewer(change.sequence, for: document.id) else {
                return .supersededDuringEmbedding
            }

            // Commit — fully synchronous, therefore atomic within the actor.
            removeChunks(of: document.id)
            var refusals = 0
            var ids: [ChunkID] = []
            ids.reserveCapacity(embedded.count)
            for (chunk, vector) in embedded {
                chunkContent[chunk.id] = StoredChunk(text: chunk.text, tier: document.tier)
                lexical.insert(id: chunk.id, text: chunk.text, tier: document.tier)
                let outcome = vectors.insert(id: chunk.id, vector: vector, tier: document.tier)
                switch outcome {
                case .stored, .replacedExisting:
                    break
                case .rejectedDimensionMismatch, .rejectedDegenerateVector, .rejectedExceedsBudget:
                    refusals += 1 // Chunk stays lexically searchable; vector tier degrades gracefully.
                }
                ids.append(chunk.id)
            }
            documentChunks[document.id] = ids
            latestSequence[document.id] = change.sequence
            return .applied(chunksIndexed: ids.count, vectorRefusals: refusals)
        }
    }

    private func isNewer(_ sequence: UInt64, for id: DocumentID) -> Bool {
        sequence > (latestSequence[id] ?? 0)
    }

    private func removeChunks(of id: DocumentID) {
        guard let ids = documentChunks.removeValue(forKey: id) else { return }
        for chunkID in ids {
            chunkContent.removeValue(forKey: chunkID)
            lexical.remove(id: chunkID)
            vectors.remove(id: chunkID)
        }
    }

    // MARK: - Search (called by sources)

    public func lexicalSearch(_ query: String, maxTier: PrivacyTier, limit: Int) -> [ScoredChunk] {
        lexical.search(query, maxTier: maxTier, limit: limit).compactMap { hit in
            guard let stored = chunkContent[hit.id] else { return nil }
            return ScoredChunk(id: hit.id, text: stored.text, tier: stored.tier, score: hit.score)
        }
    }

    public func vectorSearch(_ vector: [Float], maxTier: PrivacyTier, limit: Int) -> [ScoredChunk] {
        vectors.search(vector, maxTier: maxTier, limit: limit).compactMap { hit in
            guard let stored = chunkContent[hit.id] else { return nil }
            return ScoredChunk(id: hit.id, text: stored.text, tier: stored.tier, score: hit.score)
        }
    }

    // MARK: - Introspection

    public var chunkCount: Int { chunkContent.count }
    public var documentCount: Int { documentChunks.count }
    public var vectorUsedBytes: Int { vectors.usedBytes }
    public var vectorEvictionCount: Int { vectors.evictionCount }

    public func latestSequence(for id: DocumentID) -> UInt64? {
        latestSequence[id]
    }

    public func text(for id: ChunkID) -> String? {
        chunkContent[id]?.text
    }
}
