import XCTest
@testable import HybridRetrieval

/// Embedder whose latency is controlled by the text itself: any text containing
/// "SLOWMARKER" sleeps before returning. Sleeps in small cancellation-checking steps so
/// tests never leak long-lived work.
private struct MarkerDelayEmbedder: EmbeddingProvider {
    let inner = HashingEmbedder(dimensions: 16)
    var dimensions: Int { inner.dimensions }

    func embed(_ text: String) async throws -> [Float] {
        if text.lowercased().contains("slowmarker") {
            for _ in 0..<20 {
                try await Task.sleep(nanoseconds: 10_000_000) // 20 × 10ms = 200ms
            }
        }
        return try await inner.embed(text)
    }
}

private struct ThrowingEmbedder: EmbeddingProvider {
    struct Failure: Error {}
    var dimensions: Int { 16 }
    func embed(_ text: String) async throws -> [Float] {
        throw Failure()
    }
}

final class IndexStoreTests: XCTestCase {
    private func doc(_ id: String, _ text: String, tier: PrivacyTier = .open) -> Document {
        Document(id: DocumentID(id), text: text, tier: tier)
    }

    func testUpsertThenSearchRoundTrip() async throws {
        let store = IndexStore()
        let result = try await store.apply(.upsert(doc("d1", "Actors isolate mutable state in Swift."), sequence: 1))
        guard case .applied(let chunks, let refusals) = result else {
            return XCTFail("expected .applied, got \(result)")
        }
        XCTAssertGreaterThan(chunks, 0)
        XCTAssertEqual(refusals, 0)

        let hits = await store.lexicalSearch("actors", maxTier: .open, limit: 5)
        XCTAssertEqual(hits.count, 1)
        XCTAssertTrue(hits.first?.text.contains("Actors") == true, "store must return canonical text")
    }

    func testStaleSequenceIsIgnored() async throws {
        let store = IndexStore()
        try await store.apply(.upsert(doc("d1", "version five content"), sequence: 5))
        let stale = try await store.apply(.upsert(doc("d1", "version three content"), sequence: 3))
        XCTAssertEqual(stale, .staleIgnored)

        let hits = await store.lexicalSearch("five", maxTier: .open, limit: 5)
        XCTAssertEqual(hits.count, 1, "stale write must not replace newer content")
        let ghost = await store.lexicalSearch("three", maxTier: .open, limit: 5)
        XCTAssertTrue(ghost.isEmpty)
    }

    func testEqualSequenceIsAlsoStale() async throws {
        let store = IndexStore()
        try await store.apply(.upsert(doc("d1", "first delivery"), sequence: 7))
        let redelivery = try await store.apply(.upsert(doc("d1", "duplicate delivery"), sequence: 7))
        XCTAssertEqual(redelivery, .staleIgnored, "at-least-once delivery must be deduplicated by sequence")
    }

    /// The reentrancy test the design doc points at: an upsert that is *newest at
    /// arrival* but finishes embedding *after* a newer change committed must be
    /// discarded, not committed. Wall-clock completion order is adversarial here:
    /// seq 1 embeds slowly, seq 2 lands mid-embedding.
    func testSlowEmbedCannotOverwriteNewerCommit() async throws {
        let store = IndexStore(embedder: MarkerDelayEmbedder())

        let oldDocument = doc("d1", "old text with SLOWMARKER inside")
        let slow = Task {
            try await store.apply(.upsert(oldDocument, sequence: 1))
        }
        // Give the slow apply time to pass its first sequence check and suspend.
        try await Task.sleep(nanoseconds: 50_000_000)
        let fast = try await store.apply(.upsert(doc("d1", "new text that must win"), sequence: 2))
        guard case .applied = fast else {
            return XCTFail("fast apply should commit, got \(fast)")
        }

        let slowResult = try await slow.value
        XCTAssertEqual(slowResult, .supersededDuringEmbedding,
                       "an apply that lost the sequence race during embedding must discard its work")

        let winners = await store.lexicalSearch("win", maxTier: .open, limit: 5)
        XCTAssertEqual(winners.count, 1)
        let losers = await store.lexicalSearch("slowmarker", maxTier: .open, limit: 5)
        XCTAssertTrue(losers.isEmpty, "superseded content must not be searchable")
        let sequence = await store.latestSequence(for: DocumentID("d1"))
        XCTAssertEqual(sequence, 2)
    }

    func testDeleteTombstoneBlocksLateUpsert() async throws {
        let store = IndexStore()
        try await store.apply(.upsert(doc("d1", "soon to vanish"), sequence: 1))
        let deletion = try await store.apply(.delete(DocumentID("d1"), sequence: 3))
        XCTAssertEqual(deletion, .deleted)

        // A late (out-of-order) upsert with an older sequence must lose to the tombstone.
        let late = try await store.apply(.upsert(doc("d1", "zombie revival"), sequence: 2))
        XCTAssertEqual(late, .staleIgnored)

        let hits = await store.lexicalSearch("zombie vanish", maxTier: .open, limit: 5)
        XCTAssertTrue(hits.isEmpty)
        let count = await store.documentCount
        XCTAssertEqual(count, 0)
    }

    func testEmbedderFailureLeavesPreviousStateIntact() async throws {
        let store = IndexStore()
        try await store.apply(.upsert(doc("d1", "stable original content"), sequence: 1))

        let failing = IndexStore(embedder: ThrowingEmbedder())
        do {
            try await failing.apply(.upsert(doc("x", "anything"), sequence: 1))
            XCTFail("expected embedder failure to propagate")
        } catch {
            // Expected. Nothing must have been committed.
            let count = await failing.chunkCount
            XCTAssertEqual(count, 0)
            let sequence = await failing.latestSequence(for: DocumentID("x"))
            XCTAssertNil(sequence, "a failed apply must not advance the sequence")
        }

        // And the healthy store is unaffected by the failing one's existence.
        let hits = await store.lexicalSearch("stable", maxTier: .open, limit: 5)
        XCTAssertEqual(hits.count, 1)
    }

    func testConcurrentWritersAllCommit() async throws {
        let store = IndexStore()
        try await withThrowingTaskGroup(of: Void.self) { group in
            for i in 0..<50 {
                group.addTask {
                    _ = try await store.apply(.upsert(
                        Document(id: DocumentID("doc\(i)"), text: "unique content token\(i) payload"),
                        sequence: UInt64(i) + 1
                    ))
                }
            }
            try await group.waitForAll()
        }
        let documents = await store.documentCount
        XCTAssertEqual(documents, 50, "50 genuinely concurrent writers must all commit")

        // Spot-check a few documents' searchability.
        for i in [0, 17, 49] {
            let hits = await store.lexicalSearch("token\(i)", maxTier: .open, limit: 5)
            XCTAssertEqual(hits.count, 1, "doc\(i) must be searchable")
        }
    }

    func testReindexReplacesOldChunksCompletely() async throws {
        let store = IndexStore()
        try await store.apply(.upsert(doc("d1", "First topic: espresso machines and grinders."), sequence: 1))
        try await store.apply(.upsert(doc("d1", "Second topic: alpine touring bindings."), sequence: 2))

        let gone = await store.lexicalSearch("espresso", maxTier: .open, limit: 5)
        XCTAssertTrue(gone.isEmpty, "old chunks must be fully unindexed after re-ingest")
        let present = await store.lexicalSearch("alpine", maxTier: .open, limit: 5)
        XCTAssertEqual(present.count, 1)
    }

    func testVectorBudgetRefusalsDegradeGracefully() async throws {
        // Budget of 0 → every chunk's vector is refused, but lexical search still works.
        let store = IndexStore(vectorByteBudget: 0)
        let result = try await store.apply(.upsert(doc("d1", "gracefully degraded but findable"), sequence: 1))
        guard case .applied(let chunks, let refusals) = result else {
            return XCTFail("expected .applied, got \(result)")
        }
        XCTAssertEqual(refusals, chunks, "with zero vector budget every vector insert must be refused")

        let lexicalHits = await store.lexicalSearch("findable", maxTier: .open, limit: 5)
        XCTAssertEqual(lexicalHits.count, 1, "lexical tier must survive vector refusals")
        let vectorHits = await store.vectorSearch([Float](repeating: 0.5, count: 64), maxTier: .open, limit: 5)
        XCTAssertTrue(vectorHits.isEmpty)
    }
}
