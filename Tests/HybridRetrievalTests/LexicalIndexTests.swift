import XCTest
@testable import HybridRetrieval

final class LexicalIndexTests: XCTestCase {
    private func id(_ doc: String, _ ordinal: Int = 0) -> ChunkID {
        ChunkID(document: DocumentID(doc), ordinal: ordinal)
    }

    /// Golden-value check with a hand-derived expectation (not recomputed via the
    /// implementation's own formula — that would be a vacuous mirror test).
    /// Corpus: c1 = "swift actor isolation" (3 tokens), c2 = "swift concurrency" (2),
    /// c3 = "breakfast burrito" (2). Query "actor": df = 1, N = 3,
    /// idf = ln(1 + 2.5/1.5) = 0.980829..., avgdl = 7/3, tf = 1, k1 = 1.2, b = 0.75:
    /// denom = 1 + 1.2 * (0.25 + 0.75 * 3/(7/3)) = 2.4571428...,
    /// score = 0.980829 * 2.2 / 2.4571428 = 0.87818...
    func testBM25MatchesHandComputedGoldenValue() {
        let index = LexicalIndex()
        index.insert(id: id("c1"), text: "swift actor isolation", tier: .open)
        index.insert(id: id("c2"), text: "swift concurrency", tier: .open)
        index.insert(id: id("c3"), text: "breakfast burrito", tier: .open)

        let hits = index.search("actor", maxTier: .open, limit: 10)
        XCTAssertEqual(hits.count, 1)
        XCTAssertEqual(hits.first?.id, id("c1"))
        XCTAssertEqual(hits.first?.score ?? 0, 0.87818, accuracy: 0.0005)
    }

    /// Rarer terms must out-rank common terms — the core IDF property. This fails if
    /// scoring is gutted to constant or term-frequency-only ranking.
    func testRareTermOutranksCommonTerm() {
        let index = LexicalIndex()
        index.insert(id: id("a"), text: "swift swift swift common", tier: .open)
        index.insert(id: id("b"), text: "swift rareterm", tier: .open)
        index.insert(id: id("c"), text: "swift other words", tier: .open)

        let hits = index.search("rareterm swift", maxTier: .open, limit: 10)
        XCTAssertEqual(hits.first?.id, id("b"), "chunk containing the rare term must rank first")
    }

    func testRemovalReversesStatistics() {
        let index = LexicalIndex()
        index.insert(id: id("a"), text: "alpha beta", tier: .open)
        index.insert(id: id("b"), text: "alpha gamma", tier: .open)
        index.remove(id: id("b"))

        XCTAssertEqual(index.count, 1)
        XCTAssertTrue(index.search("gamma", maxTier: .open, limit: 5).isEmpty)
        XCTAssertEqual(index.search("alpha", maxTier: .open, limit: 5).count, 1)
    }

    func testReinsertionDoesNotDoubleCount() {
        let index = LexicalIndex()
        index.insert(id: id("a"), text: "alpha alpha", tier: .open)
        let first = index.search("alpha", maxTier: .open, limit: 5).first?.score ?? -1
        index.insert(id: id("a"), text: "alpha alpha", tier: .open)
        let second = index.search("alpha", maxTier: .open, limit: 5).first?.score ?? -2
        XCTAssertEqual(index.count, 1)
        XCTAssertEqual(first, second, accuracy: 1e-12, "re-indexing the same chunk must be idempotent")
    }

    func testTierFilteringExcludesAboveMaxTier() {
        let index = LexicalIndex()
        index.insert(id: id("open"), text: "shared secret handling guide", tier: .open)
        index.insert(id: id("vault"), text: "secret vault entry", tier: .sensitive)

        let openHits = index.search("secret", maxTier: .open, limit: 5)
        XCTAssertEqual(openHits.map(\.id), [id("open")])

        let sensitiveHits = index.search("secret", maxTier: .sensitive, limit: 5)
        XCTAssertEqual(sensitiveHits.count, 2)
    }

    func testEmptyCorpusEmptyQueryAndUnknownTermAreSafe() {
        let index = LexicalIndex()
        XCTAssertTrue(index.search("anything", maxTier: .open, limit: 5).isEmpty)
        index.insert(id: id("a"), text: "hello world", tier: .open)
        XCTAssertTrue(index.search("", maxTier: .open, limit: 5).isEmpty)
        XCTAssertTrue(index.search("   \n\t ", maxTier: .open, limit: 5).isEmpty)
        XCTAssertTrue(index.search("zzzunknown", maxTier: .open, limit: 5).isEmpty)
        XCTAssertTrue(index.search("hello", maxTier: .open, limit: 0).isEmpty)
        XCTAssertTrue(index.search("hello", maxTier: .open, limit: -3).isEmpty)
    }

    func testWhitespaceOnlyChunkIsNotIndexed() {
        let index = LexicalIndex()
        index.insert(id: id("blank"), text: "   \n\t  ", tier: .open)
        XCTAssertEqual(index.count, 0)
    }

    func testDeterministicTieBreakByChunkID() {
        let index = LexicalIndex()
        // Identical texts → identical scores → order must fall back to ChunkID.
        index.insert(id: id("b"), text: "same words here", tier: .open)
        index.insert(id: id("a"), text: "same words here", tier: .open)
        let hits = index.search("same", maxTier: .open, limit: 5)
        XCTAssertEqual(hits.map(\.id), [id("a"), id("b")])
    }

    func testHostileParametersAreClamped() {
        let params = LexicalIndex.Parameters(k1: .nan, b: .infinity)
        XCTAssertEqual(params.k1, 1.2)
        XCTAssertEqual(params.b, 0.75)
        let index = LexicalIndex(parameters: LexicalIndex.Parameters(k1: -5, b: 9))
        index.insert(id: id("a"), text: "alpha", tier: .open)
        let hits = index.search("alpha", maxTier: .open, limit: 1)
        XCTAssertEqual(hits.count, 1)
        XCTAssertTrue((hits.first?.score ?? -1) >= 0, "clamped parameters must not produce negative scores")
    }
}
