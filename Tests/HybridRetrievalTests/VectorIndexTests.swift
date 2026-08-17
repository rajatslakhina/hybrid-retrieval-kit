import XCTest
@testable import HybridRetrieval

final class VectorIndexTests: XCTestCase {
    private func id(_ doc: String, _ ordinal: Int = 0) -> ChunkID {
        ChunkID(document: DocumentID(doc), ordinal: ordinal)
    }

    private func vector(_ direction: Int, dimensions: Int = 4) -> [Float] {
        var v = [Float](repeating: 0, count: dimensions)
        // direction is always constructed in-range by callers; clamp anyway.
        let index = min(max(direction, 0), dimensions - 1)
        v[index] = 1
        return v
    }

    func testBudgetIsNeverExceededAndLRUEvicts() {
        // Each 4-dim entry costs 4*4 + 96 = 112 bytes; budget fits exactly two.
        let index = VectorIndex(dimensions: 4, byteBudget: 224)

        XCTAssertEqual(index.insert(id: id("a"), vector: vector(0), tier: .open), .stored)
        XCTAssertEqual(index.insert(id: id("b"), vector: vector(1), tier: .open), .stored)
        XCTAssertEqual(index.count, 2)
        XCTAssertLessThanOrEqual(index.usedBytes, 224)

        // Touch "a" so "b" becomes least recently used…
        _ = index.search(vector(0), maxTier: .open, limit: 1)
        // …then force an eviction. The victim must be "b", not "a".
        XCTAssertEqual(index.insert(id: id("c"), vector: vector(2), tier: .open), .stored)

        XCTAssertEqual(index.count, 2)
        XCTAssertLessThanOrEqual(index.usedBytes, 224)
        XCTAssertEqual(index.evictionCount, 1)
        XCTAssertFalse(index.search(vector(1), maxTier: .open, limit: 3).contains { $0.id == self.id("b") },
                       "LRU victim must be the untouched entry")
        XCTAssertTrue(index.search(vector(0), maxTier: .open, limit: 3).contains { $0.id == self.id("a") },
                      "recently-searched entry must survive eviction")
    }

    func testEvictionUnderSustainedPressureKeepsInvariant() {
        let index = VectorIndex(dimensions: 4, byteBudget: 400)
        for i in 0..<200 {
            index.insert(id: id("doc\(i)"), vector: vector(i % 4), tier: .open)
            XCTAssertLessThanOrEqual(index.usedBytes, 400, "budget invariant must hold after every insert")
        }
        XCTAssertGreaterThan(index.evictionCount, 0, "sustained pressure must actually evict")
        XCTAssertGreaterThan(index.count, 0)
    }

    func testZeroBudgetStoresNothingWithoutCrashing() {
        let index = VectorIndex(dimensions: 4, byteBudget: 0)
        XCTAssertEqual(index.insert(id: id("a"), vector: vector(0), tier: .open), .rejectedExceedsBudget)
        XCTAssertEqual(index.count, 0)
        XCTAssertEqual(index.usedBytes, 0)
        XCTAssertTrue(index.search(vector(0), maxTier: .open, limit: 5).isEmpty)
    }

    func testDimensionMismatchAndDegenerateVectorsAreRefusedNotCrashes() {
        let index = VectorIndex(dimensions: 4, byteBudget: 10_000)
        XCTAssertEqual(index.insert(id: id("short"), vector: [1, 0], tier: .open), .rejectedDimensionMismatch)
        XCTAssertEqual(index.insert(id: id("zero"), vector: [0, 0, 0, 0], tier: .open), .rejectedDegenerateVector)
        XCTAssertEqual(index.insert(id: id("nan"), vector: [.nan, 0, 0, 0], tier: .open), .rejectedDegenerateVector)
        XCTAssertEqual(index.insert(id: id("inf"), vector: [.infinity, 0, 0, 0], tier: .open), .rejectedDegenerateVector)
        XCTAssertEqual(index.count, 0)

        // Degenerate *queries* must also be inert.
        index.insert(id: id("ok"), vector: vector(0), tier: .open)
        XCTAssertTrue(index.search([0, 0, 0, 0], maxTier: .open, limit: 5).isEmpty)
        XCTAssertTrue(index.search([.nan, 0, 0, 0], maxTier: .open, limit: 5).isEmpty)
        XCTAssertTrue(index.search([1, 0], maxTier: .open, limit: 5).isEmpty)
    }

    /// Pure ordering check, so the floor is opted out (`minimumSimilarity: -1`) —
    /// otherwise the near-orthogonal "north" entry is correctly filtered before ranking
    /// and this test would be silently checking two elements instead of three.
    func testCosineRankingOrdersByDirection() {
        let index = VectorIndex(dimensions: 4, byteBudget: 10_000, minimumSimilarity: -1)
        index.insert(id: id("east"), vector: [1, 0, 0, 0], tier: .open)
        index.insert(id: id("northeast"), vector: [1, 1, 0, 0], tier: .open)
        index.insert(id: id("north"), vector: [0, 1, 0, 0], tier: .open)

        let hits = index.search([1, 0.1, 0, 0], maxTier: .open, limit: 3)
        XCTAssertEqual(hits.map(\.id), [id("east"), id("northeast"), id("north")])
        // Scores are cosines of normalized vectors: bounded and ordered.
        for hit in hits {
            XCTAssertTrue(hit.score >= -1.0001 && hit.score <= 1.0001)
        }
    }

    func testTierFilteringInVectorSearch() {
        let index = VectorIndex(dimensions: 4, byteBudget: 10_000)
        index.insert(id: id("open"), vector: vector(0), tier: .open)
        index.insert(id: id("secret"), vector: vector(0), tier: .sensitive)

        let openHits = index.search(vector(0), maxTier: .open, limit: 5)
        XCTAssertEqual(openHits.map(\.id), [id("open")])
        let allHits = index.search(vector(0), maxTier: .sensitive, limit: 5)
        XCTAssertEqual(allHits.count, 2)
    }

    /// The floor is what makes this a *search* rather than a ranking: an orthogonal
    /// query must return nothing at all, not the whole corpus ordered by noise.
    /// Removing the `score >= minimumSimilarity` guard makes both halves fail.
    func testSimilarityFloorRejectsOrthogonalMatches() {
        let index = VectorIndex(dimensions: 4, byteBudget: 10_000, minimumSimilarity: 0.2)
        index.insert(id: id("east"), vector: [1, 0, 0, 0], tier: .open)
        index.insert(id: id("north"), vector: [0, 1, 0, 0], tier: .open)

        // Orthogonal to both stored vectors: cosine 0 < 0.2.
        XCTAssertTrue(index.search([0, 0, 1, 0], maxTier: .open, limit: 10).isEmpty,
                      "a query matching nothing must return nothing, not noise")
        // Aligned query still returns its match.
        XCTAssertEqual(index.search([1, 0, 0, 0], maxTier: .open, limit: 10).map(\.id), [id("east")])
    }

    func testFloorOfMinusOneRestoresPureRankingMode() {
        let index = VectorIndex(dimensions: 4, byteBudget: 10_000, minimumSimilarity: -1)
        index.insert(id: id("east"), vector: [1, 0, 0, 0], tier: .open)
        index.insert(id: id("north"), vector: [0, 1, 0, 0], tier: .open)
        XCTAssertEqual(index.search([0, 0, 1, 0], maxTier: .open, limit: 10).count, 2,
                       "an opted-out floor must return every candidate")
    }

    func testHostileFloorIsClamped() {
        XCTAssertEqual(VectorIndex(dimensions: 4, byteBudget: 10, minimumSimilarity: .nan).minimumSimilarity, 0.2)
        XCTAssertEqual(VectorIndex(dimensions: 4, byteBudget: 10, minimumSimilarity: 9).minimumSimilarity, 1)
        XCTAssertEqual(VectorIndex(dimensions: 4, byteBudget: 10, minimumSimilarity: -9).minimumSimilarity, -1)
    }

    func testReplacementUpdatesAccountingExactly() {
        let index = VectorIndex(dimensions: 4, byteBudget: 10_000)
        index.insert(id: id("a"), vector: vector(0), tier: .open)
        let after = index.usedBytes
        XCTAssertEqual(index.insert(id: id("a"), vector: vector(1), tier: .open), .replacedExisting)
        XCTAssertEqual(index.usedBytes, after, "replacing an entry must not leak accounting")
        XCTAssertEqual(index.count, 1)
    }
}
