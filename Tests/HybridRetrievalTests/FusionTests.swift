import XCTest
@testable import HybridRetrieval

final class FusionTests: XCTestCase {
    private func chunk(_ doc: String, score: Double, tier: PrivacyTier = .open) -> ScoredChunk {
        ScoredChunk(id: ChunkID(document: DocumentID(doc), ordinal: 0), text: doc, tier: tier, score: score)
    }

    /// Hand-computed golden RRF values (k = 60, weights = 1):
    /// lexical list: A(rank 1), B(rank 2). vector list: B(rank 1), C(rank 2).
    /// score(A) = 1/61 = 0.016393..., score(B) = 1/62 + 1/61 = 0.032524...,
    /// score(C) = 1/62 = 0.016129... → order B, A, C.
    func testRRFMatchesHandComputedGoldenValues() {
        let fuser = ReciprocalRankFusion(k: 60)
        let hits = fuser.fuse([
            (source: "lexical", hits: [chunk("A", score: 9.0), chunk("B", score: 5.0)]),
            (source: "vector", hits: [chunk("B", score: 0.9), chunk("C", score: 0.8)])
        ], limit: 10)

        XCTAssertEqual(hits.map { $0.chunk.id.document.rawValue }, ["B", "A", "C"])
        XCTAssertEqual(hits[0].fusedScore, 1.0 / 62 + 1.0 / 61, accuracy: 1e-9)
        XCTAssertEqual(hits[1].fusedScore, 1.0 / 61, accuracy: 1e-9)
        XCTAssertEqual(hits[2].fusedScore, 1.0 / 62, accuracy: 1e-9)
        XCTAssertEqual(hits[0].contributingSources.sorted(), ["lexical", "vector"],
                       "consensus hit must credit both sources")
    }

    func testWeightsBiasFusion() {
        // Same lists; boosting the vector source must flip the winner from A to B…
        let boosted = ReciprocalRankFusion(k: 60, weights: ["vector": 3.0])
        let hits = boosted.fuse([
            (source: "lexical", hits: [chunk("A", score: 9.0)]),
            (source: "vector", hits: [chunk("B", score: 0.9)])
        ], limit: 10)
        XCTAssertEqual(hits.first?.chunk.id.document.rawValue, "B")

        // …and muting it must remove its results entirely.
        let muted = ReciprocalRankFusion(k: 60, weights: ["vector": 0])
        let mutedHits = muted.fuse([
            (source: "lexical", hits: [chunk("A", score: 9.0)]),
            (source: "vector", hits: [chunk("B", score: 0.9)])
        ], limit: 10)
        XCTAssertEqual(mutedHits.map { $0.chunk.id.document.rawValue }, ["A"])
    }

    func testDuplicateIDsWithinOneSourceCannotDoubleDip() {
        let fuser = ReciprocalRankFusion(k: 60)
        let repeated = chunk("A", score: 9.0)
        let hits = fuser.fuse([
            (source: "cheater", hits: [repeated, repeated, repeated]),
            (source: "honest", hits: [chunk("B", score: 1.0)])
        ], limit: 10)

        // A appears once at rank 1: score exactly 1/61 — tripling would give ~3/61.
        let a = hits.first { $0.chunk.id.document.rawValue == "A" }
        XCTAssertEqual(a?.fusedScore ?? 0, 1.0 / 61, accuracy: 1e-9,
                       "a source repeating an ID must not multiply its contribution")
    }

    func testFusionIsDeterministicRegardlessOfListOrder() {
        let fuser = ReciprocalRankFusion(k: 60)
        let listA = (source: SourceID("s1"), hits: [chunk("A", score: 2.0), chunk("B", score: 1.0)])
        let listB = (source: SourceID("s2"), hits: [chunk("C", score: 2.0), chunk("A", score: 1.0)])

        let forward = fuser.fuse([listA, listB], limit: 10)
        let reversed = fuser.fuse([listB, listA], limit: 10)
        XCTAssertEqual(forward.map { $0.chunk.id }, reversed.map { $0.chunk.id },
                       "fan-out completion order must never change the ranking")
    }

    func testEqualScoresTieBreakByChunkID() {
        let fuser = ReciprocalRankFusion(k: 60)
        // Two sources, each ranking a different chunk first → symmetric scores.
        let hits = fuser.fuse([
            (source: "s1", hits: [chunk("B", score: 1.0)]),
            (source: "s2", hits: [chunk("A", score: 1.0)])
        ], limit: 10)
        XCTAssertEqual(hits.map { $0.chunk.id.document.rawValue }, ["A", "B"],
                       "equal fused scores must tie-break on ChunkID for determinism")
    }

    func testEmptyListsAndZeroLimitAreSafe() {
        let fuser = ReciprocalRankFusion()
        XCTAssertTrue(fuser.fuse([], limit: 10).isEmpty)
        XCTAssertTrue(fuser.fuse([(source: "s", hits: [])], limit: 10).isEmpty)
        XCTAssertTrue(fuser.fuse([(source: "s", hits: [chunk("A", score: 1)])], limit: 0).isEmpty)
        XCTAssertTrue(fuser.fuse([(source: "s", hits: [chunk("A", score: 1)])], limit: -2).isEmpty)
    }

    func testHostileParametersAreClamped() {
        let fuser = ReciprocalRankFusion(k: .nan, weights: ["s": -4, "t": .infinity])
        XCTAssertEqual(fuser.k, 60)
        XCTAssertEqual(fuser.weights["s"], 0)
        XCTAssertEqual(fuser.weights["t"], 0)

        let tiny = ReciprocalRankFusion(k: 0.001)
        XCTAssertEqual(tiny.k, 1, "k must be clamped >= 1 so k + rank can never be zero")
    }

    func testNonFiniteInputScoresAreNeutralizedAtConstruction() {
        let poisoned = ScoredChunk(
            id: ChunkID(document: DocumentID("x"), ordinal: 0),
            text: "x", tier: .open, score: .nan
        )
        XCTAssertEqual(poisoned.score, 0, "ScoredChunk must clamp non-finite scores at the boundary")
    }
}
