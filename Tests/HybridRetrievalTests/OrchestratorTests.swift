import XCTest
@testable import HybridRetrieval

// MARK: - Test doubles

private struct StubSource: RetrievalSource {
    let id: SourceID
    let result: [ScoredChunk]

    func retrieve(_ query: RetrievalQuery) async throws -> [ScoredChunk] {
        result
    }
}

private struct FailingSource: RetrievalSource {
    struct Boom: Error {}
    let id: SourceID
    func retrieve(_ query: RetrievalQuery) async throws -> [ScoredChunk] {
        throw Boom()
    }
}

/// Cooperative slowness: sleeps far past any test deadline but exits promptly when
/// cancelled (the well-behaved-but-slow backend).
private struct SlowCooperativeSource: RetrievalSource {
    let id: SourceID
    func retrieve(_ query: RetrievalQuery) async throws -> [ScoredChunk] {
        try await Task.sleep(nanoseconds: 10_000_000_000)
        return []
    }
}

/// The adversarial case: ignores cancellation entirely for ~3s. Bounded so a test run
/// can never leak unbounded work.
private struct UncooperativeSource: RetrievalSource {
    let id: SourceID
    func retrieve(_ query: RetrievalQuery) async throws -> [ScoredChunk] {
        for _ in 0..<30 {
            try? await Task.sleep(nanoseconds: 100_000_000) // swallows cancellation
        }
        return []
    }
}

/// A source that violates the privacy contract by returning sensitive content to any
/// query — the deliberately broken implementation that must make the enforcement fire.
private struct LeakySource: RetrievalSource {
    let id: SourceID
    func retrieve(_ query: RetrievalQuery) async throws -> [ScoredChunk] {
        [
            ScoredChunk(id: ChunkID(document: DocumentID("leaked"), ordinal: 0),
                        text: "SSN 000-00-0000", tier: .sensitive, score: 99),
            ScoredChunk(id: ChunkID(document: DocumentID("fine"), ordinal: 0),
                        text: "public fact", tier: .open, score: 1)
        ]
    }
}

private func chunk(_ doc: String, score: Double, tier: PrivacyTier = .open) -> ScoredChunk {
    ScoredChunk(id: ChunkID(document: DocumentID(doc), ordinal: 0), text: doc, tier: tier, score: score)
}

// MARK: - Tests

final class OrchestratorTests: XCTestCase {
    func testFanOutFusesMultipleSources() async {
        let orchestrator = RetrievalOrchestrator(sources: [
            StubSource(id: "s1", result: [chunk("A", score: 2), chunk("B", score: 1)]),
            StubSource(id: "s2", result: [chunk("B", score: 5), chunk("C", score: 4)])
        ])
        let response = await orchestrator.retrieve(RetrievalQuery(text: "q"))

        XCTAssertTrue(response.isComplete)
        XCTAssertEqual(response.fulfilledSourceCount, 2)
        XCTAssertEqual(response.hits.first?.chunk.id.document.rawValue, "B",
                       "the consensus chunk must rank first under RRF")
        XCTAssertEqual(response.hits.count, 3)
    }

    func testTimedOutSourceYieldsPartialResultsWithinDeadline() async {
        let orchestrator = RetrievalOrchestrator(
            sources: [
                StubSource(id: "fast", result: [chunk("A", score: 1)]),
                SlowCooperativeSource(id: "slow")
            ],
            configuration: OrchestratorConfiguration(perSourceDeadline: .milliseconds(150))
        )

        let clock = ContinuousClock()
        let start = clock.now
        let response = await orchestrator.retrieve(RetrievalQuery(text: "q"))
        let elapsed = clock.now - start

        XCTAssertLessThan(elapsed, .seconds(2), "a 10s-sleep source must not stall a 150ms-deadline query")
        XCTAssertFalse(response.isComplete)
        XCTAssertEqual(response.hits.map { $0.chunk.id.document.rawValue }, ["A"],
                       "partial results from healthy sources must still be served")

        let slowReport = response.reports.first { $0.source == "slow" }
        XCTAssertEqual(slowReport?.disposition, .timedOut)
        let fastReport = response.reports.first { $0.source == "fast" }
        XCTAssertEqual(fastReport?.disposition, .fulfilled(resultCount: 1))
    }

    /// The guarantee the README leads with: even a source that IGNORES cancellation
    /// cannot stall the caller past the deadline.
    func testUncooperativeSourceCannotStallTheQuery() async {
        let orchestrator = RetrievalOrchestrator(
            sources: [
                StubSource(id: "fast", result: [chunk("A", score: 1)]),
                UncooperativeSource(id: "rogue")
            ],
            configuration: OrchestratorConfiguration(perSourceDeadline: .milliseconds(150))
        )

        let clock = ContinuousClock()
        let start = clock.now
        let response = await orchestrator.retrieve(RetrievalQuery(text: "q"))
        let elapsed = clock.now - start

        XCTAssertLessThan(elapsed, .seconds(2),
                          "cancellation-ignoring source must be abandoned at the deadline, not awaited")
        XCTAssertEqual(response.reports.first { $0.source == "rogue" }?.disposition, .timedOut)
        XCTAssertEqual(response.hits.count, 1)
    }

    func testFailingSourceIsIsolatedNotFatal() async {
        let orchestrator = RetrievalOrchestrator(sources: [
            StubSource(id: "healthy", result: [chunk("A", score: 1)]),
            FailingSource(id: "broken")
        ])
        let response = await orchestrator.retrieve(RetrievalQuery(text: "q"))

        XCTAssertEqual(response.hits.count, 1, "one broken source must not empty the response")
        guard case .failed(let message)? = response.reports.first(where: { $0.source == "broken" })?.disposition else {
            return XCTFail("broken source must be reported as failed")
        }
        XCTAssertTrue(message.contains("Boom"), "failure report should carry the underlying error")
    }

    /// Defense-in-depth check with a deliberately broken implementation: if this test
    /// fails, the privacy boundary does not actually exist outside the built-in sources.
    func testPrivacyEnforcementDropsAndCountsContractViolations() async {
        let orchestrator = RetrievalOrchestrator(sources: [LeakySource(id: "leaky")])
        let response = await orchestrator.retrieve(RetrievalQuery(text: "q", maxTier: .open))

        XCTAssertEqual(response.hits.count, 1)
        XCTAssertEqual(response.hits.first?.chunk.id.document.rawValue, "fine")
        XCTAssertFalse(response.hits.contains { $0.chunk.tier > .open },
                       "no above-tier chunk may survive orchestrator enforcement")

        let report = response.reports.first { $0.source == "leaky" }
        XCTAssertEqual(report?.privacyViolationsFiltered, 1,
                       "the violation must be counted, not silently dropped")
        XCTAssertEqual(report?.disposition, .fulfilled(resultCount: 1))
    }

    func testSensitiveQueryContextMaySeeSensitiveContent() async {
        let orchestrator = RetrievalOrchestrator(sources: [LeakySource(id: "leaky")])
        let response = await orchestrator.retrieve(RetrievalQuery(text: "q", maxTier: .sensitive))
        XCTAssertEqual(response.hits.count, 2, "a sensitive-tier context is allowed sensitive results")
        XCTAssertEqual(response.reports.first?.privacyViolationsFiltered, 0)
    }

    func testMinimumFulfilledSourcesPolicy() async {
        let config = OrchestratorConfiguration(perSourceDeadline: .milliseconds(150), minimumFulfilledSources: 2)
        let orchestrator = RetrievalOrchestrator(
            sources: [
                StubSource(id: "only", result: [chunk("A", score: 1)]),
                SlowCooperativeSource(id: "slow")
            ],
            configuration: config
        )
        let response = await orchestrator.retrieve(RetrievalQuery(text: "q"))
        XCTAssertFalse(config.meetsPolicy(response), "1 fulfilled < required 2 must fail policy")
        XCTAssertEqual(response.fulfilledSourceCount, 1)
    }

    func testNoSourcesAndDuplicateIDsAreSafe() async {
        let empty = RetrievalOrchestrator(sources: [])
        let emptyResponse = await empty.retrieve(RetrievalQuery(text: "q"))
        XCTAssertTrue(emptyResponse.hits.isEmpty)
        XCTAssertTrue(emptyResponse.reports.isEmpty)

        let duplicated = RetrievalOrchestrator(sources: [
            StubSource(id: "dup", result: [chunk("A", score: 1)]),
            StubSource(id: "dup", result: [chunk("B", score: 1)])
        ])
        let response = await duplicated.retrieve(RetrievalQuery(text: "q"))
        XCTAssertEqual(response.reports.count, 1, "duplicate source IDs must be rejected at construction")
        XCTAssertEqual(response.hits.map { $0.chunk.id.document.rawValue }, ["A"])
    }

    func testEndToEndEngineRoundTrip() async throws {
        let engine = HybridSearchEngine(
            configuration: OrchestratorConfiguration(perSourceDeadline: .seconds(2), maxResults: 5)
        )
        try await engine.apply(.upsert(Document(
            id: DocumentID("swift-actors"),
            text: "Actors protect mutable state. Actor isolation is checked at compile time."
        ), sequence: 1))
        try await engine.apply(.upsert(Document(
            id: DocumentID("coffee"),
            text: "Espresso extraction depends on grind size and water temperature."
        ), sequence: 2))

        let response = await engine.search(RetrievalQuery(text: "actor isolation"))
        XCTAssertTrue(response.isComplete)
        XCTAssertEqual(response.hits.first?.chunk.id.document, DocumentID("swift-actors"))
        XCTAssertEqual(response.reports.count, 2, "engine wires exactly lexical + vector by default")
    }
}
