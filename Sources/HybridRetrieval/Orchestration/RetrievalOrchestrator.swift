/// Per-source outcome for one query — the report is a first-class product of retrieval,
/// not an afterthought: degraded answers MUST be distinguishable from complete ones at
/// the call site (a RAG layer that silently drops a timed-out source is lying to the
/// model about its own context).
public struct SourceReport: Sendable {
    public enum Disposition: Equatable, Sendable {
        case fulfilled(resultCount: Int)
        /// The source missed its deadline.
        case timedOut
        case failed(message: String)
        /// The caller cancelled the query while this source was in flight. Never
        /// folded into `timedOut` — a cancelled query says nothing about source health.
        case cancelled
    }

    public let source: SourceID
    public let disposition: Disposition
    public let latency: Duration
    /// Results this source returned above the query's allowed tier, dropped by the
    /// orchestrator's enforcement pass. Nonzero means the source violated its contract.
    public let privacyViolationsFiltered: Int
}

/// The complete answer to one retrieval call: fused hits plus the health report.
public struct RetrievalResponse: Sendable {
    public let hits: [RankedHit]
    public let reports: [SourceReport]

    /// True when every source fulfilled within its deadline.
    public var isComplete: Bool {
        reports.allSatisfy { report in
            if case .fulfilled = report.disposition { return true }
            return false
        }
    }

    /// True when the caller cancelled this query mid-flight. Such a response is
    /// abandoned work, not a degraded answer — callers should discard it rather than
    /// render it as a partial result.
    public var wasCancelled: Bool {
        reports.contains { $0.disposition == .cancelled }
    }

    /// Count of sources that fulfilled.
    public var fulfilledSourceCount: Int {
        reports.reduce(0) { count, report in
            if case .fulfilled = report.disposition { return count + 1 }
            return count
        }
    }
}

public struct OrchestratorConfiguration: Sendable {
    /// Hard per-source answer deadline. The fan-out's total latency is bounded by this
    /// (children run concurrently), so it is also the query's tail-latency budget.
    public var perSourceDeadline: Duration
    /// Maximum fused results returned.
    public var maxResults: Int
    /// Minimum number of fulfilled sources for the response to satisfy policy; below
    /// this, `meetsPolicy(_:)` is false and callers decide (fail, retry, or serve
    /// degraded with a disclosure).
    public var minimumFulfilledSources: Int

    public init(
        perSourceDeadline: Duration = .milliseconds(250),
        maxResults: Int = 10,
        minimumFulfilledSources: Int = 1
    ) {
        self.perSourceDeadline = perSourceDeadline > .zero ? perSourceDeadline : .milliseconds(250)
        self.maxResults = min(max(maxResults, 1), 500)
        self.minimumFulfilledSources = max(0, minimumFulfilledSources)
    }

    public func meetsPolicy(_ response: RetrievalResponse) -> Bool {
        response.fulfilledSourceCount >= minimumFulfilledSources
    }
}

/// Fans a query out to every registered source with a per-source deadline, enforces the
/// privacy boundary on whatever comes back, fuses the survivors, and reports per-source
/// health.
///
/// ## System behavior under failure (the macro design)
/// - A slow source costs at most `perSourceDeadline`; it cannot stall the query
///   (`Deadline.race` guarantees return even against cancellation-ignoring sources).
/// - A failing source is isolated: its error becomes a report entry, not a query error.
/// - A source violating the privacy contract has those results dropped *and counted* —
///   trust, but verify, then report.
/// - Fusion sees only fulfilled sources; RRF degrades gracefully with list absence.
///
/// The orchestrator itself is stateless: all state lives in `IndexStore` (behind its
/// actor) and in per-call locals. That makes it trivially `Sendable` and lets callers
/// run any number of concurrent queries.
public struct RetrievalOrchestrator: Sendable {
    private let sources: [any RetrievalSource]
    private let fuser: any RankFuser
    private let configuration: OrchestratorConfiguration

    /// Duplicate source IDs are rejected at construction (kept: first occurrence) —
    /// duplicate IDs would corrupt fusion weights and reports downstream.
    public init(
        sources: [any RetrievalSource],
        fuser: any RankFuser = ReciprocalRankFusion(),
        configuration: OrchestratorConfiguration = OrchestratorConfiguration()
    ) {
        var seen = Set<SourceID>()
        self.sources = sources.filter { seen.insert($0.id).inserted }
        self.fuser = fuser
        self.configuration = configuration
    }

    public func retrieve(_ query: RetrievalQuery) async -> RetrievalResponse {
        guard !sources.isEmpty else { return RetrievalResponse(hits: [], reports: []) }

        let deadline = configuration.perSourceDeadline
        let clock = ContinuousClock()

        // Fan-out. Each child is bounded by Deadline.race, so the group's implicit
        // waitForAll is bounded too — structured concurrency with a hard latency cap.
        let raw: [(SourceID, DeadlineOutcome<[ScoredChunk]>, Duration)] = await withTaskGroup(
            of: (SourceID, DeadlineOutcome<[ScoredChunk]>, Duration).self
        ) { group in
            for source in sources {
                group.addTask {
                    let start = clock.now
                    let outcome = await Deadline.race(limit: deadline) {
                        try await source.retrieve(query)
                    }
                    return (source.id, outcome, clock.now - start)
                }
            }
            var collected: [(SourceID, DeadlineOutcome<[ScoredChunk]>, Duration)] = []
            collected.reserveCapacity(sources.count)
            for await result in group {
                collected.append(result)
            }
            return collected
        }

        var reports: [SourceReport] = []
        reports.reserveCapacity(raw.count)
        var fusionInput: [(source: SourceID, hits: [ScoredChunk])] = []

        for (id, outcome, latency) in raw.sorted(by: { $0.0 < $1.0 }) {
            switch outcome {
            case .fulfilled(let hits):
                // Privacy enforcement at the trust boundary: drop and count anything
                // above the query's tier, regardless of what the source claims.
                let allowed = hits.filter { $0.tier <= query.maxTier }
                let violations = hits.count - allowed.count
                reports.append(SourceReport(
                    source: id,
                    disposition: .fulfilled(resultCount: allowed.count),
                    latency: latency,
                    privacyViolationsFiltered: violations
                ))
                fusionInput.append((source: id, hits: allowed))

            case .timedOut:
                reports.append(SourceReport(
                    source: id,
                    disposition: .timedOut,
                    latency: latency,
                    privacyViolationsFiltered: 0
                ))

            case .failed(let error):
                reports.append(SourceReport(
                    source: id,
                    disposition: .failed(message: String(describing: error)),
                    latency: latency,
                    privacyViolationsFiltered: 0
                ))

            case .cancelled:
                reports.append(SourceReport(
                    source: id,
                    disposition: .cancelled,
                    latency: latency,
                    privacyViolationsFiltered: 0
                ))
            }
        }

        let hits = fuser.fuse(fusionInput, limit: configuration.maxResults)
        return RetrievalResponse(hits: hits, reports: reports)
    }
}
