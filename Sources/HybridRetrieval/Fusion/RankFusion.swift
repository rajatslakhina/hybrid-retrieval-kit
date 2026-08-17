/// A fused, final ranking entry.
public struct RankedHit: Sendable {
    public let chunk: ScoredChunk
    public let fusedScore: Double
    /// Which sources contributed this chunk — the observability hook that lets you see
    /// *why* a result ranked where it did (lexical-only, vector-only, or both).
    public let contributingSources: [SourceID]
}

/// Merges per-source ranked lists into one ranking.
public protocol RankFuser: Sendable {
    func fuse(_ lists: [(source: SourceID, hits: [ScoredChunk])], limit: Int) -> [RankedHit]
}

/// Weighted Reciprocal Rank Fusion: `score(c) = Σ_s w_s / (k + rank_s(c))`.
///
/// ## Why RRF and not score normalization (the load-bearing fusion decision)
/// BM25 scores are unbounded and corpus-dependent; cosine lives in [-1, 1]; a Spotlight
/// source may return no scores at all. Min-max normalizing such scales is brittle: one
/// outlier rescales an entire list, and an empty or single-element list has no range.
/// RRF uses only *ranks*, which every source can produce honestly, and degrades
/// gracefully when a source times out (its list is simply absent). The cost — absolute
/// score magnitudes are discarded — is the right trade for heterogeneous sources.
/// `k` (default 60, from the original RRF paper) damps the head: smaller k lets a #1
/// hit dominate; larger k flattens toward consensus.
///
/// Determinism: ties break on `ChunkID`, so identical inputs always produce identical
/// output order — a hard requirement for testability and for cache-key stability above.
public struct ReciprocalRankFusion: RankFuser {
    public let k: Double
    public let weights: [SourceID: Double]

    /// - Parameters:
    ///   - k: rank damping; clamped to 1...10_000 (k must be >= 1 so the denominator
    ///     `k + rank` is always positive).
    ///   - weights: per-source multipliers; negative or non-finite weights are treated
    ///     as 0 (a source can be muted, never inverted).
    public init(k: Double = 60, weights: [SourceID: Double] = [:]) {
        self.k = k.isFinite ? min(max(k, 1), 10_000) : 60
        self.weights = weights.mapValues { weight in
            weight.isFinite ? max(0, weight) : 0
        }
    }

    public func fuse(_ lists: [(source: SourceID, hits: [ScoredChunk])], limit: Int) -> [RankedHit] {
        guard limit > 0 else { return [] }

        struct Accumulated {
            var chunk: ScoredChunk
            var score: Double
            var sources: [SourceID]
        }
        var accumulated: [ChunkID: Accumulated] = [:]

        // Deterministic iteration: process sources in ID order regardless of fan-out
        // completion order.
        for (source, hits) in lists.sorted(by: { $0.source < $1.source }) {
            let weight = weights[source] ?? 1
            guard weight > 0 else { continue }

            // Defensive re-sort + de-dup: the fuser does not trust sources to order
            // correctly or to avoid repeats (a repeated ID would otherwise double-dip).
            var seen = Set<ChunkID>()
            let ordered = hits
                .sorted { lhs, rhs in
                    if lhs.score != rhs.score { return lhs.score > rhs.score }
                    return lhs.id < rhs.id
                }
                .filter { seen.insert($0.id).inserted }

            for (index, hit) in ordered.enumerated() {
                // rank is 1-based; k >= 1, so the denominator is >= 2.
                let contribution = weight / (k + Double(index + 1))
                guard contribution.isUsableScore else { continue }
                if var existing = accumulated[hit.id] {
                    existing.score += contribution
                    existing.sources.append(source)
                    accumulated[hit.id] = existing
                } else {
                    accumulated[hit.id] = Accumulated(chunk: hit, score: contribution, sources: [source])
                }
            }
        }

        return accumulated
            .values
            .sorted { lhs, rhs in
                if lhs.score != rhs.score { return lhs.score > rhs.score }
                return lhs.chunk.id < rhs.chunk.id
            }
            .prefix(limit)
            .map { RankedHit(chunk: $0.chunk, fusedScore: $0.score, contributingSources: $0.sources) }
    }
}
