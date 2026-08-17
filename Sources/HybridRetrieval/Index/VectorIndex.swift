/// A hit from the vector index: score + tier, no text (text lives once, in `IndexStore`).
public struct VectorHit: Sendable {
    public let id: ChunkID
    public let tier: PrivacyTier
    public let score: Double
}

/// Why an insert can be refused. Refusals are results, not errors — the indexing
/// pipeline treats "this chunk is not vector-searchable" as a normal, reportable state
/// (the chunk remains lexically searchable), not an exception.
public enum VectorInsertOutcome: Equatable, Sendable {
    case stored
    case replacedExisting
    /// Vector had the wrong dimension count for this index.
    case rejectedDimensionMismatch
    /// Vector was empty/zero/non-finite and cannot participate in cosine similarity.
    case rejectedDegenerateVector
    /// A single entry larger than the whole budget can never fit.
    case rejectedExceedsBudget
}

/// Brute-force cosine-similarity index under a hard byte budget with LRU eviction.
///
/// Design decisions, stated so they can be argued with:
///
/// - **Brute force, not HNSW/IVF.** On-device corpora are 10³–10⁵ chunks; at 64–768
///   dims a linear scan is microseconds-to-low-milliseconds and exactly correct.
///   An approximate-nearest-neighbor graph buys nothing at this scale and costs
///   memory, build time, and a recall knob someone must now own. The seam to revisit
///   is `IndexStore`, not this class's API.
/// - **Byte budget, not entry count.** Entries can have different real costs as
///   dimensioning changes; budgeting entry *count* is how caches "under budget" their
///   way into memory-pressure jetsam. Costs use saturating arithmetic — saturation
///   over-estimates cost, which makes eviction more aggressive, never less.
/// - **LRU on insert and on query hit.** A chunk that keeps appearing in results is
///   exactly the chunk to keep; recency updates on read are what make that true.
///
/// Concurrency: NOT thread-safe; confined inside `IndexStore` (an actor), same
/// rationale as `LexicalIndex`.
public final class VectorIndex {
    private struct Entry {
        let vector: [Float]
        let tier: PrivacyTier
        let cost: Int
        var lastTouched: UInt64
    }

    public let dimensions: Int
    public let byteBudget: Int
    private var entries: [ChunkID: Entry] = [:]
    private var clock: UInt64 = 0
    public private(set) var usedBytes = 0
    /// Total evictions since creation — surfaced for observability and tests.
    public private(set) var evictionCount = 0

    /// - Parameters:
    ///   - dimensions: clamped to 4...4096.
    ///   - byteBudget: clamped to be non-negative. A budget of 0 stores nothing (and
    ///     every insert reports `rejectedExceedsBudget`) — legal, and tested.
    public init(dimensions: Int, byteBudget: Int) {
        self.dimensions = min(max(dimensions, 4), 4096)
        self.byteBudget = max(0, byteBudget)
    }

    public var count: Int { entries.count }

    /// Estimated resident cost of one entry: vector payload + fixed bookkeeping
    /// overhead (key, tier, recency stamp, dictionary slot). The overhead constant is
    /// an engineering estimate, deliberately rounded up.
    static let perEntryOverheadBytes = 96

    private func cost(of vector: [Float]) -> Int {
        vector.count
            .multiplyingSaturated(MemoryLayout<Float>.stride)
            .addingSaturated(Self.perEntryOverheadBytes)
    }

    @discardableResult
    public func insert(id: ChunkID, vector: [Float], tier: PrivacyTier) -> VectorInsertOutcome {
        guard vector.count == dimensions else { return .rejectedDimensionMismatch }
        guard let normalized = VectorMath.l2Normalized(vector) else { return .rejectedDegenerateVector }
        let entryCost = cost(of: normalized)
        guard entryCost <= byteBudget else { return .rejectedExceedsBudget }

        let replaced = remove(id: id)
        evictUntilFits(incomingCost: entryCost)
        clock &+= 1 // Wrapping by design: UInt64 recency stamps outlive any process.
        entries[id] = Entry(vector: normalized, tier: tier, cost: entryCost, lastTouched: clock)
        usedBytes = usedBytes.addingSaturated(entryCost)
        return replaced ? .replacedExisting : .stored
    }

    @discardableResult
    public func remove(id: ChunkID) -> Bool {
        guard let entry = entries.removeValue(forKey: id) else { return false }
        usedBytes = max(0, usedBytes.subtractingSaturated(entry.cost))
        return true
    }

    /// Evicts least-recently-touched entries until `usedBytes + incomingCost <= byteBudget`.
    /// Terminates unconditionally: each pass removes one entry, and the loop exits when
    /// the dictionary is empty even if accounting were somehow inconsistent.
    private func evictUntilFits(incomingCost: Int) {
        while usedBytes.addingSaturated(incomingCost) > byteBudget {
            guard let victim = entries.min(by: { $0.value.lastTouched < $1.value.lastTouched }) else {
                return // Nothing left to evict; incoming entry fits by the budget precheck.
            }
            entries.removeValue(forKey: victim.key)
            usedBytes = max(0, usedBytes.subtractingSaturated(victim.value.cost))
            evictionCount = evictionCount.addingSaturated(1)
        }
    }

    /// Cosine-similarity search over entries with tier `<= maxTier`. The query vector
    /// is normalized here; a degenerate query returns `[]`, never a crash. Returned
    /// entries have their recency refreshed (see type doc for why).
    public func search(_ query: [Float], maxTier: PrivacyTier, limit: Int) -> [VectorHit] {
        guard limit > 0, query.count == dimensions, !entries.isEmpty else { return [] }
        guard let normalizedQuery = VectorMath.l2Normalized(query) else { return [] }

        var hits: [VectorHit] = []
        hits.reserveCapacity(min(entries.count, limit))
        var scored: [(ChunkID, Double)] = []
        scored.reserveCapacity(entries.count)
        for (id, entry) in entries where entry.tier <= maxTier {
            scored.append((id, VectorMath.dot(normalizedQuery, entry.vector)))
        }
        scored.sort { lhs, rhs in
            if lhs.1 != rhs.1 { return lhs.1 > rhs.1 }
            return lhs.0 < rhs.0
        }
        for (id, score) in scored.prefix(limit) {
            guard var entry = entries[id] else { continue }
            clock &+= 1
            entry.lastTouched = clock
            entries[id] = entry
            hits.append(VectorHit(id: id, tier: entry.tier, score: score))
        }
        return hits
    }
}
