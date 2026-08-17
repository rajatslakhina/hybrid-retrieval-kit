import Foundation

/// A hit from the lexical index: score + tier, no text. Raw text is stored exactly once,
/// in `IndexStore` — the index keeps only term statistics, so memory scales with
/// vocabulary, not corpus prose.
public struct LexicalHit: Sendable {
    public let id: ChunkID
    public let tier: PrivacyTier
    public let score: Double
}

/// In-memory inverted index scored with BM25 (Okapi).
///
/// Why BM25 and not TF-IDF cosine: BM25's term-frequency saturation and document-length
/// normalization are exactly the two properties short, chunked corpora need — a chunk
/// that repeats a term ten times is not ten times more relevant, and long chunks must
/// not dominate on raw term count. It is also what Spotlight-class engines approximate,
/// which keeps this index a faithful stand-in for the platform source.
///
/// Concurrency: NOT thread-safe by design; instances are confined inside `IndexStore`
/// (an actor). Keeping the data structure single-threaded keeps insert/remove O(terms)
/// with zero locking cost on the query path.
public final class LexicalIndex {
    public struct Parameters: Sendable {
        public let k1: Double
        public let b: Double

        /// Standard Robertson defaults; clamped so hostile configuration cannot produce
        /// negative or non-finite scores.
        public init(k1: Double = 1.2, b: Double = 0.75) {
            self.k1 = k1.isFinite ? min(max(k1, 0), 10) : 1.2
            self.b = b.isFinite ? min(max(b, 0), 1) : 0.75
        }
    }

    private struct Entry {
        let tier: PrivacyTier
        let termFrequencies: [String: Int]
        let length: Int
    }

    private let parameters: Parameters
    /// term → (chunk → term frequency)
    private var postings: [String: [ChunkID: Int]] = [:]
    private var entries: [ChunkID: Entry] = [:]
    private var totalTokens = 0

    public init(parameters: Parameters = Parameters()) {
        self.parameters = parameters
    }

    public var count: Int { entries.count }

    /// Inserts or replaces a chunk. Replacement fully reverses the previous posting
    /// contributions first, so re-indexing a document can never double-count.
    public func insert(id: ChunkID, text: String, tier: PrivacyTier) {
        remove(id: id)
        let tokens = Tokenizer.tokenize(text)
        guard !tokens.isEmpty else { return }
        var frequencies: [String: Int] = [:]
        for token in tokens {
            frequencies[token, default: 0] += 1
        }
        for (term, frequency) in frequencies {
            postings[term, default: [:]][id] = frequency
        }
        entries[id] = Entry(tier: tier, termFrequencies: frequencies, length: tokens.count)
        totalTokens = totalTokens.addingSaturated(tokens.count)
    }

    public func remove(id: ChunkID) {
        guard let entry = entries.removeValue(forKey: id) else { return }
        for term in entry.termFrequencies.keys {
            postings[term]?.removeValue(forKey: id)
            if postings[term]?.isEmpty == true {
                postings.removeValue(forKey: term)
            }
        }
        totalTokens = max(0, totalTokens.subtractingSaturated(entry.length))
    }

    /// BM25 search over chunks whose tier is `<= maxTier`.
    ///
    /// Every division below is guarded: `documentCount > 0` and `averageLength > 0`
    /// are established up front, `df >= 1` for any term that reaches the idf line, and
    /// the per-chunk denominator is `tf + k1 * (...)` with `tf >= 1`, `k1 >= 0`, and a
    /// non-negative length ratio — so it is always >= 1.
    public func search(_ query: String, maxTier: PrivacyTier, limit: Int) -> [LexicalHit] {
        guard limit > 0 else { return [] }
        let terms = Tokenizer.tokenize(query)
        guard !terms.isEmpty else { return [] }
        let documentCount = entries.count
        guard documentCount > 0, totalTokens > 0 else { return [] }
        let averageLength = Double(totalTokens) / Double(documentCount)
        guard averageLength > 0 else { return [] }

        var scores: [ChunkID: Double] = [:]
        for term in Set(terms) {
            guard let termPostings = postings[term], !termPostings.isEmpty else { continue }
            let documentFrequency = Double(termPostings.count)
            let idf = log(1 + (Double(documentCount) - documentFrequency + 0.5) / (documentFrequency + 0.5))
            guard idf.isUsableScore else { continue }
            for (chunkID, rawFrequency) in termPostings {
                guard let entry = entries[chunkID], entry.tier <= maxTier else { continue }
                let frequency = Double(rawFrequency)
                let lengthRatio = Double(entry.length) / averageLength
                let denominator = frequency + parameters.k1 * (1 - parameters.b + parameters.b * lengthRatio)
                guard denominator > 0 else { continue }
                let contribution = idf * (frequency * (parameters.k1 + 1)) / denominator
                if contribution.isUsableScore {
                    scores[chunkID, default: 0] += contribution
                }
            }
        }

        return scores
            .compactMap { id, score -> LexicalHit? in
                guard let entry = entries[id] else { return nil }
                return LexicalHit(id: id, tier: entry.tier, score: score)
            }
            .sorted { lhs, rhs in
                if lhs.score != rhs.score { return lhs.score > rhs.score }
                return lhs.id < rhs.id
            }
            .prefix(limit)
            .map { $0 }
    }
}
