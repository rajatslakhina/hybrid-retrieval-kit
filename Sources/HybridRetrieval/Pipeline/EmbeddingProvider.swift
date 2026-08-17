/// Produces a dense vector for a piece of text.
///
/// Seam rationale: this is where a real deployment plugs in Core ML / Foundation
/// Models embeddings. The core engine never depends on a model — that keeps the whole
/// retrieval subsystem testable (and CI-runnable on Linux) without ML weights, which
/// is the module boundary this package exists to demonstrate.
public protocol EmbeddingProvider: Sendable {
    var dimensions: Int { get }
    func embed(_ text: String) async throws -> [Float]
}

/// A deterministic, dependency-free embedder: hashed bag-of-words with signed buckets
/// (the classic "hashing trick"), L2-normalized.
///
/// Two deliberate decisions worth defending in review:
///
/// 1. **FNV-1a, not `Hasher`.** Swift's `Hasher` is randomly seeded per process, so
///    vectors would differ between the indexing run and every later query run —
///    an index poisoned at rest. Cross-process determinism is a correctness
///    requirement for any persisted embedding, so the hash is hand-rolled FNV-1a.
///    (Its multiply uses `&*` because *wrapping is the algorithm*, not an overflow bug.)
/// 2. **This is a fallback, not a semantic model.** Hashed BoW captures token overlap,
///    not meaning. It exists so the system is fully exercisable end-to-end with zero
///    model dependencies; production swaps in a real encoder through the same seam.
public struct HashingEmbedder: EmbeddingProvider {
    public let dimensions: Int

    public init(dimensions: Int = 64) {
        // Clamp: zero/negative dimensions would make bucket math meaningless.
        self.dimensions = min(max(dimensions, 4), 4096)
    }

    public func embed(_ text: String) async throws -> [Float] {
        var accumulator = [Double](repeating: 0, count: dimensions)
        for token in Tokenizer.tokenize(text) {
            let hash = Self.fnv1a(token)
            // dimensions >= 4, so the modulus is never zero; the result is < dimensions,
            // so the Int conversion and subscript are provably in range.
            let bucket = Int(hash % UInt64(dimensions))
            let sign: Double = (hash & 0x8000_0000) == 0 ? 1 : -1
            accumulator[bucket] += sign
        }
        let vector = accumulator.map { Float($0) }
        // All-zero (empty/degenerate text) normalizes to nil; represent as a zero
        // vector the vector index will refuse to store — callers see "no embedding",
        // never a crash.
        return VectorMath.l2Normalized(vector) ?? [Float](repeating: 0, count: dimensions)
    }

    /// FNV-1a 64-bit over UTF-8 bytes. Wrapping multiply is intentional (see type doc).
    static func fnv1a(_ text: String) -> UInt64 {
        var hash: UInt64 = 0xcbf2_9ce4_8422_2325
        for byte in text.utf8 {
            hash ^= UInt64(byte)
            hash = hash &* 0x0000_0100_0000_01b3
        }
        return hash
    }
}
