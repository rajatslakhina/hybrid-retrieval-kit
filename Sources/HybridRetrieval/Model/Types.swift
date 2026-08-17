/// Core value types for the retrieval system. Everything here is `Sendable` and cheap
/// to copy — these cross actor and task boundaries constantly.

/// Identifies a source document in the corpus.
public struct DocumentID: Hashable, Comparable, Sendable, Codable, CustomStringConvertible {
    public let rawValue: String
    public init(_ rawValue: String) { self.rawValue = rawValue }
    public static func < (lhs: Self, rhs: Self) -> Bool { lhs.rawValue < rhs.rawValue }
    public var description: String { rawValue }
}

/// Identifies one chunk of a document. `Comparable` so every ranked list in the system
/// has a deterministic total order (score first, then `ChunkID` as tie-break) — ranking
/// output that flickers between runs is untestable and undebuggable.
public struct ChunkID: Hashable, Comparable, Sendable, Codable, CustomStringConvertible {
    public let document: DocumentID
    public let ordinal: Int

    public init(document: DocumentID, ordinal: Int) {
        self.document = document
        self.ordinal = ordinal
    }

    public static func < (lhs: Self, rhs: Self) -> Bool {
        if lhs.document != rhs.document { return lhs.document < rhs.document }
        return lhs.ordinal < rhs.ordinal
    }

    public var description: String { "\(document)#\(ordinal)" }
}

/// Privacy tier of a document — the boundary between index tiers the system enforces.
///
/// Ordering is by sensitivity: a query context carries the *maximum* tier it may see,
/// and both the index layer and the orchestrator (defense in depth — third-party
/// sources are not trusted to filter correctly) drop anything above it.
public enum PrivacyTier: Int, Sendable, Codable, CaseIterable, Comparable {
    /// Content safe for any surface (e.g. bundled reference docs).
    case open = 0
    /// User-scoped content (e.g. their notes).
    case personal = 1
    /// Content that must never leave the most restricted surface (e.g. health entries).
    case sensitive = 2

    public static func < (lhs: Self, rhs: Self) -> Bool { lhs.rawValue < rhs.rawValue }
}

/// A source document to be ingested.
public struct Document: Sendable {
    public let id: DocumentID
    public let text: String
    public let tier: PrivacyTier

    public init(id: DocumentID, text: String, tier: PrivacyTier = .open) {
        self.id = id
        self.text = text
        self.tier = tier
    }
}

/// One entry in the change feed that drives incremental indexing.
///
/// `sequence` is a monotonically increasing per-corpus number assigned by the producer
/// (SwiftData history token, server cursor, etc.). The index uses it to reject stale
/// and out-of-order deliveries — see `IndexStore.apply(_:)`.
public struct DocumentChange: Sendable {
    public enum Kind: Sendable {
        case upsert(Document)
        case delete(DocumentID)
    }

    public let kind: Kind
    public let sequence: UInt64

    public init(kind: Kind, sequence: UInt64) {
        self.kind = kind
        self.sequence = sequence
    }

    public static func upsert(_ document: Document, sequence: UInt64) -> DocumentChange {
        DocumentChange(kind: .upsert(document), sequence: sequence)
    }

    public static func delete(_ id: DocumentID, sequence: UInt64) -> DocumentChange {
        DocumentChange(kind: .delete(id), sequence: sequence)
    }
}

/// A chunk of a document produced by a `Chunker`.
public struct Chunk: Sendable {
    public let id: ChunkID
    public let text: String

    public init(id: ChunkID, text: String) {
        self.id = id
        self.text = text
    }
}

/// A chunk returned by a retrieval source, with that source's own score.
/// Scores are source-relative and NOT comparable across sources — that is exactly why
/// fusion is rank-based (see `ReciprocalRankFusion`).
public struct ScoredChunk: Sendable {
    public let id: ChunkID
    public let text: String
    public let tier: PrivacyTier
    public let score: Double

    public init(id: ChunkID, text: String, tier: PrivacyTier, score: Double) {
        self.id = id
        self.text = text
        self.tier = tier
        // A non-finite score would poison every downstream sort; clamp at the boundary.
        self.score = score.isUsableScore ? score : 0
    }
}

/// A query plus the privacy context it runs under.
public struct RetrievalQuery: Sendable {
    public let text: String
    /// The most sensitive tier this query is allowed to see.
    public let maxTier: PrivacyTier
    /// How many candidates each source should return (fusion sees this many per source).
    public let perSourceLimit: Int

    public init(text: String, maxTier: PrivacyTier = .open, perSourceLimit: Int = 20) {
        self.text = text
        self.maxTier = maxTier
        // Clamp instead of precondition: a hostile/buggy caller gets a sane value, not a trap.
        self.perSourceLimit = min(max(perSourceLimit, 1), 200)
    }
}

/// Identifies a retrieval source in fan-out, fusion weights, and reports.
public struct SourceID: Hashable, Comparable, Sendable, CustomStringConvertible, ExpressibleByStringLiteral {
    public let rawValue: String
    public init(_ rawValue: String) { self.rawValue = rawValue }
    public init(stringLiteral value: String) { self.rawValue = value }
    public static func < (lhs: Self, rhs: Self) -> Bool { lhs.rawValue < rhs.rawValue }
    public var description: String { rawValue }
}
