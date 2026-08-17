import Foundation

/// Splits documents into retrievable chunks.
///
/// Seam rationale: chunking policy is a ranking-quality decision (too big → diluted
/// embeddings; too small → context-free hits) that teams tune per corpus. It is a
/// protocol so the demo, tests, and future corpora can swap policies without touching
/// the index.
public protocol Chunker: Sendable {
    func chunk(_ text: String, document: DocumentID) -> [Chunk]
}

/// Sentence-accumulating chunker with token-bounded windows and single-sentence overlap.
///
/// - Sentences are grouped until `targetTokens` is reached; the last sentence of a
///   chunk is carried into the next chunk so answers that straddle a boundary stay
///   retrievable (the classic RAG boundary-loss problem).
/// - A single sentence longer than `2 * targetTokens` is hard-split by tokens so one
///   pathological input cannot produce an unbounded chunk.
public struct SentenceWindowChunker: Chunker {
    private let targetTokens: Int

    public init(targetTokens: Int = 48) {
        // Clamp: a zero/negative target would make the accumulation loop meaningless.
        self.targetTokens = min(max(targetTokens, 8), 512)
    }

    public func chunk(_ text: String, document: DocumentID) -> [Chunk] {
        let sentences = Self.splitSentences(text)
        guard !sentences.isEmpty else { return [] }

        var chunks: [Chunk] = []
        var window: [String] = []
        var windowTokens = 0
        var ordinal = 0

        func flush(carryLast: Bool) {
            guard !window.isEmpty else { return }
            let joined = window.joined(separator: " ")
            chunks.append(Chunk(id: ChunkID(document: document, ordinal: ordinal), text: joined))
            ordinal += 1
            if carryLast, let last = window.last, window.count > 1 {
                let carryTokens = Tokenizer.tokenize(last).count
                // Only carry an overlap that leaves room for new content.
                if carryTokens < targetTokens / 2 {
                    window = [last]
                    windowTokens = carryTokens
                    return
                }
            }
            window = []
            windowTokens = 0
        }

        for sentence in sentences {
            let tokens = Tokenizer.tokenize(sentence)
            guard !tokens.isEmpty else { continue }

            if tokens.count > targetTokens.multiplyingSaturated(2) {
                // Pathologically long sentence: flush current window, then hard-split.
                flush(carryLast: false)
                var start = 0
                while start < tokens.count {
                    let end = min(start.addingSaturated(targetTokens), tokens.count)
                    // `start..<end` is always valid: start < tokens.count and end <= tokens.count.
                    let piece = tokens[start..<end].joined(separator: " ")
                    chunks.append(Chunk(id: ChunkID(document: document, ordinal: ordinal), text: piece))
                    ordinal += 1
                    start = end
                }
                continue
            }

            if windowTokens.addingSaturated(tokens.count) > targetTokens && !window.isEmpty {
                flush(carryLast: true)
            }
            window.append(sentence)
            windowTokens = windowTokens.addingSaturated(tokens.count)
        }
        flush(carryLast: false)
        return chunks
    }

    /// Splits on sentence-ending punctuation and newlines; trims whitespace;
    /// drops empties. Never traps on any input.
    static func splitSentences(_ text: String) -> [String] {
        guard !text.isEmpty else { return [] }
        var sentences: [String] = []
        var current = ""
        for character in text {
            if character == "." || character == "!" || character == "?" || character == "\n" {
                let trimmed = current.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmed.isEmpty { sentences.append(trimmed) }
                current = ""
            } else {
                current.append(character)
            }
        }
        let trimmed = current.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty { sentences.append(trimmed) }
        return sentences
    }
}
