/// One tokenizer shared by the lexical index, the hash embedder, and the chunker.
///
/// Design decision: lexical scoring and embedding hashing MUST agree on token
/// boundaries, otherwise hybrid fusion quietly compares apples to oranges (a term that
/// BM25 sees as one token but the embedder hashes as two drifts the two rankings apart
/// for reasons no one can debug from ranked output). Centralizing tokenization makes
/// that class of bug structurally impossible.
public enum Tokenizer {
    /// Lowercased alphanumeric tokens. Never traps on empty or exotic input;
    /// empty input yields an empty array.
    public static func tokenize(_ text: String) -> [String] {
        guard !text.isEmpty else { return [] }
        var tokens: [String] = []
        var current = String.UnicodeScalarView()
        for scalar in text.lowercased().unicodeScalars {
            if scalar.properties.isAlphabetic || ("0"..."9").contains(Character(scalar)) {
                current.append(scalar)
            } else if !current.isEmpty {
                tokens.append(String(current))
                current = String.UnicodeScalarView()
            }
        }
        if !current.isEmpty {
            tokens.append(String(current))
        }
        return tokens
    }
}
