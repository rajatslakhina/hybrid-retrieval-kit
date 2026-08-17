import XCTest
@testable import HybridRetrieval

final class PipelineTests: XCTestCase {
    // MARK: - Tokenizer

    func testTokenizerHandlesEdgeInputs() {
        XCTAssertEqual(Tokenizer.tokenize(""), [])
        XCTAssertEqual(Tokenizer.tokenize("   ...!!!   "), [])
        XCTAssertEqual(Tokenizer.tokenize("Swift 6.0's actors"), ["swift", "6", "0", "s", "actors"])
        XCTAssertEqual(Tokenizer.tokenize("Grüße, 東京!"), ["grüße", "東京"])
    }

    // MARK: - Chunker

    func testChunkerEmptyAndWhitespaceProduceNoChunks() {
        let chunker = SentenceWindowChunker()
        XCTAssertTrue(chunker.chunk("", document: DocumentID("d")).isEmpty)
        XCTAssertTrue(chunker.chunk("   \n\t  ", document: DocumentID("d")).isEmpty)
        XCTAssertTrue(chunker.chunk("...!!!???", document: DocumentID("d")).isEmpty)
    }

    func testChunkerBoundsChunkSizeAndAssignsOrdinals() {
        let target = 8
        let chunker = SentenceWindowChunker(targetTokens: target)
        let text = (0..<20).map { "Sentence number \($0) has five tokens." }.joined(separator: " ")
        let chunks = chunker.chunk(text, document: DocumentID("d"))

        XCTAssertGreaterThan(chunks.count, 1, "20 sentences cannot fit one 8-token window")
        for (index, chunk) in chunks.enumerated() {
            XCTAssertEqual(chunk.id.ordinal, index, "ordinals must be dense and ordered")
            XCTAssertFalse(chunk.text.isEmpty)
            // The actual size bound: a window may overshoot by at most the sentence
            // that crossed the threshold, and sentences here are 6 tokens.
            let tokens = Tokenizer.tokenize(chunk.text).count
            XCTAssertLessThanOrEqual(tokens, target + 6,
                                     "chunk \(index) is unbounded: \(tokens) tokens")
        }
        // Every source token must survive chunking (overlap may duplicate, never drop).
        let chunked = Set(chunks.flatMap { Tokenizer.tokenize($0.text) })
        XCTAssertTrue(Set(Tokenizer.tokenize(text)).isSubset(of: chunked),
                      "chunking must not lose tokens")
    }

    func testChunkerHardSplitsPathologicalSentence() {
        let chunker = SentenceWindowChunker(targetTokens: 8)
        // One "sentence" (no terminators) of 100 tokens must not become one giant chunk.
        let monster = (0..<100).map { "token\($0)" }.joined(separator: " ")
        let chunks = chunker.chunk(monster, document: DocumentID("d"))
        XCTAssertGreaterThanOrEqual(chunks.count, 10)
        for chunk in chunks {
            XCTAssertLessThanOrEqual(Tokenizer.tokenize(chunk.text).count, 8)
        }
    }

    func testChunkerOverlapCarriesBoundarySentence() {
        let chunker = SentenceWindowChunker(targetTokens: 10)
        let text = "Alpha beta gamma delta. Epsilon zeta eta theta. Iota kappa lambda mu."
        let chunks = chunker.chunk(text, document: DocumentID("d"))
        guard chunks.count >= 2 else {
            return XCTFail("expected at least two chunks, got \(chunks.count)")
        }
        // The sentence that closed chunk N must reappear opening chunk N+1.
        XCTAssertTrue(chunks[1].text.hasPrefix("Epsilon"),
                      "overlap must carry the boundary sentence; got: \(chunks[1].text)")
    }

    // MARK: - HashingEmbedder

    /// Golden values pinned at authoring time. This is the test that catches an
    /// accidental switch to the process-seeded `Hasher` — in-process "embed twice and
    /// compare" would still pass with `Hasher`, which is exactly why it is not used here.
    func testFNV1AMatchesPinnedGoldenValues() {
        XCTAssertEqual(HashingEmbedder.fnv1a(""), 0xcbf2_9ce4_8422_2325)
        XCTAssertEqual(HashingEmbedder.fnv1a("a"), 0xaf63_dc4c_8601_ec8c)
        XCTAssertEqual(HashingEmbedder.fnv1a("swift"), 0xa416_35a2_4b4e_a2d8)
    }

    func testEmbedderProducesUnitVectorsAndNoNaN() async throws {
        let embedder = HashingEmbedder(dimensions: 32)
        let vector = try await embedder.embed("Swift actors isolate mutable state.")
        XCTAssertEqual(vector.count, 32)
        let magnitude = vector.reduce(0.0) { $0 + Double($1) * Double($1) }.squareRoot()
        XCTAssertEqual(magnitude, 1.0, accuracy: 1e-5)
        XCTAssertFalse(vector.contains { !$0.isFinite })
    }

    func testEmbedderEmptyTextYieldsZeroVectorNotCrash() async throws {
        let embedder = HashingEmbedder(dimensions: 16)
        let vector = try await embedder.embed("")
        XCTAssertEqual(vector.count, 16)
        XCTAssertTrue(vector.allSatisfy { $0 == 0 })
    }

    func testEmbedderOverlappingTextIsMoreSimilarThanDisjointText() async throws {
        let embedder = HashingEmbedder(dimensions: 64)
        let base = try await embedder.embed("swift concurrency actors isolation")
        let related = try await embedder.embed("swift concurrency tasks")
        let unrelated = try await embedder.embed("breakfast burrito recipe salsa")

        let simRelated = VectorMath.dot(base, related)
        let simUnrelated = VectorMath.dot(base, unrelated)
        XCTAssertGreaterThan(simRelated, simUnrelated,
                             "token overlap must move cosine similarity — fails if embedding is gutted")
    }

    func testEmbedderClampsHostileDimensions() {
        XCTAssertEqual(HashingEmbedder(dimensions: 0).dimensions, 4)
        XCTAssertEqual(HashingEmbedder(dimensions: -7).dimensions, 4)
        XCTAssertEqual(HashingEmbedder(dimensions: 1_000_000).dimensions, 4096)
    }

    // MARK: - Saturating arithmetic

    func testSaturatingHelpersClampInsteadOfTrapping() {
        XCTAssertEqual(Int.max.addingSaturated(1), Int.max)
        XCTAssertEqual(Int.min.addingSaturated(-1), Int.min)
        XCTAssertEqual(Int.min.subtractingSaturated(1), Int.min)
        XCTAssertEqual(Int.max.multiplyingSaturated(2), Int.max)
        XCTAssertEqual(Int.min.multiplyingSaturated(2), Int.min)
        XCTAssertEqual((-1).multiplyingSaturated(Int.min), Int.max)
        XCTAssertEqual(3.addingSaturated(4), 7)
        XCTAssertEqual(3.multiplyingSaturated(4), 12)
    }

    func testNormalizationRejectsDegenerateVectors() {
        XCTAssertNil(VectorMath.l2Normalized([]))
        XCTAssertNil(VectorMath.l2Normalized([0, 0, 0]))
        XCTAssertNil(VectorMath.l2Normalized([.nan, 1]))
        XCTAssertNil(VectorMath.l2Normalized([.infinity, 1]))
        XCTAssertNotNil(VectorMath.l2Normalized([3, 4]))
    }
}
