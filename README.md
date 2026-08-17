# HybridRetrieval

**The retrieval layer you build the day Apple's "two-line RAG" stops being enough — an on-device hybrid search engine (BM25 + vectors + any source you plug in) with per-source deadlines, a privacy boundary it enforces instead of documents, and an indexing pipeline that survives out-of-order change feeds.**

WWDC26 gave Foundation Models a Spotlight-backed search tool: fully local retrieval, no embeddings, no vector DB, two lines of code. That convenience has a precise edge. The moment your corpus is *private in-app data* Spotlight can't see, your ranking needs to fuse *meaning* with *exact identifiers*, and your product people ask "why did it surface that?" — you are building a retrieval system, whether you planned to or not. This package is that system, designed the way a staff engineer would have to defend it in review.

## Why this matters

Retrieval quality problems are system design problems wearing an ML costume:

- **A slow source is an availability problem.** One stalled backend must cost you *that source*, not the query. Here, fan-out runs under per-source deadlines with a hard guarantee: even a source that **ignores cancellation** cannot stall the caller (`Deadline.race` abandons it and reports `timedOut`; the trade-off is documented in the source, not hidden).
- **A change feed is a distributed-systems problem.** Documents re-index while embeddings are in flight; deliveries arrive late, duplicated, out of order. The index commits by **sequence number, not wall clock** — a check → await → re-check → commit protocol across actor suspension points (the reentrancy bug most actor codebases ship without knowing it).
- **Privacy is a design constraint, not a filter at the end.** Documents carry tiers (`open` / `personal` / `sensitive`); queries carry the maximum tier they may see. Tiers are enforced **twice**: inside the store's searches, and again at the orchestrator on whatever any third-party source returns — violations are dropped *and counted* in the per-query report.
- **Degraded answers must be distinguishable from complete ones.** Every query returns a `RetrievalResponse` with per-source dispositions (fulfilled / timed out / failed), latency, and privacy-violation counts. A RAG layer that silently drops a source is lying to the model about its own context.

## Architecture

Macro (the system): an incremental indexing pipeline and a fan-out/fusion query path.
Micro (the seams): every stage is a protocol; the model is *absent by design*.

```
        change feed (seq-numbered)                 query + privacy context
                 │                                          │
                 ▼                                          ▼
        ┌─────────────────┐                     ┌──────────────────────┐
        │   IndexStore    │ actor               │ RetrievalOrchestrator │
        │  check → await  │                     │  fan-out, per-source  │
        │  → re-check →   │                     │  deadlines, privacy   │
        │  commit (atomic)│                     │  enforcement, report  │
        └───┬─────────┬───┘                     └──┬───────┬───────┬───┘
            │         │                            │       │       │
   ┌────────▼──┐  ┌───▼────────┐          ┌────────▼─┐ ┌───▼────┐ ┌▼─────────────┐
   │ Chunker   │  │ Embedding  │          │ Lexical  │ │ Vector │ │ your source: │
   │ (protocol)│  │ Provider   │          │ Source   │ │ Source │ │ Spotlight,   │
   └───────────┘  │ (protocol) │          │ (BM25)   │ │(cosine)│ │ FAQ, remote… │
                  └────────────┘          └──────────┘ └────────┘ └─────────────┘
                                                   │       │       │
                                                   └───────┴───────┘
                                                           ▼
                                              ReciprocalRankFusion (RRF)
```

**What's in the box:** `LexicalIndex` (BM25, guarded divisions, tier-aware), `VectorIndex` (byte-budgeted, LRU-evicting, brute-force cosine), `IndexStore` (actor; owns text exactly once), `LexicalSource` / `VectorSource`, `RetrievalOrchestrator` (+ `SourceReport`), `ReciprocalRankFusion`, `SentenceWindowChunker`, `HashingEmbedder` (deterministic fallback), and a thin `HybridSearchEngine` facade that wires the default stack.

## Design decisions & rejected alternatives

**RRF over score normalization.** BM25 scores are unbounded; cosine lives in [-1, 1]; a Spotlight-style source may return no scores at all. Min-max normalization is brittle (one outlier rescales a list; a one-element list has no range). Reciprocal Rank Fusion uses only ranks, degrades gracefully when a source's list is absent, and has one interpretable knob (`k`). Rejected: learned fusion (needs training data + a model dependency this package exists to avoid).

**Brute-force cosine over HNSW/IVF.** On-device corpora are 10³–10⁵ chunks; a linear scan at that scale is fast and *exactly* correct. An ANN graph buys nothing here and costs memory, build time, and a recall knob someone must own. The seam to revisit at 10⁶+ is `IndexStore`, not the public API.

**Byte budget, not entry count, for the vector tier.** Budgeting entries is how caches "under budget" their way into jetsam. Costs use saturating arithmetic — saturation over-estimates cost, which makes eviction *more* aggressive, never less. A refused vector degrades that chunk to lexical-only searchability; it never fails ingest.

**FNV-1a in the fallback embedder, not `Hasher`.** Swift's `Hasher` is seeded per process: vectors would differ between the indexing run and every later query run — an index poisoned at rest. Cross-process determinism is a correctness requirement for any persisted embedding. (The test suite pins golden hash values authored *outside* the implementation; an "embed twice and compare" test would pass with `Hasher` and is exactly the vacuous test this suite refuses to ship.)

**Unstructured race in `Deadline.race` — deliberately.** A `TaskGroup` race cannot guarantee return-by-deadline: the group's implicit `waitForAll` blocks on a cancellation-ignoring child. We prefer bounded caller latency over guaranteed resource reclamation, bound the leak to the source's own body, and surface the violation in the report. This is the one place the package steps outside structured concurrency, and it says so out loud.

**The store never awaits a model on the query path.** Query embedding happens in `VectorSource`, outside the actor — a slow embedder slows one source, and the per-source deadline contains it.

## Using it

```swift
import HybridRetrieval

// Default stack: BM25 + vector over a deterministic fallback embedder.
let engine = HybridSearchEngine(
    extraSources: [MySpotlightAdapter()],          // any RetrievalSource you own
    configuration: .init(perSourceDeadline: .milliseconds(250), maxResults: 8)
)

// Drive the index from your change feed (SwiftData history, server cursor, …).
try await engine.apply(.upsert(Document(id: DocumentID("note-42"),
                                        text: noteText,
                                        tier: .personal),
                               sequence: 1041))

// Query under a privacy context; inspect results AND per-source health.
let response = await engine.search(RetrievalQuery(text: "actor reentrancy",
                                                  maxTier: .personal))
for hit in response.hits { print(hit.chunk.text, hit.fusedScore, hit.contributingSources) }
if !response.isComplete { /* disclose degradation to your RAG layer */ }
```

Swapping in real semantic embeddings (Core ML, Foundation Models) is one conformance:

```swift
struct MyEmbedder: EmbeddingProvider {
    var dimensions: Int { 512 }
    func embed(_ text: String) async throws -> [Float] { try await model.encode(text) }
}
let engine = HybridSearchEngine(embedder: MyEmbedder())
```

### SPM

```swift
.package(url: "https://github.com/rajatslakhina/hybrid-retrieval-kit.git", from: "1.0.0")
```

## Tests

61 XCTest cases, and none of them are decorative. The suite's policy: any property this README brags about gets a test that feeds a *deliberately broken implementation* and asserts the guard fires.

- `testSlowEmbedCannotOverwriteNewerCommit` — the actor-reentrancy race, made deterministic: an upsert that is newest at arrival but finishes embedding after a newer commit must be discarded (`.supersededDuringEmbedding`).
- `testUncooperativeSourceCannotStallTheQuery` — a source that *ignores cancellation* for seconds; the query still returns at its deadline with partial results.
- `testPrivacyEnforcementDropsAndCountsContractViolations` — a deliberately leaky source returns `sensitive` content to an `open` query; enforcement must drop **and count** it.
- `testBM25MatchesHandComputedGoldenValue` / `testRRFMatchesHandComputedGoldenValues` — hand-derived numbers, not the implementation's formula mirrored back at itself.
- `testBudgetIsNeverExceededAndLRUEvicts`, `testDeleteTombstoneBlocksLateUpsert`, `testConcurrentWritersAllCommit` (50 real concurrent writers), zero-budget / degenerate-vector / empty-corpus edge walls, and FNV-1a golden values pinned at authoring time.

## Demo app

Demo app: (added after the companion repo is pushed — see below)

## Verification

- `swift build -Xswiftc -warnings-as-errors` from a clean tree and `swift test`: **61 tests, 0 failures** (Swift 6.0.3, Linux; strict concurrency, Swift 6 language mode). CI re-runs both on every push — see the [Actions tab](https://github.com/rajatslakhina/hybrid-retrieval-kit/actions): a Linux job (clean build with warnings-as-errors + full test suite in a `swift:6.0` container) and a macOS job compiling for `generic/platform=iOS Simulator` (no simulator runtime dependency by design).
- This library repo contains no app target; the runnable demo lives in the companion repo above, which consumes this package as a version-pinned remote dependency.

## License

MIT — see [LICENSE](LICENSE).
