# HybridRetrieval

**The retrieval layer you build the day Apple's "two-line RAG" stops being enough — an on-device hybrid search engine (BM25 + vectors + any source you plug in) with per-source deadlines, a privacy boundary it enforces instead of documents, and an indexing pipeline that survives out-of-order change feeds.**

WWDC26 gave Foundation Models a Spotlight-backed search tool: fully local retrieval, no embeddings, no vector DB, two lines of code. That convenience has a precise edge. The moment your corpus is *private in-app data* Spotlight can't see, your ranking needs to fuse *meaning* with *exact identifiers*, and your product people ask "why did it surface that?" — you are building a retrieval system, whether you planned to or not. This package is that system, designed the way a staff engineer would have to defend it in review.

## Why this matters

Retrieval quality problems are system design problems wearing an ML costume:

- **A slow source is an availability problem.** One stalled backend must cost you *that source*, not the query. Here, fan-out runs under per-source deadlines with a hard guarantee: even a source that **ignores cancellation** cannot stall the caller (`Deadline.race` abandons it and reports `timedOut`; the trade-off is documented in the source, not hidden).
- **A change feed is a distributed-systems problem.** Documents re-index while embeddings are in flight; deliveries arrive late, duplicated, out of order. The index commits by **sequence number, not wall clock** — a check → await → re-check → commit protocol across actor suspension points (the reentrancy bug most actor codebases ship without knowing it).
- **Privacy is a design constraint, not a filter at the end.** Documents carry tiers (`open` / `personal` / `sensitive`); queries carry the maximum tier they may see. Tiers are enforced **twice**: inside the store's searches, and again at the orchestrator on whatever any third-party source returns — violations are dropped *and counted* in the per-query report.
- **Degraded answers must be distinguishable from complete ones.** Every query returns a `RetrievalResponse` with per-source dispositions (fulfilled / timed out / failed / **cancelled**), latency, and privacy-violation counts. A RAG layer that silently drops a source is lying to the model about its own context — and a report that blames a healthy source for the *caller's* cancellation is lying in the other direction, which is why `cancelled` is its own case.

## Architecture

Macro (the system): an incremental indexing pipeline and a fan-out/fusion query path.
Micro (the seams): every stage is a protocol; the model is *absent by design*.

```
        change feed (seq-numbered)                 query + privacy context
                 │                                          │
                 ▼                                          ▼
        ┌─────────────────┐                     ┌───────────────────────┐
        │   IndexStore    │ actor               │ RetrievalOrchestrator │
        │  check → await  │                     │  fan-out, per-source  │
        │  → re-check →   │                     │  deadlines, privacy   │
        │  commit (atomic)│                     │  enforcement, report  │
        └───┬─────────┬───┘                     └───┬───────┬───────┬───┘
            │         │                             │       │       │
   ┌────────▼──┐  ┌───▼────────┐          ┌─────────▼┐ ┌────▼───┐ ┌▼─────────────┐
   │ Chunker   │  │ Embedding  │          │ Lexical  │ │ Vector │ │ your source: │
   │ (protocol)│  │ Provider   │          │ Source   │ │ Source │ │ Spotlight,   │
   └───────────┘  │ (protocol) │          │ (BM25)   │ │(cosine)│ │ FAQ, remote… │
                  └────────────┘          └──────────┘ └────────┘ └──────────────┘
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

**Unstructured race in `Deadline.race` — deliberately.** A `TaskGroup` race cannot guarantee return-by-deadline: the group's implicit `waitForAll` blocks on a cancellation-ignoring child. We prefer bounded caller latency over guaranteed resource reclamation, bound the leak to the source's own body, and surface the violation in the report. This is the one place the package steps outside structured concurrency, and it says so out loud. The cost of stepping outside is that the work does *not* inherit caller cancellation — so cancellation is wired back explicitly with `withTaskCancellationHandler`, and a cancelled caller gets `.cancelled`, never `.timedOut`. (In a SwiftUI search field that cancels the previous query on every keystroke, folding those into `timedOut` would make every healthy backend look like it was failing.)

**A similarity floor, so the vector tier is a search and not a ranking.** `VectorIndex` drops candidates scoring below `minimumSimilarity` (default 0.2) instead of returning the whole corpus ordered by noise. Without it, a query matching nothing still produces a full result page, and RRF then hands rank-1 weight to the least-bad garbage — which a RAG layer feeds to the model as context. Rejected: filtering downstream in the fuser (the index is the only component that knows its own score scale) and returning everything and letting the UI decide (that pushes a correctness decision into every call site). The trade-off is a tuning knob: 0.2 is calibrated for the bundled `HashingEmbedder`, where real matches measure 0.39–0.54 and collision noise mostly sits below 0.2 — though hashed bag-of-words noise *can* reach ~0.24 and overlap genuinely weak matches. That overlap is a property of the fallback embedder, not of the fusion layer; a real encoder separates the bands and should retune the floor. Pass `-1` to opt out entirely.

**The store never awaits a model on the query path.** Query embedding happens in `VectorSource`, outside the actor — a slow embedder slows one source, and the per-source deadline contains it.

**Tombstones are retained, not compacted.** `IndexStore` keeps one sequence number per distinct `DocumentID` ever seen, including deletes, so a delete can never be undone by a late redelivery. That is bounded by corpus *cardinality*, not by churn — but it is genuinely unbounded for a feed that creates and deletes unique ids forever. The alternative (a low-water mark below which tombstones are dropped) trades exactly-once semantics for memory and needs a durable watermark the producer must supply; that belongs to a persistence layer this package deliberately does not own.

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
if response.wasCancelled { return }                // abandoned work, not an answer
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
.package(url: "https://github.com/rajatslakhina/hybrid-retrieval-kit.git", from: "1.1.0")
```

## Tests

72 XCTest cases, and the suite is written to be *mutation-hostile*: every property this README brags about has a test that fails when the property is removed. Several tests exist specifically because an earlier version of the suite passed against a deliberately broken implementation.

- `testSlowEmbedCannotOverwriteNewerCommit` — the actor-reentrancy race, made deterministic: an upsert that is newest at arrival but finishes embedding after a newer commit must be discarded (`.supersededDuringEmbedding`).
- `testDocumentTierSurvivesIngestAndGatesStoreSearches` / `testEngineHonoursTierAcrossEverySource` — hardcoding `.open` inside the commit loop (making every ingested chunk world-readable) used to leave the suite green. It does not now.
- `testUncooperativeSourceCannotStallTheQuery` — a source that *ignores cancellation* for seconds; the query still returns at its deadline with partial results.
- `testCallerCancellationIsReportedAsCancelledNotTimedOut` — cancels a query mid-flight and asserts the report says `cancelled`, not `timedOut`.
- `testPrivacyEnforcementDropsAndCountsContractViolations` — a deliberately leaky source returns `sensitive` content to an `open` query; enforcement must drop **and count** it.
- `testWinnerCancelsLoserTimer` — observes timer cancellation through an injected `ProbeClock` rather than asserting it in prose; deleting `timer.cancel()` fails it.
- `testSimilarityFloorRejectsOrthogonalMatches` — an orthogonal query must return nothing, not the corpus ordered by noise.
- `testConcurrentWritersOnOneDocumentConvergeToHighestSequence` — 40 writers contending on the *same* document in shuffled order; the highest sequence must win regardless of completion order.
- `testBM25MatchesHandComputedGoldenValue` / `testRRFMatchesHandComputedGoldenValues` — hand-derived numbers, not the implementation's formula mirrored back at itself.
- Plus zero-budget / degenerate-vector / empty-corpus edge walls, delete-tombstone ordering, and FNV-1a golden values pinned at authoring time.

## Demo app

**Demo app: [rajatslakhina/hybrid-retrieval-demo-app](https://github.com/rajatslakhina/hybrid-retrieval-demo-app)** — a SwiftUI app that consumes this package as a version-pinned remote dependency (`from: 1.1.0`) and makes the failure behavior visible: toggles inject a deadline-busting source, a contract-violating source and an always-failing one, so you can watch the query return partial results while the per-source report marks them `TIMED OUT`, counts the filtered privacy violation, and shows `FAILED` — plus a privacy-context picker that moves the tier boundary live.

## Verification

Three separate facts, deliberately not conflated:

- **Local:** `swift build -Xswiftc -warnings-as-errors` from a clean tree (`.build` removed) and `swift test` — **72 tests, 0 failures** (Swift 6.0.3 on Linux; Swift 6 language mode, strict concurrency).
- **CI:** both re-run on every push — see the [Actions tab](https://github.com/rajatslakhina/hybrid-retrieval-kit/actions). A Linux job does the clean warnings-as-errors build plus the full suite in a `swift:6.0` container; a macOS job compiles for `generic/platform=iOS Simulator` (generic on purpose — a named device would tie the job to whichever simulator runtimes that day's runner image happens to ship).
- **No Simulator run.** Nobody has launched the demo app on a Simulator. This was an unattended scheduled run; screen-control access was granted, but the machine already had an unrelated production Xcode workspace open with live edits and two Simulators running unrelated apps, so the run stopped rather than clicking through someone else's work in progress. "Compiles for an iOS Simulator destination" is a strictly weaker claim and is not used here to imply the stronger one.

This library repo contains no app target of any kind; the runnable app lives in the companion repo above.

## License

MIT — see [LICENSE](LICENSE).
