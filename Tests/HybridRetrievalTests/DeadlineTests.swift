import XCTest
import Foundation
@testable import HybridRetrieval

/// A `ContinuousClock` wrapper that records how many sleeps started and how many were
/// cancelled, so tests can *observe* timer lifecycle instead of asserting it in prose.
private final class ProbeClock: Clock, @unchecked Sendable {
    typealias Instant = ContinuousClock.Instant

    private let inner = ContinuousClock()
    private let lock = NSLock()
    private var started = 0
    private var cancelled = 0

    var now: Instant { inner.now }
    var minimumResolution: Duration { inner.minimumResolution }

    var sleepsStarted: Int { lock.withLock { started } }
    var sleepsCancelled: Int { lock.withLock { cancelled } }

    func sleep(until deadline: Instant, tolerance: Duration?) async throws {
        lock.withLock { started += 1 }
        do {
            try await inner.sleep(until: deadline, tolerance: tolerance)
        } catch {
            lock.withLock { cancelled += 1 }
            throw error
        }
    }
}

private extension NSLock {
    func withLock<T>(_ body: () -> T) -> T {
        lock()
        defer { unlock() }
        return body()
    }
}

final class DeadlineTests: XCTestCase {
    func testFastOperationFulfills() async {
        let outcome = await Deadline.race(limit: .seconds(2)) { 42 }
        guard case .fulfilled(let value) = outcome else {
            return XCTFail("expected fulfilled, got \(outcome)")
        }
        XCTAssertEqual(value, 42)
    }

    func testSlowOperationTimesOutPromptly() async {
        let clock = ContinuousClock()
        let start = clock.now
        let outcome: DeadlineOutcome<Int> = await Deadline.race(limit: .milliseconds(100)) {
            try await Task.sleep(nanoseconds: 10_000_000_000)
            return 1
        }
        let elapsed = clock.now - start
        guard case .timedOut = outcome else {
            return XCTFail("expected timedOut, got \(outcome)")
        }
        XCTAssertLessThan(elapsed, .seconds(2))
    }

    func testFailureIsReportedAsFailedNotTimeout() async {
        struct Sad: Error {}
        let outcome: DeadlineOutcome<Int> = await Deadline.race(limit: .seconds(2)) {
            throw Sad()
        }
        guard case .failed(let error) = outcome else {
            return XCTFail("expected failed, got \(outcome)")
        }
        XCTAssertTrue(error is Sad)
    }

    func testZeroAndNegativeLimitsShortCircuit() async {
        let zero: DeadlineOutcome<Int> = await Deadline.race(limit: .zero) { 1 }
        guard case .timedOut = zero else { return XCTFail("zero limit must time out immediately") }
        let negative: DeadlineOutcome<Int> = await Deadline.race(limit: .milliseconds(-50)) { 1 }
        guard case .timedOut = negative else { return XCTFail("negative limit must time out immediately") }
    }

    /// Observes the timer's lifecycle through an injected clock: a fast winner must
    /// leave zero pending sleeps behind. Deleting `timer.cancel()` from the winner path
    /// in `Deadline.race` makes this fail (sleepsCancelled drops to 0) — the previous
    /// version of this test, which only asserted `.fulfilled`, did not.
    func testWinnerCancelsLoserTimer() async {
        let clock = ProbeClock()
        let outcome = await Deadline.race(limit: .seconds(30), clock: clock) { true }
        guard case .fulfilled = outcome else {
            return XCTFail("fast operation must win its race")
        }
        XCTAssertEqual(clock.sleepsStarted, 1, "the deadline timer must actually arm")

        // The timer is cancelled from the winning task; give it a moment to unwind.
        for _ in 0..<50 where clock.sleepsCancelled == 0 {
            try? await Task.sleep(nanoseconds: 10_000_000)
        }
        XCTAssertEqual(clock.sleepsCancelled, 1,
                       "the losing timer must be cancelled, not left pending for 30s")
    }

    func testRepeatedFastRacesNeverDoubleResume() async {
        // Exercises the OnceGate hot path; a double resume would trap the process.
        for _ in 0..<200 {
            let outcome = await Deadline.race(limit: .milliseconds(50)) { true }
            guard case .fulfilled = outcome else {
                return XCTFail("fast operation must win its race")
            }
        }
    }

    /// Caller cancellation must return promptly and be reported as `.cancelled` —
    /// never as `.timedOut`, which would blame a healthy source for the caller's choice.
    func testCallerCancellationReturnsPromptlyAsCancelled() async {
        let clock = ContinuousClock()
        let start = clock.now

        let task = Task { () -> DeadlineOutcome<Int> in
            await Deadline.race(limit: .seconds(30)) {
                try await Task.sleep(nanoseconds: 30_000_000_000)
                return 1
            }
        }
        try? await Task.sleep(nanoseconds: 100_000_000)
        task.cancel()

        let outcome = await task.value
        let elapsed = clock.now - start

        guard case .cancelled = outcome else {
            return XCTFail("cancelled caller must observe .cancelled, got \(outcome)")
        }
        XCTAssertLessThan(elapsed, .seconds(5),
                          "a cancelled caller must not wait out the 30s deadline")
    }

    func testAlreadyCancelledCallerShortCircuits() async {
        let task = Task { () -> DeadlineOutcome<Int> in
            // Cancellation lands before the race starts.
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 5_000_000)
            }
            return await Deadline.race(limit: .seconds(30)) { 7 }
        }
        task.cancel()
        let outcome = await task.value
        guard case .cancelled = outcome else {
            return XCTFail("a pre-cancelled caller must short-circuit to .cancelled, got \(outcome)")
        }
    }

    func testOnceGateAllowsExactlyOneClaimUnderContention() async {
        let gate = OnceGate()
        let claims = await withTaskGroup(of: Bool.self, returning: Int.self) { group in
            for _ in 0..<64 {
                group.addTask { gate.claim() }
            }
            var wins = 0
            for await didClaim in group where didClaim {
                wins += 1
            }
            return wins
        }
        XCTAssertEqual(claims, 1, "exactly one concurrent claimer may win")
    }

    func testCancelBoxHandlesCancelBeforeSet() {
        let box = CancelBox()
        box.cancel() // cancel arrives first
        let task = Task<Void, Never> {
            try? await Task.sleep(nanoseconds: 10_000_000_000)
        }
        box.set(task)
        XCTAssertTrue(task.isCancelled, "set after cancel must cancel the task immediately")
    }

    /// `ResumeBox` exists because a cancellation handler can fire before the
    /// continuation is attached. Both orderings must deliver exactly one value.
    func testResumeBoxDeliversWhenOutcomeArrivesBeforeContinuation() async {
        let box = ResumeBox<Int>()
        box.deliver(.fulfilled(9)) // outcome first
        let outcome = await withCheckedContinuation { (continuation: CheckedContinuation<DeadlineOutcome<Int>, Never>) in
            box.attach(continuation)
        }
        guard case .fulfilled(let value) = outcome else {
            return XCTFail("pending outcome must be delivered on attach, got \(outcome)")
        }
        XCTAssertEqual(value, 9)
    }
}
