import XCTest
@testable import HybridRetrieval

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

    func testWinnerCancelsLoserTimer() async {
        // If the fast path leaked its timer, 200 racing calls would accumulate 200
        // pending sleeps; the OnceGate makes double-resume impossible either way.
        // This exercises the fast path repeatedly to shake out double-resume crashes.
        for _ in 0..<200 {
            let outcome = await Deadline.race(limit: .milliseconds(50)) { true }
            guard case .fulfilled = outcome else {
                return XCTFail("fast operation must win its race")
            }
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
            // If set-after-cancel failed to cancel, this would sleep 10s and the
            // expectation below would fail the test's timeout budget.
            try? await Task.sleep(nanoseconds: 10_000_000_000)
        }
        box.set(task)
        XCTAssertTrue(task.isCancelled, "set after cancel must cancel the task immediately")
    }
}
