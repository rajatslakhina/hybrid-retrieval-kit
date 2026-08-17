import Foundation

/// Outcome of racing an operation against a deadline.
public enum DeadlineOutcome<Value: Sendable>: Sendable {
    case fulfilled(Value)
    case timedOut
    case failed(any Error)
}

/// A one-shot gate: exactly one caller wins `claim()`. Backed by `NSLock` rather than
/// `Synchronization.Mutex` because `Mutex` requires iOS 18 and this package deploys to
/// iOS 17; the lock guards a single Bool, so contention is irrelevant.
final class OnceGate: @unchecked Sendable {
    private let lock = NSLock()
    private var fired = false

    /// Returns `true` exactly once across all callers and threads.
    func claim() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard !fired else { return false }
        fired = true
        return true
    }
}

/// Holds a cancellable task with idempotent, order-independent cancellation:
/// if `cancel()` arrives before `set(_:)`, the task is cancelled at set time.
final class CancelBox: @unchecked Sendable {
    private let lock = NSLock()
    private var task: Task<Void, Never>?
    private var cancelled = false

    func set(_ task: Task<Void, Never>) {
        lock.lock()
        let wasCancelled = cancelled
        self.task = task
        lock.unlock()
        if wasCancelled { task.cancel() }
    }

    func cancel() {
        lock.lock()
        cancelled = true
        let held = task
        lock.unlock()
        held?.cancel()
    }
}

public enum Deadline {
    /// Races `operation` against `limit` and **guarantees the caller returns by roughly
    /// the deadline** — even when the operation ignores cancellation.
    ///
    /// Trade-off, stated explicitly because it is the load-bearing decision here:
    /// a `withThrowingTaskGroup` race cannot make this guarantee — the group's implicit
    /// `waitForAll` blocks on an uncooperative child, so one misbehaving retrieval
    /// source would stall every query that fans out to it. Instead the operation runs in
    /// an unstructured `Task`; on deadline it is cancelled and *not awaited*. A source
    /// that ignores cancellation keeps running (bounded by its own body) with its result
    /// discarded — we deliberately prefer bounded caller latency over guaranteed
    /// resource reclamation, and surface the contract violation as `.timedOut` in the
    /// caller's report rather than hiding it.
    public static func race<Value: Sendable>(
        limit: Duration,
        clock: ContinuousClock = ContinuousClock(),
        operation: @escaping @Sendable () async throws -> Value
    ) async -> DeadlineOutcome<Value> {
        guard limit > .zero else { return .timedOut }
        let gate = OnceGate()
        let timer = CancelBox()
        return await withCheckedContinuation { continuation in
            let work = Task {
                let outcome: DeadlineOutcome<Value>
                do {
                    let value = try await operation()
                    outcome = .fulfilled(value)
                } catch is CancellationError {
                    outcome = .timedOut
                } catch {
                    outcome = .failed(error)
                }
                if gate.claim() {
                    timer.cancel()
                    continuation.resume(returning: outcome)
                }
            }
            timer.set(Task {
                try? await clock.sleep(for: limit)
                guard !Task.isCancelled else { return }
                if gate.claim() {
                    work.cancel()
                    continuation.resume(returning: .timedOut)
                }
            })
        }
    }
}
