import Foundation

/// Outcome of racing an operation against a deadline.
public enum DeadlineOutcome<Value: Sendable>: Sendable {
    case fulfilled(Value)
    /// The operation missed its deadline. This blames the *operation*.
    case timedOut
    case failed(any Error)
    /// The **caller** was cancelled while waiting. Kept distinct from `timedOut`
    /// because conflating them makes a health report lie: a user typing in a search
    /// field cancels the previous query on every keystroke, and reporting those as
    /// source timeouts would frame healthy backends as failing.
    case cancelled
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

/// Holds the continuation so a resume can arrive **before** the continuation exists —
/// which genuinely happens when the caller is already cancelled as the race starts.
/// Resume-exactly-once is guaranteed by the caller's `OnceGate`; this type only fixes
/// the ordering.
final class ResumeBox<Value: Sendable>: @unchecked Sendable {
    private let lock = NSLock()
    private var continuation: CheckedContinuation<DeadlineOutcome<Value>, Never>?
    private var pending: DeadlineOutcome<Value>?

    func attach(_ continuation: CheckedContinuation<DeadlineOutcome<Value>, Never>) {
        lock.lock()
        if let outcome = pending {
            pending = nil
            lock.unlock()
            continuation.resume(returning: outcome)
            return
        }
        self.continuation = continuation
        lock.unlock()
    }

    func deliver(_ outcome: DeadlineOutcome<Value>) {
        lock.lock()
        if let continuation {
            self.continuation = nil
            lock.unlock()
            continuation.resume(returning: outcome)
            return
        }
        pending = outcome
        lock.unlock()
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
    ///
    /// Because the work runs in an unstructured `Task`, it does **not** inherit the
    /// caller's cancellation — so cancellation is wired back explicitly with
    /// `withTaskCancellationHandler`. A cancelled caller returns promptly with
    /// `.cancelled`, never `.timedOut`: attributing a user-initiated cancel to the
    /// source would poison the health report this package exists to produce.
    ///
    /// `clock` is generic so tests can observe timer behavior (see
    /// `DeadlineTests.testWinnerCancelsLoserTimer`) rather than assert it in prose.
    public static func race<Value: Sendable, C: Clock>(
        limit: C.Instant.Duration,
        clock: C = ContinuousClock(),
        operation: @escaping @Sendable () async throws -> Value
    ) async -> DeadlineOutcome<Value> {
        guard limit > .zero else { return .timedOut }
        guard !Task.isCancelled else { return .cancelled }

        let gate = OnceGate()
        let timer = CancelBox()
        let worker = CancelBox()
        let box = ResumeBox<Value>()

        return await withTaskCancellationHandler {
            await withCheckedContinuation { (continuation: CheckedContinuation<DeadlineOutcome<Value>, Never>) in
                box.attach(continuation)

                worker.set(Task {
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
                        box.deliver(outcome)
                    }
                })

                timer.set(Task {
                    try? await clock.sleep(for: limit)
                    guard !Task.isCancelled else { return }
                    if gate.claim() {
                        worker.cancel()
                        box.deliver(.timedOut)
                    }
                })
            }
        } onCancel: {
            if gate.claim() {
                worker.cancel()
                timer.cancel()
                box.deliver(.cancelled)
            }
        }
    }
}
