/// Small arithmetic helpers so no numeric operation reachable from the public API can trap.
///
/// Policy: byte accounting and counters saturate instead of wrapping or trapping.
/// Saturation is the right failure mode for budgets — an over-saturated byte count
/// makes eviction *more* aggressive, never less, so the invariant "usedBytes <= budget"
/// degrades safely.
extension Int {
    /// `self + other`, clamped to `Int.min...Int.max` instead of trapping on overflow.
    func addingSaturated(_ other: Int) -> Int {
        let (value, overflow) = addingReportingOverflow(other)
        guard !overflow else { return other > 0 ? Int.max : Int.min }
        return value
    }

    /// `self - other`, clamped instead of trapping on overflow.
    func subtractingSaturated(_ other: Int) -> Int {
        let (value, overflow) = subtractingReportingOverflow(other)
        guard !overflow else { return other > 0 ? Int.min : Int.max }
        return value
    }

    /// `self * other`, clamped instead of trapping on overflow.
    func multiplyingSaturated(_ other: Int) -> Int {
        let (value, overflow) = multipliedReportingOverflow(by: other)
        guard !overflow else {
            // Sign of the true result decides which bound we saturate to.
            return (self < 0) == (other < 0) ? Int.max : Int.min
        }
        return value
    }
}

extension Double {
    /// True when the value is safe to use in scoring math (no NaN/inf propagation).
    var isUsableScore: Bool { isFinite }
}

enum VectorMath {
    /// L2-normalizes a vector. Returns `nil` when the vector cannot be normalized:
    /// empty, zero magnitude, or containing non-finite components. Callers treat `nil`
    /// as "this vector does not participate in similarity search" — never a crash.
    static func l2Normalized(_ vector: [Float]) -> [Float]? {
        guard !vector.isEmpty else { return nil }
        var sumOfSquares: Double = 0
        for component in vector {
            let d = Double(component)
            guard d.isFinite else { return nil }
            sumOfSquares += d * d
        }
        // Magnitude of zero (or denormal-underflow) cannot be divided by.
        guard sumOfSquares.isFinite, sumOfSquares > 0 else { return nil }
        let magnitude = sumOfSquares.squareRoot()
        guard magnitude.isFinite, magnitude > 0 else { return nil }
        return vector.map { Float(Double($0) / magnitude) }
    }

    /// Dot product over the overlapping prefix. Both inputs are expected to be
    /// same-dimension and pre-normalized; `zip` makes a mismatch impossible to trap on.
    static func dot(_ a: [Float], _ b: [Float]) -> Double {
        var total: Double = 0
        for (x, y) in zip(a, b) {
            total += Double(x) * Double(y)
        }
        return total.isUsableScore ? total : 0
    }
}
