"""Objective-value checks shared by OPERA and the standalone case verifiers."""
from math import isfinite
from numbers import Real


DEFAULT_ABSOLUTE_TOLERANCE = 0.1


def objective_matches(output, reference, tolerance=DEFAULT_ABSOLUTE_TOLERANCE, mode="absolute"):
    """Compare finite objective values in their original units, without rounding."""
    for value in (output, reference):
        if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
            return False
    if mode == "relative":
        tolerance = tolerance * abs(reference)
    elif mode != "absolute":
        raise ValueError(f"Unsupported tolerance mode: {mode}")
    return abs(output - reference) <= tolerance
