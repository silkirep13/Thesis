""" Self-verification for tonic anchoring in _resolve_tonic_by_finalis.
    Run directly: python test_finalis.py"""
import sys

import numpy as np

from analysis import _resolve_tonic_by_finalis, TONIC_NAMES

failures: list[str] = []

MAJOR = "Greek Major (Matzore)"
MINOR = "Greek Minor (Minore)"

# Scores shaped like a real measurement: one tonic edges out the tonic a
# fourth below it by about two per cent, because the two scales differ in a
# single pitch class. That is the margin the finalis exists to overrule.
# These mode names are only convenient labels for the fixture: in the
# pipeline this function is reached for Byzantine chant, where the closing
# note is held long enough to be trusted. See the note at the call site for
# the measurements that kept it away from Greek folk.
SCORES = {
    (TONIC_NAMES.index("A"), MAJOR): 0.8921,
    (TONIC_NAMES.index("D"), MAJOR): 0.8563,
    (TONIC_NAMES.index("D"), MINOR): 0.8176,
    (TONIC_NAMES.index("A"), MINOR): 0.7900,
}


def check(condition: bool, message: str):
    """ Record a failed expectation so every case still runs.
    Collecting failures rather than raising keeps one broken case from
    hiding the state of the others."""
    if not condition:
        failures.append(message)


def frames(*, lead: float, ending: list[tuple[float, int]]) -> np.ndarray:
    """ Build a pitch track whose last 15% of frames is the given ending.
    The ending is given as (cents, frame count) pairs, because how long each
    note is held is the whole point: the counts decide the most-held pitch
    class while the spread decides the median, and the two can disagree. The
    lead is padded so the ending fills exactly the tail window."""
    tail = np.concatenate([np.full(count, cents, dtype=float) for cents, count in ending])
    head = np.full(int(np.ceil(tail.size / 0.15)) - tail.size, float(lead))
    return np.concatenate([head, tail])


D4, E4, F4, FS4, A4 = 6200.0, 6400.0, 6500.0, 6600.0, 6900.0

# A closing phrase held on the tonic: the plain case, which the old median
# handled correctly too.
check(_resolve_tonic_by_finalis(SCORES, "A", MAJOR,
                                frames(lead=A4, ending=[(D4, 100)]))[0] == "D",
      "a sustained finalis on D did not override the chroma winner")

# The regression this file exists for, shaped like the real recording: the
# closing phrase dwells on D but passes through several higher notes on the
# way. D is held longest, yet more than half the frames lie above it, so a
# median of the pitches reports a note in the middle of the phrase instead of
# the one it ends on. Counting frames per pitch class returns D.
_spread = frames(lead=A4, ending=[(D4, 50), (E4, 25), (F4, 25), (FS4, 25), (A4, 25)])
check(_resolve_tonic_by_finalis(SCORES, "A", MAJOR, _spread)[0] == "D",
      "a closing phrase spread above its final note did not resolve to D")

_tail = _spread[int(_spread.size * 0.85):]
check(abs(float(np.median(_tail)) % 1200.0 - 200.0) > 50.0,
      "test no longer exercises the median failure it was written for")

# Glykeria's recording sits about 36 cents below A440. Each frame is rounded
# on its own, so a flat recording still lands on the right degree.
check(_resolve_tonic_by_finalis(SCORES, "A", MAJOR,
                                frames(lead=A4 - 36,
                                       ending=[(D4 - 36, 60), (FS4 - 36, 40)]))[0] == "D",
      "a recording tuned flat of A440 was pushed onto the wrong degree")

# Once the tonic is settled, chroma still chooses the mode at that tonic.
check(_resolve_tonic_by_finalis(SCORES, "A", MAJOR,
                                frames(lead=A4, ending=[(D4, 100)]))[1] == MAJOR,
      "the mode was not taken from the best-scoring template at the finalis")

# Agreement between chroma and finalis must leave both untouched.
check(_resolve_tonic_by_finalis(SCORES, "A", MAJOR,
                                frames(lead=D4, ending=[(A4, 100)])) == ("A", MAJOR),
      "an agreeing chroma winner was overwritten")

# Too little material for the ending to mean anything: pass through unchanged.
check(_resolve_tonic_by_finalis(SCORES, "A", MAJOR, np.array([D4] * 10)) == ("A", MAJOR),
      "a too-short pitch track was overridden anyway")

# A finalis with no template scored at it cannot be adopted.
check(_resolve_tonic_by_finalis({(TONIC_NAMES.index("A"), MAJOR): 0.89}, "A", MAJOR,
                                frames(lead=A4, ending=[(D4, 100)])) == ("A", MAJOR),
      "a finalis with no scored template was adopted regardless")


if failures:
    print("FAILED:")
    for f in failures:
        print("  -", f)
    sys.exit(1)

print("OK - finalis anchoring verified across 8 cases.")
