""" Self-verification for the melodic-range filter in _drop_out_of_range_notes.
    Run directly: python test_note_range.py"""
import sys

from analysis import _drop_out_of_range_notes as clean

failures: list[str] = []


def mk(cents: list[float]) -> list[dict]:
    """ Build note dictionaries of the shape _segment_notes produces.
    Only midi_cents varies between them; the remaining fields are filled with
    plausible constants so the filter can be checked for leaving them alone."""
    return [{"midi_cents": c, "start_s": i * 0.5, "end_s": i * 0.5 + 0.4,
             "degree": 1, "cents_from_tonic": 0.0, "deviation_cents": 0.0,
             "accidental": "natural"} for i, c in enumerate(cents)]


def cents_of(notes: list[dict]) -> list[float]:
    """ Reduce a list of notes to the pitches it carries, in order."""
    return [n["midi_cents"] for n in notes]


def check(condition: bool, message: str):
    """ Record a failed expectation so every case still runs.
    Collecting failures rather than raising means one broken case does not
    hide the state of the others."""
    if not condition:
        failures.append(message)


# A female vocal line centred near A4 (6900 cents), spanning roughly a tenth.
MELODY = [6900, 7100, 7200, 7400, 7600, 7400, 7200, 7100, 6900, 6700,
          6500, 6700, 6900, 7100, 7200, 7400]

check(cents_of(clean(mk(MELODY))) == MELODY,
      "a clean melody was altered")

# CREPE subharmonics one and two octaves below the line: the failure mode
# that put notes on five ledger lines below the staff.
check(cents_of(clean(mk(MELODY[:8] + [4500, 4800] + MELODY[8:]))) == MELODY,
      "subharmonic octave errors survived the filter")

# A spurious pitch that is not an octave displacement goes the same way.
check(cents_of(clean(mk(MELODY[:8] + [5300] + MELODY[8:]))) == MELODY,
      "a non-octave outlier survived the filter")

# The window is deliberately generous. A leap of just over an octave below
# the line's centre is music, not an artefact, and has to survive.
_wide = MELODY[:8] + [5800] + MELODY[8:]
check(cents_of(clean(mk(_wide))) == _wide,
      "a legitimate wide leap was discarded")

# Two full octaves end to end is the widest span this repertoire realistically
# produces within one excerpt, and is kept whole.
_broad = [5900 + 100 * i for i in range(25)]
check(cents_of(clean(mk(_broad))) == _broad,
      "a two-octave instrumental line was trimmed")

# Known and accepted limit, asserted rather than left to be discovered: a line
# wider than 28 semitones loses its extremes. If that ever proves too tight,
# this is the case that will say so.
_huge = [4800 + 200 * i for i in range(25)]
_kept = cents_of(clean(mk(_huge)))
check((min(_kept), max(_kept)) == (5800, 8600),
      f"four-octave line trimmed to {min(_kept)}..{max(_kept)}, expected 5800..8600")

# Too few notes for the median to mean anything: the input passes through.
_tiny = [6900, 7100, 3000]
check(cents_of(clean(mk(_tiny))) == _tiny,
      "a short input was trimmed against an untrustworthy centre")

# A median stays on the melody even when a large minority of notes are wrong.
check(sorted(cents_of(clean(mk(MELODY + [n - 2400 for n in MELODY[:6]])))) == sorted(MELODY),
      "heavy contamination dragged the centre off the melody")

# Surviving notes keep every field the rest of the pipeline reads.
_src = mk(MELODY[:8] + [3000] + MELODY[8:])
_src[3]["degree"] = 5
_src[3]["cents_from_tonic"] = 700.0
_kept_note = [n for n in clean(_src) if n["start_s"] == 1.5][0]
check((_kept_note["degree"], _kept_note["cents_from_tonic"], _kept_note["midi_cents"])
      == (5, 700.0, 7400),
      "a surviving note lost fields the notation step depends on")

check(clean([]) == [], "an empty input was not handled")


if failures:
    print("FAILED:")
    for f in failures:
        print("  -", f)
    sys.exit(1)

print("OK - melodic range filter verified across 10 cases.")
