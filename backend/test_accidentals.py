""" Self-verification for accidental spelling in the generated MusicXML.
    Run directly: python test_accidentals.py"""
import sys
import xml.etree.ElementTree as ET

import numpy as np

from notation import build_score, score_to_musicxml

failures: list[str] = []

# Pitches deliberately placed off the piano keys: a quarter-sharp D, a
# quarter-flat E, a bent F# and two plain notes. Raw CREPE output looks like
# this on any sung line.
BENT = [6250.0, 6350.0, 6580.0, 6900.0, 7140.0, 6155.0, 6445.0]

ALLOWED_NAMES = {"sharp", "flat", "natural"}


def check(condition: bool, message: str):
    """ Record a failed expectation so every case still runs.
    Collecting failures rather than raising keeps one broken case from
    hiding the state of the others."""
    if not condition:
        failures.append(message)


def xml_for(cents: list[float], semitone_only: bool) -> str:
    """ Build a one-part score from bare pitches and return its MusicXML.
    Durations and rhythm are irrelevant here, so every note is given the same
    length and the pulse is left unmeasured."""
    notes = [{"midi_cents": c, "start_s": i * 0.5, "end_s": i * 0.5 + 0.5}
             for i, c in enumerate(cents)]
    score, _ = build_score(notes, mode="test", tonic="D", tempo_bpm=0.0,
                           pulse_strength=0.0, instrument_name="Φωνή",
                           semitone_only=semitone_only)
    return score_to_musicxml(score, metrical=False)


def alters(xml: str) -> list[float]:
    """ Collect every <alter> value the score writes, in document order."""
    return [float(a.text) for a in ET.fromstring(xml).iter("alter")]


def names(xml: str) -> list[str]:
    """ Collect every printed <accidental> name the score writes."""
    return [a.text for a in ET.fromstring(xml).iter("accidental")]


# The request this file exists for: with semitone_only nothing but sharps,
# flats and naturals may reach the page.
_plain = xml_for(BENT, semitone_only=True)
_bad_alters = [a for a in alters(_plain) if a != int(a) or abs(a) > 1]
check(not _bad_alters,
      f"semitone-only score wrote non-semitone alterations: {_bad_alters}")

_bad_names = [n for n in names(_plain) if n not in ALLOWED_NAMES]
check(not _bad_names,
      f"semitone-only score printed accidentals outside #/b/natural: {_bad_names}")

# Rounding must go to the NEAREST key rather than dropping the fraction, so a
# note 50 cents sharp of D does not silently become D.
_up = xml_for([6280.0], semitone_only=True)     # 6280c is nearest to MIDI 63
_step = ET.fromstring(_up).find(".//step").text
_alter = alters(_up)
check((_step, _alter) in (("E", [-1.0]), ("D", [1.0])),
      f"6280 cents rounded to {_step} with alter {_alter}, expected the key at MIDI 63")

# Byzantine and Arabic scores must keep the neutral intervals that define
# their modes: switching the flag off has to change the output.
_micro = xml_for(BENT, semitone_only=False)
check(any(a != int(a) for a in alters(_micro)),
      "microtonal score wrote no fractional alteration at all")

# A note already on a piano key is untouched either way.
_exact = xml_for([6200.0, 6400.0, 6900.0], semitone_only=True)
check(all(a == int(a) for a in alters(_exact)),
      "notes already in tune were given fractional alterations")

# The time signature is still stripped, as the rhythm work established.
check(not list(ET.fromstring(_plain).iter("time")),
      "a time signature reappeared in the exported score")


if failures:
    print("FAILED:")
    for f in failures:
        print("  -", f)
    sys.exit(1)

print("OK - accidental spelling verified across 6 cases.")
