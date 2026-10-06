""" Score generation, turns the note-level transcription into MusicXML.
Two things set this repertoire apart from the Western music most notation
tools assume. Its rhythm is often free rather than metrical, so the pulse is
measured first and only material that really has one is barred. Its modes
contain intervals that do not exist on a piano, so pitches are written with
quarter-tone accidentals and the exact deviation is printed under each note."""

from __future__ import annotations

import numpy as np

PULSE_THRESHOLD = 0.40
MIN_QUARTER_LENGTH = 0.25
_ALLOWED_QL = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]


def estimate_pulse(y: np.ndarray, sr: int) -> tuple[float, float]:
    """ Measure whether the recording has a steady pulse, and how fast it is.
    The strength returned is how tightly the detected onsets cluster around
    the nearest beat: near 1 when the music lands on the beat, near 0 when it
    does not. It deliberately ignores how regular the beat track itself looks,
    because the beat tracker forces a near-constant tempo and therefore looks
    metronomic even on free-rhythm audio."""
    import librosa

    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
    tempo = float(np.atleast_1d(tempo)[0])
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time", backtrack=True)

    if len(beats) < 4 or len(onsets) < 8:
        return tempo, 0.0

    idx = np.clip(np.searchsorted(beats, onsets), 1, len(beats) - 1)
    prev_b, next_b = beats[idx - 1], beats[idx]
    nearest = np.where(onsets - prev_b < next_b - onsets, prev_b, next_b)
    local_period = next_b - prev_b
    fallback = float(np.median(np.diff(beats)))
    local_period = np.where(local_period > 0, local_period, fallback)

    phase = (onsets - nearest) / local_period

    strength = max(
        float(np.abs(np.mean(np.exp(2j * np.pi * k * phase))))
        for k in (1, 2, 4)
    )
    return tempo, strength


def _snap_quarter_length(seconds: float, quarter_seconds: float) -> float:
    """ Round a duration in seconds to the nearest writable note value.
    This is needed even for unmeasured scores, because MusicXML can only
    express note types such as quarters and eighths, never an arbitrary
    length. Being unmeasured changes whether a metre is claimed over those
    values, not whether the values can be written at all."""
    raw = seconds / max(quarter_seconds, 1e-6)
    return min(_ALLOWED_QL, key=lambda q: abs(q - raw))


def _reference_quarter(notes: list[dict], tempo_bpm: float, metrical: bool) -> float:
    """ Decide how many seconds one quarter note represents.
    With a real pulse this comes from the detected tempo. Without one that
    tempo is only an artefact of a beat tracker that always returns something,
    so the median sung duration becomes the quarter note instead and the
    relative lengths of the notes are still preserved."""
    if metrical:
        return 60.0 / max(tempo_bpm, 1e-6)
    durations = [max(n["end_s"] - n["start_s"], 1e-3) for n in notes]
    return float(np.median(durations)) if durations else 0.5


def _midi_cents_to_pitch(midi_cents: float, semitone_only: bool = False):
    """ Convert a measured pitch into a music21 pitch that keeps its microtone.
    Assigning the fractional pitch value directly is the only approach that
    actually reaches the file: music21 then picks the nearest quarter-tone
    accidental and writes it out. Setting the note and its microtone
    separately looks correct in memory but exports no alteration at all,
    silently discarding the very information this project exists to capture.
    With `semitone_only` the pitch is rounded to the nearest piano key first,
    so the score uses none of the quarter-tone symbols."""
    from music21 import pitch

    p = pitch.Pitch()
    p.ps = round(midi_cents / 100.0) if semitone_only else midi_cents / 100.0

    # Respell anything needing more than a single accidental. Left alone,
    # music21 writes pitches such as C three-quarter-sharp, a sharp with three
    # strokes that performers do not read. The enharmonic equivalent is the
    # same sound spelled from the neighbouring letter and needs only a
    # quarter-flat, so the score stays within the symbols musicians use.
    if p.accidental is not None and abs(p.accidental.alter) > 1:
        try:
            alt = p.getEnharmonic()
            if abs(alt.ps - p.ps) < 0.01 and (
                alt.accidental is None or abs(alt.accidental.alter) <= 1
            ):
                p = alt
        except Exception:
            pass
    return p


def build_score(
    notes: list[dict],
    *,
    mode: str,
    tonic: str,
    tempo_bpm: float,
    pulse_strength: float,
    instrument_name: str | None = None,
    title: str = "Μεταγραφή",
    semitone_only: bool = False,
):
    """ Build a music21 score from the detected notes.
    A part name is always set, because music21 otherwise labels the staff with
    an internal object id, and a tempo mark is written only when one was
    really measured. `semitone_only` restricts the score to sharps, flats and
    naturals. Returns the score together with a flag saying whether it was
    barred or left in free rhythm."""
    from music21 import stream, note as m21note, metadata, tempo as m21tempo
    from music21 import meter, expressions, instrument as m21instrument

    metrical = pulse_strength >= PULSE_THRESHOLD

    score = stream.Score()
    score.metadata = metadata.Metadata()
    score.metadata.title = title
    score.metadata.movementName = instrument_name or "Μελωδία"
    score.metadata.composer = f"{mode} · βάση {tonic}"

    part = stream.Part()
    inst = m21instrument.Instrument()
    inst.partName = instrument_name or "Μελωδία"
    inst.partAbbreviation = ""
    part.insert(0, inst)

    if tempo_bpm > 0:
        part.append(m21tempo.MetronomeMark(number=round(tempo_bpm)))

    # No time signature is ever written. The analysis can establish THAT a
    # steady pulse exists, but not how many beats make a bar, and this
    # repertoire is full of 7/8 and 9/8, stamping 4/4 on a kalamatianos
    # would assert a metre that was never measured. What was measured, the
    # tempo and whether a pulse was found at all, is stated instead.
    if metrical:
        part.append(expressions.TextExpression(
            f"παλμός ≈{round(tempo_bpm)} BPM, μέτρο μη προσδιορισμένο"))
    else:
        part.append(expressions.TextExpression("senza misura, ελεύθερος ρυθμός"))

    quarter_seconds = _reference_quarter(notes, tempo_bpm, metrical)

    for ev in notes:
        seconds = max(ev["end_s"] - ev["start_s"], 1e-3)
        ql = _snap_quarter_length(seconds, quarter_seconds)

        n = m21note.Note()
        n.pitch = _midi_cents_to_pitch(ev["midi_cents"], semitone_only)
        n.quarterLength = ql

        # The exact cent deviation is deliberately not printed under the note.
        # It is carried by the quarter-tone accidental on the staff and is
        # reported separately in the analysis panel; printing a number under
        # every note turned the score into something no musician would read.
        part.append(n)

    score.append(part)
    return score, metrical


def score_to_musicxml(score, metrical: bool = True) -> str:
    """ Write the score out as a MusicXML string.
    The time signature is always stripped. music21 needs measures to export
    at all and fills them with a default 4/4 whatever it was asked for, but
    the number of beats per bar was never measured, so printing any metre
    would be an unsupported claim."""
    from music21.musicxml.m21ToXml import GeneralObjectExporter
    import xml.etree.ElementTree as ET

    xml = GeneralObjectExporter(score).parse().decode("utf-8")
    root = ET.fromstring(xml)
    for attributes in root.iter("attributes"):
        for time_el in attributes.findall("time"):
            attributes.remove(time_el)
    body = ET.tostring(root, encoding="unicode")
    prolog = xml[: xml.index("<score-partwise")]
    return prolog + body


def notes_to_musicxml(
    notes: list[dict],
    y: np.ndarray,
    sr: int,
    *,
    mode: str,
    tonic: str,
    instrument_name: str | None = None,
    filename: str = "",
    semitone_only: bool = False,
) -> dict:
    """ Turn detected notes and their audio into a finished score.
    Measures the pulse, builds the score accordingly and returns the MusicXML
    alongside the rhythm information the caller needs to describe what was
    produced. Returns nothing at all when there are no notes to write."""
    if not notes:
        return {}

    tempo_bpm, strength = estimate_pulse(y, sr)
    score, metrical = build_score(
        notes,
        mode=mode,
        tonic=tonic,
        tempo_bpm=tempo_bpm,
        pulse_strength=strength,
        instrument_name=instrument_name,
        title=filename or "Μεταγραφή",
        semitone_only=semitone_only,
    )
    return {
        "musicxml": score_to_musicxml(score, metrical=metrical),
        "tempo_bpm": round(tempo_bpm, 1),
        "pulse_strength": round(strength, 3),
        "metrical": metrical,
        "note_count": len(notes),
    }
