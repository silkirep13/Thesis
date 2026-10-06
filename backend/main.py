from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from analysis import analyze_audio

app = FastAPI(title="Byzantine & Eastern Music Analysis API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}


class ModeInfo(BaseModel):
    tradition: str
    alt_name: str | None = None
    description: str


class NoteEvent(BaseModel):
    start_s: float
    end_s: float
    degree: int
    cents_from_tonic: float
    deviation_cents: float
    accidental: str


class AnalysisResult(BaseModel):
    filename: str
    file_size_kb: float
    detected_mode: str
    tonic: str
    confidence: float
    mean_pitch_hz: float
    pitch_range_cents: int
    microtonal_deviations: list[str]
    microtonal_reliable: bool
    detected_instruments: list[str]
    duration_seconds: float
    voiced_frames: int
    total_frames: int
    engine: str
    status: str
    pitch_times: list[float]
    pitch_contour: list[float | None]
    tonic_hz: float | None
    mode_info: ModeInfo | None
    notation: list[NoteEvent]
    genus: str | None
    tie_group_size: int
    melody_instrument: str | None
    melody_instrument_confidence: float
    musicxml: str | None
    tempo_bpm: float | None
    pulse_strength: float | None
    metrical: bool | None


@app.get("/health")
def health():
    """ Report that the service is running.
    Used to confirm the backend started correctly before the frontend tries
    to send it an audio file."""
    return {"status": "ok", "service": "music-analysis-api"}


ALLOWED_TRADITIONS = {"byzantine", "greek", "cypriot", "arabic"}


@app.post("/api/analyze", response_model=AnalysisResult)
async def analyze(
    file: UploadFile = File(...),
    tradition: str = Form("byzantine"),
):
    """ Accept an uploaded recording and return its full analysis.
    Rejects unsupported file types and unknown traditions before doing any
    work, then hands the audio to the analysis pipeline and returns the mode,
    the instrument and the generated score together."""
    ext = ("." + file.filename.rsplit(".", 1)[-1].lower()) if "." in file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")
    if tradition not in ALLOWED_TRADITIONS:
        raise HTTPException(status_code=400, detail=f"Unknown tradition: {tradition}")

    contents = await file.read()
    result = analyze_audio(contents, file.filename, tradition=tradition)
    return result
