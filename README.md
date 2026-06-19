 Byzantine & Eastern Music Analysis — MIR Thesis

A web-based Music Information Retrieval tool for analysing Byzantine, Greek Folk, Cypriot, and Arabic Maqam recordings. The system detects modal structure, tonic, microtonal deviations from 12-TET, and present instruments using a multi-stage DSP/ML pipeline.

## Features

- **Modal detection** across 4 musical traditions (22 scale templates)
- **Microtonal analysis** — cent-level deviation from equal temperament (p10/p25/p50/p75/p90)
- **Pitch contour visualisation** — CREPE pitch track over time with tonic reference line
- **Source separation** via Demucs htdemucs (vocals / melody / bass / drums stems)
- **Instrument detection** — tradition-aware labels per Demucs stem energy
- **Tradition selector** — restricts mode templates to the correct scale family, eliminating cross-tradition confusion

### Supported traditions

| Tradition | Modes / Maqamat |
|---|---|
| Byzantine | Octoechos (8 modes) + Chromatic Hard/Soft + Enharmonic |
| Greek Folk | Greek Minor (Minore), Greek Hijaz |
| Cypriot | Cypriot Pentachord |
| Arabic Maqam | Rast, Bayati, Hijaz, Nahawand, Saba, Kurd, Ajam, Jiharkah |

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.11, FastAPI |
| Pitch tracking | CREPE (CNN, TensorFlow) |
| Source separation | Demucs htdemucs (PyTorch) |
| Audio analysis | librosa, SciPy |
| Frontend | React 19, Vite |
| Visualisation | Recharts |

## Project Structure

```
Thesis/
├── backend/
│   ├── main.py          # FastAPI app & Pydantic models
│   ├── analysis.py      # Full DSP/MIR pipeline
│   └── requirements.txt
└── frontend/
    ├── src/
    │   ├── App.jsx
    │   └── components/
    │       ├── TraditionSelector.jsx
    │       ├── Uploader.jsx
    │       ├── ResultsPanel.jsx
    │       └── PitchChart.jsx
    └── package.json
```

## Running the Project

### Backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
uvicorn main:app --reload
```

Backend runs on `http://localhost:8000`

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs on `http://localhost:5173` and proxies `/api/*` to the backend.

## Analysis Pipeline

1. **Demucs** separates the audio into 4 stems
2. **HPSS** (librosa) isolates harmonic content from the full mix
3. **Chroma + joint search** (12 tonics × N templates, ending-weighted) detects mode & tonic
4. **CREPE** extracts cent-accurate pitch from the dominant stem (vocal or melodic instrument)
5. **Refinement passes** resolve ambiguous cases:
   - Byzantine kyrios/plagal disambiguation (absolute pitch, finalis-anchored)
   - Arabic neutral-interval disambiguation (Kurd/Bayati, Ajam/Rast)
6. **Microtonal deviation statistics** computed from CREPE pitch distribution
7. **Pitch contour** downsampled and returned for frontend visualisation

## Academic References

- Chrysanthos of Madytos, *Theoretikon Mega* (1832) — 72-moira Byzantine interval system
- d'Erlanger, *La Musique Arabe* (1930–1959) — Arabic maqam corpus
- Kim et al., *CREPE: A Convolutional Representation for Pitch Estimation* (ICASSP 2018)
- Défossez et al., *Hybrid Transformers for Music Source Separation* (ISMIR 2022)
- Serra et al., *compIAM* — Computational analysis of Hindustani, Carnatic & Arab-Andalusian music
